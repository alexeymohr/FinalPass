"""Sample-index and frame-rate helpers.

FinalPass keeps all sample/time math in one place. Phase 6 refines the helper
layer so AAF placement uses exact rational edit rates for common fractional
frame rates such as 23.976 and 29.97. Phase 8T adds drop-frame counting.

Drop-frame timecode follows SMPTE ST 12-1: it is defined only for the
30000/1001 and 60000/1001 frame-rate families (29.97 and 59.94). Frame numbers
00 and 01 (00-03 for the 60 family) are skipped at the start of every minute
except minutes divisible by ten. Drop-frame strings use the semicolon
convention (``HH:MM:SS;FF``); non-drop strings keep colons throughout. The
counting mode changes only how frame counts map to labels — sample↔edit-unit
math always uses the exact rational edit rate and is identical in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class FrameRateInfo:
    edit_rate: Fraction
    nominal_fps: int
    drop_frame: bool = False


_COMMON_FRAME_RATES: tuple[tuple[float, Fraction, int], ...] = (
    (23.976, Fraction(24000, 1001), 24),
    (23.98, Fraction(24000, 1001), 24),
    (24.0, Fraction(24, 1), 24),
    (25.0, Fraction(25, 1), 25),
    (29.97, Fraction(30000, 1001), 30),
    (30.0, Fraction(30, 1), 30),
    (47.952, Fraction(48000, 1001), 48),
    (48.0, Fraction(48, 1), 48),
    (50.0, Fraction(50, 1), 50),
    (59.94, Fraction(60000, 1001), 60),
    (60.0, Fraction(60, 1), 60),
    (119.88, Fraction(120000, 1001), 120),
    (120.0, Fraction(120, 1), 120),
)
_FPS_TOLERANCE = 0.001
# ST 12-1 defines drop-frame counting only for these nominal rates.
_DROP_FRAMES_PER_MINUTE = {30: 2, 60: 4}


def frame_rate_info(fps: float, *, drop_frame: bool = False) -> FrameRateInfo:
    """Return the exact edit rate and nominal timecode rate for ``fps``."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    info: FrameRateInfo | None = None
    for candidate, edit_rate, nominal_fps in _COMMON_FRAME_RATES:
        if abs(fps - candidate) <= _FPS_TOLERANCE:
            info = FrameRateInfo(edit_rate=edit_rate, nominal_fps=nominal_fps)
            break

    if info is None:
        rounded = round(fps)
        if abs(fps - rounded) <= _FPS_TOLERANCE:
            info = FrameRateInfo(edit_rate=Fraction(int(rounded), 1), nominal_fps=int(rounded))
        else:
            exact_rate = Fraction(str(fps)).limit_denominator(1_000_000)
            nominal_fps = max(1, int(round(float(exact_rate))))
            info = FrameRateInfo(edit_rate=exact_rate, nominal_fps=nominal_fps)

    if not drop_frame:
        return info
    if info.edit_rate.denominator != 1001 or info.nominal_fps not in _DROP_FRAMES_PER_MINUTE:
        raise ValueError(
            f"drop-frame timecode is not defined for {fps} fps; "
            "SMPTE ST 12-1 defines drop-frame only for 29.97 and 59.94."
        )
    return FrameRateInfo(edit_rate=info.edit_rate, nominal_fps=info.nominal_fps, drop_frame=True)


def sample_to_edit_units(
    sample_index: int,
    sample_rate: int,
    fps: float,
    *,
    start_time_reference_samples: int | None = None,
) -> int:
    """Convert a sample index to an integer edit-unit position."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    absolute_sample_index = sample_index + (start_time_reference_samples or 0)
    info = frame_rate_info(fps)
    value = Fraction(
        absolute_sample_index * info.edit_rate.numerator,
        sample_rate * info.edit_rate.denominator,
    )
    return _round_fraction(value)


def tc_to_frames(timecode: str, fps: float, *, drop_frame: bool = False) -> int:
    """Convert an ``HH:MM:SS:FF`` / ``HH:MM:SS;FF`` timecode string to edit units."""
    info = frame_rate_info(fps, drop_frame=drop_frame)
    hh, mm, ss, ff = _parse_timecode(timecode)
    if ff >= info.nominal_fps:
        raise ValueError(f"timecode frame field {ff} is invalid for fps={fps}")
    base = ((hh * 60 + mm) * 60 + ss) * info.nominal_fps + ff
    if not drop_frame:
        return base
    dropped_per_minute = _DROP_FRAMES_PER_MINUTE[info.nominal_fps]
    if ss == 0 and mm % 10 != 0 and ff < dropped_per_minute:
        raise ValueError(
            f"timecode {timecode!r} does not exist in drop-frame counting; "
            f"frame numbers 00-{dropped_per_minute - 1:02d} are skipped at this minute."
        )
    total_minutes = hh * 60 + mm
    return base - dropped_per_minute * (total_minutes - total_minutes // 10)


def tc_to_sample_start(timecode: str, sample_rate: int, fps: float, *, drop_frame: bool = False) -> int:
    """Return the first sample at or after the given timecode."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    total_frames = tc_to_frames(timecode, fps, drop_frame=drop_frame)
    info = frame_rate_info(fps)
    value = Fraction(
        total_frames * sample_rate * info.edit_rate.denominator,
        info.edit_rate.numerator,
    )
    return _ceil_fraction(value)


def samples_to_tc(
    sample_index: int,
    sample_rate: int,
    fps: float,
    *,
    start_time_reference_samples: int | None = None,
    drop_frame: bool = False,
) -> str:
    """Convert a sample index to an ``HH:MM:SS:FF`` / ``HH:MM:SS;FF`` timecode string."""
    total_frames = sample_to_edit_units(
        sample_index,
        sample_rate,
        fps,
        start_time_reference_samples=start_time_reference_samples,
    )
    return frames_to_tc(total_frames, fps, drop_frame=drop_frame)


def frames_to_tc(total_frames: int, fps: float, *, drop_frame: bool = False) -> str:
    """Format edit units as an ``HH:MM:SS:FF`` / ``HH:MM:SS;FF`` timecode string."""
    info = frame_rate_info(fps, drop_frame=drop_frame)
    frames_per_second = info.nominal_fps

    if drop_frame:
        dropped_per_minute = _DROP_FRAMES_PER_MINUTE[info.nominal_fps]
        frames_per_nondrop_minute = frames_per_second * 60
        frames_per_drop_minute = frames_per_nondrop_minute - dropped_per_minute
        frames_per_ten_minutes = frames_per_nondrop_minute + 9 * frames_per_drop_minute

        ten_minute_blocks = total_frames // frames_per_ten_minutes
        rem = total_frames - ten_minute_blocks * frames_per_ten_minutes
        if rem < frames_per_nondrop_minute:
            total_minutes = ten_minute_blocks * 10
            frame_in_minute = rem
        else:
            rem -= frames_per_nondrop_minute
            total_minutes = ten_minute_blocks * 10 + 1 + rem // frames_per_drop_minute
            # Frame numbers in a dropped minute start at the first kept number.
            frame_in_minute = rem % frames_per_drop_minute + dropped_per_minute
        hh, mm = divmod(total_minutes, 60)
        ss, ff = divmod(frame_in_minute, frames_per_second)
        return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"

    frames_per_minute = frames_per_second * 60
    frames_per_hour = frames_per_minute * 60

    hh = total_frames // frames_per_hour
    rem = total_frames - hh * frames_per_hour
    mm = rem // frames_per_minute
    rem -= mm * frames_per_minute
    ss = rem // frames_per_second
    ff = rem - ss * frames_per_second
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _parse_timecode(timecode: str) -> tuple[int, int, int, int]:
    # Parsing accepts either frame-field separator; the drop_frame argument of
    # the caller, not the separator, selects the counting mode.
    if ";" in timecode:
        head, _, frame_field = timecode.rpartition(";")
        if ";" in head:
            raise ValueError(f"invalid timecode {timecode!r}; expected HH:MM:SS:FF or HH:MM:SS;FF")
        parts = [*head.split(":"), frame_field]
    else:
        parts = timecode.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid timecode {timecode!r}; expected HH:MM:SS:FF or HH:MM:SS;FF")
    try:
        hh, mm, ss, ff = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid timecode {timecode!r}; expected numeric HH:MM:SS:FF or HH:MM:SS;FF") from exc
    if min(hh, mm, ss, ff) < 0:
        raise ValueError(f"invalid timecode {timecode!r}; fields must be non-negative")
    if mm >= 60 or ss >= 60:
        raise ValueError(f"invalid timecode {timecode!r}; minutes and seconds must be < 60")
    return hh, mm, ss, ff


def _ceil_fraction(value: Fraction) -> int:
    if value.denominator == 1:
        return value.numerator
    if value >= 0:
        return (value.numerator + value.denominator - 1) // value.denominator
    return value.numerator // value.denominator


def _round_fraction(value: Fraction) -> int:
    if value >= 0:
        return (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    return -_round_fraction(-value)
