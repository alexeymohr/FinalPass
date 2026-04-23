"""Sample-index and frame-rate helpers.

FinalPass keeps all sample/time math in one place. Phase 6 refines the helper
layer so AAF placement uses exact rational edit rates for common fractional
frame rates such as 23.976 and 29.97.

The CLI currently exposes only ``--fps`` and does not distinguish drop-frame
from non-drop-frame timecode as a separate user choice. FinalPass therefore
formats persisted timecode strings in non-drop notation while still placing AAF
markers against the exact edit rate derived from ``--fps``.
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
)
_FPS_TOLERANCE = 0.001


def frame_rate_info(fps: float) -> FrameRateInfo:
    """Return the exact edit rate and nominal timecode rate for ``fps``."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    for candidate, edit_rate, nominal_fps in _COMMON_FRAME_RATES:
        if abs(fps - candidate) <= _FPS_TOLERANCE:
            return FrameRateInfo(edit_rate=edit_rate, nominal_fps=nominal_fps)

    rounded = round(fps)
    if abs(fps - rounded) <= _FPS_TOLERANCE:
        return FrameRateInfo(edit_rate=Fraction(int(rounded), 1), nominal_fps=int(rounded))

    exact_rate = Fraction(str(fps)).limit_denominator(1_000_000)
    nominal_fps = max(1, int(round(float(exact_rate))))
    return FrameRateInfo(edit_rate=exact_rate, nominal_fps=nominal_fps)


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


def samples_to_tc(
    sample_index: int,
    sample_rate: int,
    fps: float,
    *,
    start_time_reference_samples: int | None = None,
) -> str:
    """Convert a sample index to an ``HH:MM:SS:FF`` timecode string."""
    total_frames = sample_to_edit_units(
        sample_index,
        sample_rate,
        fps,
        start_time_reference_samples=start_time_reference_samples,
    )
    return frames_to_tc(total_frames, fps)


def frames_to_tc(total_frames: int, fps: float) -> str:
    """Format edit units as a non-drop ``HH:MM:SS:FF`` timecode string."""
    info = frame_rate_info(fps)
    frames_per_second = info.nominal_fps
    frames_per_minute = frames_per_second * 60
    frames_per_hour = frames_per_minute * 60

    hh = total_frames // frames_per_hour
    rem = total_frames - hh * frames_per_hour
    mm = rem // frames_per_minute
    rem -= mm * frames_per_minute
    ss = rem // frames_per_second
    ff = rem - ss * frames_per_second
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _round_fraction(value: Fraction) -> int:
    if value >= 0:
        return (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    return -_round_fraction(-value)
