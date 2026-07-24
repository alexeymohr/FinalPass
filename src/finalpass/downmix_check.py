"""Phase 8B downmix-consistency analysis.

When a delivery carries both a 2.0 and a 5.1/7.1 of the same program, this
pass derives a plain in-phase fold-down from the surround master and compares
it against the delivered stereo: integrated-loudness delta, windowed
similarity, and the delivered stereo's own mono compatibility.

The derived signal is a **Lo/Ro-style fold-down** and nothing more. It is not
an Lt/Rt matrix encode, and FinalPass does not emulate any matrix decoder —
there is no phase-shift network, no surround steering, and no attempt to
reproduce a proprietary process. The default gains are the published
derivation values (center and surrounds at -3 dB, LFE omitted), which is the
industry's stated reference point for this relationship.

A discrete stereo mix that is not a mechanical fold-down is normal and passes
at the default thresholds. This check exists to catch the wrong episode, a
gross level offset, sync drift between layouts, missing elements, and
phase-hostile stereo — not to demand fold-down identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.signal import butter, sosfiltfilt

from .analysis_common import (
    estimate_offset_samples,
    signed_correlation,
    to_dbfs,
    rms,
    window_bounds,
    window_rms_dbfs,
)
from .analysis_window import prepare_analysis_window
from .audio_io import AudioFile, channel_config_from_count
from .errors import (
    AlignmentError,
    FinalPassError,
    SampleRateMismatchError,
    UnsupportedChannelConfigError,
)
from .loudness import measure
from .models import AnalysisWindow, DownmixSummary, FlaggedRegion
from .timecode import TimecodeMode, samples_to_tc

DEFAULT_DOWNMIX_CENTER_DB = -3.0
DEFAULT_DOWNMIX_SURROUND_DB = -3.0
DEFAULT_DOWNMIX_LFE_LOWPASS_HZ = 120.0
DEFAULT_DOWNMIX_WINDOW_MS = 1000.0
DEFAULT_DOWNMIX_HOP_MS = 250.0
DEFAULT_DOWNMIX_ACTIVITY_DBFS = -60.0
DEFAULT_DOWNMIX_SIMILARITY_CORR = 0.60
DEFAULT_DOWNMIX_MONO_CORR = -0.20
DEFAULT_DOWNMIX_LOUDNESS_DELTA_LU = 2.0

ANALYSIS_SIGNAL_NAME = "loro_fold_down"
REASON_INSUFFICIENT_CONTENT = "insufficient_active_content"
NOTE_MATRIX_ENCODED_STEREO = "delivered_stereo_is_matrix_encoded"

# SMPTE slot order; the fold-down reads legs by position, never by remapping.
_L, _R, _C, _LFE, _LS, _RS, _LSS, _RSS = range(8)
_SUPPORTED_SURROUND_COUNTS = (6, 8)
_ALIGNMENT_RELATIVE_MARGIN = 1.05
_ALIGNMENT_ABSOLUTE_MIN = 0.25
_SIMILARITY_DETAIL = "delivered 2.0 diverges from the Lo/Ro fold-down of the surround master"
_MONO_DETAIL = "delivered 2.0 is phase-hostile; this passage largely cancels in mono"


class DownmixTunables(BaseModel):
    """Analysis knobs for the downmix-consistency pass. Runtime-only; never persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    center_db: float = DEFAULT_DOWNMIX_CENTER_DB
    surround_db: float = DEFAULT_DOWNMIX_SURROUND_DB
    lfe_db: float | None = None
    lfe_lowpass_hz: float = DEFAULT_DOWNMIX_LFE_LOWPASS_HZ
    window_ms: float = DEFAULT_DOWNMIX_WINDOW_MS
    hop_ms: float = DEFAULT_DOWNMIX_HOP_MS
    activity_dbfs: float = DEFAULT_DOWNMIX_ACTIVITY_DBFS
    similarity_corr: float = DEFAULT_DOWNMIX_SIMILARITY_CORR
    mono_corr: float = DEFAULT_DOWNMIX_MONO_CORR
    loudness_delta_lu: float = DEFAULT_DOWNMIX_LOUDNESS_DELTA_LU


@dataclass(frozen=True)
class DownmixAnalysis:
    summary: DownmixSummary
    flags: list[FlaggedRegion]
    analysis_window: AnalysisWindow
    notes: list[str]
    level_passed: bool


@dataclass(frozen=True)
class _WindowScore:
    start_sample: int
    end_sample: int
    value: float


def analyze_downmix(
    delivered: AudioFile,
    surround: AudioFile,
    *,
    mode: TimecodeMode,
    time_reference_samples: int | None = None,
    presentation_label: str | None = None,
    tunables: DownmixTunables = DownmixTunables(),
) -> DownmixAnalysis:
    """Compare a delivered 2.0 against a fold-down derived from the surround master."""
    _validate_inputs(delivered, surround)

    derived = _derive_fold_down(surround, tunables=tunables)
    prepared_window = prepare_analysis_window(
        [("Delivered 2.0", delivered), ("Derived fold-down", derived)],
        mode=mode,
        anchor_index=0,
    )
    delivered_windowed, derived_windowed = prepared_window.inputs
    delivered_data = delivered_windowed.data
    derived_data = derived_windowed.data

    delivered_mono = np.mean(delivered_data, axis=1)
    derived_mono = np.mean(derived_data, axis=1)
    _detect_alignment_error(delivered_mono, derived_mono, delivered.sample_rate)

    notes: list[str] = []
    if presentation_label is not None and "LTRT" in presentation_label.upper():
        notes.append(NOTE_MATRIX_ENCODED_STEREO)

    delivered_lufs = _integrated_lufs(delivered, delivered_data)
    derived_lufs = _integrated_lufs(derived, derived_data)
    if delivered_lufs is None or derived_lufs is None:
        loudness_delta = None
        level_passed = True
    else:
        loudness_delta = abs(delivered_lufs - derived_lufs)
        level_passed = loudness_delta <= tunables.loudness_delta_lu

    windows = window_bounds(
        sample_count=delivered_data.shape[0],
        sample_rate=delivered.sample_rate,
        window_ms=tunables.window_ms,
        hop_ms=tunables.hop_ms,
    )
    start_offset = delivered_windowed.start_sample

    similarity = _similarity_scores(
        delivered_mono=delivered_mono,
        derived_mono=derived_mono,
        windows=windows,
        activity_dbfs=tunables.activity_dbfs,
        start_offset=start_offset,
    )
    mono = _mono_compatibility_scores(
        delivered_data=delivered_data,
        windows=windows,
        activity_dbfs=tunables.activity_dbfs,
        start_offset=start_offset,
    )

    similarity_flagged = [score for score in similarity if score.value < tunables.similarity_corr]
    mono_flagged = [score for score in mono if score.value < tunables.mono_corr]

    flags = [
        *_merge_scores(
            similarity_flagged,
            metric="downmix_correlation",
            threshold=tunables.similarity_corr,
            detail=_SIMILARITY_DETAIL,
            sample_rate=delivered.sample_rate,
            mode=mode,
            time_reference_samples=time_reference_samples,
        ),
        *_merge_scores(
            mono_flagged,
            metric="stereo_correlation",
            threshold=tunables.mono_corr,
            detail=_MONO_DETAIL,
            sample_rate=delivered.sample_rate,
            mode=mode,
            time_reference_samples=time_reference_samples,
        ),
    ]
    flags.sort(key=lambda flag: (flag.start_sample, flag.metric))

    summary = DownmixSummary(
        delivered_integrated_lufs=delivered_lufs,
        derived_integrated_lufs=derived_lufs,
        loudness_delta_lu=None if loudness_delta is None else round(loudness_delta, 2),
        level_pass=level_passed,
        similarity_windows_total=len(windows),
        similarity_windows_gated_out=len(windows) - len(similarity),
        similarity_windows_analyzed=len(similarity),
        similarity_windows_flagged=len(similarity_flagged),
        similarity_min_corr=min((score.value for score in similarity), default=None),
        similarity_skipped_reason=None if similarity else REASON_INSUFFICIENT_CONTENT,
        mono_windows_analyzed=len(mono),
        mono_windows_flagged=len(mono_flagged),
        mono_min_corr=min((score.value for score in mono), default=None),
        flagged_regions=len(flags),
    )
    return DownmixAnalysis(
        summary=summary,
        flags=flags,
        analysis_window=prepared_window.analysis_window,
        notes=notes,
        level_passed=level_passed,
    )


def _validate_inputs(delivered: AudioFile, surround: AudioFile) -> None:
    if delivered.sample_rate != surround.sample_rate:
        raise SampleRateMismatchError(
            "Downmix inputs must share one sample rate: "
            f"{delivered.path.name}={delivered.sample_rate}Hz, "
            f"{surround.path.name}={surround.sample_rate}Hz"
        )
    if delivered.channel_count != 2:
        actual = channel_config_from_count(delivered.channel_count) or f"{delivered.channel_count}ch"
        raise UnsupportedChannelConfigError(
            f"{delivered.path.name}: the delivered input must be stereo; got {actual}."
        )
    if surround.channel_count not in _SUPPORTED_SURROUND_COUNTS:
        actual = channel_config_from_count(surround.channel_count) or f"{surround.channel_count}ch"
        raise UnsupportedChannelConfigError(
            f"{surround.path.name}: the surround input must be 5.1 or 7.1; got {actual}."
        )


def _derive_fold_down(surround: AudioFile, *, tunables: DownmixTunables) -> AudioFile:
    """Build the Lo/Ro-style fold-down in memory. No clipping, no normalization."""
    data = surround.data
    center_gain = 10.0 ** (tunables.center_db / 20.0)
    surround_gain = 10.0 ** (tunables.surround_db / 20.0)

    left = data[:, _L] + center_gain * data[:, _C] + surround_gain * data[:, _LS]
    right = data[:, _R] + center_gain * data[:, _C] + surround_gain * data[:, _RS]
    if surround.channel_count == 8:
        left = left + surround_gain * data[:, _LSS]
        right = right + surround_gain * data[:, _RSS]

    if tunables.lfe_db is not None:
        lfe_gain = 10.0 ** (tunables.lfe_db / 20.0)
        lfe = _low_pass(
            data[:, _LFE],
            sample_rate=surround.sample_rate,
            cutoff_hz=tunables.lfe_lowpass_hz,
        )
        left = left + lfe_gain * lfe
        right = right + lfe_gain * lfe

    derived = np.column_stack([left, right])
    return AudioFile(
        path=surround.path,
        data=derived,
        sample_rate=surround.sample_rate,
        bit_depth=surround.bit_depth,
        channel_count=2,
        duration_seconds=surround.duration_seconds,
        # Inherit the surround master's start so the shared program-window
        # logic lines the two signals up exactly as it does for null and M&E.
        time_reference_samples=surround.time_reference_samples,
    )


def _low_pass(signal: np.ndarray, *, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist or signal.shape[0] < 64:
        return signal
    sos = butter(4, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, signal)


def _detect_alignment_error(delivered: np.ndarray, derived: np.ndarray, sample_rate: int) -> None:
    lag = estimate_offset_samples(
        delivered,
        derived,
        sample_rate=sample_rate,
        relative_margin=_ALIGNMENT_RELATIVE_MARGIN,
        absolute_min=_ALIGNMENT_ABSOLUTE_MIN,
    )
    if lag != 0:
        raise AlignmentError(
            f"Detected a constant global offset of approximately {lag} samples between "
            "the delivered 2.0 and the surround fold-down. FinalPass does not auto-align "
            "downmix inputs."
        )


def _integrated_lufs(source: AudioFile, data: np.ndarray) -> float | None:
    """Integrated loudness of a windowed signal, via the existing measurement path."""
    windowed = AudioFile(
        path=source.path,
        data=data,
        sample_rate=source.sample_rate,
        bit_depth=source.bit_depth,
        channel_count=data.shape[1],
        duration_seconds=data.shape[0] / float(source.sample_rate),
        time_reference_samples=source.time_reference_samples,
    )
    return measure(windowed).integrated_lufs


def _similarity_scores(
    *,
    delivered_mono: np.ndarray,
    derived_mono: np.ndarray,
    windows: list[tuple[int, int]],
    activity_dbfs: float,
    start_offset: int,
) -> list[_WindowScore]:
    scores: list[_WindowScore] = []
    for start, end in windows:
        a = delivered_mono[start:end]
        b = derived_mono[start:end]
        if to_dbfs(rms(a)) <= activity_dbfs or to_dbfs(rms(b)) <= activity_dbfs:
            continue
        scores.append(_WindowScore(
            start_sample=start + start_offset,
            end_sample=end + start_offset,
            value=signed_correlation(a, b),
        ))
    return scores


def _mono_compatibility_scores(
    *,
    delivered_data: np.ndarray,
    windows: list[tuple[int, int]],
    activity_dbfs: float,
    start_offset: int,
) -> list[_WindowScore]:
    levels = window_rms_dbfs(delivered_data, windows=windows)
    scores: list[_WindowScore] = []
    for index, (start, end) in enumerate(windows):
        if levels[index, 0] <= activity_dbfs or levels[index, 1] <= activity_dbfs:
            continue
        scores.append(_WindowScore(
            start_sample=start + start_offset,
            end_sample=end + start_offset,
            value=signed_correlation(delivered_data[start:end, 0], delivered_data[start:end, 1]),
        ))
    return scores


def _merge_scores(
    scores: list[_WindowScore],
    *,
    metric: str,
    threshold: float,
    detail: str,
    sample_rate: int,
    mode: TimecodeMode,
    time_reference_samples: int | None,
) -> list[FlaggedRegion]:
    """Merge overlapping/touching flagged windows, keeping the worst value."""
    if not scores:
        return []

    merged: list[dict[str, float | int]] = []
    current = {
        "start_sample": scores[0].start_sample,
        "end_sample": scores[0].end_sample,
        "value": scores[0].value,
    }
    for score in scores[1:]:
        if score.start_sample <= int(current["end_sample"]):
            current["end_sample"] = max(int(current["end_sample"]), score.end_sample)
            current["value"] = min(float(current["value"]), score.value)
            continue
        merged.append(current)
        current = {
            "start_sample": score.start_sample,
            "end_sample": score.end_sample,
            "value": score.value,
        }
    merged.append(current)

    return [
        FlaggedRegion(
            code="DOWNMIX",
            metric=metric,
            value=round(float(region["value"]), 3),
            threshold=threshold,
            start_sample=int(region["start_sample"]),
            end_sample=int(region["end_sample"]),
            start_tc=samples_to_tc(
                int(region["start_sample"]),
                sample_rate,
                mode,
                start_time_reference_samples=time_reference_samples,
            ),
            end_tc=samples_to_tc(
                int(region["end_sample"]),
                sample_rate,
                mode,
                start_time_reference_samples=time_reference_samples,
            ),
            duration_seconds=(int(region["end_sample"]) - int(region["start_sample"])) / float(sample_rate),
            detail=detail,
        )
        for region in merged
    ]


def validate_windowing(*, window_ms: float, hop_ms: float) -> None:
    if window_ms <= 0:
        raise FinalPassError(f"window_ms must be positive; got {window_ms}.")
    if hop_ms <= 0:
        raise FinalPassError(f"hop_ms must be positive; got {hop_ms}.")
