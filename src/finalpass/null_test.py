"""Stem-sum null analysis for Phase 3.

The null pass is intentionally narrow: same-rate, same-channel-count stems are
reduced to a shared program window, summed, and subtracted from the
printmaster. We flag windows whose residual RMS exceeds a dBFS threshold. If
the material appears globally
offset, we fail with an alignment error rather than auto-correcting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from pydantic import BaseModel, ConfigDict

from .analysis_window import prepare_analysis_window
from .audio_io import AudioFile, channel_config_from_count
from .errors import (
    AlignmentError,
    ChannelMismatchError,
    FinalPassError,
    SampleRateMismatchError,
    UnsupportedChannelConfigError,
)
from .models import AnalysisWindow, FlaggedRegion, NullInputFile, NullSummary
from .timecode import samples_to_tc

DEFAULT_NULL_WINDOW_MS = 1000.0
DEFAULT_NULL_HOP_MS = 100.0
DEFAULT_NULL_THRESHOLD_DBFS = -40.0


class NullTunables(BaseModel):
    """Analysis knobs for the stem-sum null pass. Runtime-only; never persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_ms: float = DEFAULT_NULL_WINDOW_MS
    hop_ms: float = DEFAULT_NULL_HOP_MS
    threshold_dbfs: float = DEFAULT_NULL_THRESHOLD_DBFS
NULL_DBFS_FLOOR = -300.0
_NULL_LINEAR_FLOOR = 10 ** (NULL_DBFS_FLOOR / 20.0)
_NULL_FLAG_DETAIL = "Residual exceeded threshold after summing stems against printmaster."
_ALIGNMENT_MAX_LAG_SECONDS = 0.25
_ALIGNMENT_TARGET_POINTS = 12000
_ALIGNMENT_RELATIVE_MARGIN = 1.01


@dataclass(frozen=True)
class NullAnalysis:
    summary: NullSummary
    flags: list[FlaggedRegion]
    analysis_window: AnalysisWindow


@dataclass(frozen=True)
class _WindowResult:
    start_sample: int
    end_sample: int
    residual_rms_dbfs: float


def describe_audio(audio: AudioFile) -> NullInputFile:
    item = NullInputFile(
        path=str(audio.path),
        sample_rate=audio.sample_rate,
        bit_depth=audio.bit_depth,
        channel_count=audio.channel_count,
        channel_config_actual=channel_config_from_count(audio.channel_count),
        duration_seconds=round(audio.duration_seconds, 3),
        source_kind="interleaved",
        source_paths=[str(audio.path)],
        member_legs=[],
        presentation_label=None,
    )
    item._time_reference_samples = audio.time_reference_samples
    return item


def analyze_null(
    printmaster: AudioFile,
    stems: list[AudioFile],
    *,
    fps: float,
    drop_frame: bool = False,
    time_reference_samples: int | None = None,
    window_ms: float = DEFAULT_NULL_WINDOW_MS,
    hop_ms: float = DEFAULT_NULL_HOP_MS,
    threshold_dbfs: float = DEFAULT_NULL_THRESHOLD_DBFS,
) -> NullAnalysis:
    _validate_windowing(window_ms=window_ms, hop_ms=hop_ms)
    _validate_inputs(printmaster, stems)

    prepared_window = prepare_analysis_window(
        [("Printmaster", printmaster), *((f"Stem {index}", stem) for index, stem in enumerate(stems, start=1))],
        fps=fps,
        drop_frame=drop_frame,
        anchor_index=0,
    )
    pm_windowed = prepared_window.inputs[0]
    stem_windowed = list(prepared_window.inputs[1:])

    stem_data = [stem.data for stem in stem_windowed]
    stem_sum = np.sum(np.stack(stem_data, axis=0), axis=0)
    _detect_alignment_error(pm_windowed.data, stem_data, stem_sum, printmaster.sample_rate)

    residual = pm_windowed.data - stem_sum
    window_results = _window_residual_rms(
        residual,
        sample_rate=printmaster.sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
        start_sample_offset=pm_windowed.start_sample,
    )
    flags = _merge_flagged_windows(
        [w for w in window_results if w.residual_rms_dbfs > threshold_dbfs],
        sample_rate=printmaster.sample_rate,
        fps=fps,
        drop_frame=drop_frame,
        time_reference_samples=time_reference_samples,
        threshold_dbfs=threshold_dbfs,
    )
    summary = _summarize_windows(window_results, threshold_dbfs=threshold_dbfs, flagged_regions=len(flags))
    return NullAnalysis(summary=summary, flags=flags, analysis_window=prepared_window.analysis_window)


def _validate_windowing(*, window_ms: float, hop_ms: float) -> None:
    if window_ms <= 0:
        raise FinalPassError(f"window_ms must be positive; got {window_ms}.")
    if hop_ms <= 0:
        raise FinalPassError(f"hop_ms must be positive; got {hop_ms}.")


def _validate_inputs(printmaster: AudioFile, stems: list[AudioFile]) -> None:
    inputs = [printmaster, *stems]
    sample_rates = {audio.sample_rate for audio in inputs}
    if len(sample_rates) > 1:
        detail = ", ".join(f"{audio.path.name}={audio.sample_rate}Hz" for audio in inputs)
        raise SampleRateMismatchError(f"Null inputs must share one sample rate: {detail}")

    channel_counts = {audio.channel_count for audio in inputs}
    if len(channel_counts) > 1:
        detail = ", ".join(f"{audio.path.name}={audio.channel_count}ch" for audio in inputs)
        raise ChannelMismatchError(f"Null inputs must share one channel count: {detail}")

    for audio in inputs:
        if channel_config_from_count(audio.channel_count) is None:
            raise UnsupportedChannelConfigError(
                f"{audio.path.name}: unsupported channel count {audio.channel_count}. "
                "FinalPass v0.1 supports mono, stereo, 5.1, and 7.1 only."
            )


def _detect_alignment_error(printmaster: np.ndarray, stems: list[np.ndarray], stem_sum: np.ndarray, sample_rate: int) -> None:
    lag = _estimate_offset_samples(printmaster, stem_sum, sample_rate)
    if lag != 0:
        raise AlignmentError(
            f"Detected a constant global offset of approximately {lag} samples between "
            "the printmaster and summed stems. Phase 3 does not auto-align null inputs."
        )
    for index, stem in enumerate(stems, start=1):
        stem_lag = _estimate_offset_samples(printmaster, stem, sample_rate)
        if stem_lag != 0:
            raise AlignmentError(
                f"Detected a constant global offset of approximately {stem_lag} samples "
                f"between the printmaster and stem {index}. Phase 3 does not auto-align null inputs."
            )


def _estimate_offset_samples(printmaster: np.ndarray, stem_sum: np.ndarray, sample_rate: int) -> int:
    n_samples = int(printmaster.shape[0])
    if n_samples < 2:
        return 0

    stride = max(1, n_samples // _ALIGNMENT_TARGET_POINTS)
    pm = np.mean(printmaster, axis=1)[::stride]
    summed = np.mean(stem_sum, axis=1)[::stride]
    pm = pm - float(np.mean(pm))
    summed = summed - float(np.mean(summed))

    pm_norm = float(np.linalg.norm(pm))
    summed_norm = float(np.linalg.norm(summed))
    if pm_norm == 0.0 or summed_norm == 0.0:
        return 0

    corr = correlate(pm / pm_norm, summed / summed_norm, mode="full", method="fft")
    lags = np.arange(-len(summed) + 1, len(pm))
    max_lag = max(1, int(round(min(_ALIGNMENT_MAX_LAG_SECONDS * sample_rate, n_samples - 1) / stride)))
    mask = np.abs(lags) <= max_lag
    corr = corr[mask]
    lags = lags[mask]

    if corr.size == 0:
        return 0

    best_idx = int(np.argmax(np.abs(corr)))
    best_lag = int(lags[best_idx])
    zero_idx = int(np.where(lags == 0)[0][0])
    zero_score = float(abs(corr[zero_idx]))
    best_score = float(abs(corr[best_idx]))

    if best_lag == 0:
        return 0
    if best_score <= zero_score * _ALIGNMENT_RELATIVE_MARGIN:
        return 0

    estimated_samples = best_lag * stride
    return estimated_samples if abs(estimated_samples) >= stride else 0


def _window_residual_rms(
    residual: np.ndarray,
    *,
    sample_rate: int,
    window_ms: float,
    hop_ms: float,
    start_sample_offset: int = 0,
) -> list[_WindowResult]:
    n_samples = int(residual.shape[0])
    if n_samples <= 0:
        return []
    window_samples = max(1, int(round(window_ms * sample_rate / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    windows: list[tuple[int, int]]
    if n_samples <= window_samples:
        windows = [(0, n_samples)]
    else:
        windows = [
            (start, start + window_samples)
            for start in range(0, n_samples - window_samples + 1, hop_samples)
        ]

    out: list[_WindowResult] = []
    for start, end in windows:
        block = residual[start:end, :]
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        dbfs = 20.0 * np.log10(max(rms, _NULL_LINEAR_FLOOR))
        out.append(_WindowResult(
            start_sample=start + start_sample_offset,
            end_sample=end + start_sample_offset,
            residual_rms_dbfs=float(dbfs),
        ))
    return out


def _summarize_windows(
    window_results: list[_WindowResult],
    *,
    threshold_dbfs: float,
    flagged_regions: int,
) -> NullSummary:
    if not window_results:
        return NullSummary(
            windows_total=0,
            windows_flagged=0,
            flagged_regions=flagged_regions,
            max_residual_rms_dbfs=NULL_DBFS_FLOOR,
            mean_residual_rms_dbfs=NULL_DBFS_FLOOR,
        )

    dbfs_values = [w.residual_rms_dbfs for w in window_results]
    return NullSummary(
        windows_total=len(window_results),
        windows_flagged=sum(1 for w in window_results if w.residual_rms_dbfs > threshold_dbfs),
        flagged_regions=flagged_regions,
        max_residual_rms_dbfs=max(dbfs_values),
        mean_residual_rms_dbfs=float(np.mean(dbfs_values)),
    )


def _merge_flagged_windows(
    windows: list[_WindowResult],
    *,
    sample_rate: int,
    fps: float,
    drop_frame: bool,
    time_reference_samples: int | None,
    threshold_dbfs: float,
) -> list[FlaggedRegion]:
    if not windows:
        return []

    merged: list[dict[str, float | int]] = []
    current = {
        "start_sample": windows[0].start_sample,
        "end_sample": windows[0].end_sample,
        "value": windows[0].residual_rms_dbfs,
    }
    for window in windows[1:]:
        if window.start_sample <= int(current["end_sample"]):
            current["end_sample"] = max(int(current["end_sample"]), window.end_sample)
            current["value"] = max(float(current["value"]), window.residual_rms_dbfs)
            continue
        merged.append(current)
        current = {
            "start_sample": window.start_sample,
            "end_sample": window.end_sample,
            "value": window.residual_rms_dbfs,
        }
    merged.append(current)

    return [
        FlaggedRegion(
            code="NULL",
            metric="residual_rms_dbfs",
            value=float(region["value"]),
            threshold=threshold_dbfs,
            start_sample=int(region["start_sample"]),
            end_sample=int(region["end_sample"]),
            start_tc=samples_to_tc(
                int(region["start_sample"]),
                sample_rate,
                fps,
                start_time_reference_samples=time_reference_samples,
                drop_frame=drop_frame,
            ),
            end_tc=samples_to_tc(
                int(region["end_sample"]),
                sample_rate,
                fps,
                start_time_reference_samples=time_reference_samples,
                drop_frame=drop_frame,
            ),
            duration_seconds=(int(region["end_sample"]) - int(region["start_sample"])) / float(sample_rate),
            detail=_NULL_FLAG_DETAIL,
        )
        for region in merged
    ]
