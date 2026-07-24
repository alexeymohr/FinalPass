"""M&E dialogue-bleed analysis for Phase 4.

The M&E pass is a conservative DSP-only heuristic: downmix both files to one
speech-band analysis signal, gate out windows without meaningful DX energy, and
flag windows whose correlation and coherence indicate likely dialogue leakage.
Detected fixed offsets remain hard failures rather than auto-corrected input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.signal import butter, coherence, correlate, sosfiltfilt

from .analysis_window import prepare_analysis_window
from .audio_io import AudioFile, channel_config_from_count
from .errors import (
    AlignmentError,
    FinalPassError,
    SampleRateMismatchError,
    UnsupportedChannelConfigError,
)
from .models import AnalysisInputFile, AnalysisWindow, FlaggedRegion, MECheckSummary
from .timecode import TimecodeMode, samples_to_tc

DEFAULT_ME_WINDOW_MS = 500.0
DEFAULT_ME_HOP_MS = 100.0
DEFAULT_ME_BAND_LOW_HZ = 200.0
DEFAULT_ME_BAND_HIGH_HZ = 4000.0
DEFAULT_ME_CORR_THRESHOLD = 0.65
DEFAULT_ME_COHERENCE_THRESHOLD = 0.60
DEFAULT_ME_DX_GATE_DBFS = -45.0
DEFAULT_ME_ME_FLOOR_DBFS = -60.0
ME_DBFS_FLOOR = -300.0
ANALYSIS_SIGNAL_NAME = "mono_downmix_excluding_lfe"

_ME_LINEAR_FLOOR = 10 ** (ME_DBFS_FLOOR / 20.0)
_ALIGNMENT_MAX_LAG_SECONDS = 0.25
_ALIGNMENT_TARGET_POINTS = 12000
_ALIGNMENT_RELATIVE_MARGIN = 1.05
_ALIGNMENT_ABSOLUTE_MIN = 0.25


class METunables(BaseModel):
    """Analysis knobs for the M&E dialogue-bleed pass. Runtime-only; never persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_ms: float = DEFAULT_ME_WINDOW_MS
    hop_ms: float = DEFAULT_ME_HOP_MS
    band_low_hz: float = DEFAULT_ME_BAND_LOW_HZ
    band_high_hz: float = DEFAULT_ME_BAND_HIGH_HZ
    corr_threshold: float = DEFAULT_ME_CORR_THRESHOLD
    coherence_threshold: float = DEFAULT_ME_COHERENCE_THRESHOLD
    dx_gate_dbfs: float = DEFAULT_ME_DX_GATE_DBFS
    me_floor_dbfs: float = DEFAULT_ME_ME_FLOOR_DBFS


@dataclass(frozen=True)
class MEAnalysis:
    summary: MECheckSummary
    flags: list[FlaggedRegion]
    analysis_window: AnalysisWindow


@dataclass(frozen=True)
class _WindowResult:
    start_sample: int
    end_sample: int
    dx_band_rms_dbfs: float
    me_band_rms_dbfs: float
    corr_abs: float
    coherence_mean: float
    dialog_bleed_score: float
    gated_out: bool


def describe_audio(audio: AudioFile) -> AnalysisInputFile:
    item = AnalysisInputFile(
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


def analyze_me(
    me_file: AudioFile,
    dx_file: AudioFile,
    *,
    mode: TimecodeMode,
    time_reference_samples: int | None = None,
    window_ms: float = DEFAULT_ME_WINDOW_MS,
    hop_ms: float = DEFAULT_ME_HOP_MS,
    band_low_hz: float = DEFAULT_ME_BAND_LOW_HZ,
    band_high_hz: float = DEFAULT_ME_BAND_HIGH_HZ,
    corr_threshold: float = DEFAULT_ME_CORR_THRESHOLD,
    coherence_threshold: float = DEFAULT_ME_COHERENCE_THRESHOLD,
    dx_gate_dbfs: float = DEFAULT_ME_DX_GATE_DBFS,
    me_floor_dbfs: float = DEFAULT_ME_ME_FLOOR_DBFS,
) -> MEAnalysis:
    _validate_inputs(me_file, dx_file)
    _validate_settings(
        sample_rate=me_file.sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        corr_threshold=corr_threshold,
        coherence_threshold=coherence_threshold,
    )
    prepared_window = prepare_analysis_window(
        [("M&E", me_file), ("DX", dx_file)],
        mode=mode,
        anchor_index=0,
    )
    me_windowed, dx_windowed = prepared_window.inputs

    dx_signal = _downmix_to_analysis_signal(dx_windowed.data)
    me_signal = _downmix_to_analysis_signal(me_windowed.data)
    dx_band = _band_limit(dx_signal, sample_rate=dx_file.sample_rate, low_hz=band_low_hz, high_hz=band_high_hz)
    me_band = _band_limit(me_signal, sample_rate=me_file.sample_rate, low_hz=band_low_hz, high_hz=band_high_hz)

    _detect_alignment_error(dx_band, me_band, dx_file.sample_rate)

    windows = _analyze_windows(
        dx_band=dx_band,
        me_band=me_band,
        sample_rate=dx_file.sample_rate,
        window_ms=window_ms,
        hop_ms=hop_ms,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        corr_threshold=corr_threshold,
        coherence_threshold=coherence_threshold,
        dx_gate_dbfs=dx_gate_dbfs,
        me_floor_dbfs=me_floor_dbfs,
        start_sample_offset=me_windowed.start_sample,
    )
    flagged_windows = [
        window
        for window in windows
        if (not window.gated_out)
        and window.me_band_rms_dbfs >= me_floor_dbfs
        and window.dialog_bleed_score >= 1.0
    ]
    flags = _merge_flagged_windows(
        flagged_windows,
        sample_rate=dx_file.sample_rate,
        mode=mode,
        time_reference_samples=time_reference_samples,
    )
    analyzed = [window for window in windows if not window.gated_out]
    summary = MECheckSummary(
        windows_total=len(windows),
        windows_gated_out=sum(1 for window in windows if window.gated_out),
        windows_analyzed=len(analyzed),
        windows_flagged=len(flagged_windows),
        flagged_regions=len(flags),
        max_dialog_bleed_score=max((window.dialog_bleed_score for window in analyzed), default=0.0),
        max_corr_abs=max((window.corr_abs for window in analyzed), default=0.0),
        max_coherence_mean=max((window.coherence_mean for window in analyzed), default=0.0),
    )
    return MEAnalysis(summary=summary, flags=flags, analysis_window=prepared_window.analysis_window)


def _validate_inputs(me_file: AudioFile, dx_file: AudioFile) -> None:
    inputs = [me_file, dx_file]
    sample_rates = {audio.sample_rate for audio in inputs}
    if len(sample_rates) > 1:
        detail = ", ".join(f"{audio.path.name}={audio.sample_rate}Hz" for audio in inputs)
        raise SampleRateMismatchError(f"M&E inputs must share one sample rate: {detail}")

    # The M&E and DX inputs may carry different channel layouts (a mono DX
    # against a stereo M&E is a normal delivery shape). Both signals are
    # reduced to the same mono analysis downmix before any comparison, so only
    # per-file layout support is validated here.
    for audio in inputs:
        if channel_config_from_count(audio.channel_count) is None:
            raise UnsupportedChannelConfigError(
                f"{audio.path.name}: unsupported channel count {audio.channel_count}. "
                "FinalPass v0.1 supports mono, stereo, 5.1, and 7.1 only."
            )


def _validate_settings(
    *,
    sample_rate: int,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
) -> None:
    if window_ms <= 0:
        raise FinalPassError(f"window_ms must be positive; got {window_ms}.")
    if hop_ms <= 0:
        raise FinalPassError(f"hop_ms must be positive; got {hop_ms}.")
    if band_low_hz <= 0:
        raise FinalPassError(f"band_low_hz must be positive; got {band_low_hz}.")
    if band_high_hz <= band_low_hz:
        raise FinalPassError(
            f"band_high_hz must be greater than band_low_hz; got {band_low_hz} .. {band_high_hz}."
        )
    nyquist = sample_rate / 2.0
    if band_high_hz >= nyquist:
        raise FinalPassError(
            f"band_high_hz must be below Nyquist ({nyquist:g} Hz); got {band_high_hz}."
        )
    if corr_threshold <= 0:
        raise FinalPassError(f"corr_threshold must be positive; got {corr_threshold}.")
    if coherence_threshold <= 0:
        raise FinalPassError(
            f"coherence_threshold must be positive; got {coherence_threshold}."
        )


def _downmix_to_analysis_signal(data: np.ndarray) -> np.ndarray:
    channel_count = data.shape[1]
    if channel_count == 1:
        return data[:, 0].astype(np.float64, copy=False)
    if channel_count == 2:
        return np.mean(data, axis=1, dtype=np.float64)
    if channel_count == 6:
        return np.mean(data[:, [0, 1, 2, 4, 5]], axis=1, dtype=np.float64)
    if channel_count == 8:
        return np.mean(data[:, [0, 1, 2, 4, 5, 6, 7]], axis=1, dtype=np.float64)
    raise UnsupportedChannelConfigError(
        f"unsupported channel count {channel_count}. FinalPass v0.1 supports mono, stereo, 5.1, and 7.1 only."
    )


def _band_limit(signal: np.ndarray, *, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    if signal.size < 4:
        return signal.astype(np.float64, copy=True)
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal.astype(np.float64, copy=False), padlen=0)


def _detect_alignment_error(dx_band: np.ndarray, me_band: np.ndarray, sample_rate: int) -> None:
    lag = _estimate_offset_samples(dx_band, me_band, sample_rate)
    if lag != 0:
        raise AlignmentError(
            f"Detected a constant global offset of approximately {lag} samples between "
            "DX and M&E. Phase 4 does not auto-align M&E inputs."
        )


def _estimate_offset_samples(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> int:
    n_samples = int(reference.shape[0])
    if n_samples < 2:
        return 0

    stride = max(1, n_samples // _ALIGNMENT_TARGET_POINTS)
    ref = reference[::stride] - float(np.mean(reference[::stride]))
    cand = candidate[::stride] - float(np.mean(candidate[::stride]))

    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 or cand_norm == 0.0:
        return 0

    corr = correlate(ref / ref_norm, cand / cand_norm, mode="full", method="fft")
    lags = np.arange(-len(cand) + 1, len(ref))
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
    if best_score < _ALIGNMENT_ABSOLUTE_MIN:
        return 0
    if best_score <= zero_score * _ALIGNMENT_RELATIVE_MARGIN:
        return 0

    estimated_samples = best_lag * stride
    return estimated_samples if abs(estimated_samples) >= stride else 0


def _analyze_windows(
    *,
    dx_band: np.ndarray,
    me_band: np.ndarray,
    sample_rate: int,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
    start_sample_offset: int = 0,
) -> list[_WindowResult]:
    n_samples = int(dx_band.shape[0])
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
        dx_window = dx_band[start:end]
        me_window = me_band[start:end]
        dx_rms_dbfs = _rms_dbfs(dx_window)
        me_rms_dbfs = _rms_dbfs(me_window)
        if dx_rms_dbfs < dx_gate_dbfs:
            out.append(
                _WindowResult(
                    start_sample=start,
                    end_sample=end,
                    dx_band_rms_dbfs=dx_rms_dbfs,
                    me_band_rms_dbfs=me_rms_dbfs,
                    corr_abs=0.0,
                    coherence_mean=0.0,
                    dialog_bleed_score=0.0,
                    gated_out=True,
                )
            )
            continue

        corr_abs = _zero_lag_corr_abs(dx_window, me_window)
        coherence_mean = _coherence_mean(
            dx_window,
            me_window,
            sample_rate=sample_rate,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
        )
        dialog_bleed_score = min(
            corr_abs / corr_threshold,
            coherence_mean / coherence_threshold,
        )
        if me_rms_dbfs < me_floor_dbfs:
            dialog_bleed_score = min(dialog_bleed_score, 0.999999)

        out.append(
            _WindowResult(
                start_sample=start,
                end_sample=end,
                dx_band_rms_dbfs=dx_rms_dbfs,
                me_band_rms_dbfs=me_rms_dbfs,
                corr_abs=corr_abs,
                coherence_mean=coherence_mean,
                dialog_bleed_score=dialog_bleed_score,
                gated_out=False,
            )
        )
    if start_sample_offset == 0:
        return out
    return [
        _WindowResult(
            start_sample=window.start_sample + start_sample_offset,
            end_sample=window.end_sample + start_sample_offset,
            dx_band_rms_dbfs=window.dx_band_rms_dbfs,
            me_band_rms_dbfs=window.me_band_rms_dbfs,
            corr_abs=window.corr_abs,
            coherence_mean=window.coherence_mean,
            dialog_bleed_score=window.dialog_bleed_score,
            gated_out=window.gated_out,
        )
        for window in out
    ]


def _rms_dbfs(window: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(window ** 2))) if window.size else 0.0
    return float(20.0 * np.log10(max(rms, _ME_LINEAR_FLOOR)))


def _zero_lag_corr_abs(dx_window: np.ndarray, me_window: np.ndarray) -> float:
    dx_centered = dx_window - float(np.mean(dx_window))
    me_centered = me_window - float(np.mean(me_window))
    dx_norm = float(np.linalg.norm(dx_centered))
    me_norm = float(np.linalg.norm(me_centered))
    if dx_norm == 0.0 or me_norm == 0.0:
        return 0.0
    return float(abs(np.dot(dx_centered, me_centered) / (dx_norm * me_norm)))


def _coherence_mean(
    dx_window: np.ndarray,
    me_window: np.ndarray,
    *,
    sample_rate: int,
    band_low_hz: float,
    band_high_hz: float,
) -> float:
    if len(dx_window) < 8:
        return 0.0
    if float(np.linalg.norm(dx_window)) == 0.0 or float(np.linalg.norm(me_window)) == 0.0:
        return 0.0
    nperseg = min(2048, len(dx_window))
    # A window with silent segments yields zero-power spectra; scipy's
    # coherence divide then produces NaN bins and a RuntimeWarning. Suppress
    # the warning and drop the undefined bins instead of averaging NaNs.
    with np.errstate(divide="ignore", invalid="ignore"):
        freqs, values = coherence(dx_window, me_window, fs=sample_rate, nperseg=nperseg)
    mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
    band = values[mask]
    band = band[np.isfinite(band)]
    if band.size == 0:
        return 0.0
    return float(np.mean(band))


def _merge_flagged_windows(
    windows: list[_WindowResult],
    *,
    sample_rate: int,
    mode: TimecodeMode,
    time_reference_samples: int | None,
) -> list[FlaggedRegion]:
    if not windows:
        return []

    merged: list[dict[str, object]] = []
    current = {
        "start_sample": windows[0].start_sample,
        "end_sample": windows[0].end_sample,
        "peak": windows[0],
    }
    for window in windows[1:]:
        if window.start_sample <= int(current["end_sample"]):
            current["end_sample"] = max(int(current["end_sample"]), window.end_sample)
            peak = current["peak"]
            if isinstance(peak, _WindowResult) and window.dialog_bleed_score > peak.dialog_bleed_score:
                current["peak"] = window
            continue
        merged.append(current)
        current = {
            "start_sample": window.start_sample,
            "end_sample": window.end_sample,
            "peak": window,
        }
    merged.append(current)

    out: list[FlaggedRegion] = []
    for region in merged:
        peak = region["peak"]
        assert isinstance(peak, _WindowResult)
        out.append(
            FlaggedRegion(
                code="ME",
                metric="dialog_bleed_score",
                value=peak.dialog_bleed_score,
                threshold=1.0,
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
                detail=(
                    f"corr={peak.corr_abs:.2f} coh={peak.coherence_mean:.2f} "
                    f"dx={peak.dx_band_rms_dbfs:.1f} me={peak.me_band_rms_dbfs:.1f}"
                ),
            )
        )
    return out
