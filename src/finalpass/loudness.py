"""Loudness measurement per ITU-R BS.1770-4 / EBU Tech 3341 / 3342.

Responsibilities split across helpers:

- :func:`_integrated` — integrated LUFS via pyloudnorm.
- :func:`_lra` — loudness range via pyloudnorm (EBU Tech 3342).
- :func:`_true_peak` — our own 4× oversampled true peak (pyloudnorm has no TP API).
- :func:`_short_term_max` / :func:`_momentary_max` — windowed max using the
  K-weighting filter chain from the pyloudnorm :class:`Meter` we already built.

All functions return Python floats or ``None``; any narrative error is appended
to the ``errors`` list on the returned :class:`LoudnessMeasurement`. Callers
(the CLI or ``check``) translate ``None`` into failed checks with the error
string passed through to the JSON report.

True peak oversampling: ``scipy.signal.resample_poly`` with a Kaiser β=14
window. The polyphase path is the FIR-equivalent of a windowed-sinc low-pass,
audited in scipy and cheaper than hand-rolling one. 4× is the minimum ratio
BS.1770-4 Annex 2 permits; inter-sample peaks stay within ~0.1 dB.
"""

from __future__ import annotations

import math
from typing import Literal
import warnings

import numpy as np
import pyloudnorm as pyln
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import resample_poly

from .audio_io import AudioFile
from .errors import LoudnessError
from .models import CheckResult, Measurements
from .specs import Spec

ABS_GATE_LUFS = -70.0
TP_OVERSAMPLE_RATIO = 4

# BS.1770-4 channel weights (up to 7.1). SMPTE order: L R C LFE Ls Rs Lss Rss.
# LFE is dropped (weight 0); surround channels get 1.41.
_CHANNEL_WEIGHTS: dict[int, list[float]] = {
    1: [1.0],
    2: [1.0, 1.0],
    6: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
    8: [1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41],
}


class LoudnessMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    lra: float | None = None
    short_term_max_lufs: float | None = None
    momentary_max_lufs: float | None = None
    errors: list[str] = Field(default_factory=list)

    def as_measurements(self) -> Measurements:
        return Measurements(
            integrated_lufs=self.integrated_lufs,
            true_peak_dbtp=self.true_peak_dbtp,
            lra=self.lra,
            short_term_max_lufs=self.short_term_max_lufs,
            momentary_max_lufs=self.momentary_max_lufs,
        )


def _finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _integrated(data: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    if data.shape[0] < 3 * sr:
        return None, "file_too_short_for_integrated"
    if data.ndim == 2 and data.shape[1] > 5:
        return _integrated_extended(data, sr)
    meter = pyln.Meter(sr)
    value = meter.integrated_loudness(data)
    cleaned = _finite_or_none(value)
    if cleaned is None:
        return None, "silent_or_below_gate"
    return cleaned, None


def _lra(data: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    if data.shape[0] < 3 * sr:
        return None, "file_too_short_for_integrated"
    if data.ndim == 2 and data.shape[1] > 5:
        return _lra_extended(data, sr)
    meter = pyln.Meter(sr)
    try:
        value = meter.loudness_range(data)
    except Exception as exc:  # pyloudnorm raises ValueError on very quiet input
        return None, f"lra_undefined_signal_too_quiet ({exc})"
    cleaned = _finite_or_none(value)
    if cleaned is None:
        return None, "lra_undefined_signal_too_quiet"
    return cleaned, None


def _true_peak(data: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    """4× oversampled true peak in dBTP, max across channels."""
    if data.size == 0:
        return None, "empty_signal"

    n_channels = data.shape[1]
    peak = 0.0
    for ch in range(n_channels):
        signal = data[:, ch]
        if signal.size == 0:
            continue
        # Kaiser-window polyphase upsample. resample_poly chooses an FIR of
        # appropriate order for the given (up, down, window) combination.
        oversampled = resample_poly(
            signal, up=TP_OVERSAMPLE_RATIO, down=1, window=("kaiser", 14.0)
        )
        ch_peak = float(np.max(np.abs(oversampled)))
        if ch_peak > peak:
            peak = ch_peak

    if peak <= 0.0:
        return None, "silent_or_below_gate"
    return 20.0 * math.log10(peak), None


def _make_meter(sr: int) -> pyln.Meter:
    return pyln.Meter(sr)


def _k_weight(data: np.ndarray, meter: pyln.Meter) -> np.ndarray:
    """Apply the same K-weighting filter chain pyloudnorm's Meter uses.

    ``meter._filters`` is an ordered dict of IIR filter stages (pre-filter
    shelving + RLB high-pass). pyloudnorm is a single-author library with a
    stable internal API; if this breaks on a future version the tests will
    catch it and we'll switch to hand-rolled K-weighting biquads.
    """
    filters = getattr(meter, "_filters", None)
    if not filters:
        raise LoudnessError(
            "pyloudnorm.Meter has no _filters attribute; cannot reuse "
            "K-weighting. Update finalpass.loudness to implement it directly."
        )
    weighted = data.copy()
    for stage in filters.values():
        weighted = stage.apply_filter(weighted)
    return weighted


def _windowed_max_lufs(
    weighted: np.ndarray,
    sr: int,
    window_seconds: float,
    hop_seconds: float,
    channel_weights: list[float],
) -> float | None:
    n = weighted.shape[0]
    window = int(round(window_seconds * sr))
    hop = int(round(hop_seconds * sr))
    if n < window:
        return None

    max_lufs = -math.inf
    for start in range(0, n - window + 1, hop):
        block = weighted[start : start + window, :]
        ms = np.mean(block ** 2, axis=0)
        z = float(sum(w * m for w, m in zip(channel_weights, ms)))
        if z <= 0.0:
            continue
        lufs = -0.691 + 10.0 * math.log10(z)
        if lufs < ABS_GATE_LUFS:
            continue
        if lufs > max_lufs:
            max_lufs = lufs

    if not math.isfinite(max_lufs):
        return None
    return max_lufs


def _short_term_max(data: np.ndarray, sr: int, weights: list[float]) -> tuple[float | None, str | None]:
    if data.shape[0] < 3 * sr:
        return None, "file_too_short_for_integrated"
    weighted = _k_weight(data, _make_meter(sr))
    value = _windowed_max_lufs(weighted, sr, 3.0, 0.1, weights)
    return value, (None if value is not None else "silent_or_below_gate")


def _momentary_max(data: np.ndarray, sr: int, weights: list[float]) -> tuple[float | None, str | None]:
    min_samples = int(round(0.4 * sr))
    if data.shape[0] < min_samples:
        return None, "file_too_short_for_momentary"
    weighted = _k_weight(data, _make_meter(sr))
    value = _windowed_max_lufs(weighted, sr, 0.4, 0.1, weights)
    return value, (None if value is not None else "silent_or_below_gate")


def _weights_for_channels(n_channels: int) -> list[float]:
    if n_channels not in _CHANNEL_WEIGHTS:
        raise LoudnessError(
            f"Unsupported channel count for BS.1770 weighting: {n_channels}"
        )
    return _CHANNEL_WEIGHTS[n_channels]


def _apply_k_weighting(data: np.ndarray, sr: int) -> np.ndarray:
    weighted = data.copy()
    for filter_stage in _make_meter(sr)._filters.values():
        for ch in range(weighted.shape[1]):
            weighted[:, ch] = filter_stage.apply_filter(weighted[:, ch])
    return weighted


def _blockwise_bs1770_loudness(
    weighted: np.ndarray,
    *,
    sr: int,
    block_size: float,
    overlap: float,
    gains: list[float],
) -> list[float]:
    num_channels = weighted.shape[1]
    num_samples = weighted.shape[0]
    step = 1.0 - overlap
    duration = num_samples / sr
    num_blocks = int(np.round(((duration - block_size) / (block_size * step))) + 1)
    if num_blocks <= 0:
        return []

    z = np.zeros((num_channels, num_blocks), dtype=np.float64)
    for i in range(num_channels):
        for j in range(num_blocks):
            lower = int(block_size * (j * step) * sr)
            upper = int(block_size * (j * step + 1) * sr)
            z[i, j] = (1.0 / (block_size * sr)) * np.sum(np.square(weighted[lower:upper, i]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return [
            -0.691 + 10.0 * np.log10(np.sum([gains[i] * z[i, j] for i in range(num_channels)]))
            for j in range(num_blocks)
        ]


def _integrated_extended(data: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    gains = _weights_for_channels(data.shape[1])
    weighted = _apply_k_weighting(data, sr)
    blockwise = _blockwise_bs1770_loudness(
        weighted,
        sr=sr,
        block_size=0.4,
        overlap=0.75,
        gains=gains,
    )
    gated = [value for value in blockwise if value >= ABS_GATE_LUFS]
    if not gated:
        return None, "silent_or_below_gate"

    z_avg_gated_power = np.sum(np.power(10.0, np.divide(np.array(gated) + 0.691, 10.0))) / len(gated)
    relative_threshold = -0.691 + 10.0 * np.log10(z_avg_gated_power) - 10.0
    doubly_gated = [value for value in gated if value > relative_threshold]
    if not doubly_gated:
        return None, "silent_or_below_gate"

    gated_power = np.sum(np.power(10.0, np.divide(np.array(doubly_gated) + 0.691, 10.0))) / len(doubly_gated)
    integrated = -0.691 + 10.0 * np.log10(gated_power)
    return _finite_or_none(integrated), (None if math.isfinite(integrated) else "silent_or_below_gate")


def _lra_extended(data: np.ndarray, sr: int) -> tuple[float | None, str | None]:
    gains = _weights_for_channels(data.shape[1])
    padded = np.vstack([data, np.zeros((int(round(1.5 * sr)), data.shape[1]), dtype=np.float64)])
    weighted = _apply_k_weighting(padded, sr)
    blockwise = _blockwise_bs1770_loudness(
        weighted,
        sr=sr,
        block_size=3.0,
        overlap=0.97,
        gains=gains,
    )
    if not blockwise:
        return None, "lra_undefined_signal_too_quiet"
    absolute_gated = [value for value in blockwise if value >= ABS_GATE_LUFS]
    if not absolute_gated:
        return None, "lra_undefined_signal_too_quiet"

    power = np.sum(np.power(10.0, np.divide(absolute_gated, 10.0))) / len(absolute_gated)
    relative_threshold = 10.0 * np.log10(power) - 20.0
    relative_gated = [value for value in absolute_gated if value >= relative_threshold]
    if not relative_gated:
        return None, "lra_undefined_signal_too_quiet"

    lra = float(np.percentile(relative_gated, 95) - np.percentile(relative_gated, 10))
    return _finite_or_none(lra), (None if math.isfinite(lra) else "lra_undefined_signal_too_quiet")


def measure(audio: AudioFile) -> LoudnessMeasurement:
    """Full loudness measurement for one file.

    Errors are accumulated on the returned object — library code does not
    print. The CLI layer formats ``errors`` for the user.
    """
    sr = audio.sample_rate
    data = audio.data
    weights = _weights_for_channels(audio.channel_count)

    errors: list[str] = []

    integrated, err = _integrated(data, sr)
    if err:
        errors.append(err)

    tp, err = _true_peak(data, sr)
    if err and err != "silent_or_below_gate":
        errors.append(err)

    lra, err = _lra(data, sr)
    if err and err not in errors:
        errors.append(err)

    st_max, err = _short_term_max(data, sr, weights)
    if err and err not in errors:
        errors.append(err)

    m_max, err = _momentary_max(data, sr, weights)
    if err and err not in errors:
        errors.append(err)

    return LoudnessMeasurement(
        integrated_lufs=integrated,
        true_peak_dbtp=tp,
        lra=lra,
        short_term_max_lufs=st_max,
        momentary_max_lufs=m_max,
        errors=errors,
    )


Role = Literal["primary", "dx"]


def check(measurement: LoudnessMeasurement, spec: Spec, role: Role) -> list[CheckResult]:
    """Compare a measurement against a spec and produce per-metric check results."""
    checks: list[CheckResult] = []

    if role == "dx":
        if spec.dialog_lufs is None:
            return checks
        checks.append(_check_target("dialog_lufs", measurement.integrated_lufs, spec.dialog_lufs.target, spec.dialog_lufs.tolerance, measurement.errors))
        return checks

    checks.append(_check_target("integrated_lufs", measurement.integrated_lufs, spec.integrated_lufs.target, spec.integrated_lufs.tolerance, measurement.errors))
    checks.append(_check_limit("true_peak_dbtp", measurement.true_peak_dbtp, spec.true_peak_max_dbtp, measurement.errors))
    if spec.lra_max is not None:
        checks.append(_check_limit("lra", measurement.lra, spec.lra_max, measurement.errors))
    return checks


def _first_error_for(metric: str, errors: list[str]) -> str | None:
    mapping = {
        "integrated_lufs": {"file_too_short_for_integrated", "silent_or_below_gate"},
        "dialog_lufs": {"file_too_short_for_integrated", "silent_or_below_gate"},
        "true_peak_dbtp": {"silent_or_below_gate", "empty_signal"},
        "lra": {"lra_undefined_signal_too_quiet", "file_too_short_for_integrated"},
    }
    relevant = mapping.get(metric, set())
    for e in errors:
        for r in relevant:
            if e.startswith(r):
                return e
    return None


def _check_target(metric: str, measured: float | None, target: float, tolerance: float, errors: list[str]) -> CheckResult:
    if measured is None:
        return CheckResult(
            metric=metric, target=target, tolerance=tolerance, measured=None,
            **{"pass": False}, error=_first_error_for(metric, errors),
        )
    passed = abs(measured - target) <= tolerance
    return CheckResult(
        metric=metric, target=target, tolerance=tolerance, measured=measured,
        **{"pass": passed},
    )


def _check_limit(metric: str, measured: float | None, limit: float, errors: list[str]) -> CheckResult:
    if measured is None:
        # A silent file legitimately has no peak — TP has nothing to violate,
        # so flipping TP to FAIL on silence is noise. The silent/short verdict
        # is already carried by the integrated_lufs check (which fails). TP
        # stays "pass" here so the summary reflects the one real problem, not
        # two. See Phase 1 report decision #2.
        if metric == "true_peak_dbtp":
            return CheckResult(
                metric=metric, limit=limit, measured=None,
                **{"pass": True}, error=_first_error_for(metric, errors),
            )
        # LRA on insufficient gated content (too-short or too-quiet file) is
        # not a mix-quality problem; the lra_max check has no meaningful
        # verdict. Mark it skipped so the summary counts neither pass nor fail.
        if metric == "lra":
            return CheckResult(
                metric=metric, limit=limit, measured=None,
                **{"pass": None}, skipped=True, reason="insufficient_gated_content",
            )
        return CheckResult(
            metric=metric, limit=limit, measured=None,
            **{"pass": False}, error=_first_error_for(metric, errors),
        )
    return CheckResult(
        metric=metric, limit=limit, measured=measured,
        **{"pass": measured <= limit},
    )
