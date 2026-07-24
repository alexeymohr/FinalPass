"""Windowing and signal statistics shared by the comparison-style checks.

These are the primitives more than one check needs: how a file is cut into
analysis windows, how a window's level is measured, and the pairwise
statistics (optimal-gain null depth, signed correlation, band energy ratio)
that channel integrity and downmix consistency both reason about.

Kept deliberately free of any check's vocabulary — nothing here knows what a
"leg" or a "fold-down" is. Thresholds, severities, and findings belong to the
checks that own them.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

ANALYSIS_DBFS_FLOOR = -300.0

_LINEAR_FLOOR = 10 ** (ANALYSIS_DBFS_FLOOR / 20.0)
_MIN_FILTER_SAMPLES = 64


def window_bounds(
    *,
    sample_count: int,
    sample_rate: int,
    window_ms: float,
    hop_ms: float,
) -> list[tuple[int, int]]:
    """Cut a span into (start, end) windows, matching the null/M&E convention.

    Only whole windows are produced; a file shorter than one window yields a
    single window covering everything it has.
    """
    if sample_count <= 0:
        return []
    window_samples = max(1, int(round(window_ms * sample_rate / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    if sample_count <= window_samples:
        return [(0, sample_count)]
    return [
        (start, start + window_samples)
        for start in range(0, sample_count - window_samples + 1, hop_samples)
    ]


def window_rms_dbfs(data: np.ndarray, *, windows: list[tuple[int, int]]) -> np.ndarray:
    """Per-window, per-channel RMS in dBFS as a (windows × channels) array."""
    out = np.full((len(windows), data.shape[1]), ANALYSIS_DBFS_FLOOR, dtype=np.float64)
    for index, (start, end) in enumerate(windows):
        block = data[start:end, :]
        if block.size == 0:
            continue
        block_rms = np.sqrt(np.mean(block ** 2, axis=0))
        out[index, :] = 20.0 * np.log10(np.maximum(block_rms, _LINEAR_FLOOR))
    return out


def rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def to_dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, _LINEAR_FLOOR)))


def optimal_gain(a: np.ndarray, b: np.ndarray) -> float:
    """Least-squares gain that best cancels ``a`` with ``b``: <a,b>/<b,b>."""
    denominator = float(np.dot(b, b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def null_depth_db(a: np.ndarray, b: np.ndarray, gain: float, *, max_depth_db: float) -> float:
    """How far the gain-fitted difference residual sits below the content.

    Capped at ``max_depth_db`` because a bit-identical copy nulls to an exact
    zero residual, which would otherwise read as a meaningless ~280 dB.
    """
    residual_rms = rms(a - gain * b)
    depth = to_dbfs(rms(a)) - to_dbfs(residual_rms)
    return float(min(depth, max_depth_db))


def signed_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Mean-removed zero-lag correlation, sign preserved."""
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    a_norm = float(np.linalg.norm(a_centered))
    b_norm = float(np.linalg.norm(b_centered))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / (a_norm * b_norm))


def energy_ratio_above(
    signal: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float,
) -> float | None:
    """Fraction of total energy surviving a 4th-order zero-phase high-pass.

    Returns None when the question is unanswerable: an out-of-range cutoff, a
    silent signal, or too few samples for zero-phase filtering to mean much.
    """
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        return None
    total_energy = float(np.dot(signal, signal))
    if total_energy <= 0.0:
        return None
    if signal.shape[0] < _MIN_FILTER_SAMPLES:
        return None
    sos = butter(4, cutoff_hz / nyquist, btype="highpass", output="sos")
    high = sosfiltfilt(sos, signal)
    return float(np.dot(high, high) / total_energy)
