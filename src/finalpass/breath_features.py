"""Per-frame acoustic features for audiobook breath detection. Numbers only.

10 ms hop. Each frame reports level, voicing (normalised autocorrelation peak in
the speaking pitch range), and spectral shape (centroid, flatness, band
balance). Breaths are unpitched, noise-like and soft; vowels are pitched;
sibilants are noise-like but concentrated very high. These are the quantities
that separate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP_S = 0.010
WIN_S = 0.035          # long enough for a 60 Hz autocorrelation lag
PITCH_MIN_HZ = 70.0
PITCH_MAX_HZ = 400.0
BLOCK_FRAMES = 4000


@dataclass
class Frames:
    sample_rate: int
    hop: int
    rms_db: np.ndarray        # dBFS
    voicing: np.ndarray       # 0..1, normalised autocorrelation peak
    centroid_hz: np.ndarray
    flatness: np.ndarray      # 0..1 over 300 Hz - 12 kHz
    low_ratio: np.ndarray     # energy share below 1.5 kHz (of 100 Hz-12 kHz)
    high_ratio: np.ndarray    # energy share above 5 kHz
    zero: np.ndarray          # frame is digital black

    def __len__(self) -> int:
        return len(self.rms_db)

    def frame_of(self, sample: int) -> int:
        return sample // self.hop


def compute(x: np.ndarray, sample_rate: int) -> Frames:
    """Frame features for a mono float signal in [-1, 1]."""
    hop = int(round(HOP_S * sample_rate))
    win = int(round(WIN_S * sample_rate))
    nfft = 1 << (2 * win - 1).bit_length()
    n = max(0, (len(x) - win) // hop + 1)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    band = (freqs >= 100) & (freqs <= 12000)
    flat_band = (freqs >= 300) & (freqs <= 12000)
    low = band & (freqs < 1500)
    high = band & (freqs >= 5000)
    lag_lo = int(sample_rate / PITCH_MAX_HZ)
    lag_hi = int(sample_rate / PITCH_MIN_HZ)
    window = np.hanning(win)

    out = {k: np.empty(n) for k in ("rms", "voi", "cen", "flat", "lowr", "highr")}
    zero = np.empty(n, dtype=bool)
    for b0 in range(0, n, BLOCK_FRAMES):
        b1 = min(n, b0 + BLOCK_FRAMES)
        idx = (np.arange(b0, b1) * hop)[:, None] + np.arange(win)[None, :]
        fr = x[idx]
        zero[b0:b1] = ~np.any(fr != 0.0, axis=1)
        e = np.mean(fr * fr, axis=1)
        out["rms"][b0:b1] = 10 * np.log10(np.maximum(e, 1e-20))
        # voicing: normalised autocorrelation, mean-removed, via FFT
        c = fr - fr.mean(axis=1, keepdims=True)
        spec = np.fft.rfft(c, nfft, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, : lag_hi + 1]
        r0 = np.maximum(ac[:, 0], 1e-20)
        # unbiased normalisation so long lags are not penalised
        scale = win / (win - np.arange(lag_lo, lag_hi + 1))
        out["voi"][b0:b1] = np.clip((ac[:, lag_lo:lag_hi + 1] * scale).max(axis=1) / r0, 0, 1)
        p = np.abs(np.fft.rfft(fr * window, nfft, axis=1)) ** 2
        pb = p[:, band]
        tot = np.maximum(pb.sum(axis=1), 1e-30)
        out["cen"][b0:b1] = (pb * freqs[band]).sum(axis=1) / tot
        pf = np.maximum(p[:, flat_band], 1e-30)
        out["flat"][b0:b1] = np.exp(np.mean(np.log(pf), axis=1)) / np.mean(pf, axis=1)
        out["lowr"][b0:b1] = p[:, low].sum(axis=1) / tot
        out["highr"][b0:b1] = p[:, high].sum(axis=1) / tot
    return Frames(sample_rate, hop, out["rms"], out["voi"], out["cen"], out["flat"],
                  out["lowr"], out["highr"], zero)
