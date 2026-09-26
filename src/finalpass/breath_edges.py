"""What happens at the edges of a breath: its onset, and the sound after it.

Numbers only. Two questions are answered here.

**Does the breath open with a mouth-release burst (a "T-inhale")?**

A "T-inhale" is a breath that opens with a mouth or tongue release: a burst a few
milliseconds long, bright in the high frequencies, usually out of a closed-mouth
silence, before the inhale noise. A breath that follows a word ending in a hard
consonant can also open with a burst — but that burst belongs to the word, and
a vowel sits just before it. These features measure both halves of that
description around the start of each detected breath, and a small fitted score
combines them.

**Is the "breath" really an "h" sound?** An "h" is breath noise through a vocal
tract already shaped for the vowel that follows, so its spectrum carries that
vowel's resonances; an inhale does not. `following_similarity` measures exactly
that.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

MS = 0.001
HP_HZ = 2000.0
LOOK_BEFORE_S = 0.060    # the release can sit just ahead of the tightened start
LOOK_AFTER_S = 0.100
CONTEXT_S = 0.300        # how far back to look for the preceding word


def _env(x: np.ndarray, n: int) -> np.ndarray:
    k = len(x) // n
    if k == 0:
        return np.array([-200.0])
    e = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1))
    return 20 * np.log10(np.maximum(e, 1e-10))


def onset_features(x: np.ndarray, sr: int, start: int, end: int,
                   narration_dbfs: float) -> dict:
    """Features of the release burst (if any) at the start of one breath."""
    ms = max(1, int(MS * sr))
    lo = max(0, start - int((CONTEXT_S + LOOK_BEFORE_S) * sr))
    seg = x[lo:end].astype(np.float64)
    if len(seg) < 30 * ms:
        return {}
    hp = sosfiltfilt(butter(4, HP_HZ, btype="high", fs=sr, output="sos"), seg)
    full = _env(seg, ms)
    high = _env(hp, ms)
    s0 = (start - lo) // ms
    w0 = max(0, s0 - int(LOOK_BEFORE_S / MS))
    w1 = min(len(high), s0 + int(LOOK_AFTER_S / MS))
    body = high[s0:] if len(high) > s0 else high
    body_full = full[s0:] if len(full) > s0 else full
    k = w0 + int(np.argmax(high[w0:w1]))           # the burst: brightest ms at onset

    peak = high[k]
    above = high >= peak - 6.0
    a = k
    while a > 0 and above[a - 1]:
        a -= 1
    b = k
    while b < len(high) - 1 and above[b + 1]:
        b += 1
    pre = high[max(0, k - 5):k]
    closure = full[max(0, k - 20):max(0, k - 2)]
    context = full[max(0, k - int(CONTEXT_S / MS)):max(0, k - 30)]

    c0 = lo + k * ms
    burst = seg[max(0, k * ms - 2 * ms):k * ms + 3 * ms]
    if burst.size >= 8:
        spec = np.abs(np.fft.rfft(burst * np.hanning(len(burst)))) ** 2
        f = np.fft.rfftfreq(len(burst), 1.0 / sr)
        tot = max(spec.sum(), 1e-30)
        centroid = float((spec * f).sum() / tot)
        above4k = float(spec[f >= 4000].sum() / tot)
    else:
        centroid, above4k = 0.0, 0.0

    return {
        "burst_offset_ms": round((c0 - start) * 1000 / sr, 1),
        "burst_prominence_db": round(float(peak - np.median(body)), 2),
        "burst_width_ms": int(b - a + 1),
        "burst_rise_db": round(float(peak - pre.min()), 2) if pre.size else 0.0,
        "burst_centroid_hz": round(centroid, 1),
        "burst_above_4k": round(above4k, 4),
        "closure_below_body_db": round(float(np.median(body_full) - np.median(closure)), 2)
        if closure.size else 0.0,
        "context_rel_db": round(float(context.max() - narration_dbfs), 2) if context.size else -200.0,
    }


# Fitted on the operator's own labels from one delivered audiobook: 643 events,
# 40 T-inhales (chapters 1-3 tagged without context, plus two grading rounds
# heard in context across 50 further chapters). Leave-one-group-out AUC 0.90.
# Least reliable distinction in the labels: T-inhale vs a breath that directly
# follows a word ending in a hard consonant.
T_INHALE_FEATURES = ("burst_rise_db", "closure_below_body_db", "context_rel_db",
                     "gap_before_ms", "burst_width_ms", "burst_above_4k")
T_INHALE_MEAN = (10.026267, -5.024557, -3.104277, 2.125929, 11.094868, 0.254480)
T_INHALE_STD = (5.891307, 11.762975, 11.028919, 0.382813, 26.059604, 0.369541)
T_INHALE_INTERCEPT = -1.917985
T_INHALE_WEIGHTS = (0.784254, 0.709311, -0.262321, 0.442854, -2.687740, 1.328779)
T_INHALE_THRESHOLD = 1.764326   # flags about as many as the operator labelled


def t_inhale_score(features: dict, gap_before_s: float) -> float:
    """Logit that a breath opens with a mouth-release burst. Higher is more T-like."""
    vals = []
    for name in T_INHALE_FEATURES:
        if name == "gap_before_ms":
            vals.append(np.log10(max(gap_before_s * 1000.0, 10.0)))
        else:
            vals.append(features[name])
    z = (np.asarray(vals) - np.asarray(T_INHALE_MEAN)) / np.asarray(T_INHALE_STD)
    return float(T_INHALE_INTERCEPT + z @ np.asarray(T_INHALE_WEIGHTS))


SIM_LO_HZ, SIM_HI_HZ, SIM_BANDS = 150.0, 5000.0, 24
FOLLOW_S = 0.080


def _band_logspec(x: np.ndarray, sr: int) -> np.ndarray:
    """Mean log spectrum in ~1/6-octave bands: resonances, not harmonics."""
    n = 2048
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    w = np.hanning(n)
    frames = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    p = np.mean([np.abs(np.fft.rfft(fr * w)) ** 2 for fr in frames], axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f >= SIM_LO_HZ) & (f <= SIM_HI_HZ)
    edges = np.geomspace(SIM_LO_HZ, SIM_HI_HZ, SIM_BANDS + 1)
    b = np.digitize(f[m], edges)
    s = np.array([np.log10(p[m][b == k].mean() + 1e-20)
                  for k in range(1, len(edges)) if (b == k).any()])
    return s - s.mean()


def following_similarity(x: np.ndarray, sr: int, start: int, end: int,
                         gap_after_s: float) -> float | None:
    """Spectral-shape correlation between a breath and the next 80 ms of sound.

    Near or above ~0.1 the event shares the following vowel's resonances, as an
    "h" does; real inhales sit well below (median about -0.4). None when there
    is not enough audio after the event to judge.
    """
    after_start = end + int(gap_after_s * sr)
    after = x[after_start:after_start + int(FOLLOW_S * sr)]
    event = x[start:end]
    if len(after) <= 512 or len(event) < 64:
        return None
    return float(np.corrcoef(_band_logspec(event, sr), _band_logspec(after, sr))[0, 1])
