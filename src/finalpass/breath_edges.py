"""What happens at the onset of a breath: is there a mouth-release burst?

Numbers only. A "T-inhale" is a breath that opens with a mouth or tongue release:
a burst a few milliseconds long, close to full-band — a low thump as well as a
bright click — usually out of a closed-mouth silence, before an inhale that has
almost nothing below ~350 Hz. A breath that follows a word ending in a hard
consonant can also open with a burst — but that burst belongs to the word, and a
vowel sits just before it. These features measure both halves of that
description around the start of each detected breath, and a small fitted score
combines them.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

MS = 0.001
HP_HZ = 2000.0
THUMP_HZ = 300.0        # below this, the burst's thump; the inhale has almost none
LOOK_BEFORE_S = 0.060    # the release can sit just ahead of the tightened start
LOOK_AFTER_S = 0.100
CONTEXT_S = 0.300        # how far back to look for the preceding word


def _env(x: np.ndarray, n: int) -> np.ndarray:
    k = len(x) // n
    if k == 0:
        return np.array([-200.0])
    e = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1))
    return 20 * np.log10(np.maximum(e, 1e-10))


def _brightest_ms(high: np.ndarray, s0: int) -> int:
    """The burst: the brightest millisecond around the breath's start (index into `high`)."""
    w0 = max(0, s0 - int(LOOK_BEFORE_S / MS))
    w1 = min(len(high), s0 + int(LOOK_AFTER_S / MS))
    return w0 + int(np.argmax(high[w0:w1]))


def onset_features(x: np.ndarray, sr: int, start: int, end: int,
                   narration_dbfs: float) -> dict:
    """Features of the release burst (if any) at the start of one breath."""
    ms = max(1, int(MS * sr))
    lo = max(0, start - int((CONTEXT_S + LOOK_BEFORE_S) * sr))
    seg = x[lo:end].astype(np.float64)
    if len(seg) < 30 * ms:
        return {}
    hp = sosfiltfilt(butter(4, HP_HZ, btype="high", fs=sr, output="sos"), seg)
    lp = sosfiltfilt(butter(4, THUMP_HZ, btype="low", fs=sr, output="sos"), seg)
    full = _env(seg, ms)
    high = _env(hp, ms)
    low = _env(lp, ms)
    s0 = (start - lo) // ms
    body = high[s0:] if len(high) > s0 else high
    body_full = full[s0:] if len(full) > s0 else full
    k = _brightest_ms(high, s0)

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
    inhale = low[s0 + 30:] if len(low) - s0 > 60 else low[s0:]
    thump = low[max(0, k - 3):k + 4]
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
        "burst_low_rise_db": round(float(thump.max() - np.median(inhale)), 2)
        if thump.size and inhale.size else 0.0,
    }


CLICK_RMS_SAMPLES = 32   # short enough to see a few-millisecond gap
CLICK_GAP_DBFS = -60.0
CLICK_GAP_SEARCH_S = 0.030
CLICK_ONSET_DROP_DB = 20.0
CLICK_ONSET_MAX_S = 0.010


def click_gap(x: np.ndarray, sr: int, start: int, end: int, narration_dbfs: float) -> dict:
    """Does the breath's opening burst stand apart from what came before it?

    A mouth click out of a closed-mouth silence has a short gap (at or below
    -60 dBFS) just before it; a word's own final consonant running into the
    breath has none, or only its stop closure. Measured on the same burst as
    `onset_features`: walk back from the burst's peak to its onset (20 dB under
    the peak, at most 10 ms), then take the longest run at or below -60 dBFS in
    the 30 ms before that onset (32-sample RMS). Also the burst's peak level
    relative to the narration.

    On the operator's context-tagged breaths from one audiobook, 32-sample RMS:
    T-inhales 9/11 (one chapter) and 15/19 (other chapters) had a gap of at
    least 250 samples at 44.1 kHz; breaths after a word's hard consonant 2/11.
    """
    ms = max(1, int(MS * sr))
    lo = max(0, start - int((CONTEXT_S + LOOK_BEFORE_S) * sr))
    seg = x[lo:end].astype(np.float64)
    if len(seg) < 30 * ms:
        return {}
    high = _env(sosfiltfilt(butter(4, HP_HZ, btype="high", fs=sr, output="sos"), seg), ms)
    burst = lo + _brightest_ms(high, (start - lo) // ms) * ms

    a0 = max(0, burst - int(0.100 * sr))
    y = x[a0:min(len(x), burst + ms + CLICK_RMS_SAMPLES)].astype(np.float64)
    power = np.convolve(y ** 2, np.ones(CLICK_RMS_SAMPLES) / CLICK_RMS_SAMPLES, mode="same")
    db = 10 * np.log10(np.maximum(power, 1e-20))
    b = burst - a0
    i = b + int(np.argmax(db[b:b + ms]))
    peak = float(db[i])
    stop = b - int(CLICK_ONSET_MAX_S * sr)
    while i > max(stop, 0) and db[i] > peak - CLICK_ONSET_DROP_DB:
        i -= 1
    quiet = db[max(0, i - int(CLICK_GAP_SEARCH_S * sr)):i] <= CLICK_GAP_DBFS
    edges = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    gap = int(runs.max()) if runs.size else 0
    return {
        "click_gap_samples": gap,
        "click_gap_ms": round(gap * 1000.0 / sr, 2),
        "click_rel_db": round(peak - narration_dbfs, 2),
    }


# Fitted on the operator's own labels from one delivered audiobook: 647 breaths,
# 64 T-inhales (chapters 1-3 tagged without context, plus three rounds heard in
# context across the other chapters), measured on spans after the pause rule.
# Leave-one-chapter-out AUC 0.90 (0.83 without the low thump, which
# carries the largest weight). Least reliable distinction in the labels:
# T-inhale vs a breath that directly follows a word ending in a hard consonant.
# Source: tools/breaths/t_inhale_model_v3.json (tools/breaths/fit_t_inhale.py).
T_INHALE_FEATURES = ("burst_rise_db", "closure_below_body_db", "context_rel_db", "gap_before_ms",
                     "burst_width_ms", "burst_above_4k", "burst_low_rise_db")
T_INHALE_MEAN = (10.145981, -4.388624, -3.388099, 2.141938, 10.927357, 0.248661, 15.648006)
T_INHALE_STD = (5.933957, 11.975214, 11.034126, 0.370517, 25.986549, 0.365992, 12.915639)
T_INHALE_INTERCEPT = -3.459185
T_INHALE_WEIGHTS = (0.372199, 0.526229, -0.627225, 0.200679, -0.396684, 0.828397, 1.160400)
T_INHALE_THRESHOLD = -0.604201   # flags about as many as the operator labelled


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
