"""Does a breath fade into a pause before the next word? Numbers only.

An experienced re-recording mixer's description, checked against 683 events they
labelled in one audiobook: a breath fades down into a pause — a short
stretch where the level drops to about -60 dBFS — before the next word, while an
"h" is a frame or so of breath noise that runs straight into its vowel. So a
detected breath counts only when such a pause lies between it and the next
pitched speech. When sound follows that pause inside the detected span (an "h",
or a consonant leading into the word), the breath ends where the pause begins.

On those labels this kept 628 of 640 breaths and 21 of 23 breaths followed by an
"h", and rejected all 4 lone "h" sounds and 9 of 16 other non-breaths.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PAUSE_DBFS = -60.0
DEFAULT_PAUSE_MIN_MS = 10.0
LOOKAHEAD_S = 1.0     # how far past the breath a pause is looked for


def rms_dbfs(y: np.ndarray, w: int) -> np.ndarray:
    """RMS level of every ``w``-sample stretch of ``y``: ``out[i]`` covers ``y[i:i + w]``."""
    c = np.concatenate(([0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)))
    return 10.0 * np.log10(np.maximum((c[w:] - c[:-w]) / w, 1e-20))


def breath_end(x: np.ndarray, sr: int, start: int, end: int, next_speech: int,
               pause_dbfs: float = DEFAULT_PAUSE_DBFS,
               pause_min_ms: float = DEFAULT_PAUSE_MIN_MS) -> int | None:
    """Sample where the breath ends, or None when no pause follows it.

    A pause is a stretch of at least ``pause_min_ms`` whose RMS is at or below
    ``pause_dbfs``. The last pause before ``next_speech`` (the first sample of
    pitched speech after the breath, or the end of the file) is the one that
    separates the breath from the word.
    """
    w = max(1, int(round(pause_min_ms * sr / 1000.0)))
    stop = min(len(x), next_speech, end + int(LOOKAHEAD_S * sr))
    if stop - start < w:
        return None
    quiet = np.flatnonzero(rms_dbfs(x[start:stop], w) <= pause_dbfs)
    if quiet.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(quiet) > 1)
    first = int(quiet[breaks[-1] + 1] if breaks.size else quiet[0])
    pause_start, pause_stop = start + first, start + int(quiet[-1]) + w
    return pause_start if pause_stop < end else end
