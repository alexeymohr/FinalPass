"""Speech-content annotation for scored regions.

The retired LAION evaluation failed for a reason worth carrying forward: a
detector can pass a workload gate while firing almost exclusively on silence.
Every candidate therefore carries the speech content of its own region, and the
report gives a speech-gated view beside the primary one.

This is annotation, not filtering. The primary candidate set is exactly what
the frozen threshold/smoothing/duration pipeline produces; nothing here removes
a candidate from it.
"""
from __future__ import annotations

import math
from typing import Sequence

from .config import FROZEN


def frame_rms_dbfs(block: Sequence[float]) -> float:
    """Full-scale RMS of one analysis frame, in dB."""
    if not len(block):
        return -math.inf
    total = 0.0
    for v in block:
        total += float(v) * float(v)
    rms = math.sqrt(total / len(block))
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms)


def speech_fraction(levels_dbfs: Sequence[float], floor_dbfs: float = FROZEN.speech_floor_dbfs) -> float:
    """Fraction of frames whose level sits above the speech floor."""
    if not len(levels_dbfs):
        return 0.0
    above = sum(1 for lv in levels_dbfs if lv > floor_dbfs)
    return above / len(levels_dbfs)
