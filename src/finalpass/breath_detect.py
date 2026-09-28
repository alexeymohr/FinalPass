"""Find breath events in narration from per-frame features. Numbers only.

A breath, as measured on operator-confirmed QC cuts in a delivered audiobook:
0.2-0.6 s long, peaking 19-33 dB under the narration, energy centred around
1.5-2.6 kHz with almost nothing above 5 kHz, and separated from pitched speech
by a short gap on both sides. The rules below encode exactly those properties:

* the narration level is the median of loud, pitched frames in the file;
* a candidate frame is audible, not loud-and-pitched, not sibilant-bright;
* candidate runs become events when their length, peak level, spectral balance
  and voicing look like a breath;
* an event must be clear of pitched speech on both sides, which is what keeps
  word-attached fricatives ("s", "sh", "f", "h") out;
* each event is then tightened to the breath body. A 35 ms analysis window
  starting inside the gap can reach the next word, so the breath's own peak is
  taken from interior frames only, edge frames louder than that are dropped as
  bleed, quiet lead-in and tail below the body floor are trimmed, and the sample
  span uses frame centres rather than window extents.

Settings: the looser speech gap (10 ms) and 120 ms minimum were adopted after a
blind operator spot check found all 10 events only they produce to be breaths.
The 150 ms minimum on the tightened breath body follows the operator's review of
chapters 1-3 (593 labelled events): it removes 15 of 18 non-breaths and 18 of 20
unidentifiable events for 22 of 503 plain breaths.

Every threshold lives in `BreathParams` so it can be reported and tuned openly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .breath_features import Frames


@dataclass(frozen=True)
class BreathParams:
    speech_voicing: float = 0.70       # pitched enough to be narration
    speech_min_rel_db: float = -25.0   # ...and loud enough
    floor_rel_db: float = -52.0        # quieter frames are pause, not breath
    ceiling_rel_db: float = -6.0       # louder frames are speech
    frame_max_voicing: float = 0.85    # decaying vowels sit above this
    frame_max_high_ratio: float = 0.25 # sibilants carry most energy above 5 kHz
    frame_centroid_hz: tuple[float, float] = (600.0, 4500.0)
    bridge_frames: int = 2             # close holes this short inside a run
    min_s: float = 0.12
    max_s: float = 1.2
    min_peak_rel_db: float = -42.0
    max_median_voicing: float = 0.65
    event_centroid_hz: tuple[float, float] = (900.0, 4000.0)
    max_median_high_ratio: float = 0.15
    min_gap_to_speech_frames: int = 1  # 10 ms clear of pitched speech each side
    edge_guard_frames: int = 3         # frames whose window can overlap a neighbour
    body_floor_db: float = 20.0        # breath body: within this of its own peak
    bleed_db: float = 3.0              # edge frames this far over the peak are bleed
    min_body_s: float = 0.15           # operator: shorter is rarely identifiable


@dataclass
class Breath:
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    duration_s: float
    peak_rel_db: float
    median_rel_db: float
    median_voicing: float
    median_centroid_hz: float
    median_high_ratio: float
    gap_before_s: float
    gap_after_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def narration_level(f: Frames, p: BreathParams = BreathParams()) -> float:
    loud = (~f.zero) & (f.voicing > p.speech_voicing) & (f.rms_db > -50)
    return float(np.median(f.rms_db[loud])) if loud.any() else float("nan")


def speech_mask(f: Frames, level: float, p: BreathParams = BreathParams()) -> np.ndarray:
    """Frames of pitched, narration-loud speech."""
    return (~f.zero) & (f.voicing > p.speech_voicing) & (f.rms_db - level > p.speech_min_rel_db)


def frame_centre(f: Frames) -> int:
    """Sample offset from a frame's start to the point it represents."""
    return (int(round(0.035 * f.sample_rate)) - f.hop) // 2


def shorten(b: Breath, end_sample: int, f: Frames, level: float) -> Breath:
    """The same breath ending earlier, with its level measures taken again."""
    end_frame = max(b.start_frame + 1, (end_sample - frame_centre(f)) // f.hop)
    body = f.rms_db[b.start_frame:end_frame] - level
    return replace(
        b, end_sample=end_sample, end_frame=end_frame,
        duration_s=round((end_sample - b.start_sample) / f.sample_rate, 3),
        peak_rel_db=round(float(body.max()), 2), median_rel_db=round(float(np.median(body)), 2),
    )


def _runs(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > bridge + 1)
    starts = np.concatenate(([idx[0]], idx[splits + 1]))
    ends = np.concatenate((idx[splits], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _body(rel: np.ndarray, a: int, b: int, p: BreathParams) -> tuple[int, int, float]:
    """Tighten a candidate run to the breath itself; return (a, b, own peak)."""
    g = p.edge_guard_frames
    inner = rel[a + g:b - g] if b - a > 2 * g + 1 else rel[a:b]
    ref = float(inner.max())
    lo, hi = ref - p.body_floor_db, ref + p.bleed_db
    while a < b and not (lo <= rel[a] <= hi):
        a += 1
    while b > a and not (lo <= rel[b - 1] <= hi):
        b -= 1
    return a, b, ref


def detect(f: Frames, p: BreathParams = BreathParams()) -> tuple[list[Breath], float]:
    level = narration_level(f, p)
    if not np.isfinite(level):
        return [], level
    rel = f.rms_db - level
    audible = ~f.zero
    speech = speech_mask(f, level, p)
    cand =(audible & ~speech & (rel > p.floor_rel_db) & (rel < p.ceiling_rel_db)
            & (f.voicing < p.frame_max_voicing) & (f.high_ratio < p.frame_max_high_ratio)
            & (f.centroid_hz > p.frame_centroid_hz[0]) & (f.centroid_hz < p.frame_centroid_hz[1]))
    speech_idx = np.flatnonzero(speech)
    hop_s = f.hop / f.sample_rate

    out: list[Breath] = []
    for a, b in _runs(cand, p.bridge_frames):
        dur = (b - a) * hop_s
        if not (p.min_s <= dur <= p.max_s):
            continue
        seg = slice(a, b)
        live = audible[seg]
        r = rel[seg][live]
        if r.size == 0 or r.max() < p.min_peak_rel_db:
            continue
        voi = float(np.median(f.voicing[seg][live]))
        cen = float(np.median(f.centroid_hz[seg][live]))
        hi = float(np.median(f.high_ratio[seg][live]))
        if voi > p.max_median_voicing or hi > p.max_median_high_ratio:
            continue
        if not (p.event_centroid_hz[0] <= cen <= p.event_centroid_hz[1]):
            continue
        # The speech-gap guard judges the detected run itself (what was spot
        # checked); tightening below only moves the reported edges.
        k = np.searchsorted(speech_idx, a)
        gap_before = a - speech_idx[k - 1] - 1 if k > 0 else 10**6
        gap_after = speech_idx[k] - b if k < len(speech_idx) else 10**6
        if gap_before < p.min_gap_to_speech_frames or gap_after < p.min_gap_to_speech_frames:
            continue
        a, b, ref = _body(rel, a, b, p)
        if (b - a) * hop_s < p.min_body_s:
            continue
        centre = frame_centre(f)   # frame i represents its centre +/- hop/2
        body = rel[a:b]
        out.append(Breath(
            start_sample=a * f.hop + centre, end_sample=b * f.hop + centre,
            start_frame=a, end_frame=b, duration_s=round((b - a) * hop_s, 3),
            peak_rel_db=round(ref, 2), median_rel_db=round(float(np.median(body)), 2),
            median_voicing=round(voi, 3), median_centroid_hz=round(cen, 1),
            median_high_ratio=round(hi, 4),
            gap_before_s=round(min(gap_before, 10**5) * hop_s, 3),
            gap_after_s=round(min(gap_after, 10**5) * hop_s, 3),
        ))
    return out, level
