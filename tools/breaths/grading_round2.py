"""Grading round 2: regular breaths spanning very minor to quite noticeable. LOCAL ONLY.

Draws breaths the T-inhale score does not flag, at least 150 ms long, from
chapters and clips the operator has not reviewed, at even steps of level relative
to the local narration (the likeliest driver of how noticeable a breath is) from
the quietest to the loudest regular breaths in the book.
Shuffled, named blindly, same context format as round 1. The key keeps every
measured property for fitting a noticeability rule to the operator's grades.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breaths.evaluate import load_frames  # noqa: E402
from breaths.grading_set import write_context_clip  # noqa: E402

SEED = 20260926
N_CLIPS = 50
LOCAL_S = 3.0
EXCLUDE_CHAPTERS = ("001", "002", "003")

README = """Breath grading — round 2 (regular breaths)

Every clip starts 1.0 s before the breath and runs 0.5 s past it.
The breath itself always begins at exactly 1.000 s.

Add a score to the end of each file name:
   _1   very minor
   _2   noticeable
   _3   quite noticeable
e.g.  breath_07.wav  ->  breath_07_2.wav

If one is actually a T-inhale, or not a breath, tag it as before instead.
"""


def local_level(f, frame: int, level: float) -> float:
    """Median level of loud, pitched frames within +-LOCAL_S of the breath."""
    k = int(LOCAL_S / (f.hop / f.sample_rate))
    lo, hi = max(0, frame - k), min(len(f), frame + k)
    v = f.voicing[lo:hi]; r = f.rms_db[lo:hi]; z = f.zero[lo:hi]
    m = (~z) & (v > 0.7) & (r > level - 25)
    return float(np.median(r[m])) if m.any() else level


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--exclude-key", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    det = json.loads(Path(a.scored).read_text())
    level = {f["file"]: f["narration_level_dbfs"] for f in json.loads(Path(a.eval).read_text())["per_file"]}
    seen = set()
    for k in a.exclude_key:
        for v in json.loads(Path(k).read_text()).values():
            seen.add((v["file"], v["start_sample"]))

    pool = []
    for fn, evs in det.items():
        if fn[:3] in EXCLUDE_CHAPTERS or not evs:
            continue
        f = load_frames(Path(a.frames) / (fn[:-4] + ".npz"))
        for b in evs:
            if b["likely_t_inhale"] or b["duration_s"] < 0.15 or (fn, b["start_sample"]) in seen:
                continue
            loc = local_level(f, b["start_frame"], level[fn])
            peak_abs = b["peak_rel_db"] + level[fn]
            pool.append((fn, b, round(peak_abs - loc, 2), round(loc, 2)))

    # Even steps across the loudness scale, not in proportion to the book: loud
    # breaths are rare (about 2 % sit within 15 dB of the narration), and the
    # grading has to see them to learn where "quite noticeable" begins.
    rng = random.Random(SEED)
    rel = np.array([p[2] for p in pool])
    targets = np.linspace(np.percentile(rel, 0.5), np.percentile(rel, 99.8), N_CLIPS)
    picks, used = [], set()
    for t in targets:
        near = sorted((abs(p[2] - t), i) for i, p in enumerate(pool) if i not in used)
        window = [i for d, i in near[:12] if d <= max(0.75, near[0][0])]
        i = rng.choice(window)
        used.add(i)
        picks.append(pool[i])
    rng.shuffle(picks)

    key, cache = {}, {}
    for i, (fn, b, rel_local, loc) in enumerate(picks, 1):
        if fn not in cache:
            cache.clear()
            cache[fn] = sf.read(str(Path(a.audio_dir) / fn), dtype="int16", always_2d=False)
        x, sr = cache[fn]
        write_context_clip(out / f"breath_{i:02d}.wav", x, sr, b["start_sample"], b["end_sample"])
        key[i] = {"file": fn, "start_sample": b["start_sample"], "end_sample": b["end_sample"],
                  "duration_s": b["duration_s"], "peak_rel_narration_db": b["peak_rel_db"],
                  "median_rel_narration_db": b["median_rel_db"],
                  "peak_rel_local_db": rel_local, "local_speech_dbfs": loc,
                  "gap_before_s": b["gap_before_s"], "gap_after_s": b["gap_after_s"],
                  "median_centroid_hz": b["median_centroid_hz"], "t_inhale_score": b["t_inhale_score"]}
    (out / "breath_round2_key.json").write_text(json.dumps(key, indent=1))
    (out / "README.txt").write_text(README)
    r = np.array([v["peak_rel_local_db"] for v in key.values()])
    d = np.array([v["duration_s"] for v in key.values()])
    print(f"{len(key)} clips from {len({v['file'] for v in key.values()})} chapters | "
          f"peak vs local narration {r.min():.1f} to {r.max():.1f} dB | durations {d.min():.2f}-{d.max():.2f} s")


if __name__ == "__main__":
    main()
