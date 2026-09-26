"""Build a blind grading set of breaths WITH context. LOCAL ONLY.

Each excerpt carries a fixed lead-in and tail so the operator hears the word a
breath follows and the phrase it leads into, and the breath always begins at
exactly LEAD_S into the file. The sample is drawn from chapters nobody has
labelled yet, mixes breaths the T-inhale score flags with breaths spread across
the level range, and is shuffled with no score shown — so the operator's grades
also measure the T-inhale score on unseen chapters.

These are excerpts of client audio: written locally for the operator's own
listening, never uploaded, attached or sent anywhere, and never read back.
"""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.report import clock  # noqa: E402

LEAD_S = 1.0
TAIL_S = 0.5
EDGE_FADE_S = 0.005
SEED = 20260925
N_FLAGGED = 20
N_PER_LEVEL_BAND = 10
LEVEL_BANDS = 4
EXCLUDE = ("001", "002", "003")   # already labelled

README = """Breath grading — round 1

Every clip starts 1.0 s before the breath and runs 0.5 s past it.
The breath itself always begins at exactly 1.000 s.

For each row in grading_round1.csv:
  grade      0 = natural, leave it
             1 = noticeable but acceptable
             2 = distracting, should be reduced or fixed
             3 = must fix
  t_inhale   yes / no / unsure  (mouth-release "T" onset before the inhale)
  notes      anything else

If this scale isn't how you think about breaths, say so and it changes.
"""


def write_context_clip(path: Path, x: np.ndarray, sr: int, s: int, e: int) -> None:
    """Breath at exactly LEAD_S into the clip, TAIL_S after it, faded file edges."""
    lead, tail = int(LEAD_S * sr), int(TAIL_S * sr)
    pad_front = max(0, lead - s)
    clip = np.concatenate([np.zeros(pad_front, np.int16),
                           x[max(0, s - lead):min(len(x), e + tail)]]).astype(np.float64)
    n = int(EDGE_FADE_S * sr)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    sf.write(str(path), np.round(clip).astype(np.int16), sr, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--scored", required=True, help="breaths_scored.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    det = json.loads(Path(a.scored).read_text())
    pool = [(fn, b) for fn, v in det.items() if fn[:3] not in EXCLUDE for b in v]
    rng = random.Random(SEED)
    flagged = [p for p in pool if p[1]["likely_t_inhale"]]
    rest = [p for p in pool if not p[1]["likely_t_inhale"]]
    picks = rng.sample(flagged, min(N_FLAGGED, len(flagged)))
    edges = np.quantile([b["peak_rel_db"] for _, b in rest], np.linspace(0, 1, LEVEL_BANDS + 1))
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = [p for p in rest if lo <= p[1]["peak_rel_db"] <= hi]
        picks += rng.sample(band, min(N_PER_LEVEL_BAND, len(band)))
    rng.shuffle(picks)

    rows, key, cache = [], {}, {}
    for i, (fn, b) in enumerate(picks, 1):
        if fn not in cache:
            cache.clear()
            cache[fn] = sf.read(str(Path(a.audio_dir) / fn), dtype="int16", always_2d=False)
        x, sr = cache[fn]
        s, e = b["start_sample"], b["end_sample"]
        name = f"grade_{i:02d}.wav"
        write_context_clip(out / name, x, sr, s, e)
        rows.append({"id": i, "clip": name, "chapter_file": fn,
                     "breath_time": clock(s, sr), "breath_ms": (e - s) * 1000 // sr,
                     "grade": "", "t_inhale": "", "notes": ""})
        key[i] = {"file": fn, "start_sample": s, "end_sample": e,
                  "t_inhale_score": b["t_inhale_score"], "likely_t_inhale": b["likely_t_inhale"],
                  "peak_rel_db": b["peak_rel_db"]}

    with open(out / "grading_round1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "grading_round1_key.json").write_text(json.dumps(key, indent=1))
    (out / "README.txt").write_text(README)
    print(f"{len(rows)} context clips written to {out}")


if __name__ == "__main__":
    main()
