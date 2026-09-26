"""Add onset features and a T-inhale score to every detected breath. Numbers only."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breaths.onset import load_model, onset_features, t_inhale_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--breaths", required=True, help="breaths_detected.json")
    ap.add_argument("--eval", required=True, help="breath_eval.json (narration levels)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = load_model()
    det = json.loads(Path(a.breaths).read_text())
    level = {f["file"]: f["narration_level_dbfs"] for f in json.loads(Path(a.eval).read_text())["per_file"]}
    flagged = 0
    for fn, evs in det.items():
        if not evs:
            continue
        x, sr = sf.read(str(Path(a.audio_dir) / fn), dtype="float64", always_2d=False)
        for b in evs:
            f = onset_features(x, sr, b["start_sample"], b["end_sample"], level[fn])
            b["onset"] = f
            b["t_inhale_score"] = round(t_inhale_score(f, b["gap_before_s"], model), 3) if f else None
            b["likely_t_inhale"] = bool(f) and b["t_inhale_score"] >= model["flag_threshold"]
            flagged += b["likely_t_inhale"]
    Path(a.out).write_text(json.dumps(det, indent=1))
    total = sum(len(v) for v in det.values())
    print(f"{total} breaths scored; {flagged} flagged as likely T-inhales ({flagged / total:.1%})")


if __name__ == "__main__":
    main()
