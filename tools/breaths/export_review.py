"""Write every detected breath to its own WAV for the operator to audition.

LOCAL ONLY. These are excerpts of client audio: they are written to a local,
gitignored folder for the operator's own listening and must never be uploaded,
attached, or sent anywhere — including to a chat. The tool that writes them
never reads them back.

Each excerpt is the exact detected span, copied sample-for-sample from the
submitted file, with a 3 ms raised-cosine fade at each end so a hard cut does not
add a click of its own (a click there could be mistaken for a mouth-release
onset). Per-chapter review reels place every breath back to back with a fixed
silence between them, for fast skimming; the index maps both back to the source.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.report import clock  # noqa: E402

FADE_S = 0.003
REEL_GAP_S = 0.6


def _faded(x: np.ndarray, sr: int) -> np.ndarray:
    y = x.astype(np.float64)
    n = min(int(FADE_S * sr), len(y) // 2)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]
    return np.round(y).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--breaths", required=True, help="breaths_detected.json")
    ap.add_argument("--qc-cuts", default="", help="edit_inventory.json (optional)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "reels").mkdir(parents=True, exist_ok=True)
    events = json.loads(Path(a.breaths).read_text())
    cuts = {}
    if a.qc_cuts:
        for fn, d in json.loads(Path(a.qc_cuts).read_text()).items():
            cuts[fn] = [(e["before_start"], e["before_end"]) for e in d["edits"]
                        if e["kind"] == "audio_to_silence"]

    rows, n_id = [], 0
    for fn in sorted(events):
        evs = sorted(events[fn], key=lambda b: b["start_sample"])
        if not evs:
            continue
        x, sr = sf.read(str(Path(a.audio_dir) / fn), dtype="int16", always_2d=False)
        chap = fn[:3]
        cdir = out / chap
        cdir.mkdir(exist_ok=True)
        gap = np.zeros(int(REEL_GAP_S * sr), dtype=np.int16)
        reel, pos = [], 0
        for b in evs:
            n_id += 1
            clip = _faded(x[b["start_sample"]:b["end_sample"]], sr)
            t = clock(b["start_sample"], sr).replace(":", "-")
            name = f"{chap}_{t}_{len(clip) * 1000 // sr}ms.wav"
            sf.write(str(cdir / name), clip, sr, subtype="PCM_16")
            rows.append({
                "breath_id": n_id, "chapter_file": fn,
                "time": clock(b["start_sample"], sr), "end_time": clock(b["end_sample"], sr),
                "duration_ms": len(clip) * 1000 // sr,
                "peak_below_narration_db": round(-b["peak_rel_db"], 1),
                "wav": f"{chap}/{name}",
                "reel": f"reels/{chap}_breath_reel.wav", "reel_time": clock(pos, sr),
                "qc_cut": "yes" if any(s < b["end_sample"] and e > b["start_sample"]
                                       for s, e in cuts.get(fn, [])) else "",
                "start_sample": b["start_sample"], "end_sample": b["end_sample"],
            })
            reel += [clip, gap]
            pos += len(clip) + len(gap)
        sf.write(str(out / "reels" / f"{chap}_breath_reel.wav"),
                 np.concatenate(reel), sr, subtype="PCM_16")
        print(f"  {chap}: {len(evs)} breaths", flush=True)

    with open(out / "breath_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} breath excerpts written under {out}")


if __name__ == "__main__":
    main()
