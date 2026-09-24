"""Cross-tabulate frame scores against speech content. Numbers only.

Predeclared before the scan (`config.SPEECH_FLOOR_DBFS`,
`config.SPEECH_RICH_FRACTION`) because the retired LAION evaluation established
that a detector can pass a workload gate while firing only on silence. This
answers the question that gate cannot: are the flagged regions *in the speech*?

Needs no model — it is RMS over the source audio on the same frame grid.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.chunking import frame_source_samples, total_frames  # noqa: E402
from sqa_eval.config import FROZEN  # noqa: E402


def frame_levels(path: Path, n_frames: int, step: int) -> np.ndarray:
    """Per-frame dBFS of the source, computed in bounded blocks."""
    out = np.empty(n_frames, dtype=np.float64)
    block_frames = 20000
    pos = 0
    with sf.SoundFile(str(path)) as fh:
        while pos < n_frames:
            take = min(block_frames, n_frames - pos)
            fh.seek(pos * step)
            data = fh.read(take * step, dtype="float64", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]
            if len(data) < take * step:
                data = np.pad(data, (0, take * step - len(data)))
            rms = np.sqrt(np.mean(data.reshape(take, step) ** 2, axis=1))
            with np.errstate(divide="ignore"):
                out[pos:pos + take] = 20.0 * np.log10(rms)
            pos += take
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scan = Path(a.scan)
    meta = json.loads((scan / "run_meta.json").read_text())
    per_file = json.loads((scan / "per_file.json").read_text())
    threshold = meta["threshold"]

    all_scores, all_levels = [], []
    for f in per_file:
        path = Path(a.audio_dir) / f["source_id"]
        step = frame_source_samples(f["sample_rate"])
        n = total_frames(f["total_samples"], f["sample_rate"])
        scores = np.load(scan / "frame_scores" / f"{Path(f['source_id']).stem}.npy").astype(np.float64)
        assert len(scores) == n, (len(scores), n)
        all_scores.append(scores)
        all_levels.append(frame_levels(path, n, step))
        print(f"  {f['source_id']}: {n} frames", flush=True)

    scores = np.concatenate(all_scores)
    levels = np.concatenate(all_levels)
    speech = levels > FROZEN.speech_floor_dbfs
    below = scores < threshold

    bands = []
    edges = [(-np.inf, -80), (-80, -60), (-60, -45), (-45, -30), (-30, -20), (-20, 0)]
    for lo, hi in edges:
        m = (levels > lo) & (levels <= hi)
        bands.append({
            "level_band_dbfs": f"({lo:g}, {hi:g}]",
            "frames": int(m.sum()),
            "frame_fraction": round(float(m.mean()), 6),
            "median_score": round(float(np.median(scores[m])), 4) if m.any() else None,
            "below_threshold": int(below[m].sum()),
            "below_threshold_rate": round(float(below[m].mean()), 6) if m.any() else None,
            "share_of_all_below_threshold": round(float(below[m].sum() / below.sum()), 6)
            if below.any() else None,
        })

    rows = json.loads((scan / "candidates.json").read_text())
    sc = sorted(r.get("speech_content") or 0.0 for r in rows)
    result = {
        "threshold": threshold,
        "frames": int(len(scores)),
        "speech_floor_dbfs": FROZEN.speech_floor_dbfs,
        "speech_frames": int(speech.sum()),
        "speech_frame_fraction": round(float(speech.mean()), 6),
        "below_threshold_frames": int(below.sum()),
        "below_threshold_fraction": round(float(below.mean()), 6),
        "below_threshold_in_speech": int((below & speech).sum()),
        "below_threshold_in_silence": int((below & ~speech).sum()),
        "share_of_below_threshold_that_is_silence": round(
            float((below & ~speech).sum() / below.sum()), 6) if below.any() else None,
        "median_score_speech": round(float(np.median(scores[speech])), 4),
        "median_score_silence": round(float(np.median(scores[~speech])), 4) if (~speech).any() else None,
        "p1_score_speech": round(float(np.percentile(scores[speech], 1)), 4),
        "below_threshold_rate_in_speech": round(float(below[speech].mean()), 6),
        "level_bands": bands,
        "candidate_speech_content": {
            "candidates": len(rows),
            "min": round(sc[0], 4) if sc else None,
            "median": round(sc[len(sc) // 2], 4) if sc else None,
            "max": round(sc[-1], 4) if sc else None,
            "at_or_above_0.5": sum(1 for v in sc if v >= 0.5),
            "at_or_above_0.25": sum(1 for v in sc if v >= 0.25),
            "exactly_zero": sum(1 for v in sc if v == 0.0),
        },
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "level_bands"}, indent=1))
    print("\nlevel bands:")
    for b in result["level_bands"]:
        print(f"  {b['level_band_dbfs']:>16}  frames={b['frames']:>7} "
              f"({b['frame_fraction']*100:5.2f}%)  median={b['median_score']}  "
              f"below={b['below_threshold']:>5} rate={b['below_threshold_rate']}")


if __name__ == "__main__":
    main()
