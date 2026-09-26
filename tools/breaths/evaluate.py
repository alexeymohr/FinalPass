"""Score the breath detector against the exact QC cuts from the before/after diff.

Recall is measured on every audio-to-silence cut the QC operator made, split by
what their comment called it. Precision cannot be measured without listening, so
the detector's overall rate and a spot-check list are reported instead.
"""
from __future__ import annotations

import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breaths.detect import BreathParams, detect  # noqa: E402
from breaths.features import Frames  # noqa: E402
from sqa_eval.report import clock  # noqa: E402

SPOT_SEED = 20260925
SPOT_DEFAULT = 30   # random events from the default setting
SPOT_LOOSE = 10     # random events only a looser setting finds
LOOSE = {"min_gap_to_speech_frames": 1, "min_s": 0.12}


def load_frames(path: Path) -> Frames:
    z = np.load(path)
    return Frames(int(z["sr"]), int(z["hop"]), z["rms"].astype(np.float64),
                  z["voi"].astype(np.float64), z["cen"].astype(np.float64),
                  z["flat"].astype(np.float64), z["lowr"].astype(np.float64),
                  z["highr"].astype(np.float64), z["zero"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--diff", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot-check", action="store_true",
                    help="draw a NEW blind spot-check sample (overwrites the key)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    inv = json.loads((Path(a.diff) / "edit_inventory.json").read_text())
    labels = json.loads((Path(a.diff) / "qc_labels_exact.json").read_text())
    cat = {}
    for l in labels:
        if l["matched"]:
            cat.setdefault((l["file"], l["edit_index"]), l["qc_category"])

    p = BreathParams()
    events, per_file, audible_s = {}, [], 0.0
    for fpath in sorted(Path(a.frames).glob("*.npz")):
        name = fpath.stem + ".wav"
        f = load_frames(fpath)
        ev, level = detect(f, p)
        events[name] = ev
        live = float((~f.zero).sum()) * f.hop / f.sample_rate
        audible_s += live
        per_file.append({"file": name, "narration_level_dbfs": round(level, 2),
                         "breaths": len(ev), "audible_minutes": round(live / 60, 2),
                         "breaths_per_minute": round(len(ev) / (live / 60), 2) if live else None})

    recall = defaultdict(lambda: [0, 0])
    missed = defaultdict(list)
    hit_events = set()
    for fname, d in inv.items():
        for n, e in enumerate(d["edits"]):
            if e["kind"] != "audio_to_silence":
                continue
            c = cat.get((fname, n), "uncommented")
            hits = [k for k, b in enumerate(events.get(fname, []))
                    if b.start_sample < e["before_end"] and b.end_sample > e["before_start"]]
            recall[c][1] += 1
            if hits:
                recall[c][0] += 1
                hit_events.update((fname, k) for k in hits)
            else:
                missed[c].append({"file": fname, "time": clock(e["before_start"], d["sample_rate"]),
                                  "duration_s": round((e["before_end"] - e["before_start"]) / d["sample_rate"], 3),
                                  "peak_dbfs": e["before_peak_dbfs"]})

    total = sum(len(v) for v in events.values())

    # Blind spot check: default-setting events plus events only a looser setting
    # finds, shuffled together, with no hint of which is which or what QC did.
    from dataclasses import replace
    loose_p = replace(p, **LOOSE)
    loose_only = []
    for fpath in (sorted(Path(a.frames).glob("*.npz")) if a.spot_check else []):
        name = fpath.stem + ".wav"
        lev, _ = detect(load_frames(fpath), loose_p)
        for b in lev:
            if not any(x.start_sample < b.end_sample and x.end_sample > b.start_sample
                       for x in events[name]):
                loose_only.append((name, b))
    rng = random.Random(SPOT_SEED)
    pool = [(fn, b) for fn, v in events.items() for b in v] if a.spot_check else []
    picks = ([("default", fn, b) for fn, b in rng.sample(pool, min(SPOT_DEFAULT, len(pool)))]
             + [("loose_only", fn, b) for fn, b in rng.sample(loose_only, min(SPOT_LOOSE, len(loose_only)))])
    rng.shuffle(picks)
    rows, key = [], {}
    for i, (setting, fn, b) in enumerate(picks, 1):
        rows.append({"id": i, "file": fn, "time": clock(b.start_sample, 44100),
                     "duration_ms": round(b.duration_s * 1000), "is_breath": "", "notes": ""})
        key[i] = {"setting": setting, "start_sample": b.start_sample, "end_sample": b.end_sample}
    if a.spot_check:
        import csv
        with open(out / "spot_check_breaths.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        (out / "spot_check_key.json").write_text(json.dumps(key, indent=1))

    all_ev = [b for v in events.values() for b in v]
    summary = {
        "params": {k: v for k, v in p.__dict__.items()},
        "breaths_detected": total,
        "spot_check_drawn": a.spot_check,
        "audible_hours": round(audible_s / 3600, 3),
        "breaths_per_audible_minute": round(total / (audible_s / 60), 2),
        "recall_on_qc_cuts": {c: {"found": v[0], "of": v[1], "rate": round(v[0] / v[1], 3)}
                              for c, v in sorted(recall.items())},
        "detected_event_medians": {
            k: round(float(np.median([getattr(b, k) for b in all_ev])), 3)
            for k in ("duration_s", "peak_rel_db", "median_voicing", "median_centroid_hz")},
        "missed": missed,
        "per_file": per_file,
    }
    (out / "breath_eval.json").write_text(json.dumps(summary, indent=1))
    (out / "breaths_detected.json").write_text(json.dumps(
        {fn: [b.as_dict() for b in v] for fn, v in events.items()}, indent=1))
    print(json.dumps({k: summary[k] for k in ("breaths_detected", "audible_hours",
                      "breaths_per_audible_minute", "recall_on_qc_cuts",
                      "detected_event_medians")}, indent=1))


if __name__ == "__main__":
    main()
