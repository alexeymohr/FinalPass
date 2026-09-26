"""Build listening lists from the exact QC diff. Timecodes only, never audio.

Two lists, because the QC "artifacts" turned out to be two different things:

* short sounds the QC operator cut to silence — exact sample spans in the
  submitted file;
* problems inside speech, fixed by regenerating the whole text block — the
  block is exact, the artefact's place inside it is estimated from where the
  comment sits in the regenerated block.

Each row has an empty `your_label` column for the taxonomy pass.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqa_eval.report import clock  # noqa: E402

QUICK_PASS_EVERY = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", required=True, help="qc_diff output directory")
    ap.add_argument("--category", default="artifact")
    a = ap.parse_args()
    d = Path(a.diff)
    inv = json.loads((d / "edit_inventory.json").read_text())
    labels = json.loads((d / "qc_labels_exact.json").read_text())

    short, speech = [], []
    for l in labels:
        if not l["matched"] or l["qc_category"] != a.category:
            continue
        e = inv[l["file"]]["edits"][l["edit_index"]]
        sr = l["sample_rate"]
        if e["kind"] == "audio_to_silence":
            short.append({
                "file": l["file"], "time": clock(e["before_start"], sr),
                "duration_ms": round((e["before_end"] - e["before_start"]) * 1000 / sr),
                "qc_comment": l["comment"], "peak_dbfs": e["before_peak_dbfs"],
                "start_sample": e["before_start"], "end_sample": e["before_end"],
                "your_label": "",
            })
        elif e["kind"] == "audio_rerendered":
            span_a = max(1, e["after_end"] - e["after_start"])
            rel = min(1.0, max(0.0, (l["qc_time_after_s"] * sr - e["after_start"]) / span_a))
            approx = e["before_start"] + int(rel * (e["before_end"] - e["before_start"]))
            speech.append({
                "file": l["file"], "time_approx": clock(approx, sr),
                "qc_comment": l["comment"],
                "block_start": clock(e["before_start"], sr),
                "block_end": clock(e["before_end"], sr),
                "block_seconds": round((e["before_end"] - e["before_start"]) / sr, 2),
                "approx_sample": approx, "your_label": "",
            })

    for name, rows, key in (("listen_short_artifacts.csv", short, "start_sample"),
                            ("listen_in_speech_artifacts.csv", speech, "approx_sample")):
        rows.sort(key=lambda r: (r["file"], r[key]))
        for n, r in enumerate(rows):
            r["quick_pass"] = "yes" if n % QUICK_PASS_EVERY == 0 else ""
        with open(d / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"{name}: {len(rows)} rows, {sum(1 for r in rows if r['quick_pass'])} marked for a quick pass")


if __name__ == "__main__":
    main()
