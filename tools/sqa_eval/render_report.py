"""Render the numeric tables of the evaluation report from the scan artifacts.

Keeps every figure in the written report tied to a file on disk rather than
retyped, so the prose cannot drift from the data. Numbers only.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scan, pkg, audit = Path(a.scan), Path(a.package), Path(a.audit)
    meta = json.loads((scan / "run_meta.json").read_text())
    summary = json.loads((pkg / "summary.json").read_text())
    cal = json.loads(Path(a.calibration).read_text())
    rows = json.loads((pkg / "candidates_ranked.json").read_text())
    dep = json.loads((audit / "dependency_audit.json").read_text())

    parts = []
    parts.append("## Dependency and package-age audit\n\n" + table(
        ["package", "version", "published", "age (days)", "license"],
        [[p["pkg"], p["ver"], p["published"], p["age_days"], p["license"]]
         for p in dep["packages"]]) +
        f"\n\nMinimum age required: {dep['min_age_days_required']} days. "
        f"Violations: **{dep['violations'] or 'none'}**.\n")

    b = summary["burden_all"]; bi = summary["burden_interior"]
    bb = summary["burden_boundary_adjacent"]; b100 = summary["burden_worst_100"]
    bs = summary["burden_speech_rich"]
    parts.append("## Triage burden\n\n" + table(
        ["set", "events", "events/finished h", "unique candidate s",
         "audition min / finished h", "mean event s"],
        [[name, x["events"], x["events_per_finished_hour"],
          x["unique_candidate_seconds"], x["audition_minutes_per_finished_hour"],
          x["mean_event_seconds"]]
         for name, x in (("all candidates", b), ("interior only", bi),
                         ("boundary-adjacent", bb), ("worst 100", b100),
                         ("speech-rich (>=50 % speech)", bs))]) + "\n")

    parts.append("## Score distribution: calibration vs delivery\n\n" + table(
        ["", "clean public reference", "audiobook"],
        [["frames", cal["threshold"]["frames_total"], meta["total_frames"]],
         ["median", round(cal["distribution"]["median"], 4),
          round(summary["score_distribution"]["file_median_of_medians"], 4)],
         ["minimum", round(cal["distribution"]["min"], 4),
          round(summary["score_distribution"]["file_min"], 4)],
         ["fraction below threshold",
          round(cal["threshold"]["fraction_below"], 6),
          summary["score_distribution"]["fraction_below_threshold"]]]) + "\n")

    per_file = summary["per_file"]
    parts.append("## Per file\n\n" + table(
        ["file", "duration s", "frames", "clips", "below-thr frames",
         "candidates", "min score", "median score", "speech frac"],
        [[f["source_id"], round(f["seconds"], 1), f["frames"], f["clip_count"],
          f["frames_below_threshold"], f["candidates"], round(f["score_min"], 3),
          round(f["score_median"], 3), round(f["speech_frame_fraction"], 3)]
         for f in per_file]) + "\n")

    top = rows[:100]
    parts.append("## Top 100 candidates\n\n" + table(
        ["rank", "file", "start TC", "dur s", "min", "mean", "below", "speech", "class"],
        [[r["rank"], Path(r["source_id"]).stem.rsplit("_", 1)[-1],
          r["start_tc"], round(r["duration_seconds"], 3), round(r["min_score"], 3),
          round(r["mean_score"], 3), r["frames_below_threshold"],
          round(r.get("speech_content") or 0.0, 2),
          "interior" if r["interior"] else "boundary"] for r in top]) + "\n")

    Path(a.out).write_text("\n".join(parts))
    print(f"wrote {a.out} ({len(parts)} sections)")


if __name__ == "__main__":
    main()
