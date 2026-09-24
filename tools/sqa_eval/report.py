"""Build the human-audition package from a completed scan. Numbers only.

Runs in FinalPass's own environment so the shipped AAF writer and timecode
helpers can be reused. Emits:

* a ranked JSON and CSV of candidate regions;
* the same ranking restricted to interior candidates;
* an audition list with fixed pre/post-roll context;
* an experimental AAF marker set;
* the triage-burden figures the mission's workload gate is decided on.

No transcript, no audio, no waveform, no rendered excerpt.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.config import (  # noqa: E402
    AUDITION_CONTEXT_SECONDS, FROZEN, SPEECH_RICH_FRACTION,
)

# The three columns an operator actually works from lead; everything after
# them is supporting detail.
CSV_FIELDS = [
    "file", "time", "problem",
    "rank", "source_id", "start_sample", "end_sample", "sample_rate",
    "start_seconds", "end_seconds", "duration_seconds", "start_tc", "end_tc",
    "min_score", "mean_score", "threshold", "frames_below_threshold",
    "speech_content", "interior", "overlaps_boundary_gap",
    "distance_to_clip_boundary_samples",
]


SEVERE_BELOW = 1.5
CLEAR_BELOW = 2.2


def clock(sample: int, sample_rate: int) -> str:
    """H:MM:SS.mmm from the start of the file, in exact integer arithmetic.

    Chapter WAVs carry no BWF start time, so this is where the event sits when
    the file is opened on its own — no frame rate involved.
    """
    ms = (sample * 1000) // sample_rate
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{h}:{m:02d}:{sec:02d}.{ms:03d}"


def describe(row: dict) -> str:
    """Plain-English summary of what the model reported for one region.

    The model scores quality; it does not say what is wrong. So this states how
    bad, how long, and where — never a guessed defect type. Naming the problem is
    what the listening pass is for. The severity bands only word the score; they
    play no part in deciding what gets flagged.
    """
    score = row["min_score"]
    if score < SEVERE_BELOW:
        severity = "Severe"
    elif score < CLEAR_BELOW:
        severity = "Clear"
    else:
        severity = "Slight"
    ms = int(round(row["duration_seconds"] * 1000))
    speech = (row.get("speech_content") or 0.0) >= SPEECH_RICH_FRACTION
    where = "during speech" if speech else "in a pause or near-silence"
    edge = "" if row["interior"] else ", at a clip join"
    return f"{severity} drop in voice quality {where}{edge} ({ms} ms)"


CONTROL_SEED = 20260911
CONTROL_COUNT = 25


def _control_regions(rows, per_file, mode, samples_to_tc, count: int = CONTROL_COUNT) -> list[dict]:
    """`count` unflagged regions, chosen by a seeded RNG so the list reproduces."""
    import random

    flagged: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        flagged.setdefault(r["source_id"], []).append((r["start_sample"], r["end_sample"]))
    durations = sorted(r["duration_seconds"] for r in rows)
    span = durations[len(durations) // 2] if durations else 0.5

    rng = random.Random(CONTROL_SEED)
    files = [f for f in per_file if f["total_samples"] > 0]
    out: list[dict] = []
    attempts = 0
    while len(out) < count and attempts < count * 200 and files:
        attempts += 1
        f = files[rng.randrange(len(files))]
        sr = f["sample_rate"]
        width = int(round(span * sr))
        if f["total_samples"] <= width:
            continue
        start = rng.randrange(0, f["total_samples"] - width)
        end = start + width
        if any(fs < end and fe > start for fs, fe in flagged.get(f["source_id"], [])):
            continue
        if any(c["source_id"] == f["source_id"] and c["start_sample"] < end
               and c["end_sample"] > start for c in out):
            continue
        out.append({
            "control_index": len(out) + 1, "source_id": f["source_id"],
            "start_sample": start, "end_sample": end, "sample_rate": sr,
            "start_tc": samples_to_tc(start, sr, mode),
            "end_tc": samples_to_tc(end, sr, mode),
            "duration_seconds": round(width / sr, 4),
            "flagged": False,
        })
    out.sort(key=lambda c: (c["source_id"], c["start_sample"]))
    for i, c in enumerate(out):
        c["control_index"] = i + 1
    return out


def burden(rows: list[dict], hours: float) -> dict:
    unique_seconds = sum(r["duration_seconds"] for r in rows)
    return {
        "events": len(rows),
        "events_per_finished_hour": round(len(rows) / hours, 3) if hours else None,
        "unique_candidate_seconds": round(unique_seconds, 3),
        "audition_minutes_per_finished_hour": round((unique_seconds / 60.0) / hours, 3) if hours else None,
        "mean_event_seconds": round(unique_seconds / len(rows), 4) if rows else None,
        "median_min_score": round(sorted(r["min_score"] for r in rows)[len(rows) // 2], 4) if rows else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=23.976)
    ap.add_argument("--drop-frame", action="store_true")
    ap.add_argument("--aaf", action="store_true")
    a = ap.parse_args()

    scan = Path(a.scan)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((scan / "candidates.json").read_text())
    meta = json.loads((scan / "run_meta.json").read_text())
    per_file = json.loads((scan / "per_file.json").read_text())
    hours = meta["total_audio_hours"]

    from finalpass.timecode import samples_to_tc, timecode_mode
    mode = timecode_mode(a.fps, drop_frame=a.drop_frame)
    for r in rows:
        r["file"] = r["source_id"]
        r["time"] = clock(r["start_sample"], r["sample_rate"])
        r["problem"] = describe(r)
        r["start_tc"] = samples_to_tc(r["start_sample"], r["sample_rate"], mode)
        r["end_tc"] = samples_to_tc(r["end_sample"], r["sample_rate"], mode)

    interior = [r for r in rows if r["interior"]]
    for i, r in enumerate(interior):
        r = dict(r); r["rank"] = i + 1
        interior[i] = r
    boundary = [r for r in rows if not r["interior"]]
    worst100 = rows[:100]
    speech_rich = [r for r in rows
                   if (r.get("speech_content") or 0.0) >= SPEECH_RICH_FRACTION]

    (out / "candidates_ranked.json").write_text(json.dumps(rows, indent=1))
    (out / "candidates_interior_ranked.json").write_text(json.dumps(interior, indent=1))
    for name, data in (("candidates_ranked.csv", rows),
                       ("candidates_interior_ranked.csv", interior)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)

    context = AUDITION_CONTEXT_SECONDS
    total_by_file = {f["source_id"]: f["total_samples"] for f in per_file}
    audition = []
    for r in rows[:50]:
        sr = r["sample_rate"]; pad = int(round(context * sr))
        lo = max(0, r["start_sample"] - pad)
        hi = min(total_by_file.get(r["source_id"], r["end_sample"] + pad), r["end_sample"] + pad)
        audition.append({
            "rank": r["rank"], "source_id": r["source_id"],
            "audition_start_sample": lo, "audition_end_sample": hi,
            "audition_start_tc": samples_to_tc(lo, sr, mode),
            "audition_end_tc": samples_to_tc(hi, sr, mode),
            "event_start_tc": r["start_tc"], "event_end_tc": r["end_tc"],
            "context_seconds": context,
            "min_score": r["min_score"], "interior": r["interior"],
        })
    (out / "audition_top50.json").write_text(json.dumps(audition, indent=1))

    # Interior-only audition list, same shape.
    audition_interior = []
    for r in interior[:50]:
        sr = r["sample_rate"]; pad = int(round(context * sr))
        lo = max(0, r["start_sample"] - pad)
        hi = min(total_by_file.get(r["source_id"], r["end_sample"] + pad), r["end_sample"] + pad)
        audition_interior.append({
            "rank": r["rank"], "source_id": r["source_id"],
            "audition_start_sample": lo, "audition_end_sample": hi,
            "audition_start_tc": samples_to_tc(lo, sr, mode),
            "audition_end_tc": samples_to_tc(hi, sr, mode),
            "event_start_tc": r["start_tc"], "event_end_tc": r["end_tc"],
            "context_seconds": context,
            "min_score": r["min_score"], "interior": True,
        })
    (out / "audition_interior_top50.json").write_text(json.dumps(audition_interior, indent=1))

    # Deterministic control regions: unflagged spans matched in duration to the
    # median candidate, so the operator can listen blind and the precision
    # estimate is not built only from the detector's own positives.
    controls = _control_regions(rows, per_file, mode, samples_to_tc)
    (out / "control_regions.json").write_text(json.dumps(controls, indent=1))

    # One AAF per source file. A single combined lane would stack markers from
    # 28 chapters that each start at sample 0 on top of each other; the operator
    # imports markers into the session for the chapter they are auditioning.
    markers_written = 0
    marker_files: list[str] = []
    if a.aaf and rows:
        from sqa_eval.markers import write_candidate_markers
        marker_dir = out / "markers"
        marker_dir.mkdir(exist_ok=True)
        by_source: dict[str, list[dict]] = {}
        for r in rows:
            by_source.setdefault(r["source_id"], []).append(r)
        for source_id, group in sorted(by_source.items()):
            target = marker_dir / f"{Path(source_id).stem}.aaf"
            written = write_candidate_markers(target, group, mode=mode)
            markers_written += written
            marker_files.append(str(target.relative_to(out)))

    summary = {
        "corpus": {
            "files": meta["files_scanned"],
            "hours": hours,
            "frames": meta["total_frames"],
            "sample_rates": sorted({f["sample_rate"] for f in per_file}),
            "channels": sorted({f["channels"] for f in per_file}),
            "subtypes": sorted({f["subtype"] for f in per_file}),
        },
        "threshold": meta["threshold"],
        "runtime": {
            "elapsed_seconds": meta["elapsed_seconds"],
            "real_time_factor": meta["real_time_factor"],
            "audio_hours_per_wall_hour": round(meta["real_time_factor"], 2)
            if meta["real_time_factor"] else None,
        },
        "network_attempts": meta["network_attempts"],
        "score_distribution": {
            "file_min": min(f["score_min"] for f in per_file),
            "file_median_of_medians": sorted(f["score_median"] for f in per_file)[len(per_file) // 2],
            "frames_below_threshold": sum(f["frames_below_threshold"] for f in per_file),
            "fraction_below_threshold": round(
                sum(f["frames_below_threshold"] for f in per_file) / meta["total_frames"], 6),
            "speech_frame_fraction": round(
                sum(f["speech_frame_fraction"] * f["frames"] for f in per_file) / meta["total_frames"], 6),
        },
        "burden_all": burden(rows, hours),
        "burden_interior": burden(interior, hours),
        "burden_boundary_adjacent": burden(boundary, hours),
        "burden_worst_100": burden(worst100, hours),
        "burden_speech_rich": burden(speech_rich, hours),
        "speech_rich_fraction_threshold": SPEECH_RICH_FRACTION,
        "counts": {
            "candidates": len(rows), "interior": len(interior),
            "boundary_adjacent": len(boundary),
            "speech_rich": len(speech_rich),
            "control_regions": len(controls),
            "markers_written": markers_written,
            "marker_files": len(marker_files),
        },
        "verdict_gate": {
            "limit_audition_minutes_per_hour": 10.0,
            "measured": burden(rows, hours)["audition_minutes_per_finished_hour"],
            "result": ("PROMISING"
                       if (burden(rows, hours)["audition_minutes_per_finished_hour"] or 0) <= 10.0
                       else "TOO_NOISY"),
        },
        "per_file": per_file,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_file"}, indent=1))


if __name__ == "__main__":
    main()
