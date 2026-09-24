"""Experimental AAF marker set for candidate regions.

Reuses FinalPass's shipped marker writer (`finalpass.aaf_export`) and its exact
sample -> edit-unit / timecode helpers, so a marker lands on the same original
sample the JSON and CSV report. Nothing in the released report schema is
touched: these markers are built directly from candidate rows and written to a
separate experimental file.

Marker labels carry the score, the interior/boundary classification and the
duration, and never a defect name — the model reports quality, not a diagnosis.
Runs in FinalPass's own environment (pyaaf2), not the ML evaluation venv.
"""
from __future__ import annotations

from pathlib import Path

from finalpass.aaf_export import MarkerCandidate, write_markers_aaf
from finalpass.timecode import TimecodeMode, sample_to_edit_units, samples_to_tc

MARKER_CODE = "low_frame_quality_region"
MARKER_METRIC = "frame_mos"


def marker_label(row: dict) -> str:
    kind = "interior" if row["interior"] else "boundary"
    ms = int(round(row["duration_seconds"] * 1000))
    return f"SQA LOW {row['min_score']:.3f} | {kind} | {ms} ms"


def marker_comment(row: dict) -> str:
    return (
        f"min {row['min_score']:.3f} / mean {row['mean_score']:.3f} MOS "
        f"vs threshold {row['threshold']:.3f}; "
        f"{row['frames_below_threshold']} frames below; "
        f"speech content {row['speech_content']:.2f}; "
        f"samples {row['start_sample']}-{row['end_sample']} @ {row['sample_rate']} Hz. "
        "Experimental frame-level speech-quality flag: unvalidated, informational, "
        "no defect class implied."
    )


def candidates_to_markers(rows: list[dict], *, mode: TimecodeMode) -> list[MarkerCandidate]:
    out: list[MarkerCandidate] = []
    for row in rows:
        sr = int(row["sample_rate"])
        start_eu = sample_to_edit_units(int(row["start_sample"]), sr, mode)
        end_eu = sample_to_edit_units(int(row["end_sample"]), sr, mode)
        out.append(MarkerCandidate(
            name=marker_label(row),
            comment=marker_comment(row),
            code=MARKER_CODE,
            metric=MARKER_METRIC,
            source_start_edit_unit=sample_to_edit_units(0, sr, mode),
            start_sample=int(row["start_sample"]),
            end_sample=int(row["end_sample"]),
            start_tc=samples_to_tc(int(row["start_sample"]), sr, mode),
            end_tc=samples_to_tc(int(row["end_sample"]), sr, mode),
            sample_rate=sr,
            start_edit_unit=start_eu,
            length_edit_units=max(1, end_eu - start_eu),
            group_id=None,
            asset_label=row["source_id"],
        ))
    return out


def write_candidate_markers(path: Path, rows: list[dict], *, mode: TimecodeMode) -> int:
    markers = candidates_to_markers(rows, mode=mode)
    if markers:
        write_markers_aaf(Path(path), markers, mode=mode)
    return len(markers)
