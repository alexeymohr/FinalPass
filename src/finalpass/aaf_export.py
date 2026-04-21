"""Phase 6 AAF marker export.

The runtime dependency is the ``pyaaf2`` package, which imports as ``aaf2``.
This module exports only already-persisted timed flags from the ``null`` and
``me`` report paths; it never reopens audio or recomputes analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .errors import AAFExportError
from .models import AllReport, FlaggedRegion, Group, MEReport, NullReport, Report
from .timecode import frame_rate_info, sample_to_edit_units

try:  # pragma: no cover - exercised by integration tests
    import aaf2
    from aaf2.rational import AAFRational
except ImportError:  # pragma: no cover - explicit tool error path
    aaf2 = None
    AAFRational = None

ExportableReport: TypeAlias = Report | NullReport | MEReport | AllReport

EXPORT_TRACK_NAME = "FinalPass Markers"


@dataclass(frozen=True)
class MarkerCandidate:
    name: str
    comment: str
    code: str
    metric: str
    start_sample: int
    end_sample: int
    start_tc: str
    end_tc: str
    sample_rate: int
    start_edit_unit: int
    length_edit_units: int
    group_id: str | None = None


def collect_marker_candidates(report: ExportableReport) -> list[MarkerCandidate]:
    """Return the timed marker candidates that Phase 6 is allowed to export."""
    candidates: list[MarkerCandidate] = []

    if isinstance(report, Report):
        return []

    if isinstance(report, NullReport):
        candidates.extend(
            _candidates_from_flags(
                flags=report.null_test.flags,
                sample_rate=report.printmaster.sample_rate,
                fps=report.fps,
            )
        )
    elif isinstance(report, MEReport):
        candidates.extend(
            _candidates_from_flags(
                flags=report.me_check.flags,
                sample_rate=report.me_file.sample_rate,
                fps=report.fps,
            )
        )
    else:
        for group in report.groups:
            sample_rate = _group_sample_rate(group)
            if sample_rate is None:
                continue
            if group.null_test is not None:
                candidates.extend(
                    _candidates_from_flags(
                        flags=group.null_test.flags,
                        sample_rate=sample_rate,
                        fps=report.fps,
                        group_id=group.group_id,
                    )
                )
            if group.me_check is not None:
                candidates.extend(
                    _candidates_from_flags(
                        flags=group.me_check.flags,
                        sample_rate=sample_rate,
                        fps=report.fps,
                        group_id=group.group_id,
                    )
                )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.start_edit_unit,
            candidate.group_id or "",
            candidate.code,
            candidate.metric,
        ),
    )


def write_markers_aaf(path: Path, candidates: list[MarkerCandidate], *, fps: float) -> None:
    """Write a minimal top-level marker AAF."""
    if not candidates:
        return
    if aaf2 is None or AAFRational is None:
        raise AAFExportError(
            "AAF export requires the pyaaf2 package (import path: aaf2)."
        )

    edit_rate = frame_rate_info(fps).edit_rate
    try:
        with aaf2.open(str(path), "w") as handle:
            composition = handle.create.CompositionMob(EXPORT_TRACK_NAME)
            composition.usage = "Usage_TopLevel"
            handle.content.mobs.append(composition)

            slot = handle.create.EventMobSlot(1)
            slot.name = EXPORT_TRACK_NAME
            slot.edit_rate = AAFRational(edit_rate.numerator, edit_rate.denominator)
            sequence = handle.create.Sequence(media_kind="DescriptiveMetadata")
            slot.segment = sequence
            composition.slots.append(slot)

            for candidate in candidates:
                marker = handle.create.DescriptiveMarker()
                marker["Position"].value = candidate.start_edit_unit
                marker["Length"].value = max(1, candidate.length_edit_units)
                marker["Comment"].value = candidate.comment
                marker["CommentMarkerAnnotationList"].value = candidate.name
                marker["CommentMarkerTime"].value = candidate.start_tc
                sequence.components.append(marker)
    except Exception as exc:  # pragma: no cover - exercised via CLI/tests
        raise AAFExportError(f"Could not write markers.aaf: {exc}") from exc


def _candidates_from_flags(
    *,
    flags: list[FlaggedRegion],
    sample_rate: int,
    fps: float,
    group_id: str | None = None,
) -> list[MarkerCandidate]:
    out: list[MarkerCandidate] = []
    for flag in flags:
        start_edit_unit = sample_to_edit_units(flag.start_sample, sample_rate, fps)
        end_edit_unit = sample_to_edit_units(flag.end_sample, sample_rate, fps)
        out.append(
            MarkerCandidate(
                name=_marker_name(flag.code, group_id=group_id),
                comment=_marker_comment(flag, group_id=group_id),
                code=flag.code,
                metric=flag.metric,
                start_sample=flag.start_sample,
                end_sample=flag.end_sample,
                start_tc=flag.start_tc,
                end_tc=flag.end_tc,
                sample_rate=sample_rate,
                start_edit_unit=start_edit_unit,
                length_edit_units=max(1, end_edit_unit - start_edit_unit),
                group_id=group_id,
            )
        )
    return out


def _group_sample_rate(group: Group) -> int | None:
    for file_report in group.files:
        if file_report.role == "pm":
            return file_report.sample_rate
    return group.files[0].sample_rate if group.files else None


def _marker_name(code: str, *, group_id: str | None) -> str:
    return f"{group_id} {code}" if group_id else code


def _marker_comment(flag: FlaggedRegion, *, group_id: str | None) -> str:
    group_context = f"{group_id} — " if group_id else ""
    return f"[{flag.code}] {flag.metric} {_format_metric_value(flag)} — {group_context}{flag.detail}"


def _format_metric_value(flag: FlaggedRegion) -> str:
    if flag.metric == "dialog_bleed_score":
        return f"{flag.value:.2f}"
    return f"{flag.value:.1f}"
