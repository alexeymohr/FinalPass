"""Phase 6 AAF marker export.

The runtime dependency is the ``pyaaf2`` package, which imports as ``aaf2``.
This module exports only already-persisted timed flags from the report model;
it never reopens audio or recomputes analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

from .errors import AAFExportError
from .models import AllReport, ChannelsReport, DownmixReport, FlaggedRegion, Group, MEReport, NullReport, Report
from .presentation import logical_asset_display_name
from .timecode import TimecodeMode, sample_to_edit_units, timecode_mode

try:  # pragma: no cover - exercised by integration tests
    import aaf2
    from aaf2 import components as component_module
    from aaf2.rational import AAFRational
except ImportError:  # pragma: no cover - explicit tool error path
    aaf2 = None
    component_module = None
    AAFRational = None

ExportableReport: TypeAlias = Report | NullReport | MEReport | AllReport | ChannelsReport | DownmixReport

EXPORT_TRACK_NAME = "FinalPass Markers"
LOUDNESS_MARKER_MIN_SPACING_SECONDS = 1.0


@dataclass(frozen=True)
class MarkerCandidate:
    name: str
    comment: str
    code: str
    metric: str
    source_start_edit_unit: int
    start_sample: int
    end_sample: int
    start_tc: str
    end_tc: str
    sample_rate: int
    start_edit_unit: int
    length_edit_units: int
    group_id: str | None = None
    asset_label: str | None = None


def collect_marker_candidates(report: ExportableReport) -> list[MarkerCandidate]:
    """Return the timed marker candidates that Phase 6 is allowed to export."""
    candidates: list[MarkerCandidate] = []
    if isinstance(report, ChannelsReport):
        # Channel-integrity findings describe a channel, not a moment. Phase 6's
        # rule stands: no timeline position, no marker, no synthetic AAF.
        return candidates
    # The envelope persists the run's timecode mode; rebuild it once here so
    # placement and marker labels use exactly what the run was measured with.
    mode = timecode_mode(report.fps, drop_frame=report.drop_frame)

    if isinstance(report, Report):
        for file_report in report.files:
            candidates.extend(
                _candidates_from_flags(
                    flags=file_report.flags,
                    sample_rate=file_report.sample_rate,
                    mode=mode,
                    time_reference_samples=_time_reference_samples(file_report),
                    asset_label=logical_asset_display_name(file_report),
                )
            )
    elif isinstance(report, NullReport):
        candidates.extend(
                _candidates_from_flags(
                    flags=report.null_test.flags,
                    sample_rate=report.printmaster.sample_rate,
                    mode=mode,
                    time_reference_samples=_time_reference_samples(report.printmaster),
                )
            )

    elif isinstance(report, MEReport):
        candidates.extend(
                _candidates_from_flags(
                    flags=report.me_check.flags,
                    sample_rate=report.me_file.sample_rate,
                    mode=mode,
                    time_reference_samples=_time_reference_samples(report.me_file),
                )
            )
    elif isinstance(report, DownmixReport):
        candidates.extend(
                _candidates_from_flags(
                    flags=report.downmix_check.flags,
                    sample_rate=report.stereo_file.sample_rate,
                    mode=mode,
                    time_reference_samples=_time_reference_samples(report.stereo_file),
                )
            )
    else:
        for group in report.groups:
            sample_rate = _group_sample_rate(group)
            if sample_rate is None:
                continue
            for file_report in group.files:
                candidates.extend(
                    _candidates_from_flags(
                        flags=file_report.flags,
                        sample_rate=file_report.sample_rate,
                        mode=mode,
                        time_reference_samples=_time_reference_samples(file_report),
                        group_id=group.group_id,
                        asset_label=logical_asset_display_name(file_report),
                    )
                )
            if group.null_test is not None:
                candidates.extend(
                    _candidates_from_flags(
                        flags=group.null_test.flags,
                        sample_rate=sample_rate,
                        mode=mode,
                        time_reference_samples=_time_reference_samples(group.null_test),
                        group_id=group.group_id,
                    )
                )
            if group.me_check is not None:
                me_sample_rate = _analysis_sample_rate(group.me_check) or sample_rate
                candidates.extend(
                    _candidates_from_flags(
                        flags=group.me_check.flags,
                        sample_rate=me_sample_rate,
                        mode=mode,
                        time_reference_samples=_time_reference_samples(group.me_check),
                        group_id=group.group_id,
                    )
                )

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start_edit_unit,
            candidate.group_id or "",
            candidate.asset_label or "",
            candidate.code,
            candidate.metric,
        ),
    )
    candidates = _throttle_loudness_candidates(candidates)
    return [
        replace(
            candidate,
            name=f"[{index}] FAIL: {_short_failure_label(candidate)}",
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def write_markers_aaf(path: Path, candidates: list[MarkerCandidate], *, mode: TimecodeMode) -> None:
    """Write a Pro Tools-friendly AAF with a marker lane on a real composition."""
    if not candidates:
        return
    if aaf2 is None or AAFRational is None or component_module is None:
        raise AAFExportError(
            "AAF export requires the pyaaf2 package (import path: aaf2)."
        )

    edit_rate = mode.edit_rate
    timeline_start = _timeline_start_edit_units(candidates)
    timeline_length = _timeline_length_edit_units(candidates, timeline_start=timeline_start)
    try:
        with aaf2.open(str(path), "w") as handle:
            composition = handle.create.CompositionMob(EXPORT_TRACK_NAME)
            composition.usage = "Usage_TopLevel"
            handle.content.mobs.append(composition)

            edit_rate_value = AAFRational(edit_rate.numerator, edit_rate.denominator)

            timecode_slot = composition.create_timeline_slot(
                edit_rate=edit_rate_value,
                slot_id=1,
            )
            timecode_slot.name = "Timecode"
            timecode_slot.origin = 0
            timecode_segment = handle.create.Timecode(length=timeline_length)
            timecode_segment.start = timeline_start
            timecode_segment.fps = mode.nominal_fps
            timecode_segment.drop = mode.drop_frame
            timecode_slot.segment = timecode_segment

            audio_slot = composition.create_timeline_slot(
                edit_rate=edit_rate_value,
                slot_id=2,
            )
            audio_slot.name = "Marker Guide"
            audio_slot.origin = 0
            audio_slot["PhysicalTrackNumber"].value = 1
            audio_sequence = handle.create.Sequence(media_kind="sound")
            audio_sequence["Components"].value = []
            audio_sequence.components.append(
                handle.create.Filler(media_kind="sound", length=timeline_length)
            )
            audio_slot.segment = audio_sequence

            marker_slot = handle.create.EventMobSlot()
            marker_slot.slot_id = 2001
            marker_slot.name = EXPORT_TRACK_NAME
            marker_slot.edit_rate = edit_rate_value
            marker_slot["EventSlotOrigin"].value = 0
            marker_sequence = handle.create.Sequence()
            marker_sequence["Components"].value = []
            marker_sequence["DataDefinition"].value = handle.dictionary.lookup_datadef(
                "DescriptiveMetadata"
            )
            marker_slot.segment = marker_sequence
            composition.slots.append(marker_slot)

            for candidate in candidates:
                marker = _create_comment_marker(
                    handle,
                    candidate.start_edit_unit - timeline_start,
                    max(0, candidate.length_edit_units),
                    candidate.name,
                    candidate.comment,
                    candidate.start_tc,
                )
                marker_sequence.components.append(marker)
    except Exception as exc:  # pragma: no cover - exercised via CLI/tests
        raise AAFExportError(f"Could not write markers.aaf: {exc}") from exc


def _candidates_from_flags(
    *,
    flags: list[FlaggedRegion],
    sample_rate: int,
    mode: TimecodeMode,
    time_reference_samples: int | None = None,
    group_id: str | None = None,
    asset_label: str | None = None,
) -> list[MarkerCandidate]:
    out: list[MarkerCandidate] = []
    source_start_edit_unit = sample_to_edit_units(
        0,
        sample_rate,
        mode,
        start_time_reference_samples=time_reference_samples,
    )
    for flag in flags:
        start_edit_unit = sample_to_edit_units(
            flag.start_sample,
            sample_rate,
            mode,
            start_time_reference_samples=time_reference_samples,
        )
        end_edit_unit = sample_to_edit_units(
            flag.end_sample,
            sample_rate,
            mode,
            start_time_reference_samples=time_reference_samples,
        )
        out.append(
            MarkerCandidate(
                name=_marker_name(flag.code, group_id=group_id, asset_label=asset_label),
                comment=_marker_comment(
                    flag,
                    group_id=group_id,
                    asset_label=asset_label,
                ),
                code=flag.code,
                metric=flag.metric,
                source_start_edit_unit=source_start_edit_unit,
                start_sample=flag.start_sample,
                end_sample=flag.end_sample,
                start_tc=flag.start_tc,
                end_tc=flag.end_tc,
                sample_rate=sample_rate,
                start_edit_unit=start_edit_unit,
                length_edit_units=max(1, end_edit_unit - start_edit_unit),
                group_id=group_id,
                asset_label=asset_label,
            )
        )
    return out


def _group_sample_rate(group: Group) -> int | None:
    for file_report in group.files:
        if file_report.role == "pm":
            return file_report.sample_rate
    return group.files[0].sample_rate if group.files else None


def _marker_name(
    code: str,
    *,
    group_id: str | None,
    asset_label: str | None,
) -> str:
    parts = [part for part in (group_id, asset_label, code) if part]
    return " ".join(parts)


def _marker_comment(
    flag: FlaggedRegion,
    *,
    group_id: str | None,
    asset_label: str | None,
) -> str:
    context = " — ".join(part for part in (group_id, asset_label) if part)
    context_prefix = f"{context} — " if context else ""
    return f"[{flag.code}] {flag.metric} {_format_metric_value(flag)} — {context_prefix}{flag.detail}"


def _format_metric_value(flag: FlaggedRegion) -> str:
    if flag.metric == "dialog_bleed_score":
        return f"{flag.value:.2f}"
    return f"{flag.value:.1f}"


def _short_failure_label(candidate: MarkerCandidate) -> str:
    if candidate.code == "NULL":
        return "null mismatch"
    if candidate.code == "ME":
        return "dialog bleed"
    if candidate.code == "LOUDNESS":
        return "true peak over"
    if candidate.code == "DOWNMIX":
        return (
            "mono compatibility"
            if candidate.metric == "stereo_correlation"
            else "downmix mismatch"
        )
    return candidate.metric.replace("_", " ").lower()


def _timeline_start_edit_units(candidates: list[MarkerCandidate]) -> int:
    if not candidates:
        return 0
    return min(candidate.source_start_edit_unit for candidate in candidates)


def _timeline_length_edit_units(candidates: list[MarkerCandidate], *, timeline_start: int) -> int:
    if not candidates:
        return 1
    return max(
        1,
        max(
            (candidate.start_edit_unit - timeline_start) + max(1, candidate.length_edit_units)
            for candidate in candidates
        ),
    )


def _throttle_loudness_candidates(
    candidates: list[MarkerCandidate],
) -> list[MarkerCandidate]:
    if not candidates:
        return []

    throttled: list[MarkerCandidate] = []
    last_kept_seconds: dict[tuple[str | None, str | None], float] = {}
    for candidate in candidates:
        if candidate.code != "LOUDNESS":
            throttled.append(candidate)
            continue

        source_key = (candidate.group_id, candidate.asset_label)
        start_seconds = candidate.start_sample / float(candidate.sample_rate)
        last_seconds = last_kept_seconds.get(source_key)
        if (
            last_seconds is not None
            and start_seconds < last_seconds + LOUDNESS_MARKER_MIN_SPACING_SECONDS
        ):
            continue
        last_kept_seconds[source_key] = start_seconds
        throttled.append(candidate)
    return throttled


def _create_comment_marker(
    handle,
    position: int,
    length: int,
    title: str,
    comment: str,
    start_tc: str,
):
    marker = component_module.CommentMarker.__new__(component_module.CommentMarker)
    marker.root = handle
    component_module.Component.__init__(
        marker,
        media_kind="DescriptiveMetadata",
        length=max(0, length),
    )
    try:
        marker.name = title
    except Exception:
        pass

    marker["Position"].value = position
    marker["Comment"].value = comment
    marker["CommentMarkerTime"].value = start_tc

    try:
        marker["Annotation"].value = comment
    except Exception:
        pass
    try:
        marker["UserComments"].append(
            _create_string_tagged_value(handle, "Comment", comment)
        )
        marker["UserComments"].append(
            _create_string_tagged_value(handle, "Label", title)
        )
        marker["UserComments"].append(
            _create_string_tagged_value(handle, "Detail", comment)
        )
    except Exception:
        pass
    try:
        marker["CommentMarkerAnnotationList"].value = comment
    except Exception:
        pass
    try:
        marker["CommentMarkerAttributeList"].append(
            _create_string_tagged_value(handle, "_ATN_CRM_COM", comment)
        )
        marker["CommentMarkerAttributeList"].append(
            _create_string_tagged_value(handle, "Label", title)
        )
    except Exception:
        pass
    return marker


def _create_string_tagged_value(handle, name: str, value: str):
    tag = handle.create.TaggedValue()
    tag.name = name
    tag.encode_value(value, handle.dictionary.lookup_typedef("aafString"))
    return tag


def _time_reference_samples(item) -> int | None:
    return getattr(item, "_time_reference_samples", None)


def _analysis_sample_rate(item) -> int | None:
    return getattr(item, "_sample_rate", None)
