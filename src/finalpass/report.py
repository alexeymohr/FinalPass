"""Phase 5 HTML report renderer.

Rendering is deliberately model-only: this module consumes already-built
Pydantic report models and never reopens audio files or reruns analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TypeAlias

from jinja2 import Environment, FunctionLoader, TemplateNotFound, select_autoescape

from .errors import ReportRenderError
from .models import AllReport, FileReport, FlaggedRegion, Group, MEReport, NullReport, Report
from .presentation import (
    blocking_issue_count,
    display_group_name,
    format_metric_value,
    format_target_limit,
    humanize_code,
    humanize_code_list,
    logical_asset_display_name,
    skip_detail,
    source_kind_label,
    source_summary,
    verdict_explainer,
)

RenderableReport: TypeAlias = Report | NullReport | MEReport | AllReport

_TEMPLATE_PACKAGE = "finalpass.templates"
_SVG_VIEWBOX_WIDTH = 1000.0
_SVG_TRACK_X = 32.0
_SVG_TRACK_WIDTH = 936.0


@dataclass(frozen=True)
class TimelineRect:
    x: float
    width: float
    title: str


@dataclass(frozen=True)
class TimelineLane:
    view_box_width: float
    track_x: float
    track_width: float
    duration_seconds: float
    duration_label: str
    caption: str
    empty_message: str
    rects: list[TimelineRect]


@dataclass(frozen=True)
class SourcePathEntry:
    name: str
    folder: str
    path: str


def render_report_html(
    report: RenderableReport,
    *,
    artifacts: tuple[str, ...] = ("report.json", "report.html"),
) -> str:
    """Render self-contained HTML for an existing report model."""
    try:
        template = _template_name(report)
        return _environment().get_template(template).render(
            report=report,
            command=_command_name(report),
            artifacts=artifacts,
            title=_title_for_report(report),
            subtitle=_subtitle_for_report(report),
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI/tests
        raise ReportRenderError(f"Could not render report.html: {exc}") from exc


def grouped_templates_present() -> bool:
    """Lightweight packaging probe for tests."""
    required = ("base.html.j2", "standalone.html.j2", "all.html.j2", "_macros.html.j2")
    root = resources.files(_TEMPLATE_PACKAGE)
    return all(root.joinpath(name).is_file() for name in required)


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=FunctionLoader(_load_template),
        autoescape=select_autoescape(enabled_extensions=("html", "xml"), default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt_float"] = _fmt_float
    env.filters["fmt_seconds"] = _fmt_seconds
    env.filters["basename"] = _basename
    env.filters["verdict_label"] = _verdict_label
    env.filters["verdict_class"] = _verdict_class
    env.filters["metric_value"] = format_metric_value
    env.globals.update(
        format_target_limit=format_target_limit,
        timeline_lane=_timeline_lane,
        group_anchor_file=_group_anchor_file,
        group_duration_seconds=_group_duration_seconds,
        group_sample_rate=_group_sample_rate,
        channel_display=_channel_display,
        source_path_manifest=_source_path_manifest,
        blocking_issue_count=blocking_issue_count,
        display_label=logical_asset_display_name,
        display_group_name=display_group_name,
        humanize_code=humanize_code,
        humanize_code_list=humanize_code_list,
        skip_detail=skip_detail,
        source_summary=source_summary,
        source_kind_label=source_kind_label,
        verdict_explainer=verdict_explainer,
    )
    return env


def _load_template(name: str) -> str:
    try:
        return resources.files(_TEMPLATE_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TemplateNotFound(name) from exc


def _template_name(report: RenderableReport) -> str:
    return "all.html.j2" if isinstance(report, AllReport) else "standalone.html.j2"


def _command_name(report: RenderableReport) -> str:
    return report.command if hasattr(report, "command") else "loudness"


def _title_for_report(report: RenderableReport) -> str:
    if isinstance(report, Report):
        return "Loudness Report"
    if isinstance(report, NullReport):
        return "Stem-Sum Null Report"
    if isinstance(report, MEReport):
        return "M&E Dialogue-Bleed Report"
    return "Folder QC Report"


def _subtitle_for_report(report: RenderableReport) -> str:
    if isinstance(report, (Report, AllReport)):
        return f"{report.spec.display_name} ({report.spec.source})"
    if isinstance(report, NullReport):
        return logical_asset_display_name(report.printmaster)
    if isinstance(report, MEReport):
        return logical_asset_display_name(report.me_file)
    return ""


def _fmt_float(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_seconds(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}s"


def _basename(value: str) -> str:
    return Path(value).name


def _source_path_manifest(report: RenderableReport) -> list[SourcePathEntry]:
    paths = list(dict.fromkeys(_source_paths_for_report(report)))
    return [
        SourcePathEntry(
            name=Path(path).name,
            folder=str(Path(path).parent),
            path=str(path),
        )
        for path in paths
    ]


def _source_paths_for_report(report: RenderableReport) -> list[str]:
    if isinstance(report, AllReport):
        items = []
        for group in report.groups:
            items.extend(group.assets)
            items.extend(group.files)
        items.extend(report.files)
        items.extend(report.unclassified)
        items.extend(report.discovery_errors)
        return [
            source_path
            for item in items
            for source_path in _source_paths_for_item(item)
        ]
    if isinstance(report, Report):
        return [
            source_path
            for file_report in report.files
            for source_path in _source_paths_for_item(file_report)
        ]
    if isinstance(report, NullReport):
        return [
            source_path
            for item in (report.printmaster, *report.stems)
            for source_path in _source_paths_for_item(item)
        ]
    if isinstance(report, MEReport):
        return [
            source_path
            for item in (report.me_file, report.dx_file)
            for source_path in _source_paths_for_item(item)
        ]
    return []


def _source_paths_for_item(item) -> list[str]:
    paths = [str(path) for path in (getattr(item, "source_paths", []) or [])]
    path = getattr(item, "path", None)
    if path:
        paths.append(str(path))
    return paths


def _verdict_label(value: bool | None, skipped: bool = False) -> str:
    if skipped or value is None:
        return "SKIPPED"
    return "PASS" if value else "FAIL"


def _verdict_class(value: bool | None, skipped: bool = False) -> str:
    if skipped or value is None:
        return "badge-skipped"
    return "badge-pass" if value else "badge-fail"


def _channel_display(channel_config_actual: str | None, channel_count: int) -> str:
    return channel_config_actual or f"{channel_count}ch"


def _group_anchor_file(group: Group) -> FileReport | None:
    for file_report in group.files:
        if file_report.role == "pm":
            return file_report
    return group.files[0] if group.files else None


def _group_duration_seconds(group: Group) -> float:
    anchor = _group_anchor_file(group)
    return anchor.duration_seconds if anchor is not None else 0.0


def _group_sample_rate(group: Group) -> int:
    anchor = _group_anchor_file(group)
    return anchor.sample_rate if anchor is not None else 1


def _timeline_lane(flags: list[FlaggedRegion], *, sample_rate: int, duration_seconds: float) -> TimelineLane:
    duration_seconds = max(float(duration_seconds), 0.0)
    total_samples = max(1, int(round(duration_seconds * sample_rate)))
    rects: list[TimelineRect] = []

    for flag in flags:
        start_ratio = min(max(flag.start_sample / total_samples, 0.0), 1.0)
        end_ratio = min(max(flag.end_sample / total_samples, start_ratio), 1.0)
        x = _SVG_TRACK_X + (_SVG_TRACK_WIDTH * start_ratio)
        x = min(x, (_SVG_TRACK_X + _SVG_TRACK_WIDTH) - 4.0)
        width = max(4.0, _SVG_TRACK_WIDTH * (end_ratio - start_ratio))
        max_width = (_SVG_TRACK_X + _SVG_TRACK_WIDTH) - x
        width = max(4.0, min(width, max_width))
        rects.append(TimelineRect(
            x=x,
            width=width,
            title=_flag_title(flag),
        ))

    return TimelineLane(
        view_box_width=_SVG_VIEWBOX_WIDTH,
        track_x=_SVG_TRACK_X,
        track_width=_SVG_TRACK_WIDTH,
        duration_seconds=duration_seconds,
        duration_label=_fmt_seconds(duration_seconds),
        caption="Merged flagged regions only; no per-window trace persisted.",
        empty_message="No flagged regions.",
        rects=rects,
    )


def _flag_title(flag: FlaggedRegion) -> str:
    value_label = format_metric_value(flag.value, metric=flag.metric)
    threshold_label = format_metric_value(flag.threshold, metric=flag.metric)
    return "\n".join([
        f"{flag.code} {flag.metric}",
        f"{flag.start_tc} → {flag.end_tc}",
        f"duration: {flag.duration_seconds:.2f}s",
        f"value: {value_label}",
        f"threshold: {threshold_label}",
        flag.detail,
    ])
