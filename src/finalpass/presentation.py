from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class DisplayItem:
    path: Path
    source_kind: str | None
    source_paths: tuple[str, ...]
    member_legs: tuple[str, ...]
    presentation_label: str | None
    channel_config_actual: str | None
    role: str | None


_CODE_LABELS = {
    "alternate_presentation": "Other layout available",
    "AmbiguousFamilyKey": "Ambiguous split family",
    "audio_format_error": "Audio format error",
    "broadband_lfe": "Broadband content in the LFE slot",
    "channel_imbalance": "Channel imbalance",
    "channel_config_label_mismatch": "Channel label does not match audio header",
    "ChannelConfigLabelMismatch": "Channel label does not match audio header",
    "channel_count_mismatch": "Channel-count mismatch",
    "ChannelMismatchError": "Channel-count mismatch",
    "DialogFallbackAmbiguity": "Alternate-layout DX fallback is ambiguous",
    "dialog_fallback_ambiguous": "Alternate-layout DX fallback is ambiguous",
    "dialog_loudness": "Dialog loudness",
    "duplicate_target_layout": "Competing same-layout asset",
    "DuplicateTargetLayoutRoleError": "Competing same-layout asset",
    "duplicate_channels": "Duplicated channels",
    "dx": "DX",
    "empty_signal": "Empty signal",
    "insufficient_active_content": "Not enough active content to compare channels",
    "lfe_synthesized_from_5_0_source": "LFE is FinalPass-synthesized silence from a 5.0 source; LFE checks suppressed",
    "not_applicable_mono": "Not applicable to a mono asset",
    "polarity_inversion": "Polarity inversion",
    "silent_leg": "Silent leg",
    "file_too_short_for_integrated": "File is too short for integrated loudness",
    "fx": "FX",
    "IncompatiblePresentation": "Incompatible split presentation",
    "insufficient_gated_content": "Not enough gated content to measure",
    "insufficient_stems_for_auto_null": "Auto-null skipped: no valid same-layout DX+MX+FX or DX+ME set",
    "InvalidFamily": "Invalid split family",
    "InvalidLegSet": "Invalid split-leg set",
    "me": "M&E",
    "missing_dx_or_me": "M&E skipped: missing DX or M&E asset",
    "MissingFamilyLayoutPrintmaster": "No printmaster matches the selected loudness family",
    "MissingLeg": "Missing split leg",
    "MissingPath": "Missing source file",
    "missing_target_layout_printmaster": "Auto-null skipped: missing target-layout printmaster",
    "MissingTargetLayoutPrintmaster": "Missing target-layout printmaster",
    "MissingTargetLayoutRole": "Missing target-layout asset",
    "MixedSampleCount": "Mixed sample counts",
    "MixedSampleRate": "Mixed sample rates",
    "mono_downmix_excluding_lfe": "Speech-band downmix excluding LFE",
    "mx": "MX",
    "NonMonoMember": "Non-mono split member",
    "no_role_pattern_match": "No role token matched",
    "null": "Auto-null",
    "opt": "Optional",
    "pm_loudness": "PM loudness",
    "pm": "PM",
    "selected_for_dialog_fallback": "Used as dialog fallback from another layout",
    "selected_for_family_dialog_loudness": "Included for matched family dialog loudness",
    "selected_for_family_loudness": "Included for matched family loudness",
    "selected_for_loudness_only": "Included for loudness only",
    "selected_for_target_layout": "Chosen for target layout",
    "sample_count_mismatch": "Sample-count mismatch",
    "sample_rate_mismatch": "Sample-rate mismatch",
    "silent_or_below_gate": "Silent or below gate",
    "SubtypeMismatch": "Mixed audio subtypes",
    "unknown_loudness": "Unclassified loudness",
    "unknown": "Unknown",
    "unknown_skipped": "Unclassified asset was not used",
    "UnsupportedChannelConfig": "Unsupported channel layout",
}
_SKIP_DETAIL_LABELS = {
    "insufficient_stems_for_auto_null": "Auto-null was not run because no same-layout DX+MX+FX or DX+ME set was available.",
    "missing_dx_or_me": "M&E was not run because no DX and M&E pair was available.",
    "missing_target_layout_printmaster": "Auto-null was not run because no target-layout printmaster was available.",
}
_SPECIAL_GROUP_LABELS = {
    "AUDIO FILES": "Ungrouped delivery (Audio Files)",
}
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# How each flagged-region metric reads: decimal places, unit suffix, and the
# terminal column header. One entry per metric; every surface (terminal tables,
# HTML tables, timeline tooltips) formats through here so a new metric is a
# single edit.
_METRIC_DISPLAY = {
    "residual_rms_dbfs": (1, "dBFS", "peak residual"),
    "dialog_bleed_score": (2, "", "bleed score"),
    "true_peak_dbtp": (1, "dBTP", "peak true peak"),
}
_METRIC_DISPLAY_FALLBACK = (1, "dBFS", "value")


def logical_asset_display_name(item: Any) -> str:
    display = _to_display_item(item)
    stem = display.path.stem
    if display.source_kind == "split_mono" and display.member_legs:
        suffix = f".{display.member_legs[0]}"
        if stem.upper().endswith(suffix.upper()):
            return stem[:-len(suffix)]
    return display.path.name


def source_kind_label(source_kind: str | None) -> str:
    if source_kind == "split_mono":
        return "split mono"
    if source_kind == "interleaved":
        return "interleaved"
    return "unknown"


def source_summary(item: Any) -> str:
    display = _to_display_item(item)
    parts = [source_kind_label(display.source_kind)]
    if display.channel_config_actual:
        parts.append(display.channel_config_actual)
    if _is_five_point_zero_source(display):
        parts.append("5.0 source; silent LFE")
    elif display.presentation_label and display.presentation_label != display.channel_config_actual:
        parts.append(display.presentation_label)
    if display.source_kind == "split_mono":
        count = max(len(display.source_paths), len(display.member_legs), 1)
        noun = "mono file" if count == 1 else "mono files"
        parts.append(f"{count} {noun}")
    return " · ".join(parts)


def asset_menu_label(item: Any) -> str:
    display = _to_display_item(item)
    parts: list[str] = []
    if display.role:
        parts.append(display.role)
    parts.append(logical_asset_display_name(item))
    if display.channel_config_actual:
        if _is_five_point_zero_source(display):
            parts.append("5.0 source -> 5.1")
        else:
            parts.append(display.channel_config_actual)
    if display.source_kind == "split_mono":
        count = max(len(display.source_paths), len(display.member_legs), 1)
        noun = "mono file" if count == 1 else "mono files"
        parts.append(f"split mono ({count} {noun})")
    else:
        parts.append(source_kind_label(display.source_kind))
    return " · ".join(parts)


def display_group_name(group_id: str) -> str:
    return _SPECIAL_GROUP_LABELS.get(group_id, group_id)


def humanize_code(code: str | None) -> str:
    if not code:
        return "—"
    parts = [part.strip() for part in code.split(";") if part.strip()]
    if not parts:
        return "—"
    return "; ".join(_humanize_single_code(part) for part in parts)


def humanize_code_list(codes: list[str] | tuple[str, ...]) -> str:
    if not codes:
        return "—"
    return ", ".join(_humanize_single_code(code) for code in codes)


def skip_detail(reason: str | None) -> str:
    if not reason:
        return "This analysis was not run."
    return _SKIP_DETAIL_LABELS.get(reason, humanize_code(reason))


def format_metric_value(value: float | None, *, metric: str) -> str:
    """Render a flagged-region value or threshold in its metric's own terms."""
    if value is None:
        return "—"
    digits, unit, _ = _METRIC_DISPLAY.get(metric, _METRIC_DISPLAY_FALLBACK)
    text = f"{value:.{digits}f}"
    return f"{text} {unit}" if unit else text


def stem_strategy_label(strategy: str | None) -> str:
    """Render an auto-null stem strategy for display."""
    if not strategy:
        return "—"
    return strategy.replace("_", "+")


def metric_value_header(metric: str) -> str:
    """Column header for the value column of a flagged-region table."""
    _, _, header = _METRIC_DISPLAY.get(metric, _METRIC_DISPLAY_FALLBACK)
    return header


def format_target_limit(check: Any) -> str:
    """Render a loudness check's target ± tolerance, or its limit."""
    if check.target is not None and check.tolerance is not None:
        return f"{check.target:.1f} ±{check.tolerance:.1f}"
    if check.limit is not None:
        return f"≤ {check.limit:.1f}"
    return "—"


def blocking_issue_count(item: Any) -> int:
    if hasattr(item, "discovery_errors") and hasattr(item, "groups"):
        return len(item.discovery_errors) + sum(len(group.errors) for group in item.groups)
    if hasattr(item, "errors"):
        return len(item.errors)
    return 0


def verdict_explainer(*, overall_pass: bool, failed: int, blocking_issues: int) -> str:
    verdict = "PASS" if overall_pass else "FAIL"
    checks_noun = "check failure" if failed == 1 else "check failures"
    issues_noun = "blocking issue" if blocking_issues == 1 else "blocking issues"
    return f"{verdict} — {failed} {checks_noun}, {blocking_issues} {issues_noun}"


def _to_display_item(item: Any) -> DisplayItem:
    if hasattr(item, "logical_asset"):
        logical_asset = item.logical_asset
        return DisplayItem(
            path=logical_asset.canonical_path,
            source_kind=logical_asset.source_kind,
            source_paths=tuple(str(path) for path in logical_asset.source_paths),
            member_legs=tuple(logical_asset.member_legs),
            presentation_label=logical_asset.presentation_label,
            channel_config_actual=logical_asset.channel_config_actual,
            role=getattr(item, "role", None),
        )

    return DisplayItem(
        path=Path(getattr(item, "path")),
        source_kind=getattr(item, "source_kind", None),
        source_paths=tuple(getattr(item, "source_paths", []) or ()),
        member_legs=tuple(getattr(item, "member_legs", []) or ()),
        presentation_label=getattr(item, "presentation_label", None),
        channel_config_actual=getattr(item, "channel_config_actual", None),
        role=getattr(item, "role", None),
    )


def _is_five_point_zero_source(display: DisplayItem) -> bool:
    return (
        display.source_kind == "split_mono"
        and display.channel_config_actual == "5.1"
        and set(display.member_legs) == {"L", "R", "C", "Ls", "Rs"}
    )


def _humanize_single_code(code: str) -> str:
    mapped = _CODE_LABELS.get(code)
    if mapped is not None:
        return mapped

    text = code.replace("_", " ").replace("-", " ")
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "—"
    return text[:1].upper() + text[1:]
