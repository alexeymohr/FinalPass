from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    if display.presentation_label and display.presentation_label != display.channel_config_actual:
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
        parts.append(display.channel_config_actual)
    if display.source_kind == "split_mono":
        count = max(len(display.source_paths), len(display.member_legs), 1)
        noun = "mono file" if count == 1 else "mono files"
        parts.append(f"split mono ({count} {noun})")
    else:
        parts.append(source_kind_label(display.source_kind))
    return " · ".join(parts)


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
