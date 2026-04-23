"""Logical-asset discovery and lightweight classification for ``finalpass all``.

SM-3 moves the folder-level pipeline from raw files to logical assets while
preserving the existing classifier patterns as an optional supplement.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from .assets import DiscoveryError, LogicalAsset, discover_logical_assets, discover_logical_assets_in_folder, read_logical_asset
from .audio_io import AudioFile
from .classify import ClassifiedFile, ClassifierConfig, classify_file
from .errors import AmbiguousClassificationError
from .models import FileRole
from .prep_folders import PrepBucketHint

_COMPOUND_ROLE_PATTERN = re.compile(
    r"(?i)(PM|PRINTMASTER|MIX|FINAL|COMP|DX|DIA|DIALOG|DIALOGUE|MX|MUS|MUSIC|FX|SFX|EFFECTS|ME|M_AND_E|MANDE|OPT|OPTIONAL|NARR|NARRATION)\s*-\s*"
    r"(PM|PRINTMASTER|MIX|FINAL|COMP|DX|DIA|DIALOG|DIALOGUE|MX|MUS|MUSIC|FX|SFX|EFFECTS|ME|M_AND_E|MANDE|OPT|OPTIONAL|NARR|NARRATION)"
)


@dataclass(frozen=True)
class ClassifiedLogicalAsset:
    logical_asset: LogicalAsset
    role: FileRole
    channel_config_hint: str | None


@dataclass(frozen=True)
class AssetFolderScan:
    groups: "OrderedDict[str, list[ClassifiedLogicalAsset]]"
    unclassified: list[ClassifiedLogicalAsset]
    discovery_errors: list[DiscoveryError]
    ignored_paths: list[Path]


def discover_folder_assets(folder: Path, cfg: ClassifierConfig) -> AssetFolderScan:
    discovery = discover_logical_assets_in_folder(folder)
    return _build_asset_scan(
        assets=discovery.assets,
        discovery_errors=discovery.errors,
        ignored_paths=discovery.ignored_paths,
        cfg=cfg,
        path_hints=None,
    )


def discover_assets_from_paths(
    paths: list[Path] | tuple[Path, ...],
    cfg: ClassifierConfig,
    *,
    path_hints: Mapping[Path, PrepBucketHint] | None = None,
) -> AssetFolderScan:
    discovery = discover_logical_assets(paths, strict_explicit_split_members=True)
    assets = _apply_prep_hints_to_assets(discovery.assets, path_hints)
    return _build_asset_scan(
        assets=assets,
        discovery_errors=discovery.errors,
        ignored_paths=discovery.ignored_paths,
        cfg=cfg,
        path_hints=path_hints,
    )


def _build_asset_scan(
    *,
    assets: list[LogicalAsset],
    discovery_errors: list[DiscoveryError],
    ignored_paths: list[Path],
    cfg: ClassifierConfig,
    path_hints: Mapping[Path, PrepBucketHint] | None,
) -> AssetFolderScan:
    groups: "OrderedDict[str, list[ClassifiedLogicalAsset]]" = OrderedDict()
    unclassified: list[ClassifiedLogicalAsset] = []

    for logical_asset in assets:
        classified = classify_logical_asset(logical_asset, cfg, path_hints=path_hints)
        groups.setdefault(logical_asset.group_id, []).append(classified)
        if classified.role == "unknown":
            unclassified.append(classified)

    return AssetFolderScan(
        groups=groups,
        unclassified=unclassified,
        discovery_errors=discovery_errors,
        ignored_paths=ignored_paths,
    )


def classify_logical_asset(
    logical_asset: LogicalAsset,
    cfg: ClassifierConfig,
    *,
    path_hints: Mapping[Path, PrepBucketHint] | None = None,
) -> ClassifiedLogicalAsset:
    try:
        classified_file = classify_file(logical_asset.canonical_path, cfg)
    except AmbiguousClassificationError:
        if logical_asset.role_hint != "unknown" or not _has_hyphenated_role_compound(logical_asset.canonical_path.stem):
            raise
        classified_file = ClassifiedFile(
            path=logical_asset.canonical_path,
            role="unknown",
            group_id=logical_asset.group_id,
            channel_config_hint=logical_asset.channel_config_hint,
        )
    prep_hint = None
    if path_hints is not None:
        prep_hint = path_hints.get(logical_asset.canonical_path.resolve())
    resolved_role = _resolve_role(
        logical_asset=logical_asset,
        classifier_role=classified_file.role,
        prep_role_hint=prep_hint.role_hint if prep_hint is not None else None,
    )
    channel_hint = logical_asset.channel_config_hint or classified_file.channel_config_hint
    if channel_hint is None and prep_hint is not None:
        channel_hint = prep_hint.layout_hint
    return ClassifiedLogicalAsset(
        logical_asset=logical_asset,
        role=resolved_role,
        channel_config_hint=channel_hint,
    )


def read_classified_audio(asset: ClassifiedLogicalAsset) -> AudioFile:
    logical_asset = asset.logical_asset
    audio_asset = read_logical_asset(logical_asset)
    return AudioFile(
        path=logical_asset.canonical_path,
        data=audio_asset.samples,
        sample_rate=audio_asset.sample_rate,
        bit_depth=logical_asset.bit_depth or 0,
        channel_count=audio_asset.channel_count,
        duration_seconds=logical_asset.sample_count / float(logical_asset.sample_rate),
        time_reference_samples=audio_asset.time_reference_samples,
    )


def to_classified_file(asset: ClassifiedLogicalAsset) -> ClassifiedFile:
    return ClassifiedFile(
        path=asset.logical_asset.canonical_path,
        role=asset.role,
        group_id=asset.logical_asset.group_id,
        channel_config_hint=asset.channel_config_hint,
    )


def _resolve_role(
    *,
    logical_asset: LogicalAsset,
    classifier_role: FileRole,
    prep_role_hint: FileRole | None = None,
) -> FileRole:
    asset_role = logical_asset.role_hint or "unknown"
    if asset_role == "unknown" and classifier_role == "unknown":
        return prep_role_hint or "unknown"
    if asset_role == "unknown":
        return classifier_role
    if classifier_role == "unknown":
        return asset_role
    if asset_role == classifier_role:
        return asset_role
    raise AmbiguousClassificationError(
        f"{logical_asset.canonical_path.name}: logical-asset discovery resolved role "
        f"{asset_role!r} but classifier patterns resolved {classifier_role!r}. "
        "Rename the asset or adjust the classifier config so only one role applies."
    )


def _apply_prep_hints_to_assets(
    assets: list[LogicalAsset],
    path_hints: Mapping[Path, PrepBucketHint] | None,
) -> list[LogicalAsset]:
    if path_hints is None:
        return assets

    out: list[LogicalAsset] = []
    for asset in assets:
        prep_hints = {
            hint.group_hint
            for path in asset.source_paths
            if (hint := path_hints.get(path.resolve())) is not None and hint.group_hint
        }
        if len(prep_hints) == 1:
            group_hint = next(iter(prep_hints))
            out.append(replace(asset, group_id=group_hint, group_hint=group_hint))
            continue
        out.append(asset)
    return out


def _has_hyphenated_role_compound(stem: str) -> bool:
    for match in _COMPOUND_ROLE_PATTERN.finditer(stem):
        if match.group(1).upper() != match.group(2).upper():
            return True
    return False
