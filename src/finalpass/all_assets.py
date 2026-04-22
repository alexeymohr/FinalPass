"""Logical-asset discovery and lightweight classification for ``finalpass all``.

SM-3 moves the folder-level pipeline from raw files to logical assets while
preserving the existing classifier patterns as an optional supplement.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .assets import DiscoveryError, LogicalAsset, discover_logical_assets_in_folder, read_logical_asset
from .audio_io import AudioFile
from .classify import ClassifiedFile, ClassifierConfig, classify_file
from .errors import AmbiguousClassificationError
from .models import FileRole


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
    groups: "OrderedDict[str, list[ClassifiedLogicalAsset]]" = OrderedDict()
    unclassified: list[ClassifiedLogicalAsset] = []

    for logical_asset in discovery.assets:
        classified = classify_logical_asset(logical_asset, cfg)
        groups.setdefault(logical_asset.group_id, []).append(classified)
        if classified.role == "unknown":
            unclassified.append(classified)

    return AssetFolderScan(
        groups=groups,
        unclassified=unclassified,
        discovery_errors=discovery.errors,
        ignored_paths=discovery.ignored_paths,
    )


def classify_logical_asset(logical_asset: LogicalAsset, cfg: ClassifierConfig) -> ClassifiedLogicalAsset:
    classified_file = classify_file(logical_asset.canonical_path, cfg)
    resolved_role = _resolve_role(
        logical_asset=logical_asset,
        classifier_role=classified_file.role,
    )
    channel_hint = logical_asset.channel_config_hint or classified_file.channel_config_hint
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
    )


def to_classified_file(asset: ClassifiedLogicalAsset) -> ClassifiedFile:
    return ClassifiedFile(
        path=asset.logical_asset.canonical_path,
        role=asset.role,
        group_id=asset.logical_asset.group_id,
        channel_config_hint=asset.channel_config_hint,
    )


def _resolve_role(*, logical_asset: LogicalAsset, classifier_role: FileRole) -> FileRole:
    asset_role = logical_asset.role_hint or "unknown"
    if asset_role == "unknown" and classifier_role == "unknown":
        return "unknown"
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
