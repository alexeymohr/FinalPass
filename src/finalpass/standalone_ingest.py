"""Standalone logical-asset resolution helpers for SM-2.

This module is the one shared standalone-ingest seam. It builds on the
accepted SM-1 :mod:`finalpass.assets` layer and keeps split/interleaved
resolution out of the individual CLI subcommands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assets import AssetResolutionError, LogicalAsset, read_logical_asset, resolve_logical_asset
from .audio_io import AudioFile
from .models import AnalysisInputFile


@dataclass(frozen=True)
class ResolvedStandaloneAsset:
    logical_asset: LogicalAsset
    audio: AudioFile


def resolve_asset_reference(path: Path) -> LogicalAsset:
    """Resolve one standalone input path to an interleaved or split logical asset."""
    seed_path = Path(path).expanduser().resolve()
    try:
        return resolve_logical_asset(seed_path)
    except AssetResolutionError as exc:
        raise AssetResolutionError(_seed_specific_error(seed_path, exc.error)) from exc


def resolve_standalone_asset(path: Path) -> ResolvedStandaloneAsset:
    logical_asset = resolve_asset_reference(path)
    audio_asset = read_logical_asset(logical_asset)
    return ResolvedStandaloneAsset(
        logical_asset=logical_asset,
        audio=AudioFile(
            path=logical_asset.canonical_path,
            data=audio_asset.samples,
            sample_rate=audio_asset.sample_rate,
            bit_depth=logical_asset.bit_depth or 0,
            channel_count=audio_asset.channel_count,
            duration_seconds=logical_asset.sample_count / float(logical_asset.sample_rate),
        ),
    )


def describe_analysis_input(resolved: ResolvedStandaloneAsset) -> AnalysisInputFile:
    logical_asset = resolved.logical_asset
    audio = resolved.audio
    return AnalysisInputFile(
        path=str(logical_asset.canonical_path),
        sample_rate=audio.sample_rate,
        bit_depth=audio.bit_depth,
        channel_count=audio.channel_count,
        channel_config_actual=logical_asset.channel_config_actual,
        duration_seconds=round(audio.duration_seconds, 3),
        source_kind=logical_asset.source_kind,
        source_paths=[str(path) for path in logical_asset.source_paths],
        member_legs=list(logical_asset.member_legs),
        presentation_label=logical_asset.presentation_label,
    )


def _seed_specific_error(seed_path: Path, error):
    return type(error)(
        type=error.type,
        message=f"{seed_path.name}: {error.message}",
        paths=error.paths,
        family_key=error.family_key,
        group_hint=error.group_hint,
    )
