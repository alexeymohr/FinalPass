"""Internal logical-asset discovery and split-mono assembly.

SM-1 adds an ingest layer above raw WAV/BWF files. A logical asset is either:

- one interleaved file already carrying its full channel layout, or
- a validated split-mono family assembled in memory in SMPTE order.

SM-2 wires this layer into the standalone `loudness`, `null`, and `me`
commands. The integrated `all` path still uses the older per-file flow until a
later split-mono redesign lands.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Literal

import numpy as np

from .audio_io import AudioFile, channel_config_from_count, probe_wav, read_wav
from .errors import FinalPassError

RoleHint = Literal["pm", "dx", "mx", "fx", "me", "opt", "unknown"]
ChannelConfigActual = Literal["mono", "stereo", "5.1", "7.1"]
SourceKind = Literal["interleaved", "split_mono"]

_ALLOWED_EXTENSIONS = {".wav", ".bwf"}
_SKIP_PREFIXES = ("._", ".")

_TOKEN_SEPARATORS = r"[ _\-.]"
_TOKEN_BOUNDARY = rf"(^|{_TOKEN_SEPARATORS})"
_TOKEN_END = rf"(?=$|{_TOKEN_SEPARATORS})"

_GROUP_PATTERNS = [
    re.compile(r"(?i)S(\d+)E(\d+)"),
    re.compile(r"(?i)EP(\d+)"),
    re.compile(r"(?i)R(\d+)"),
]

_ROLE_PATTERNS: list[tuple[RoleHint, re.Pattern[str]]] = [
    ("pm", re.compile(rf"(?i){_TOKEN_BOUNDARY}(PM|PRINTMASTER|MIX|FINAL|COMP){_TOKEN_END}")),
    ("dx", re.compile(rf"(?i){_TOKEN_BOUNDARY}(DX|DIA|DIALOG|DIALOGUE){_TOKEN_END}")),
    ("mx", re.compile(rf"(?i){_TOKEN_BOUNDARY}(MX|MUS|MUSIC){_TOKEN_END}")),
    ("fx", re.compile(rf"(?i){_TOKEN_BOUNDARY}(FX|SFX|EFFECTS){_TOKEN_END}")),
    ("me", re.compile(rf"(?i){_TOKEN_BOUNDARY}(ME|M&E|M_AND_E|MANDE){_TOKEN_END}")),
    ("opt", re.compile(rf"(?i){_TOKEN_BOUNDARY}(OPT|OPTIONAL|NARR|NARRATION){_TOKEN_END}")),
]

_CHANNEL_HINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mono", re.compile(rf"(?i){_TOKEN_BOUNDARY}(MONO|1{_TOKEN_SEPARATORS}?0){_TOKEN_END}")),
    ("stereo", re.compile(rf"(?i){_TOKEN_BOUNDARY}(STEREO|ST|2{_TOKEN_SEPARATORS}?0){_TOKEN_END}")),
    ("5.1", re.compile(rf"(?i){_TOKEN_BOUNDARY}(5{_TOKEN_SEPARATORS}?1|51){_TOKEN_END}")),
    ("7.1", re.compile(rf"(?i){_TOKEN_BOUNDARY}(7{_TOKEN_SEPARATORS}?1|71){_TOKEN_END}")),
]

_PRESENTATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("7.1", re.compile(rf"(?i){_TOKEN_BOUNDARY}(7{_TOKEN_SEPARATORS}?1|71){_TOKEN_END}")),
    ("5.1", re.compile(rf"(?i){_TOKEN_BOUNDARY}(5{_TOKEN_SEPARATORS}?1|51){_TOKEN_END}")),
    ("LtRt", re.compile(rf"(?i){_TOKEN_BOUNDARY}(LT{_TOKEN_SEPARATORS}?RT){_TOKEN_END}")),
    ("Stereo", re.compile(rf"(?i){_TOKEN_BOUNDARY}(STEREO){_TOKEN_END}")),
]

_LEG_PATTERN = re.compile(rf"(?i){_TOKEN_BOUNDARY}(LFE|LSS|RSS|LS|RS|L|R|C)$")
_LEG_CANONICAL = {
    "L": "L",
    "R": "R",
    "C": "C",
    "LFE": "LFE",
    "LS": "Ls",
    "RS": "Rs",
    "LSS": "Lss",
    "RSS": "Rss",
}
_CANONICAL_LAYOUTS: dict[ChannelConfigActual, list[str]] = {
    "stereo": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Ls", "Rs", "Lss", "Rss"],
}


@dataclass(frozen=True)
class AssetMember:
    path: Path
    basename: str
    sample_rate: int
    sample_count: int
    bit_depth: int | None
    subtype: str | None
    channel_count: int
    role_hint: RoleHint | None
    channel_config_hint: str | None
    presentation_label: str | None
    leg_label: str | None
    group_hint: str | None
    normalized_stem: str
    family_base_key: str | None


@dataclass(frozen=True)
class DiscoveryError:
    type: str
    message: str
    paths: list[Path]
    family_key: str | None
    group_hint: str | None = None


@dataclass(frozen=True)
class LogicalAsset:
    asset_id: str
    role_hint: RoleHint | None
    group_id: str
    group_hint: str | None
    source_kind: SourceKind
    channel_config_actual: ChannelConfigActual
    channel_config_hint: str | None
    presentation_label: str | None
    canonical_path: Path
    source_paths: list[Path]
    member_legs: list[str]
    sample_rate: int
    sample_count: int
    bit_depth: int | None
    errors: list[DiscoveryError] = field(default_factory=list)


@dataclass(frozen=True)
class AudioAsset:
    samples: np.ndarray
    sample_rate: int
    sample_count: int
    channel_count: int
    channel_config_actual: str
    source_kind: str
    source_paths: list[Path]


@dataclass(frozen=True)
class DiscoveryResult:
    assets: list[LogicalAsset]
    errors: list[DiscoveryError]
    ignored_paths: list[Path]


class AssetResolutionError(FinalPassError):
    """Seed path could not be resolved to a valid logical asset."""

    def __init__(self, error: DiscoveryError):
        self.error = error
        super().__init__(error.message)


def discover_logical_assets_in_folder(folder: Path) -> DiscoveryResult:
    folder = Path(folder)
    ignored: list[Path] = []
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    paths: list[Path] = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if _should_ignore(entry):
            ignored.append(entry.resolve())
            continue
        if not entry.is_file():
            ignored.append(entry.resolve())
            continue
        if entry.suffix.lower() not in _ALLOWED_EXTENSIONS:
            ignored.append(entry.resolve())
            continue
        paths.append(entry.resolve())
    result = discover_logical_assets(paths, strict_explicit_split_members=True)
    return DiscoveryResult(
        assets=result.assets,
        errors=result.errors,
        ignored_paths=sorted({*result.ignored_paths, *ignored}),
    )


def discover_logical_assets(
    paths: Iterable[Path],
    *,
    strict_explicit_split_members: bool = False,
) -> DiscoveryResult:
    ignored_paths: list[Path] = []
    members: list[AssetMember] = []
    errors: list[DiscoveryError] = []

    seen_paths: set[Path] = set()
    for raw_path in sorted((Path(path).resolve() for path in paths), key=lambda p: str(p).lower()):
        if raw_path in seen_paths:
            continue
        seen_paths.add(raw_path)
        if _should_ignore(raw_path):
            ignored_paths.append(raw_path)
            continue
        if not raw_path.is_file():
            ignored_paths.append(raw_path)
            continue
        if raw_path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            ignored_paths.append(raw_path)
            continue
        try:
            members.append(_probe_member(raw_path))
        except FinalPassError as exc:
            errors.append(DiscoveryError(
                type=exc.__class__.__name__,
                message=str(exc),
                paths=[raw_path],
                family_key=None,
                group_hint=None,
            ))

    assets: list[LogicalAsset] = []
    consumed_paths: set[Path] = set()
    family_buckets: dict[str, list[AssetMember]] = defaultdict(list)

    for member in members:
        if member.leg_label is None or member.family_base_key is None:
            continue
        family_buckets[member.family_base_key].append(member)

    for family_base_key, bucket in family_buckets.items():
        mono_members = [member for member in bucket if member.channel_count == 1]
        if len(mono_members) < 2:
            continue

        partitioned, partition_errors = _partition_family_bucket(bucket, family_base_key)
        if partition_errors:
            errors.extend(partition_errors)
            consumed_paths.update(member.path for member in bucket)
            continue

        for family_key, members_in_family in partitioned:
            asset, error = _build_split_asset(members_in_family, family_key)
            if error is not None:
                errors.append(error)
            elif asset is not None:
                assets.append(asset)
            consumed_paths.update(member.path for member in members_in_family)

    for member in members:
        if member.path in consumed_paths:
            continue
        if strict_explicit_split_members and _looks_like_explicit_split_member(member):
            errors.append(DiscoveryError(
                type="MissingLeg",
                message=(
                    f"{member.path.name} looks like an explicit split-mono member, "
                    "but no complete sibling family was found."
                ),
                paths=[member.path],
                family_key=member.family_base_key,
                group_hint=member.group_hint,
            ))
            continue
        asset, error = _build_single_asset(member)
        if error is not None:
            errors.append(error)
            continue
        assets.append(asset)

    assets.sort(key=lambda asset: (str(asset.canonical_path).lower(), asset.asset_id))
    errors.sort(key=lambda error: (error.family_key or "", str(error.paths[0]).lower() if error.paths else ""))
    ignored_paths.sort(key=lambda path: str(path).lower())
    return DiscoveryResult(assets=assets, errors=errors, ignored_paths=ignored_paths)


def resolve_logical_asset(seed_path: Path) -> LogicalAsset:
    seed_path = Path(seed_path).resolve()
    if not seed_path.exists():
        raise AssetResolutionError(DiscoveryError(
            type="MissingPath",
            message=f"{seed_path} does not exist.",
            paths=[seed_path],
            family_key=None,
            group_hint=None,
        ))

    seed_member = _probe_member(seed_path)
    if seed_member.channel_count > 1 or seed_member.leg_label is None or seed_member.family_base_key is None:
        asset, error = _build_single_asset(seed_member)
        if error is not None:
            raise AssetResolutionError(error)
        return asset

    directory_members: list[AssetMember] = []
    for sibling in sorted(seed_path.parent.iterdir(), key=lambda p: p.name.lower()):
        if _should_ignore(sibling):
            continue
        if not sibling.is_file() or sibling.suffix.lower() not in _ALLOWED_EXTENSIONS:
            continue
        try:
            member = _probe_member(sibling.resolve())
        except FinalPassError:
            continue
        if member.family_base_key == seed_member.family_base_key and member.leg_label is not None:
            directory_members.append(member)

    mono_count = sum(member.channel_count == 1 for member in directory_members)
    if mono_count < 2:
        raise AssetResolutionError(DiscoveryError(
            type="MissingLeg",
            message=f"{seed_path.name} looks like a split-mono member, but no complete sibling family was found.",
            paths=[member.path for member in directory_members] or [seed_path],
            family_key=seed_member.family_base_key,
            group_hint=seed_member.group_hint,
        ))

    partitioned, partition_errors = _partition_family_bucket(directory_members, seed_member.family_base_key)
    if partition_errors:
        raise AssetResolutionError(partition_errors[0])

    matching_family: tuple[str, list[AssetMember]] | None = None
    for family_key, members_in_family in partitioned:
        if any(member.path == seed_path for member in members_in_family):
            matching_family = (family_key, members_in_family)
            break

    if matching_family is None:
        raise AssetResolutionError(DiscoveryError(
            type="AmbiguousFamilyKey",
            message=f"{seed_path.name} could not be resolved to one split-mono family.",
            paths=[member.path for member in directory_members],
            family_key=seed_member.family_base_key,
            group_hint=seed_member.group_hint,
        ))

    asset, error = _build_split_asset(matching_family[1], matching_family[0])
    if error is not None or asset is None:
        raise AssetResolutionError(error or DiscoveryError(
            type="InvalidFamily",
            message=f"{seed_path.name} did not resolve to a valid split-mono family.",
            paths=[member.path for member in matching_family[1]],
            family_key=matching_family[0],
            group_hint=seed_member.group_hint,
        ))
    return asset


def read_logical_asset(asset: LogicalAsset) -> AudioAsset:
    if asset.source_kind == "interleaved":
        audio = read_wav(asset.canonical_path)
        return AudioAsset(
            samples=audio.data,
            sample_rate=audio.sample_rate,
            sample_count=audio.sample_count,
            channel_count=audio.channel_count,
            channel_config_actual=asset.channel_config_actual,
            source_kind=asset.source_kind,
            source_paths=list(asset.source_paths),
        )

    leg_order = _CANONICAL_LAYOUTS[asset.channel_config_actual]
    leg_to_path = dict(zip(asset.member_legs, asset.source_paths, strict=True))
    channels: list[np.ndarray] = []
    first_audio: AudioFile | None = None
    expected_subtype: str | None = None
    expected_bit_depth: int | None = None

    for leg in leg_order:
        path = leg_to_path[leg]
        header = probe_wav(path)
        if header.channel_count != 1:
            raise AssetResolutionError(DiscoveryError(
                type="NonMonoMember",
                message=f"{path.name} is {header.channel_count}ch; split-mono family members must all be mono.",
                paths=[path],
                family_key=asset.asset_id,
                group_hint=asset.group_hint,
            ))
        if first_audio is None:
            first_audio = read_wav(path)
            expected_subtype = header.subtype
            expected_bit_depth = header.bit_depth
            channels.append(first_audio.data[:, 0])
            continue
        if header.sample_rate != asset.sample_rate:
            raise AssetResolutionError(DiscoveryError(
                type="MixedSampleRate",
                message=f"{path.name} has {header.sample_rate} Hz; expected {asset.sample_rate} Hz.",
                paths=list(asset.source_paths),
                family_key=asset.asset_id,
                group_hint=asset.group_hint,
            ))
        if header.sample_count != asset.sample_count:
            raise AssetResolutionError(DiscoveryError(
                type="MixedSampleCount",
                message=f"{path.name} has {header.sample_count} samples; expected {asset.sample_count}.",
                paths=list(asset.source_paths),
                family_key=asset.asset_id,
                group_hint=asset.group_hint,
            ))
        if header.subtype != expected_subtype or header.bit_depth != expected_bit_depth:
            raise AssetResolutionError(DiscoveryError(
                type="SubtypeMismatch",
                message=f"{path.name} subtype/bit depth does not match the rest of the split family.",
                paths=list(asset.source_paths),
                family_key=asset.asset_id,
                group_hint=asset.group_hint,
            ))
        audio = read_wav(path)
        channels.append(audio.data[:, 0])

    samples = np.column_stack(channels).astype(np.float64, copy=False)
    return AudioAsset(
        samples=samples,
        sample_rate=asset.sample_rate,
        sample_count=asset.sample_count,
        channel_count=samples.shape[1],
        channel_config_actual=asset.channel_config_actual,
        source_kind=asset.source_kind,
        source_paths=list(asset.source_paths),
    )


def _probe_member(path: Path) -> AssetMember:
    header = probe_wav(path)
    stem = path.stem
    role_hint = _parse_role_hint(stem)
    group_hint = _parse_group_hint(stem)
    leg_label = _parse_leg_label(stem)
    presentation_label = _parse_presentation_label(stem)
    channel_config_hint = _parse_channel_config_hint(stem, presentation_label)
    normalized_stem = _normalize_stem(stem)
    family_base_key = None
    if leg_label is not None:
        family_base_key = _family_base_key(
            parent=path.parent,
            normalized_stem=normalized_stem,
            role_hint=role_hint,
            group_hint=group_hint,
        )
    return AssetMember(
        path=path,
        basename=path.name,
        sample_rate=header.sample_rate,
        sample_count=header.sample_count,
        bit_depth=header.bit_depth,
        subtype=header.subtype,
        channel_count=header.channel_count,
        role_hint=role_hint,
        channel_config_hint=channel_config_hint,
        presentation_label=presentation_label,
        leg_label=leg_label,
        group_hint=group_hint,
        normalized_stem=normalized_stem,
        family_base_key=family_base_key,
    )


def _build_single_asset(member: AssetMember) -> tuple[LogicalAsset, None] | tuple[None, DiscoveryError]:
    actual = channel_config_from_count(member.channel_count)
    if actual is None:
        return None, DiscoveryError(
            type="UnsupportedChannelConfig",
            message=f"{member.path.name}: unsupported channel count {member.channel_count}.",
            paths=[member.path],
            family_key=None,
            group_hint=member.group_hint,
        )
    asset_id = f"interleaved::{member.path}"
    return LogicalAsset(
        asset_id=asset_id,
        role_hint=member.role_hint,
        group_id=_logical_group_id(member.group_hint, member.normalized_stem, member.path),
        group_hint=member.group_hint,
        source_kind="interleaved",
        channel_config_actual=actual,
        channel_config_hint=member.channel_config_hint,
        presentation_label=member.presentation_label,
        canonical_path=member.path,
        source_paths=[member.path],
        member_legs=[],
        sample_rate=member.sample_rate,
        sample_count=member.sample_count,
        bit_depth=member.bit_depth,
        errors=[],
    ), None


def _partition_family_bucket(
    members: list[AssetMember],
    family_base_key: str,
) -> tuple[list[tuple[str, list[AssetMember]]], list[DiscoveryError]]:
    by_presentation: dict[str | None, list[AssetMember]] = defaultdict(list)
    for member in members:
        by_presentation[member.presentation_label].append(member)

    labels = {label for label in by_presentation if label is not None}
    if None in by_presentation and labels:
        return [], [DiscoveryError(
            type="AmbiguousFamilyKey",
            message="Ambiguous split-mono family: candidates mix explicit and missing presentation labels; refusing to guess family membership.",
            paths=[member.path for member in members],
            family_key=family_base_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )]

    if len(labels) <= 1:
        label = next(iter(labels), None)
        family_key = _family_key_from_base(family_base_key, label)
        return [(family_key, sorted(members, key=lambda member: str(member.path).lower()))], []

    out: list[tuple[str, list[AssetMember]]] = []
    for label, group in sorted(by_presentation.items(), key=lambda item: item[0] or ""):
        family_key = _family_key_from_base(family_base_key, label)
        out.append((family_key, sorted(group, key=lambda member: str(member.path).lower())))
    return out, []


def _build_split_asset(
    members: list[AssetMember],
    family_key: str,
) -> tuple[LogicalAsset | None, DiscoveryError | None]:
    non_mono = [member for member in members if member.channel_count != 1]
    if non_mono:
        return None, DiscoveryError(
            type="NonMonoMember",
            message="Split-mono family contains a non-mono member.",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    leg_counter = Counter(member.leg_label for member in members)
    duplicate_legs = sorted(leg for leg, count in leg_counter.items() if leg is not None and count > 1)
    if duplicate_legs:
        return None, DiscoveryError(
            type="DuplicateLeg",
            message=f"Split-mono family contains duplicate leg(s): {', '.join(duplicate_legs)}.",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    sample_rates = {member.sample_rate for member in members}
    if len(sample_rates) > 1:
        detail = ", ".join(f"{member.path.name}={member.sample_rate}Hz" for member in members)
        return None, DiscoveryError(
            type="MixedSampleRate",
            message=f"Split-mono family members must share one sample rate: {detail}",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    sample_counts = {member.sample_count for member in members}
    if len(sample_counts) > 1:
        detail = ", ".join(f"{member.path.name}={member.sample_count} samples" for member in members)
        return None, DiscoveryError(
            type="MixedSampleCount",
            message=f"Split-mono family members must share one sample count: {detail}",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    subtypes = {member.subtype for member in members}
    bit_depths = {member.bit_depth for member in members}
    if len(subtypes) > 1 or len(bit_depths) > 1:
        detail = ", ".join(
            f"{member.path.name}={member.subtype or 'unknown'}/{member.bit_depth or 'unknown'}"
            for member in members
        )
        return None, DiscoveryError(
            type="SubtypeMismatch",
            message=f"Split-mono family members must share one subtype/bit depth: {detail}",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    legs = [member.leg_label for member in members if member.leg_label is not None]
    presentation = _single_presentation_label(members)
    explicit_missing = _missing_legs_for_presentation(presentation, legs)
    if explicit_missing is not None:
        return None, DiscoveryError(
            type="MissingLeg",
            message=f"Split-mono family is missing required {presentation} leg(s): {', '.join(explicit_missing)}.",
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    layout = _infer_layout(legs)
    if isinstance(layout, DiscoveryError):
        return None, DiscoveryError(
            type=layout.type,
            message=layout.message,
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    compatibility_error = _validate_presentation_compatibility(presentation, layout)
    if compatibility_error is not None:
        return None, DiscoveryError(
            type=compatibility_error.type,
            message=compatibility_error.message,
            paths=[member.path for member in members],
            family_key=family_key,
            group_hint=_unified_or_none(member.group_hint for member in members),
        )

    ordered_legs = _CANONICAL_LAYOUTS[layout]
    leg_to_member = {member.leg_label: member for member in members if member.leg_label is not None}
    ordered_members = [leg_to_member[leg] for leg in ordered_legs]
    canonical_path = ordered_members[0].path
    role_hint = _unified_or_unknown(member.role_hint for member in ordered_members)
    group_hint = _unified_or_none(member.group_hint for member in ordered_members)
    channel_hint = _unified_channel_hint(ordered_members, layout)
    asset_id = f"split_mono::{family_key}"
    return LogicalAsset(
        asset_id=asset_id,
        role_hint=role_hint,
        group_id=_logical_group_id(group_hint, ordered_members[0].normalized_stem, ordered_members[0].path),
        group_hint=group_hint,
        source_kind="split_mono",
        channel_config_actual=layout,
        channel_config_hint=channel_hint,
        presentation_label=presentation,
        canonical_path=canonical_path,
        source_paths=[member.path for member in ordered_members],
        member_legs=ordered_legs,
        sample_rate=ordered_members[0].sample_rate,
        sample_count=ordered_members[0].sample_count,
        bit_depth=ordered_members[0].bit_depth,
        errors=[],
    ), None


def _infer_layout(legs: list[str]) -> ChannelConfigActual | DiscoveryError:
    unique = set(legs)
    for layout, expected in _CANONICAL_LAYOUTS.items():
        if unique == set(expected):
            return layout

    expected_missing: tuple[ChannelConfigActual, list[str]] | None = None
    for layout, expected in sorted(_CANONICAL_LAYOUTS.items(), key=lambda item: len(item[1])):
        expected_set = set(expected)
        if unique.issubset(expected_set):
            missing = [leg for leg in expected if leg not in unique]
            expected_missing = (layout, missing)
            break

    if expected_missing is not None:
        layout, missing = expected_missing
        return DiscoveryError(
            type="MissingLeg",
            message=f"Split-mono family is missing required {layout} leg(s): {', '.join(missing)}.",
            paths=[],
            family_key=None,
        )

    return DiscoveryError(
        type="InvalidLegSet",
        message=f"Split-mono family has unsupported leg set: {', '.join(sorted(unique))}.",
        paths=[],
        family_key=None,
    )


def _validate_presentation_compatibility(
    presentation_label: str | None,
    inferred_layout: ChannelConfigActual,
) -> DiscoveryError | None:
    if presentation_label is None:
        return None
    if presentation_label in {"Stereo", "LtRt"} and inferred_layout != "stereo":
        return DiscoveryError(
            type="IncompatiblePresentation",
            message=f"Presentation hint {presentation_label} is incompatible with inferred layout {inferred_layout}.",
            paths=[],
            family_key=None,
        )
    if presentation_label == "5.1" and inferred_layout != "5.1":
        return DiscoveryError(
            type="IncompatiblePresentation",
            message=f"Presentation hint {presentation_label} is incompatible with inferred layout {inferred_layout}.",
            paths=[],
            family_key=None,
        )
    if presentation_label == "7.1" and inferred_layout != "7.1":
        return DiscoveryError(
            type="IncompatiblePresentation",
            message=f"Presentation hint {presentation_label} is incompatible with inferred layout {inferred_layout}.",
            paths=[],
            family_key=None,
        )
    return None


def _missing_legs_for_presentation(
    presentation_label: str | None,
    legs: list[str],
) -> list[str] | None:
    if presentation_label == "5.1":
        expected = _CANONICAL_LAYOUTS["5.1"]
    elif presentation_label == "7.1":
        expected = _CANONICAL_LAYOUTS["7.1"]
    elif presentation_label in {"Stereo", "LtRt"}:
        expected = _CANONICAL_LAYOUTS["stereo"]
    else:
        return None

    unique = set(legs)
    expected_set = set(expected)
    if unique.issubset(expected_set) and unique != expected_set:
        return [leg for leg in expected if leg not in unique]
    return None


def _single_presentation_label(members: list[AssetMember]) -> str | None:
    labels = {member.presentation_label for member in members if member.presentation_label is not None}
    if not labels:
        return None
    return sorted(labels)[0]


def _parse_role_hint(stem: str) -> RoleHint | None:
    hits: list[RoleHint] = []
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(stem):
            hits.append(role)
    if not hits:
        return "unknown"
    return hits[0] if len(hits) == 1 else "unknown"


def _parse_group_hint(stem: str) -> str | None:
    for pattern in _GROUP_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(0).upper()
    return None


def _parse_channel_config_hint(stem: str, presentation_label: str | None) -> str | None:
    if presentation_label == "5.1":
        return "5.1"
    if presentation_label == "7.1":
        return "7.1"
    if presentation_label in {"Stereo", "LtRt"}:
        return "stereo"
    for hint, pattern in _CHANNEL_HINT_PATTERNS:
        if pattern.search(stem):
            return hint
    return None


def _parse_presentation_label(stem: str) -> str | None:
    for label, pattern in _PRESENTATION_PATTERNS:
        if pattern.search(stem):
            return label
    return None


def _parse_leg_label(stem: str) -> str | None:
    match = _LEG_PATTERN.search(stem)
    if match is None:
        return None
    return _LEG_CANONICAL[match.group(2).upper()]


def _normalize_stem(stem: str) -> str:
    cleaned = stem
    leg_match = _LEG_PATTERN.search(cleaned)
    if leg_match is not None:
        cleaned = cleaned[:leg_match.start(2)]
    for _, pattern in _ROLE_PATTERNS:
        cleaned = pattern.sub("_", cleaned)
    for _, pattern in _PRESENTATION_PATTERNS:
        cleaned = pattern.sub("_", cleaned)
    for _, pattern in _CHANNEL_HINT_PATTERNS:
        cleaned = pattern.sub("_", cleaned)
    for pattern in _GROUP_PATTERNS:
        cleaned = pattern.sub("_", cleaned)
    cleaned = re.sub(r"[ _\-.]+", "_", cleaned).strip("_")
    return cleaned.lower()


def _family_base_key(
    *,
    parent: Path,
    normalized_stem: str,
    role_hint: RoleHint | None,
    group_hint: str | None,
) -> str:
    return "|".join([
        str(parent.resolve()),
        normalized_stem or "-",
        role_hint or "-",
        group_hint or "-",
    ])


def _family_key_from_base(family_base_key: str, presentation_label: str | None) -> str:
    return f"{family_base_key}|{presentation_label or '-'}"


def _logical_group_id(group_hint: str | None, normalized_stem: str, path: Path) -> str:
    if group_hint:
        return group_hint.upper()
    if normalized_stem:
        return normalized_stem.upper()
    return path.parent.name.upper()


def _unified_or_unknown(values: Iterable[RoleHint | None]) -> RoleHint | None:
    present = {value for value in values if value is not None}
    if not present:
        return None
    if len(present) == 1:
        return next(iter(present))
    return "unknown"


def _unified_or_none(values: Iterable[str | None]) -> str | None:
    present = {value for value in values if value is not None}
    if not present:
        return None
    if len(present) == 1:
        return next(iter(present))
    return None


def _unified_channel_hint(members: list[AssetMember], inferred_layout: ChannelConfigActual) -> str | None:
    hints = {member.channel_config_hint for member in members if member.channel_config_hint is not None}
    if not hints:
        return None
    if len(hints) == 1:
        hint = next(iter(hints))
        if hint == inferred_layout:
            return hint
        if hint == "stereo" and inferred_layout == "stereo":
            return hint
    return None


def _should_ignore(path: Path) -> bool:
    name = path.name
    if name.startswith(_SKIP_PREFIXES):
        return True
    if name == ".DS_Store":
        return True
    return False


def _looks_like_explicit_split_member(member: AssetMember) -> bool:
    return (
        member.channel_count == 1
        and member.leg_label is not None
        and member.presentation_label is not None
    )
