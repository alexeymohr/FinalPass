"""Filename classifier. Drives :command:`finalpass all`.

The classifier is *regex-pattern driven* and loads from YAML. Bundled patterns
ship at :file:`src/finalpass/_bundled_patterns.yaml`. A user override via
``--patterns`` replaces the bundled set entirely (same top-level keys).

Design contract (see PHASE_2_SPEC.md):

- Every filename is tested against *all* role patterns. One match → role
  assigned. Zero matches → ``role == "unknown"``. Two or more matches →
  :class:`AmbiguousClassificationError` naming both matches. Do not guess.
- Hidden files, macOS ``._*`` resource forks, ``.DS_Store``, non-WAV
  extensions → silently skipped (not even listed as unclassified).
- Symlinks are followed; a seen-inode guard prevents cycles.
- Group id: first ``group_identifier`` pattern that matches wins (uppercased).
  Fallback is the stem with role + channel tokens stripped.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import AmbiguousClassificationError, ClassifierConfigError

Role = Literal["pm", "dx", "mx", "fx", "me", "opt", "unknown"]

_ALLOWED_EXT = {".wav", ".bwf"}
_SKIP_PREFIXES = ("._", ".")  # dotfiles + AppleDouble resource forks


class RolePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["pm", "dx", "mx", "fx", "me", "opt"]
    regex: str


class GroupIdentifier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns: list[str]


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns: list[RolePattern]
    group_identifier: GroupIdentifier
    channel_config_tokens: dict[str, list[str]]


@dataclass(frozen=True)
class ClassifiedFile:
    path: Path
    role: Role
    group_id: str
    channel_config_hint: str | None


@dataclass(frozen=True)
class UnclassifiedFile:
    path: Path
    reason: str


@dataclass(frozen=True)
class FolderScan:
    """Result of scanning a folder once.

    ``groups`` is an ``OrderedDict`` keyed by group_id, preserving the folder
    traversal order so the CLI prints groups the way a user sees them in
    the file manager.
    """
    groups: "OrderedDict[str, list[ClassifiedFile]]"
    unclassified: list[UnclassifiedFile]


def load_config(path: Path | None = None) -> ClassifierConfig:
    """Load bundled patterns, or a user-provided YAML if ``path`` is given."""
    if path is not None:
        if not path.is_file():
            raise ClassifierConfigError(f"Classifier patterns file not found: {path}")
        raw_text = path.read_text(encoding="utf-8")
        source = str(path)
    else:
        pkg = resources.files("finalpass").joinpath("_bundled_patterns.yaml")
        raw_text = pkg.read_text(encoding="utf-8")
        source = "bundled"

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ClassifierConfigError(f"Classifier patterns {source} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ClassifierConfigError(f"Classifier patterns {source} must be a YAML mapping.")

    try:
        cfg = ClassifierConfig.model_validate(raw)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        raise ClassifierConfigError(f"Classifier patterns {source} is invalid ({fields}): {exc}") from exc

    _compile_patterns(cfg)
    return cfg


def _compile_patterns(cfg: ClassifierConfig) -> None:
    """Pre-compile every regex — fail fast on bad patterns."""
    try:
        for rp in cfg.patterns:
            re.compile(rp.regex)
        for p in cfg.group_identifier.patterns:
            re.compile(p)
        for patterns in cfg.channel_config_tokens.values():
            for p in patterns:
                re.compile(p)
    except re.error as exc:
        raise ClassifierConfigError(f"Classifier patterns contain an invalid regex: {exc}") from exc


def classify_file(path: Path, cfg: ClassifierConfig) -> ClassifiedFile:
    """Classify a single file by its stem. Caller is responsible for filtering
    hidden files and non-WAV extensions (see :func:`scan_folder`)."""
    stem = path.stem
    roles_hit: list[tuple[str, str]] = []
    for rp in cfg.patterns:
        m = re.search(rp.regex, stem)
        if m:
            roles_hit.append((rp.role, m.group(0)))

    if len(roles_hit) > 1:
        matches = ", ".join(f"{r!r} (matched {frag!r})" for r, frag in roles_hit)
        raise AmbiguousClassificationError(
            f"{path.name}: matched multiple role patterns — {matches}. "
            "Rename the file or adjust the classifier config so only one role matches."
        )

    role: Role = roles_hit[0][0] if roles_hit else "unknown"  # type: ignore[assignment]
    group_id = _extract_group_id(stem, cfg)
    hint = _extract_channel_hint(stem, cfg)
    return ClassifiedFile(path=path, role=role, group_id=group_id, channel_config_hint=hint)


def _extract_group_id(stem: str, cfg: ClassifierConfig) -> str:
    for pattern in cfg.group_identifier.patterns:
        m = re.search(pattern, stem)
        if m:
            return m.group(0).upper()
    return _fallback_group_id(stem, cfg)


def _fallback_group_id(stem: str, cfg: ClassifierConfig) -> str:
    """Strip role + channel tokens from the stem; collapse separators."""
    cleaned = stem
    for rp in cfg.patterns:
        # Replace matched token including its trailing boundary with a single
        # separator so adjacent separators collapse cleanly below.
        cleaned = re.sub(rp.regex, "_", cleaned)
    for patterns in cfg.channel_config_tokens.values():
        for pattern in patterns:
            cleaned = re.sub(pattern, "_", cleaned)
    cleaned = re.sub(r"[_\-\.]+", "_", cleaned).strip("_-.")
    if not cleaned:
        return stem.upper()
    return cleaned.upper()


def _extract_channel_hint(stem: str, cfg: ClassifierConfig) -> str | None:
    for channel_name, patterns in cfg.channel_config_tokens.items():
        for pattern in patterns:
            if re.search(pattern, stem):
                return channel_name
    return None


def scan_folder(folder: Path, cfg: ClassifierConfig) -> FolderScan:
    """Walk ``folder`` one level deep (no recursion), classify, group."""
    if not folder.is_dir():
        raise ClassifierConfigError(f"Not a directory: {folder}")

    groups: OrderedDict[str, list[ClassifiedFile]] = OrderedDict()
    unclassified: list[UnclassifiedFile] = []
    seen_inodes: set[tuple[int, int]] = set()

    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if _should_skip(entry):
            continue

        resolved = entry.resolve()
        try:
            stat = resolved.stat()
        except OSError:
            continue
        key = (stat.st_dev, stat.st_ino)
        if key in seen_inodes:
            continue
        seen_inodes.add(key)

        if not resolved.is_file():
            continue
        if resolved.suffix.lower() not in _ALLOWED_EXT:
            continue

        classified = classify_file(resolved, cfg)
        if classified.role == "unknown":
            unclassified.append(UnclassifiedFile(path=resolved, reason="no_role_pattern_match"))
            # Still include in grouping via its group_id so callers can show it
            # under its group if they opt into --include-unclassified.
        groups.setdefault(classified.group_id, []).append(classified)

    return FolderScan(groups=groups, unclassified=unclassified)


def _should_skip(path: Path) -> bool:
    name = path.name
    if name.startswith(_SKIP_PREFIXES):
        return True
    if name == ".DS_Store":
        return True
    return False
