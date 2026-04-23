"""Derive human-readable artifact filename stems from delivery filenames."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .models import AllReport, MEReport, NullReport, Report


@dataclass(frozen=True)
class DerivedProgramName:
    show_slug: str
    slug: str
    episodes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ParsedStem:
    show_slug: str
    episode: str | None
    episode_sort_key: tuple[int, int, str] | None


_SEPARATOR_CHARS = " _-."
_ROLE_TOKENS = {
    "PM",
    "PRINTMASTER",
    "MIX",
    "FINAL",
    "COMP",
    "DX",
    "DIA",
    "DIALOG",
    "DIALOGUE",
    "MX",
    "MUS",
    "MUSIC",
    "FX",
    "SFX",
    "EFFECTS",
    "ME",
    "M_AND_E",
    "MANDE",
    "OPT",
    "OPTIONAL",
    "NARR",
    "NARRATION",
}
_CHANNEL_TOKENS = {
    "MONO",
    "STEREO",
    "ST",
    "10",
    "1_0",
    "20",
    "2_0",
    "51",
    "5_1",
    "71",
    "7_1",
    "LTRT",
}
_LEG_TOKENS = {"L", "R", "C", "LFE", "LS", "RS", "LSS", "RSS"}
_NOISE_TOKENS = _ROLE_TOKENS | _CHANNEL_TOKENS | _LEG_TOKENS
_EPISODE_PATTERNS = (
    re.compile(r"(?i)(^|[ _\-.])S(?P<season>\d{1,2})[ _\-.]?E(?P<episode>\d{1,3})(?=$|[ _\-.])"),
    re.compile(r"(?i)(^|[ _\-.])(?P<season>\d{1,2})X(?P<episode>\d{1,3})(?=$|[ _\-.])"),
    re.compile(r"(?i)(^|[ _\-.])EP(?:ISODE)?[ _\-.]?(?P<episode>\d{1,3})(?=$|[ _\-.])"),
    re.compile(r"(?i)(^|[ _\-.])E(?P<episode>\d{1,3})(?=$|[ _\-.])"),
)


def derive_program_name_from_report(report: Report | NullReport | MEReport | AllReport) -> DerivedProgramName | None:
    """Infer a stable program slug from the report's source filenames."""
    return derive_program_name_from_paths(_report_source_paths(report))


def derive_program_name_from_paths(paths: Iterable[str | Path]) -> DerivedProgramName | None:
    parsed = [
        candidate
        for path in paths
        if (candidate := _parse_stem(Path(path).stem)) is not None
    ]
    if not parsed:
        return None

    show_counts = Counter(candidate.show_slug for candidate in parsed)
    show_slug, count = show_counts.most_common(1)[0]
    if count <= len(parsed) / 2:
        return None

    selected = [candidate for candidate in parsed if candidate.show_slug == show_slug]
    episode_items = {
        candidate.episode: candidate.episode_sort_key
        for candidate in selected
        if candidate.episode is not None and candidate.episode_sort_key is not None
    }
    episodes = tuple(
        episode
        for episode, _ in sorted(episode_items.items(), key=lambda item: item[1] or (9999, 9999, item[0]))
    )
    episode_suffix = "-".join(episode.lower() for episode in episodes[:3])
    slug = f"{show_slug}-{episode_suffix}" if episode_suffix else show_slug
    return DerivedProgramName(show_slug=show_slug, slug=slug, episodes=episodes)


def _report_source_paths(report: Report | NullReport | MEReport | AllReport) -> list[str]:
    if isinstance(report, Report):
        return [
            source_path
            for file_report in report.files
            for source_path in (file_report.source_paths or [file_report.path])
        ]
    if isinstance(report, NullReport):
        return [
            source_path
            for item in (report.printmaster, *report.stems)
            for source_path in (item.source_paths or [item.path])
        ]
    if isinstance(report, MEReport):
        return [
            source_path
            for item in (report.me_file, report.dx_file)
            for source_path in (item.source_paths or [item.path])
        ]
    return [
        source_path
        for group in report.groups
        for asset in group.assets
        for source_path in (asset.source_paths or [asset.path])
    ]


def _parse_stem(stem: str) -> _ParsedStem | None:
    episode_match = _find_episode(stem)
    has_role_token = _has_role_token(stem)
    if episode_match is None and not has_role_token:
        return None

    if episode_match is not None:
        match, episode, sort_key = episode_match
        boundary = match.group(1) or ""
        episode_start = match.start() + len(boundary)
        show_source = stem[:episode_start]
    else:
        episode = None
        sort_key = None
        show_source = stem

    show_slug = _slug_from_title(show_source)
    if show_slug:
        return _ParsedStem(show_slug=show_slug, episode=episode, episode_sort_key=sort_key)
    return None


def _find_episode(stem: str) -> tuple[re.Match[str], str, tuple[int, int, str]] | None:
    matches: list[tuple[int, re.Match[str], str, tuple[int, int, str]]] = []
    for pattern in _EPISODE_PATTERNS:
        for match in pattern.finditer(stem):
            boundary = match.group(1) or ""
            start = match.start() + len(boundary)
            if "season" in match.groupdict() and match.group("season") is not None:
                season = int(match.group("season"))
                episode = int(match.group("episode"))
                canonical = f"S{season:02d}E{episode:02d}"
                sort_key = (season, episode, canonical)
            else:
                episode = int(match.group("episode"))
                prefix = "EP" if match.group(0).lstrip(_SEPARATOR_CHARS).upper().startswith("EP") else "E"
                canonical = f"{prefix}{episode:02d}"
                sort_key = (0, episode, canonical)
            matches.append((start, match, canonical, sort_key))
    if not matches:
        return None
    _, match, canonical, sort_key = min(matches, key=lambda item: item[0])
    return match, canonical, sort_key


def _has_role_token(stem: str) -> bool:
    return any(token in _ROLE_TOKENS for token in _tokens(stem))


def _slug_from_title(text: str) -> str | None:
    tokens = [token.lower() for token in _tokens(text) if token not in _NOISE_TOKENS]
    if not tokens:
        return None
    slug = "-".join(tokens)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80].rstrip("-") or None


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"(?i)M[ _\-.]*AND[ _\-.]*E", " MANDE ", text)
    normalized = re.sub(r"(?i)M[ _\-.]*&[ _\-.]*E", " MANDE ", normalized)
    normalized = normalized.replace("&", " AND ")
    normalized = re.sub(r"(?i)(\d)[ _\-.](\d)", r"\1_\2", normalized)
    normalized = re.sub(r"(?<!\d)_(?!\d)", " ", normalized)
    normalized = re.sub(r"[^A-Za-z0-9_]+", " ", normalized)
    return [token.upper().strip("_") for token in normalized.split() if token.strip("_")]
