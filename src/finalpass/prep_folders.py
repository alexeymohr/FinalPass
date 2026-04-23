from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import FileRole

PREP_ROOT_NAME = "FinalPass Prep"
IGNORE_BUCKET_NAME = "Ignore"


@dataclass(frozen=True)
class PrepBucketHint:
    role_hint: FileRole | None
    layout_hint: str | None


@dataclass(frozen=True)
class PrepBucket:
    name: str
    path: Path
    role_hint: FileRole | None
    layout_hint: str | None
    populated: bool
    file_paths: tuple[Path, ...]


@dataclass(frozen=True)
class PrepLayout:
    folder: Path
    prep_root: Path
    buckets: tuple[PrepBucket, ...]
    recognized: bool


@dataclass(frozen=True)
class PrepScanResult:
    layout: PrepLayout
    analyzable_buckets: tuple[PrepBucket, ...]
    ignored_buckets: tuple[PrepBucket, ...]
    analyzable_paths: tuple[Path, ...]
    path_hints: dict[Path, PrepBucketHint]

    @property
    def empty(self) -> bool:
        return not self.analyzable_paths


BUCKET_SPECS: tuple[tuple[str, FileRole | None, str | None], ...] = (
    ("5.1 Printmaster", "pm", "5.1"),
    ("5.1 Dialogue", "dx", "5.1"),
    ("5.1 Music", "mx", "5.1"),
    ("5.1 Effects", "fx", "5.1"),
    ("5.1 M&E", "me", "5.1"),
    ("Stereo Printmaster", "pm", "stereo"),
    ("Stereo Dialogue", "dx", "stereo"),
    ("Stereo Music", "mx", "stereo"),
    ("Stereo Effects", "fx", "stereo"),
    ("Stereo M&E", "me", "stereo"),
    (IGNORE_BUCKET_NAME, None, None),
)


def prep_root_for(folder: Path) -> Path:
    return Path(folder).resolve() / PREP_ROOT_NAME


def create_prep_layout(folder: Path) -> PrepLayout:
    folder = Path(folder).resolve()
    prep_root = prep_root_for(folder)
    prep_root.mkdir(parents=True, exist_ok=True)
    for bucket_name, _, _ in BUCKET_SPECS:
        (prep_root / bucket_name).mkdir(parents=True, exist_ok=True)
    return inspect_prep_layout(folder)


def detect_prep_layout(folder: Path) -> PrepLayout | None:
    layout = inspect_prep_layout(folder)
    if not layout.recognized:
        return None
    return layout


def inspect_prep_layout(folder: Path) -> PrepLayout:
    folder = Path(folder).resolve()
    prep_root = prep_root_for(folder)
    recognized = prep_root.is_dir()
    buckets: list[PrepBucket] = []

    for bucket_name, role_hint, layout_hint in BUCKET_SPECS:
        bucket_path = prep_root / bucket_name
        if not bucket_path.is_dir():
            recognized = False
            file_paths: tuple[Path, ...] = ()
        else:
            file_paths = _direct_files(bucket_path)
        buckets.append(PrepBucket(
            name=bucket_name,
            path=bucket_path,
            role_hint=role_hint,
            layout_hint=layout_hint,
            populated=bool(file_paths),
            file_paths=file_paths,
        ))

    return PrepLayout(
        folder=folder,
        prep_root=prep_root,
        buckets=tuple(buckets),
        recognized=recognized,
    )


def scan_prep_layout(layout: PrepLayout) -> PrepScanResult:
    refreshed = inspect_prep_layout(layout.folder)
    analyzable_buckets: list[PrepBucket] = []
    ignored_buckets: list[PrepBucket] = []
    analyzable_paths: list[Path] = []
    path_hints: dict[Path, PrepBucketHint] = {}

    for bucket in refreshed.buckets:
        if not bucket.populated:
            continue
        if bucket.name == IGNORE_BUCKET_NAME:
            ignored_buckets.append(bucket)
            continue
        analyzable_buckets.append(bucket)
        analyzable_paths.extend(bucket.file_paths)
        hint = PrepBucketHint(role_hint=bucket.role_hint, layout_hint=bucket.layout_hint)
        for path in bucket.file_paths:
            path_hints[path.resolve()] = hint

    return PrepScanResult(
        layout=refreshed,
        analyzable_buckets=tuple(analyzable_buckets),
        ignored_buckets=tuple(ignored_buckets),
        analyzable_paths=tuple(analyzable_paths),
        path_hints=path_hints,
    )


def _direct_files(folder: Path) -> tuple[Path, ...]:
    if not folder.is_dir():
        return ()
    return tuple(
        entry.resolve()
        for entry in sorted(folder.iterdir(), key=lambda path: path.name.lower())
        if entry.is_file()
    )
