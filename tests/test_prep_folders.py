from __future__ import annotations

from pathlib import Path

from finalpass.prep_folders import BUCKET_SPECS, IGNORE_BUCKET_NAME, PREP_ROOT_NAME, create_prep_layout, detect_prep_layout, scan_prep_layout


def test_create_prep_layout_is_idempotent(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()

    first = create_prep_layout(folder)
    second = create_prep_layout(folder)

    prep_root = folder / PREP_ROOT_NAME
    assert first.recognized is True
    assert second.recognized is True
    assert prep_root.is_dir()
    assert [bucket.name for bucket in second.buckets] == [name for name, _, _ in BUCKET_SPECS]
    assert all((prep_root / name).is_dir() for name, _, _ in BUCKET_SPECS)


def test_scan_prep_layout_reads_direct_files_only_and_excludes_ignore(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    create_prep_layout(folder)
    prep_root = folder / PREP_ROOT_NAME

    direct = prep_root / "Stereo Printmaster" / "pm.wav"
    direct.write_text("pm\n", encoding="utf-8")

    nested_dir = prep_root / "Stereo Dialogue" / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "dx.wav"
    nested.write_text("dx\n", encoding="utf-8")

    ignored = prep_root / IGNORE_BUCKET_NAME / "ignore.wav"
    ignored.write_text("ignore\n", encoding="utf-8")

    layout = detect_prep_layout(folder)
    assert layout is not None
    scan = scan_prep_layout(layout)

    assert scan.empty is False
    assert scan.analyzable_paths == (direct.resolve(),)
    assert len(scan.ignored_buckets) == 1
    assert scan.ignored_buckets[0].name == IGNORE_BUCKET_NAME
    assert ignored.resolve() not in scan.analyzable_paths
    assert nested.resolve() not in scan.analyzable_paths
    assert scan.path_hints[direct.resolve()].role_hint == "pm"
    assert scan.path_hints[direct.resolve()].layout_hint == "stereo"
    assert scan.path_hints[direct.resolve()].group_hint == folder.name.upper()
