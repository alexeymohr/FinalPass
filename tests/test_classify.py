from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finalpass.classify import (
    ClassifierConfig,
    classify_file,
    load_config,
    scan_folder,
)
from finalpass.errors import (
    AmbiguousClassificationError,
    ClassifierConfigError,
)


@pytest.fixture(scope="session")
def default_cfg() -> ClassifierConfig:
    return load_config(None)


# ---------------------------------------------------------------------------
# Role matching


@pytest.mark.parametrize(
    "stem,expected_role",
    [
        ("SHOW_S01E03_PM_STEREO", "pm"),
        ("SHOW_S01E03_DX_STEREO", "dx"),
        ("SHOW_S01E03_MX_STEREO", "mx"),
        ("SHOW_S01E03_FX_STEREO", "fx"),
        ("SHOW_S01E03_ME_STEREO", "me"),
        ("SHOW_S01E03_OPT_STEREO", "opt"),
        ("SHOW_PRINTMASTER_STEREO", "pm"),
        ("SHOW_DIALOG_STEREO", "dx"),
        ("SHOW_MUSIC_STEREO", "mx"),
        ("SHOW_SFX_STEREO", "fx"),
        ("SHOW_NARRATION_STEREO", "opt"),
    ],
)
def test_role_matches(tmp_path: Path, default_cfg, stem: str, expected_role: str) -> None:
    path = tmp_path / f"{stem}.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    assert cf.role == expected_role


@pytest.mark.parametrize(
    "stem",
    [
        "FIXING",      # contains 'FX' substring but not as a bounded token
        "PREMIX",      # contains 'MIX' but 'pre' is not a boundary
        "DIAMOND",     # contains 'DIA' substring but not bounded
        "OPTICAL",     # contains 'OPT' substring but not bounded
        "RANDOM_FILE",
    ],
)
def test_adversarial_substrings_are_unknown(tmp_path: Path, default_cfg, stem: str) -> None:
    path = tmp_path / f"{stem}.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    assert cf.role == "unknown", f"{stem} should not match any role"


# ---------------------------------------------------------------------------
# Group-id extraction


@pytest.mark.parametrize(
    "stem,expected_gid",
    [
        ("SHOW_S01E03_PM_STEREO", "S01E03"),
        ("SHOW_s01e03_PM_STEREO", "S01E03"),
        ("PROJECT_EP07_PM_STEREO", "EP07"),
        ("FILM_R02_PM_51", "R02"),
    ],
)
def test_group_id_extraction(tmp_path: Path, default_cfg, stem: str, expected_gid: str) -> None:
    path = tmp_path / f"{stem}.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    assert cf.group_id == expected_gid


def test_group_id_fallback_strips_role_and_channel_tokens(tmp_path: Path, default_cfg) -> None:
    """Open question #2: single-program delivery with no S##E##/EP##/R## token.
    Both files must fall back to the same group id so they group together."""
    pm = tmp_path / "MYSHOW_PM_STEREO.wav"
    dx = tmp_path / "MYSHOW_DX_STEREO.wav"
    for p in (pm, dx):
        p.touch()
    pm_cf = classify_file(pm, default_cfg)
    dx_cf = classify_file(dx, default_cfg)
    assert pm_cf.group_id == dx_cf.group_id == "MYSHOW"


def test_group_id_fallback_empty_falls_back_to_stem(tmp_path: Path, default_cfg) -> None:
    path = tmp_path / "PM.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    # Stripping leaves an empty string; we fall back to the uppercased stem.
    assert cf.group_id == "PM"
    assert cf.role == "pm"


# ---------------------------------------------------------------------------
# Ambiguity


def test_ambiguous_filename_raises(tmp_path: Path, default_cfg) -> None:
    """PM and MX both match on 'MIX' and 'MUSIC' tokens — ambiguous → error."""
    path = tmp_path / "SHOW_MIX_MUSIC_STEREO.wav"
    path.touch()
    with pytest.raises(AmbiguousClassificationError) as exc:
        classify_file(path, default_cfg)
    assert "pm" in str(exc.value)
    assert "mx" in str(exc.value)


def test_compound_role_with_space_boundary_is_ambiguous(tmp_path: Path, default_cfg) -> None:
    path = tmp_path / "SHOW_S01E03_MX-FX LtRt.L.wav"
    path.touch()
    with pytest.raises(AmbiguousClassificationError) as exc:
        classify_file(path, default_cfg)
    assert "mx" in str(exc.value)
    assert "fx" in str(exc.value)


# ---------------------------------------------------------------------------
# Channel-config hint


@pytest.mark.parametrize(
    "stem,hint",
    [
        ("SHOW_PM_STEREO", "stereo"),
        ("SHOW_PM_ST", "stereo"),
        ("SHOW_PM_5.1", "5.1"),
        ("SHOW_PM_51", "5.1"),
        ("SHOW_PM_MONO", "mono"),
        ("SHOW_PM_7_1", "7.1"),
    ],
)
def test_channel_hint(tmp_path: Path, default_cfg, stem: str, hint: str) -> None:
    path = tmp_path / f"{stem}.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    assert cf.channel_config_hint == hint


def test_channel_hint_absent(tmp_path: Path, default_cfg) -> None:
    path = tmp_path / "SHOW_PM_NO_CHANNEL_TAG.wav"
    path.touch()
    cf = classify_file(path, default_cfg)
    assert cf.channel_config_hint is None


# ---------------------------------------------------------------------------
# User override


def test_user_pattern_override(tmp_path: Path) -> None:
    config_yaml = tmp_path / "patterns.yaml"
    config_yaml.write_text(yaml.safe_dump({
        "patterns": [
            {"role": "pm", "regex": r"(?i)(^|_)MAIN(_|$)"},
            {"role": "dx", "regex": r"(?i)(^|_)VO(_|$)"},
            {"role": "mx", "regex": r"(?i)(^|_)SCORE(_|$)"},
            {"role": "fx", "regex": r"(?i)(^|_)SOUND(_|$)"},
            {"role": "me", "regex": r"(?i)(^|_)ME(_|$)"},
            {"role": "opt", "regex": r"(?i)(^|_)NARR(_|$)"},
        ],
        "group_identifier": {"patterns": [r"(?i)EP(\d+)"]},
        "channel_config_tokens": {
            "stereo": [r"(?i)(^|_)STEREO(_|$)"],
            "mono": [r"(?i)(^|_)MONO(_|$)"],
            "5.1": [r"(?i)(^|_)51(_|$)"],
            "7.1": [r"(?i)(^|_)71(_|$)"],
        },
    }))
    cfg = load_config(config_yaml)

    main_wav = tmp_path / "SHOW_EP02_MAIN_STEREO.wav"
    pm_wav = tmp_path / "SHOW_EP02_PM_STEREO.wav"
    for p in (main_wav, pm_wav):
        p.touch()
    assert classify_file(main_wav, cfg).role == "pm"
    # `PM` is no longer a role token under this override.
    assert classify_file(pm_wav, cfg).role == "unknown"


def test_missing_user_pattern_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ClassifierConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_bad_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(":: not yaml ::")
    with pytest.raises(ClassifierConfigError):
        load_config(bad)


def test_invalid_regex_in_config_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({
        "patterns": [{"role": "pm", "regex": "("}],  # unbalanced paren
        "group_identifier": {"patterns": []},
        "channel_config_tokens": {},
    }))
    with pytest.raises(ClassifierConfigError):
        load_config(bad)


# ---------------------------------------------------------------------------
# scan_folder


def _touch_wav(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_scan_folder_groups_by_group_id(tmp_path: Path, default_cfg) -> None:
    for name in [
        "SHOW_S01E03_PM_STEREO.wav",
        "SHOW_S01E03_DX_STEREO.wav",
        "SHOW_S01E04_PM_STEREO.wav",
        "SHOW_S01E04_DX_STEREO.wav",
        ".DS_Store",
        "._hidden.wav",
        "notes.txt",
    ]:
        _touch_wav(tmp_path / name)

    scan = scan_folder(tmp_path, default_cfg)
    assert set(scan.groups.keys()) == {"S01E03", "S01E04"}
    assert len(scan.groups["S01E03"]) == 2
    assert len(scan.groups["S01E04"]) == 2
    assert scan.unclassified == []


def test_scan_folder_case_insensitive_extension(tmp_path: Path, default_cfg) -> None:
    """Open question #3: .WAV, .Wav, .wav all accepted."""
    names = ["A_S01E01_PM_STEREO.WAV", "A_S01E01_DX_STEREO.Wav", "A_S01E01_MX_STEREO.wav"]
    for n in names:
        _touch_wav(tmp_path / n)
    scan = scan_folder(tmp_path, default_cfg)
    assert len(scan.groups["S01E01"]) == 3


def test_scan_folder_unclassified_goes_to_bucket(tmp_path: Path, default_cfg) -> None:
    _touch_wav(tmp_path / "SHOW_S01E01_PM_STEREO.wav")
    _touch_wav(tmp_path / "MYSTERY.wav")
    scan = scan_folder(tmp_path, default_cfg)
    assert any("MYSTERY" in u.path.name for u in scan.unclassified)
    assert len(scan.unclassified) == 1


def test_scan_folder_ignores_non_wav(tmp_path: Path, default_cfg) -> None:
    _touch_wav(tmp_path / "ignore.txt")
    _touch_wav(tmp_path / "ignore.mp3")
    _touch_wav(tmp_path / "SHOW_S01E01_PM_STEREO.wav")
    scan = scan_folder(tmp_path, default_cfg)
    assert sum(len(files) for files in scan.groups.values()) == 1


def test_scan_folder_not_a_directory_raises(tmp_path: Path, default_cfg) -> None:
    f = tmp_path / "file.wav"
    f.write_bytes(b"")
    with pytest.raises(ClassifierConfigError):
        scan_folder(f, default_cfg)
