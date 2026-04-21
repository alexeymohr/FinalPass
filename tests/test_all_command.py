from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from finalpass.cli import main

SR = 48000
SEED = 0xC0DE


def _pink(n_samples: int, n_channels: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_samples, n_channels))
    pink = np.zeros_like(white)
    pink[0] = 0.05 * white[0]
    for i in range(1, n_samples):
        pink[i] = 0.99 * pink[i - 1] + 0.05 * white[i]
    rms = float(np.sqrt(np.mean(pink ** 2)))
    if rms > 0:
        pink *= (10 ** (-23.0 / 20.0)) / rms
    return pink.astype(np.float64)


def _write_pink_stereo(path: Path, seconds: float = 10.0, gain_db: float = 0.0, seed: int = SEED, sr: int = SR) -> None:
    data = _pink(int(round(seconds * sr)), 2, seed=seed) * (10 ** (gain_db / 20.0))
    data = np.clip(data, -0.999, 0.999)
    sf.write(str(path), data, sr, subtype="PCM_24")


def _build_two_episodes(root: Path, hot_e04_pm: bool = False) -> Path:
    """Create a two-episode delivery folder and return the folder path."""
    root.mkdir(parents=True, exist_ok=True)
    roles_stems = [
        ("SHOW_S01E03_PM_STEREO.wav", SEED, 0.0),
        ("SHOW_S01E03_DX_STEREO.wav", SEED + 1, 0.0),
        ("SHOW_S01E03_MX_STEREO.wav", SEED + 2, 0.0),
        ("SHOW_S01E03_FX_STEREO.wav", SEED + 3, 0.0),
        ("SHOW_S01E04_PM_STEREO.wav", SEED + 4, 15.0 if hot_e04_pm else 0.0),
        ("SHOW_S01E04_DX_STEREO.wav", SEED + 5, 0.0),
        ("SHOW_S01E04_MX_STEREO.wav", SEED + 6, 0.0),
        ("SHOW_S01E04_FX_STEREO.wav", SEED + 7, 0.0),
    ]
    for name, seed, gain in roles_stems:
        _write_pink_stereo(root / name, seed=seed, gain_db=gain)
    return root


# ---------------------------------------------------------------------------


def test_all_two_episodes_both_pass(tmp_path: Path) -> None:
    folder = _build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 3
    assert data["command"] == "all"
    assert data["summary"]["groups_total"] == 2
    assert data["summary"]["groups_passed"] == 2
    assert data["summary"]["overall_pass"] is True
    gids = {g["group_id"] for g in data["groups"]}
    assert gids == {"S01E03", "S01E04"}


def test_all_hot_e04_fails_just_that_group(tmp_path: Path) -> None:
    folder = _build_two_episodes(tmp_path / "delivery", hot_e04_pm=True)
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert data["summary"]["overall_pass"] is False
    by_gid = {g["group_id"]: g for g in data["groups"]}
    assert by_gid["S01E03"]["group_summary"]["overall_pass"] is True
    assert by_gid["S01E04"]["group_summary"]["overall_pass"] is False


def test_all_netflix_spec_runs_dialog_check_without_dx_flag(tmp_path: Path) -> None:
    folder = _build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "netflix_stereo",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    data = json.loads(result.output)
    # Each group should have a dialog_lufs check on its DX file.
    for g in data["groups"]:
        checks = {(f["role"], c["metric"]) for f in g["files"] for c in f["checks"]}
        assert ("dx", "dialog_lufs") in checks


def test_all_duplicate_pm_isolates_to_group(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    _write_pink_stereo(folder / "SHOW_S01E03_PM_STEREO.wav", seed=1)
    _write_pink_stereo(folder / "SHOW_S01E03_PRINTMASTER_STEREO.wav", seed=2)
    _write_pink_stereo(folder / "SHOW_S01E04_PM_STEREO.wav", seed=3)
    _write_pink_stereo(folder / "SHOW_S01E04_DX_STEREO.wav", seed=4)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    by_gid = {g["group_id"]: g for g in data["groups"]}
    e03 = by_gid["S01E03"]
    assert any(e["type"] == "DuplicateRoleError" for e in e03["errors"])
    assert e03["files"] == []  # nothing measured for the failed group
    assert by_gid["S01E04"]["group_summary"]["overall_pass"] is True


def test_all_sample_rate_mismatch_within_group(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    # Two SRs in the same group → that group fails, other group runs.
    _write_pink_stereo(folder / "SHOW_S01E03_PM_STEREO.wav", sr=48000, seed=1)
    _write_pink_stereo(folder / "SHOW_S01E03_DX_STEREO.wav", sr=44100, seed=2)
    _write_pink_stereo(folder / "SHOW_S01E04_PM_STEREO.wav", sr=48000, seed=3)
    _write_pink_stereo(folder / "SHOW_S01E04_DX_STEREO.wav", sr=48000, seed=4)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    by_gid = {g["group_id"]: g for g in data["groups"]}
    assert any(e["type"] == "SampleRateMismatch" for e in by_gid["S01E03"]["errors"])
    assert by_gid["S01E04"]["group_summary"]["overall_pass"] is True


def test_all_zero_wavs_exit_two(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("nothing here\n")
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(empty),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "No WAV" in result.output or "no wav" in result.output.lower()


def test_all_ambiguous_filename_exit_two(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    # Contains tokens matching both pm ('MIX') and mx ('MUSIC').
    _write_pink_stereo(folder / "SHOW_S01E01_MIX_MUSIC_STEREO.wav", seed=1)
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "multiple role patterns" in result.output or "matched" in result.output.lower()


def test_all_surfaces_channel_config_hint_and_actual(tmp_path: Path) -> None:
    folder = _build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    for g in data["groups"]:
        for f in g["files"]:
            # 2ch stereo audio, filenames contain `_STEREO_` → both derive stereo.
            assert f["channel_config_hint"] == "stereo", f
            assert f["channel_config_actual"] == "stereo", f


def test_all_include_unclassified_measures_unknown_role(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    _write_pink_stereo(folder / "SHOW_S01E01_PM_STEREO.wav", seed=1)
    _write_pink_stereo(folder / "SHOW_S01E01_MYSTERY_STEREO.wav", seed=2)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
        "--include-unclassified",
    ])
    assert result.exit_code in (0, 1), result.output
    data = json.loads(result.output)
    roles = {f["role"] for g in data["groups"] for f in g["files"]}
    assert "unknown" in roles
    # The unknown file is ALSO listed in unclassified[] for traceability.
    assert any("MYSTERY" in u["path"] for u in data["unclassified"])
