from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from tests.asset_helpers import write_interleaved, write_split_from_array
from tests.audio_cases import SEED, SR, exact_sum_components, me_check_components, write_audio


def test_split_stereo_seed_path_loudness_succeeds(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 100, n_channels=2)
    family = write_split_from_array(tmp_path / "case", "Comp LtRt", "stereo", data["pm"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    file_report = payload["files"][0]
    assert payload["schema_version"] == 4
    assert file_report["source_kind"] == "split_mono"
    assert file_report["presentation_label"] == "LtRt"
    assert file_report["member_legs"] == ["L", "R"]
    assert file_report["source_paths"] == [str(family["L"]), str(family["R"])]
    assert file_report["channel_count"] == 2
    assert file_report["channel_config_actual"] == "stereo"


def test_split_51_seed_path_loudness_succeeds(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 101, n_channels=6)
    family = write_split_from_array(tmp_path / "case", "Comp 5.1", "5.1", data["pm"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "netflix_51",
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    payload = json.loads(result.output)
    file_report = payload["files"][0]
    assert payload["schema_version"] == 4
    assert file_report["source_kind"] == "split_mono"
    assert file_report["presentation_label"] == "5.1"
    assert file_report["member_legs"] == ["L", "R", "C", "LFE", "Ls", "Rs"]
    assert file_report["channel_count"] == 6
    assert file_report["channel_config_actual"] == "5.1"


def test_split_50_seed_path_loudness_pads_lfe_and_reports_real_members(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 1011, n_channels=6)
    family = write_split_from_array(
        tmp_path / "case",
        "DX 5.0",
        "5.0",
        data["dx"][:, [0, 1, 2, 4, 5]],
    )

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "netflix_51",
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    payload = json.loads(result.output)
    file_report = payload["files"][0]
    assert file_report["source_kind"] == "split_mono"
    assert file_report["presentation_label"] == "5.0"
    assert file_report["member_legs"] == ["L", "R", "C", "Ls", "Rs"]
    assert len(file_report["source_paths"]) == 5
    assert file_report["channel_count"] == 6
    assert file_report["channel_config_actual"] == "5.1"
    assert file_report["channel_config_hint"] == "5.0"


def test_incomplete_split_family_loudness_fails_cleanly(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 102, n_channels=6)
    family = write_split_from_array(tmp_path / "case", "Comp 5.1", "5.1", data["pm"])
    family["Rs"].unlink()

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "netflix_51",
    ])
    assert result.exit_code == 2
    assert "Comp 5.1.L.wav" in result.output
    assert "missing" in result.output.lower() or "leg" in result.output.lower()


def test_split_51_pm_and_stems_null_succeeds(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 103, n_channels=6)
    pm = write_split_from_array(tmp_path / "case", "Comp 5.1", "5.1", data["pm"])
    dx = write_split_from_array(tmp_path / "case", "DX 5.1", "5.1", data["dx"])
    mx = write_split_from_array(tmp_path / "case", "MX 5.1", "5.1", data["mx"])
    fx = write_split_from_array(tmp_path / "case", "FX 5.1", "5.1", data["fx"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(pm["L"]),
        str(dx["L"]),
        str(mx["L"]),
        str(fx["L"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 4
    assert payload["printmaster"]["source_kind"] == "split_mono"
    assert payload["printmaster"]["member_legs"] == ["L", "R", "C", "LFE", "Ls", "Rs"]
    assert all(stem["source_kind"] == "split_mono" for stem in payload["stems"])
    assert payload["null_test"]["pass"] is True


def test_mixed_interleaved_and_split_null_succeeds(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 104, n_channels=6)
    root = tmp_path / "case"
    pm = write_audio(root / "Comp 5.1.wav", data["pm"])
    dx = write_split_from_array(root, "DX 5.1", "5.1", data["dx"])
    mx = write_split_from_array(root, "MX 5.1", "5.1", data["mx"])
    fx = write_split_from_array(root, "FX 5.1", "5.1", data["fx"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(pm),
        str(dx["L"]),
        str(mx["L"]),
        str(fx["L"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["printmaster"]["source_kind"] == "interleaved"
    assert payload["stems"][0]["source_kind"] == "split_mono"


def test_invalid_split_stem_family_null_fails_cleanly(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 105, n_channels=6)
    root = tmp_path / "case"
    pm = write_audio(root / "Comp 5.1.wav", data["pm"])
    dx = write_split_from_array(root, "DX 5.1", "5.1", data["dx"])
    mx = write_audio(root / "MX 5.1.wav", data["mx"])
    fx = write_audio(root / "FX 5.1.wav", data["fx"])
    write_audio(dx["Rs"], data["dx"][:-100, [5]])

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(pm),
        str(dx["L"]),
        str(mx),
        str(fx),
    ])
    assert result.exit_code == 2
    assert "DX 5.1.L.wav" in result.output
    assert "sample count" in result.output.lower() or "samples" in result.output.lower()


def test_split_me_and_split_dx_succeed(tmp_path: Path) -> None:
    data = me_check_components(base_seed=SEED + 106, n_channels=6)
    me = write_split_from_array(tmp_path / "case", "ME 5.1", "5.1", data["me"])
    dx = write_split_from_array(tmp_path / "case", "DX 5.1", "5.1", data["dx"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(me["L"]),
        "--dx", str(dx["L"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 4
    assert payload["me_file"]["source_kind"] == "split_mono"
    assert payload["dx_file"]["source_kind"] == "split_mono"
    assert payload["me_check"]["pass"] is True


def test_ambiguous_split_family_me_fails_cleanly(tmp_path: Path) -> None:
    data = me_check_components(base_seed=SEED + 107, n_channels=6)
    root = tmp_path / "case"
    me = write_split_from_array(root, "ME 5.1", "5.1", data["me"])
    write_split_from_array(root, "DX 5.1", "5.1", data["dx"])
    unlabeled_dx = write_split_from_array(root, "DX", "5.1", data["dx"])

    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(me["L"]),
        "--dx", str(unlabeled_dx["L"]),
    ])
    assert result.exit_code == 2
    assert "DX.L.wav" in result.output
    assert "ambiguous" in result.output.lower()


def test_split_null_failing_case_writes_aaf(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 108, n_channels=2)
    start = int(round(5.0 * SR))
    end = int(round(7.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    root = tmp_path / "case"
    pm = write_split_from_array(root, "Comp LtRt", "stereo", data["pm"])
    dx = write_split_from_array(root, "DX LtRt", "stereo", data["dx"])
    mx = write_split_from_array(root, "MX LtRt", "stereo", data["mx"])
    fx = write_split_from_array(root, "FX LtRt", "stereo", data["fx"])
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(pm["L"]),
        str(dx["L"]),
        str(mx["L"]),
        str(fx["L"]),
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    assert (out_dir / "markers.aaf").exists()


def test_split_me_failing_case_writes_aaf(tmp_path: Path) -> None:
    data = me_check_components(
        base_seed=SEED + 109,
        n_channels=2,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    root = tmp_path / "case"
    me = write_split_from_array(root, "ME LtRt", "stereo", data["me"])
    dx = write_split_from_array(root, "DX LtRt", "stereo", data["dx"])
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(me["L"]),
        "--dx", str(dx["L"]),
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    assert (out_dir / "markers.aaf").exists()


def test_split_loudness_html_shows_provenance(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 110, n_channels=2)
    family = write_split_from_array(tmp_path / "case", "Comp LtRt", "stereo", data["pm"])
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "Source provenance" in html
    assert "split mono" in html
    assert "Comp LtRt" in html
    assert "Comp LtRt.L.wav" in html
    assert "Comp LtRt.R.wav" in html


def test_split_loudness_terminal_output_uses_polished_label_and_raw_path(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 112, n_channels=2)
    family = write_split_from_array(tmp_path / "case", "Comp LtRt", "stereo", data["pm"])
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    assert "Comp LtRt" in result.output
    assert "source:" in result.output
    assert "split mono" in result.output
    assert "path:" in result.output
    assert "Comp LtRt.L.wav" in result.output


def test_split_json_only_writes_no_files(tmp_path: Path) -> None:
    data = exact_sum_components(base_seed=SEED + 111, n_channels=2)
    family = write_split_from_array(tmp_path / "case", "Comp LtRt", "stereo", data["pm"])
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(family["L"]),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    assert not (out_dir / "report.json").exists()
    assert not (out_dir / "report.html").exists()
    assert not (out_dir / "markers.aaf").exists()
