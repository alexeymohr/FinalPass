from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from tests.audio_cases import SEED, build_split_group


def test_wizard_all_flow_succeeds_on_split_folder(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 2000,
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", "--out", str(out_dir)],
        input=f"{folder}\n1\n2\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    report_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    assert report_path.exists()
    assert html_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 6
    assert payload["command"] == "all"
    assert "Folder summary:" in result.output
    assert "Status: PASS" in result.output


def test_wizard_loudness_flow_succeeds_on_split_asset(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx"),
        base_seed=SEED + 2010,
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="2\n1\n2\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["files"][0]["source_kind"] == "split_mono"
    assert payload["files"][0]["member_legs"] == ["L", "R"]
    assert "split_mono · 2 files" in result.output
    assert ".L.wav" not in result.output


def test_wizard_null_flow_succeeds_with_auto_stem_plan(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx"),
        base_seed=SEED + 2020,
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="3\n1\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    report_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    aaf_path = out_dir / "markers.aaf"
    assert report_path.exists()
    assert html_path.exists()
    assert not aaf_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["command"] == "null"
    assert payload["null_test"]["pass"] is True


def test_wizard_me_flow_succeeds(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 2030,
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="4\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    report_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    assert report_path.exists()
    assert html_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["command"] == "me"
    assert payload["me_check"]["pass"] is True


def test_wizard_invalid_menu_choice_reprompts_cleanly(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(folder, "S01E03", layout="stereo", write_roles=("pm",), base_seed=SEED + 2040)

    runner = CliRunner()
    result = runner.invoke(main, ["wizard", str(folder)], input="99\n6\n")
    assert result.exit_code == 0, result.output
    assert "Invalid choice. Enter 1-6 or q to quit." in result.output
    assert "Traceback" not in result.output


def test_wizard_back_and_quit_work(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(folder, "S01E03", layout="stereo", write_roles=("pm",), base_seed=SEED + 2050)

    runner = CliRunner()
    result = runner.invoke(main, ["wizard", str(folder)], input="1\n0\nq\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Select a job type.") == 2


def test_wizard_discovery_errors_can_be_viewed_and_continue(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 2060,
    )
    broken = build_split_group(
        folder,
        "S01E04",
        layout="stereo",
        write_roles=("pm",),
        base_seed=SEED + 2061,
    )["pm"]
    broken["R"].unlink()
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="2\n1\n1\n2\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    assert "Discovery errors:" in result.output
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 6
    assert payload["discovery_errors"]


def test_wizard_single_candidate_auto_select_is_visible(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(
        folder,
        "S01E03",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 2070,
    )
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="4\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    assert "Auto-selected the only group in this folder." in result.output
    assert "Auto-selected the only M&E asset in this group." in result.output
    assert "Auto-selected the only same-layout DX asset." in result.output
