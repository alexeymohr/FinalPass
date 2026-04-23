from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from finalpass.prep_folders import BUCKET_SPECS, PREP_ROOT_NAME, create_prep_layout
from tests.audio_cases import SEED, build_split_group


def _copy_family(members: dict[str, Path], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in members.values():
        shutil.copy2(path, target_dir / path.name)


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
        input=f"{folder}\n1\n1\n2\n1\n1\n1\n3\n",
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
    assert "Job: all" in result.output
    assert "Status: PASS" in result.output
    assert "Artifacts written to:" in result.output
    assert "Produced artifacts:" in result.output


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
        input="1\n2\n1\n2\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["files"][0]["source_kind"] == "split_mono"
    assert payload["files"][0]["member_legs"] == ["L", "R"]
    assert "split mono (2 mono files)" in result.output or "split mono · stereo · LtRt · 2 mono files" in result.output
    assert "SHOW_S01E03_Comp_LtRt" in result.output
    assert ".L.wav" not in result.output
    assert "Job: loudness" in result.output
    assert "Artifacts written to:" in result.output
    assert "Produced artifacts:" in result.output


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
        input="1\n3\n1\n1\n1\n1\n3\n",
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
        input="1\n4\n1\n1\n1\n3\n",
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
    result = runner.invoke(main, ["wizard", str(folder)], input="99\n4\n")
    assert result.exit_code == 0, result.output
    assert "Invalid choice. Enter 1-4 or q to quit." in result.output
    assert "Traceback" not in result.output


def test_wizard_back_and_quit_work(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(folder, "S01E03", layout="stereo", write_roles=("pm",), base_seed=SEED + 2050)

    runner = CliRunner()
    result = runner.invoke(main, ["wizard", str(folder)], input="1\n1\n0\nq\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Select a job type.") == 2


def test_wizard_review_step_requires_confirmation(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_split_group(folder, "S01E03", layout="stereo", write_roles=("pm",), base_seed=SEED + 2055)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="1\n2\n1\n1\n1\n0\nq\n",
    )
    assert result.exit_code == 0, result.output
    assert "Review:" in result.output
    assert "Ready to run?" in result.output
    assert "Run job" in result.output
    assert not (out_dir / "report.json").exists()


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
        input="1\n2\n1\n1\n1\n1\n1\n1\n3\n",
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
        input="1\n4\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    assert "Auto-selected the only group in this folder." in result.output
    assert "Auto-selected the only M&E asset in this group." in result.output
    assert "Auto-selected the only same-layout DX asset." in result.output


def test_wizard_can_create_prep_folders_and_quit(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()

    runner = CliRunner()
    result = runner.invoke(main, ["wizard", str(folder)], input="2\n2\n")
    assert result.exit_code == 0, result.output

    prep_root = folder / PREP_ROOT_NAME
    assert prep_root.is_dir()
    assert all((prep_root / bucket_name).is_dir() for bucket_name, _, _ in BUCKET_SPECS)
    assert "Prep folders created:" in result.output
    assert "Move or copy the stems you want analyzed into the appropriate prep buckets." in result.output


def test_wizard_auto_resumes_existing_prep_layout_and_ignores_outside_files(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    families = build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx", "mx", "fx"),
        base_seed=SEED + 2080,
    )
    create_prep_layout(folder)
    prep_root = folder / PREP_ROOT_NAME
    _copy_family(families["pm"], prep_root / "Stereo Printmaster")
    _copy_family(families["dx"], prep_root / "Stereo Dialogue")

    ignored = prep_root / "Ignore"
    ignored.mkdir(exist_ok=True)
    shutil.copy2(next(iter(families["mx"].values())), ignored / "SHOW_S01E99_MIX_MUSIC_STEREO.wav")
    shutil.copy2(next(iter(families["mx"].values())), folder / "SHOW_S01E98_MIX_MUSIC_STEREO.wav")

    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="1\n2\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    assert "I found an existing FinalPass prep layout." in result.output
    assert "Create FinalPass prep folders" not in result.output

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["command"] == "all"
    assert payload["discovery_errors"] == []
    asset_paths = {
        asset["path"]
        for group in payload["groups"]
        for asset in group["assets"]
    }
    assert asset_paths
    assert all(PREP_ROOT_NAME in path for path in asset_paths)
    assert not any("S01E98" in path for path in asset_paths)
    assert not any("S01E99" in path for path in asset_paths)


def test_wizard_empty_prep_layout_is_handled_cleanly(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    create_prep_layout(folder)

    runner = CliRunner()
    result = runner.invoke(main, ["wizard", str(folder)], input="3\n")
    assert result.exit_code == 0, result.output
    assert "I found an existing FinalPass prep layout." in result.output
    assert "I found the FinalPass prep folders, but none of the analysis buckets contain files yet." in result.output
    assert "Re-scan prep folders" in result.output


def test_wizard_resume_after_restart_uses_populated_prep_buckets(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()

    runner = CliRunner()
    first = runner.invoke(main, ["wizard", str(folder)], input="2\n2\n")
    assert first.exit_code == 0, first.output

    families = build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm",),
        base_seed=SEED + 2090,
    )
    prep_root = folder / PREP_ROOT_NAME
    _copy_family(families["pm"], prep_root / "Stereo Printmaster")

    out_dir = tmp_path / "out"
    second = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="2\n1\n1\n1\n1\n3\n",
    )
    assert second.exit_code == 0, second.output
    assert "I found an existing FinalPass prep layout." in second.output
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["files"][0]["source_kind"] == "split_mono"


def test_wizard_prep_mode_handles_mixed_presentations_coherently(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    stereo = build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx"),
        base_seed=SEED + 2100,
    )
    surround = build_split_group(
        folder,
        "S01E03",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx"),
        base_seed=SEED + 2110,
    )
    create_prep_layout(folder)
    prep_root = folder / PREP_ROOT_NAME
    _copy_family(stereo["pm"], prep_root / "Stereo Printmaster")
    _copy_family(stereo["dx"], prep_root / "Stereo Dialogue")
    _copy_family(surround["pm"], prep_root / "5.1 Printmaster")
    _copy_family(surround["dx"], prep_root / "5.1 Dialogue")
    _copy_family(surround["mx"], prep_root / "5.1 Music")
    _copy_family(surround["fx"], prep_root / "5.1 Effects")

    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["wizard", str(folder), "--out", str(out_dir)],
        input="1\n3\n1\n1\n1\n3\n",
    )
    assert result.exit_code == 0, result.output
    assert "Mode: prep" in result.output
    assert "Populated prep buckets: 6" in result.output
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["command"] == "all"
    assert payload["spec"]["name"] == "netflix_51"
    assert payload["summary"]["groups_total"] == 1
    assert payload["groups"][0]["group_id"] not in {
        "5.1 DIALOGUE",
        "5.1 EFFECTS",
        "5.1 MUSIC",
        "5.1 PRINTMASTER",
        "STEREO DIALOGUE",
        "STEREO PRINTMASTER",
    }
    assert payload["groups"][0]["errors"] == []
    assert payload["groups"][0]["null_test"]["pass"] is True
    assert payload["groups"][0]["null_test"]["stem_strategy"] == "dx_mx_fx"
    assert payload["groups"][0]["me_check"]["skipped"] is True
    assert any(
        asset["channel_config_actual"] == "5.1"
        for group in payload["groups"]
        for asset in group["assets"]
    )
