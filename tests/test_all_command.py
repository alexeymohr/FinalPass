from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from click.testing import CliRunner

from tests.audio_cases import SEED, SR, build_group, build_two_episodes, exact_sum_components, write_audio
from finalpass.cli import main


# ---------------------------------------------------------------------------


def test_all_two_episodes_both_pass(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 5
    assert data["command"] == "all"
    assert data["summary"]["groups_total"] == 2
    assert data["summary"]["groups_passed"] == 2
    assert data["summary"]["overall_pass"] is True
    gids = {g["group_id"] for g in data["groups"]}
    assert gids == {"S01E03", "S01E04"}
    for g in data["groups"]:
        assert g["null_test"]["pass"] is True
        assert g["null_test"]["stem_strategy"] == "dx_mx_fx"
        assert g["me_check"]["pass"] is True
        assert g["me_check"]["flags"] == []


def test_all_hot_e04_fails_just_that_group(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", hot_e04_pm=True)
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
    folder = build_two_episodes(tmp_path / "delivery")
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
    group = build_group(folder, "S01E03", write_roles=("pm", "dx", "mx", "fx"), base_seed=SEED + 100)
    write_audio(folder / "SHOW_S01E03_PRINTMASTER_STEREO.wav", sf.read(str(group["pm"]), dtype="float64", always_2d=True)[0], sr=SR)
    build_group(folder, "S01E04", write_roles=("pm", "dx"), base_seed=SEED + 110)

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
    data = exact_sum_components(base_seed=SEED + 200)
    write_audio(folder / "SHOW_S01E03_PM_STEREO.wav", data["pm"], sr=48000)
    write_audio(folder / "SHOW_S01E03_DX_STEREO.wav", data["dx"], sr=44100)
    build_group(folder, "S01E04", write_roles=("pm", "dx", "mx", "fx"), base_seed=SEED + 210)

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
    data = exact_sum_components(base_seed=SEED + 300)["pm"]
    write_audio(folder / "SHOW_S01E01_MIX_MUSIC_STEREO.wav", data)
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "multiple role patterns" in result.output or "matched" in result.output.lower()


def test_all_surfaces_channel_config_hint_and_actual(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")
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
    build_group(folder, "S01E01", write_roles=("pm",), base_seed=SEED + 400)
    mystery = exact_sum_components(base_seed=SEED + 401)["mx"]
    write_audio(folder / "SHOW_S01E01_MYSTERY_STEREO.wav", mystery)

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


def test_all_auto_null_fallback_to_dx_me(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_group(folder, "S01E03", write_roles=("pm", "dx", "me"), base_seed=SEED + 500)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    group = data["groups"][0]
    assert group["null_test"]["pass"] is True
    assert group["null_test"]["stem_strategy"] == "dx_me"
    assert group["null_test"]["selected_roles"] == ["dx", "me"]


def test_all_insufficient_stems_skips_auto_null(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_group(folder, "S01E03", write_roles=("pm", "dx"), base_seed=SEED + 600)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    group = data["groups"][0]
    assert group["null_test"]["pass"] is None
    assert group["null_test"]["skipped"] is True
    assert group["null_test"]["reason"] == "insufficient_stems_for_auto_null"


def test_all_channel_label_mismatch_fails_null_preflight_only(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_group(
        folder,
        "S01E03",
        write_roles=("pm", "dx", "mx", "fx"),
        role_labels={"pm": "51", "dx": "51", "mx": "51", "fx": "51"},
        base_seed=SEED + 700,
    )

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    group = data["groups"][0]
    assert group["null_test"]["pass"] is False
    assert group["null_test"]["reason"] == "channel_config_label_mismatch"
    assert group["null_test"]["summary"] is None
    assert group["null_test"]["errors"][0]["type"] == "ChannelConfigLabelMismatch"
    # Loudness still ran on the PM file.
    assert group["files"][0]["role"] == "pm"


def test_all_auto_me_success(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    for group in data["groups"]:
        assert group["me_check"]["pass"] is True
        assert group["me_check"]["summary"]["flagged_regions"] == 0


def test_all_auto_me_fail(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_me_bleed_defect=True)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    by_gid = {group["group_id"]: group for group in data["groups"]}
    assert by_gid["S01E03"]["me_check"]["pass"] is True
    assert by_gid["S01E04"]["me_check"]["pass"] is False
    assert by_gid["S01E04"]["me_check"]["flags"]
    assert by_gid["S01E04"]["me_check"]["flags"][0]["code"] == "ME"
    assert by_gid["S01E04"]["me_check"]["flags"][0]["metric"] == "dialog_bleed_score"


def test_all_normal_run_writes_json_and_html(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_null_defect=True, e04_me_bleed_defect=True)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    aaf_path = out_dir / "markers.aaf"
    assert json_path.exists()
    assert html_path.exists()
    assert aaf_path.exists()
    assert "report.json" in result.output
    assert "report.html" in result.output
    assert "markers.aaf" in result.output
    html = html_path.read_text(encoding="utf-8")
    assert "Measured files" in html
    assert "Null check" in html
    assert "M&amp;E check" in html


def test_all_json_only_writes_no_files(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    assert not (out_dir / "report.json").exists()
    assert not (out_dir / "report.html").exists()
    assert not (out_dir / "markers.aaf").exists()


def test_all_clean_normal_run_writes_no_aaf(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()
    assert not (out_dir / "markers.aaf").exists()
    assert "No exportable timed markers" in result.output


def test_all_missing_dx_or_me_skips_auto_me(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_group(folder, "S01E03", write_roles=("pm", "dx"), base_seed=SEED + 800)

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    group = data["groups"][0]
    assert group["me_check"]["pass"] is None
    assert group["me_check"]["skipped"] is True
    assert group["me_check"]["reason"] == "missing_dx_or_me"


def test_all_channel_label_mismatch_fails_me_preflight_only(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_group(
        folder,
        "S01E03",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        role_labels={"me": "51"},
        base_seed=SEED + 900,
    )

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    group = data["groups"][0]
    assert group["null_test"]["pass"] is True
    assert group["me_check"]["pass"] is False
    assert group["me_check"]["reason"] == "channel_config_label_mismatch"
    assert group["me_check"]["summary"] is None
    assert group["me_check"]["errors"][0]["type"] == "ChannelConfigLabelMismatch"
