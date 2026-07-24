from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from finalpass.jobs import run_all, run_loudness, run_me, run_null
from finalpass.models import AllReport, MEReport
from finalpass.report import grouped_templates_present, render_report_html
from tests.audio_cases import (
    SEED,
    SR,
    build_split_group,
    build_two_episodes,
    exact_sum_components,
    me_check_components,
    true_peak_over_program,
    write_audio,
)


def _build_null_fail_report(root: Path):
    data = exact_sum_components(base_seed=SEED + 1000)
    start = int(round(5.0 * SR))
    end = int(round(7.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    pm = write_audio(root / "SHOW_S01E04_PM_STEREO.wav", data["pm"])
    dx = write_audio(root / "SHOW_S01E04_DX_STEREO.wav", data["dx"])
    mx = write_audio(root / "SHOW_S01E04_MX_STEREO.wav", data["mx"])
    fx = write_audio(root / "SHOW_S01E04_FX_STEREO.wav", data["fx"])
    return run_null(pm=pm, stems=(dx, mx, fx), fps=23.976)


def _build_me_fail_report(root: Path):
    data = me_check_components(base_seed=SEED + 1001, bleed_region=(5.0, 7.0), bleed_gain=0.08)
    me = write_audio(root / "SHOW_S01E04_ME_STEREO.wav", data["me"])
    dx = write_audio(root / "SHOW_S01E04_DX_STEREO.wav", data["dx"])
    return run_me(me_file=me, dx_file=dx, fps=23.976)


def test_render_loudness_html_smoke(pink_stereo_10s: Path) -> None:
    report = run_loudness(files=(pink_stereo_10s,), spec_name="ebu_r128", dx_file=None, fps=23.976)
    html = render_report_html(report)
    assert "<html" in html.lower()
    assert "<style" in html.lower()
    assert "Loudness Summary" in html
    assert "pink_stereo_10s.wav" in html
    assert "Integrated LUFS" in html
    assert "Checks" in html


def test_render_loudness_html_shows_true_peak_flags(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "tp_over_primary.wav", true_peak_over_program())
    report = run_loudness(files=(path,), spec_name="ebu_r128", dx_file=None, fps=23.976)
    html = render_report_html(report)
    assert "True-peak flagged-region timeline" in html
    assert "LOUDNESS" in html
    assert "true_peak_dbtp" in html
    assert "over by" in html


def test_render_null_html_smoke(tmp_path: Path) -> None:
    report = _build_null_fail_report(tmp_path / "null_case")
    html = render_report_html(report)
    assert "<svg" in html.lower()
    assert "Null Summary" in html
    assert "Flagged regions" in html
    assert "NULL" in html
    assert "residual_rms_dbfs" in html


def test_render_me_html_smoke(tmp_path: Path) -> None:
    report = _build_me_fail_report(tmp_path / "me_case")
    html = render_report_html(report)
    assert "<svg" in html.lower()
    assert "M&amp;E Summary" in html
    assert "Flagged regions" in html
    assert "ME" in html
    assert "dialog_bleed_score" in html


def test_render_all_html_smoke(tmp_path: Path) -> None:
    folder = build_two_episodes(
        tmp_path / "delivery",
        hot_e04_pm=True,
        e04_null_defect=True,
        e04_me_bleed_defect=True,
    )
    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=23.976,
    )
    html = render_report_html(report)
    assert "Run Summary" in html
    assert "S01E03" in html
    assert "S01E04" in html
    assert "Measured assets" in html
    assert "Null check" in html
    assert "M&amp;E check" in html


def test_render_all_html_shows_true_peak_flags(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    write_audio(folder / "SHOW_S01E05_PM_STEREO.wav", true_peak_over_program())
    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=23.976,
    )
    html = render_report_html(report)
    assert "True-peak flagged-region timeline" in html
    assert "LOUDNESS" in html
    assert "S01E05" in html


def test_rendered_html_has_no_external_assets(pink_stereo_10s: Path) -> None:
    report = run_loudness(files=(pink_stereo_10s,), spec_name="ebu_r128", dx_file=None, fps=23.976)
    html = render_report_html(report)
    lowered = html.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<script" not in lowered
    assert "<link rel=" not in lowered
    assert "@import" not in lowered


def test_packaged_templates_load_via_importlib_resources(pink_stereo_10s: Path) -> None:
    assert grouped_templates_present() is True
    template_root = importlib.resources.files("finalpass.templates")
    assert template_root.joinpath("base.html.j2").is_file()
    report = run_loudness(files=(pink_stereo_10s,), spec_name="ebu_r128", dx_file=None, fps=23.976)
    html = render_report_html(report)
    assert "FinalPass HTML Report" in html


def test_render_split_loudness_html_uses_polished_display_label(tmp_path: Path) -> None:
    family = build_split_group(
        tmp_path / "delivery",
        "S01E03",
        layout="stereo",
        write_roles=("pm",),
        base_seed=SEED + 1100,
    )["pm"]
    report = run_loudness(files=(family["L"],), spec_name="ebu_r128", dx_file=None, fps=23.976)
    html = render_report_html(report)
    assert "SHOW_S01E03_Comp_LtRt" in html
    assert "SHOW_S01E03_Comp_LtRt.L.wav" in html
    assert "split mono" in html
    assert "2 mono files" in html


def test_render_split_all_html_uses_polished_display_label(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    families = build_split_group(
        folder,
        "S01E03",
        layout="stereo",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 1110,
    )
    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=23.976,
    )
    html = render_report_html(report)
    assert "SHOW_S01E03_Comp_LtRt" in html
    assert "Source files" in html
    assert str(families["pm"]["L"]) in html
    logical_assets_section = html.split("Logical assets", 1)[1].split("Measured assets", 1)[0]
    assert "SHOW_S01E03_Comp_LtRt.L.wav" in logical_assets_section
    assert "SHOW_S01E03_Comp_LtRt.R.wav" in logical_assets_section
    assert str(families["pm"]["L"]) not in logical_assets_section
    assert "split mono" in html


def test_round_trip_me_json_renders_html(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_me_bleed_defect=True)
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(folder / "SHOW_S01E04_ME_STEREO.wav"),
        "--dx", str(folder / "SHOW_S01E04_DX_STEREO.wav"),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    report = MEReport.model_validate(json.loads(result.output))
    html = render_report_html(report)
    assert "SHOW_S01E04_ME_STEREO.wav" in html
    assert "dialog_bleed_score" in html


def test_round_trip_all_json_renders_html(tmp_path: Path) -> None:
    folder = build_two_episodes(
        tmp_path / "delivery",
        hot_e04_pm=True,
        e04_null_defect=True,
        e04_me_bleed_defect=True,
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "all",
        str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    report = AllReport.model_validate(json.loads(result.output))
    html = render_report_html(report)
    assert "Measured assets" in html
    assert "Null flagged-region timeline" in html
    assert "M&E flagged-region timeline" in html


def test_render_all_html_surfaces_blocking_issues_summary(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_two_episodes(folder)
    # Duplicate the S01E03 printmaster to force a blocking selection issue.
    pm = folder / "SHOW_S01E03_PM_STEREO.wav"
    duplicate = folder / "SHOW_S01E03_PRINTMASTER_STEREO.wav"
    duplicate.write_bytes(pm.read_bytes())

    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=23.976,
    )
    html = render_report_html(report)
    assert "Blocking issues" in html
    assert "FAIL — 0 check failures, 1 blocking issue" in html
    assert "Competing same-layout asset" in html


def test_render_all_html_skipped_analyses_do_not_render_empty_timelines(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    build_two_episodes(folder)
    for path in folder.glob("*_MX_STEREO.wav"):
        path.unlink()
    for path in folder.glob("*_FX_STEREO.wav"):
        path.unlink()
    for path in folder.glob("*_ME_STEREO.wav"):
        path.unlink()

    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=23.976,
    )
    html = render_report_html(report)
    assert "Auto-null was not run because no same-layout DX+MX+FX or DX+ME set was available." in html
    assert "M&E was not run because no DX and M&E pair was available." in html
    assert "Null flagged-region timeline" not in html
    assert "M&amp;E flagged-region timeline" not in html
