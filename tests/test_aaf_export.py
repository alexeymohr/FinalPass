from __future__ import annotations

import json
from pathlib import Path

import aaf2
import pytest
from click.testing import CliRunner

from finalpass import aaf_export
from finalpass.cli import main
from finalpass.errors import AAFExportError
from finalpass.jobs import run_all
from finalpass.models import AllReport
from finalpass.timecode import sample_to_edit_units
from tests.audio_cases import (
    SEED,
    SR,
    build_two_episodes,
    exact_sum_components,
    me_check_components,
    true_peak_over_program,
    write_audio,
    write_bext_time_reference,
)


def _write_null_case(
    root: Path,
    *,
    base_seed: int = SEED,
    defect_region: tuple[float, float] | None = None,
) -> dict[str, Path]:
    data = exact_sum_components(base_seed=base_seed)
    if defect_region is not None:
        start = int(round(defect_region[0] * SR))
        end = int(round(defect_region[1] * SR))
        data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    root.mkdir(parents=True, exist_ok=True)
    return {
        "pm": write_audio(root / "SHOW_S01E04_PM_STEREO.wav", data["pm"]),
        "dx": write_audio(root / "SHOW_S01E04_DX_STEREO.wav", data["dx"]),
        "mx": write_audio(root / "SHOW_S01E04_MX_STEREO.wav", data["mx"]),
        "fx": write_audio(root / "SHOW_S01E04_FX_STEREO.wav", data["fx"]),
    }


def _write_me_case(
    root: Path,
    *,
    base_seed: int = SEED,
    bleed_region: tuple[float, float] | None = None,
    bleed_gain: float = 0.08,
) -> dict[str, Path]:
    data = me_check_components(
        base_seed=base_seed,
        bleed_region=bleed_region,
        bleed_gain=bleed_gain,
    )
    root.mkdir(parents=True, exist_ok=True)
    return {
        "dx": write_audio(root / "SHOW_S01E04_DX_STEREO.wav", data["dx"]),
        "me": write_audio(root / "SHOW_S01E04_ME_STEREO.wav", data["me"]),
    }


def _write_tp_case(
    root: Path,
    *,
    filename: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return write_audio(root / filename, true_peak_over_program())


def _read_marker_aaf(path: Path) -> dict[str, object]:
    with aaf2.open(str(path), "r") as handle:
        composition = next(handle.content.compositionmobs())
        slot = next(slot for slot in composition.slots if type(slot).__name__ == "EventMobSlot")
        timecode_slot = next(
            candidate for candidate in composition.slots
            if type(candidate).__name__ == "TimelineMobSlot" and getattr(candidate, "name", "") == "Timecode"
        )
        timeline_slot_types = [type(candidate).__name__ for candidate in composition.slots]
        markers = [
            {
                "type": type(marker).__name__,
                "position": marker["Position"].value,
                "length": marker["Length"].value,
                "title": marker.get("Comment").value,
                "annotation": marker.get("CommentMarkerAnnotationList").value,
                "time": marker.get("CommentMarkerTime").value,
                "user_comments": {
                    tag["Name"].value: tag["Value"].value for tag in marker.get("UserComments").value
                }
                if marker.get("UserComments") is not None
                else {},
            }
            for marker in slot.segment.components
        ]
        return {
            "composition_name": composition.name,
            "slot_types": timeline_slot_types,
            "slot_name": slot.name,
            "edit_rate": str(slot.edit_rate),
            "timecode_start": timecode_slot.segment.start,
            "timecode_fps": timecode_slot.segment.fps,
            "timecode_drop": timecode_slot.segment.drop,
            "markers": markers,
        }


def _assert_no_artifacts(out_dir: Path) -> None:
    assert list(out_dir.glob("*report*.json")) == []
    assert list(out_dir.glob("*report*.html")) == []
    assert list(out_dir.glob("*markers*.aaf")) == []


def test_standalone_loudness_tp_only_fail_writes_aaf(tmp_path: Path) -> None:
    file_path = _write_tp_case(tmp_path / "case", filename="tp_over_primary.wav")
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness",
        str(file_path),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    markers = parsed["markers"]
    assert len(markers) == 1
    marker = markers[0]
    assert "[LOUDNESS]" in marker["title"]
    assert "true_peak_dbtp" in marker["title"]
    assert "over by" in marker["title"]
    assert marker["title"] == marker["annotation"]
    assert marker["position"] > 0
    assert marker["length"] >= 1
    assert marker["time"].startswith("00:00:05:")
    assert marker["user_comments"]["Label"] == "[1] FAIL: true peak over"
    assert "[LOUDNESS]" in marker["user_comments"]["Detail"]


def test_standalone_loudness_tp_markers_are_limited_to_one_per_second(tmp_path: Path) -> None:
    root = tmp_path / "case"
    file_path = write_audio(
        root / "tp_over_dense.wav",
        true_peak_over_program(
            burst_regions=((5.0, 5.002), (5.4, 5.402), (6.2, 6.202)),
        ),
    )
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness",
        str(file_path),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output

    report_data = (out_dir / "report.json").read_text(encoding="utf-8")
    parsed = _read_marker_aaf(out_dir / "markers.aaf")
    markers = parsed["markers"]
    assert len(markers) == 2
    assert markers[0]["time"].startswith("00:00:05:")
    assert markers[1]["time"].startswith("00:00:06:")

    data = json.loads(report_data)
    assert len(data["files"][0]["flags"]) == 3


def test_standalone_loudness_bext_start_shifts_aaf_timeline_and_marker_time(tmp_path: Path) -> None:
    file_path = _write_tp_case(tmp_path / "case", filename="tp_over_primary.wav")
    write_bext_time_reference(file_path, 168648480)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness",
        str(file_path),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    parsed = _read_marker_aaf(out_dir / "markers.aaf")
    marker = parsed["markers"][0]
    assert parsed["timecode_start"] == sample_to_edit_units(0, SR, 23.976, start_time_reference_samples=168648480)
    assert marker["time"].startswith("00:58:35:")
    assert 0 < marker["position"] < 1000


def test_standalone_null_failing_case_writes_aaf(tmp_path: Path) -> None:
    files = _write_null_case(tmp_path / "case", base_seed=SEED + 100, defect_region=(5.0, 7.0))
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "show-s01e04-markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    assert parsed["slot_name"] == "FinalPass Markers"
    assert parsed["edit_rate"] == "24000/1001"
    assert parsed["timecode_drop"] is False
    assert parsed["slot_types"] == ["TimelineMobSlot", "TimelineMobSlot", "EventMobSlot"]
    markers = parsed["markers"]
    assert markers
    assert all(marker["type"] == "CommentMarker" for marker in markers)
    assert all("[NULL]" in marker["title"] for marker in markers)
    assert all(marker["title"] == marker["annotation"] for marker in markers)
    positions = [marker["position"] for marker in markers]
    assert positions == sorted(positions)
    assert positions[0] > 0
    assert all(marker["length"] >= 0 for marker in markers)
    assert all(marker["user_comments"]["Comment"] == marker["title"] for marker in markers)
    assert all(marker["user_comments"]["Label"] == "[1] FAIL: null mismatch" for marker in markers)
    assert all("[NULL]" in marker["user_comments"]["Detail"] for marker in markers)


def test_standalone_null_drop_frame_aaf_sets_drop_and_semicolon_marker_time(tmp_path: Path) -> None:
    files = _write_null_case(tmp_path / "case", base_seed=SEED + 110, defect_region=(5.0, 7.0))
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--fps", "29.97",
        "--drop-frame",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "show-s01e04-markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    assert parsed["edit_rate"] == "30000/1001"
    assert parsed["timecode_fps"] == 30
    assert parsed["timecode_drop"] is True
    markers = parsed["markers"]
    assert markers
    assert all(";" in marker["time"] for marker in markers)
    assert markers[0]["time"].startswith("00:00:0")


def test_standalone_me_failing_case_writes_aaf(tmp_path: Path) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 200,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "show-s01e04-markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    markers = parsed["markers"]
    assert markers
    assert all("[ME]" in marker["title"] for marker in markers)
    assert all(marker["title"] == marker["annotation"] for marker in markers)
    assert all(marker["user_comments"]["Comment"] == marker["title"] for marker in markers)
    assert all(marker["user_comments"]["Label"] == "[1] FAIL: dialog bleed" for marker in markers)
    assert all("[ME]" in marker["user_comments"]["Detail"] for marker in markers)
    assert any("corr=" in marker["annotation"] for marker in markers)


@pytest.mark.parametrize("command_name", ["null", "me"])
def test_clean_standalone_runs_write_no_aaf(tmp_path: Path, command_name: str) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "out"
    if command_name == "null":
        files = _write_null_case(tmp_path / "case", base_seed=SEED + 300)
        argv = [
            "null",
            str(files["pm"]),
            str(files["dx"]),
            str(files["mx"]),
            str(files["fx"]),
            "--out", str(out_dir),
        ]
    else:
        files = _write_me_case(tmp_path / "case", base_seed=SEED + 301)
        argv = [
            "me",
            str(files["me"]),
            "--dx", str(files["dx"]),
            "--out", str(out_dir),
        ]

    result = runner.invoke(main, argv)
    assert result.exit_code == 0, result.output
    assert (out_dir / "show-s01e04-report.json").exists()
    assert (out_dir / "show-s01e04-report.html").exists()
    assert not (out_dir / "show-s01e04-markers.aaf").exists()
    assert "No exportable timed markers" in result.output


def test_clean_loudness_run_writes_no_aaf(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness",
        str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code in (0, 1), result.output
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()
    assert not (out_dir / "markers.aaf").exists()
    assert "No exportable timed markers" in result.output


def test_all_run_writes_combined_aaf(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_null_defect=True, e04_me_bleed_defect=True)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "show-s01e03-s01e04-markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    markers = parsed["markers"]
    assert markers
    assert [marker["user_comments"]["Label"] for marker in markers] == [
        "[1] FAIL: null mismatch",
        "[2] FAIL: dialog bleed",
    ]
    assert all(marker["user_comments"]["Comment"] == marker["title"] for marker in markers)
    assert all(marker["title"] == marker["annotation"] for marker in markers)
    assert any("S01E04" in marker["title"] for marker in markers)
    assert not any("[LOUDNESS]" in marker["title"] for marker in markers)
    assert all("Detail" in marker["user_comments"] for marker in markers)


def test_all_run_with_tp_only_group_writes_loudness_aaf(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    _write_tp_case(folder, filename="SHOW_S01E05_PM_STEREO.wav")
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    aaf_path = out_dir / "show-s01e05-markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    markers = parsed["markers"]
    assert len(markers) == 1
    marker = markers[0]
    assert "[LOUDNESS]" in marker["title"]
    assert "S01E05" in marker["title"]
    assert "true_peak_dbtp" in marker["title"]
    assert marker["title"] == marker["annotation"]
    assert marker["user_comments"]["Label"] == "[1] FAIL: true peak over"


def test_all_run_bext_start_shifts_combined_aaf_timeline_and_marker_times(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_null_defect=True, e04_me_bleed_defect=True)
    for wav_path in folder.glob("*.wav"):
        write_bext_time_reference(wav_path, 168648480)
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 1, result.output
    parsed = _read_marker_aaf(out_dir / "show-s01e03-s01e04-markers.aaf")
    assert parsed["timecode_start"] == sample_to_edit_units(0, SR, 23.976, start_time_reference_samples=168648480)
    assert any(marker["time"].startswith("00:58:35:") and "[ME]" in marker["title"] for marker in parsed["markers"])
    assert any("[NULL]" in marker["title"] for marker in parsed["markers"])


def test_clean_all_run_writes_no_aaf(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery")
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    assert (out_dir / "show-s01e03-s01e04-report.json").exists()
    assert (out_dir / "show-s01e03-s01e04-report.html").exists()
    assert not (out_dir / "show-s01e03-s01e04-markers.aaf").exists()
    assert "No exportable timed markers" in result.output


def test_json_only_writes_no_aaf(tmp_path: Path) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 400,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--out", str(out_dir),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    _assert_no_artifacts(out_dir)


def test_no_partial_aaf_left_on_export_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 500,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "show-s01e04-markers.aaf"
    stale.write_bytes(b"stale")

    def boom(*args, **kwargs) -> None:
        raise AAFExportError("boom")

    monkeypatch.setattr(aaf_export, "write_markers_aaf", boom)
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--out", str(out_dir),
    ])
    assert result.exit_code == 2
    assert stale.exists()
    assert not (out_dir / "show-s01e04-markers-01.aaf").exists()


def test_repeated_aaf_runs_reserve_new_marker_filename(tmp_path: Path) -> None:
    files = _write_null_case(tmp_path / "case", base_seed=SEED + 510, defect_region=(5.0, 7.0))
    runner = CliRunner()
    out_dir = tmp_path / "out"

    first = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--out", str(out_dir),
    ])
    second = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--out", str(out_dir),
    ])

    assert first.exit_code == 1, first.output
    assert second.exit_code == 1, second.output
    assert (out_dir / "show-s01e04-markers.aaf").exists()
    assert (out_dir / "show-s01e04-markers-01.aaf").exists()
    assert (out_dir / "show-s01e04-report.json").exists()
    assert (out_dir / "show-s01e04-report-01.json").exists()
    assert "show-s01e04-markers-01.aaf" in second.output


def test_collect_marker_candidates_from_all_uses_only_timed_flags(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", e04_null_defect=True, e04_me_bleed_defect=True)
    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    report = AllReport.model_validate_json(result.output)
    candidates = aaf_export.collect_marker_candidates(report)
    assert candidates
    assert {candidate.code for candidate in candidates} == {"NULL", "ME"}
    assert all(candidate.group_id == "S01E04" for candidate in candidates)


def test_all_me_marker_candidates_use_me_analysis_sample_rate(tmp_path: Path) -> None:
    folder = tmp_path / "delivery"
    folder.mkdir()
    pm_samples = exact_sum_components(seconds=10.0, base_seed=SEED + 520, sr=48000)["pm"]
    me_samples = me_check_components(
        seconds=10.0,
        base_seed=SEED + 521,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
        sr=44100,
    )
    write_audio(folder / "SHOW_S01E04_PM_STEREO.wav", pm_samples, sr=48000)
    write_audio(folder / "SHOW_S01E04_DX_STEREO.wav", me_samples["dx"], sr=44100)
    write_audio(folder / "SHOW_S01E04_ME_STEREO.wav", me_samples["me"], sr=44100)

    report = run_all(
        folder=folder,
        spec_name="ebu_r128",
        patterns_path=None,
        fps=24.0,
    )

    flag = report.groups[0].me_check.flags[0]
    candidate = next(candidate for candidate in aaf_export.collect_marker_candidates(report) if candidate.code == "ME")
    assert candidate.sample_rate == 44100
    assert candidate.start_edit_unit == sample_to_edit_units(flag.start_sample, 44100, 24.0)
    assert candidate.start_edit_unit != sample_to_edit_units(flag.start_sample, 48000, 24.0)
