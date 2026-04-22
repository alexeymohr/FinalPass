from __future__ import annotations

from pathlib import Path

import aaf2
import pytest
from click.testing import CliRunner

from finalpass import aaf_export
from finalpass.cli import main
from finalpass.errors import AAFExportError
from finalpass.models import AllReport
from tests.audio_cases import SEED, SR, build_two_episodes, exact_sum_components, me_check_components, write_audio


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


def _read_marker_aaf(path: Path) -> dict[str, object]:
    with aaf2.open(str(path), "r") as handle:
        composition = next(handle.content.compositionmobs())
        slot = next(slot for slot in composition.slots if type(slot).__name__ == "EventMobSlot")
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
            "markers": markers,
        }


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
    aaf_path = out_dir / "markers.aaf"
    assert aaf_path.exists()
    parsed = _read_marker_aaf(aaf_path)
    assert parsed["slot_name"] == "FinalPass Markers"
    assert parsed["edit_rate"] == "24000/1001"
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
    aaf_path = out_dir / "markers.aaf"
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
    aaf_path = out_dir / "markers.aaf"
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
    assert not (out_dir / "report.json").exists()
    assert not (out_dir / "report.html").exists()
    assert not (out_dir / "markers.aaf").exists()


def test_no_partial_aaf_left_on_export_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 500,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "markers.aaf"
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
    assert not stale.exists()


def test_collect_marker_candidates_from_all_uses_only_timed_flags(tmp_path: Path) -> None:
    folder = build_two_episodes(tmp_path / "delivery", hot_e04_pm=True, e04_null_defect=True, e04_me_bleed_defect=True)
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
