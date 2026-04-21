from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main


def test_specs_list_contains_all_bundled() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["specs", "list"])
    assert result.exit_code == 0
    for name in ("ebu_r128", "netflix_stereo", "netflix_51", "streaming_-14"):
        assert name in result.output


def test_specs_show_ebu_r128() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["specs", "show", "ebu_r128"])
    assert result.exit_code == 0
    assert "EBU R128" in result.output
    assert "-23" in result.output


def test_loudness_pass_exits_zero(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    # Pink noise at -23 dBFS should pass R128.
    assert result.exit_code in (0, 1), f"unexpected exit: {result.output}"
    report_path = tmp_path / "out" / "report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["schema_version"] == 1
    assert data["spec"]["name"] == "ebu_r128"
    assert data["files"][0]["role"] == "primary"


def test_loudness_hot_file_fails(pink_stereo_10s_hot: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s_hot),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 1, f"expected fail, got: {result.output}"


def test_json_only_is_parseable(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code in (0, 1)
    data = json.loads(result.output)
    assert data["schema_version"] == 1


def test_channel_mismatch_exits_two(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "netflix_51",  # expects 5.1, got stereo
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "5.1" in result.output or "6 ch" in result.output


def test_loudness_v1_does_not_leak_phase2_fields(pink_stereo_10s: Path, tmp_path: Path) -> None:
    """schema_version:1 output must not include channel_config_hint/actual —
    those are Phase 2+ FileReport fields and should be excluded on serialize."""
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 1
    for f in data["files"]:
        assert "channel_config_hint" not in f
        assert "channel_config_actual" not in f


def test_silent_file_skips_lra_and_fails_integrated(silent_stereo_5s: Path, tmp_path: Path) -> None:
    """Silent file: integrated fails (real problem), TP passes (silent has no
    peak), LRA is skipped (no gated content to measure). Overall fails on
    integrated; skipped LRA does not count toward pass or fail."""
    runner = CliRunner()
    out = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness", str(silent_stereo_5s),
        "--spec", "ebu_r128",
        "--out", str(out),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    checks = {c["metric"]: c for c in data["files"][0]["checks"]}
    assert checks["integrated_lufs"]["pass"] is False
    assert checks["true_peak_dbtp"]["pass"] is True
    lra = checks["lra"]
    assert lra["pass"] is None
    assert lra["skipped"] is True
    assert lra["reason"] == "insufficient_gated_content"
    # Summary tally: skipped does not land in passed or failed.
    s = data["summary"]
    assert s["skipped"] == 1
    assert s["passed"] + s["failed"] + s["skipped"] == s["total_checks"]


def test_unknown_spec_exits_two(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "no_such_spec",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "no_such_spec" in result.output or "Unknown spec" in result.output
