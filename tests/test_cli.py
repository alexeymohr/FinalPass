from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from click.testing import CliRunner

from finalpass import __version__
from finalpass.cli import main
from finalpass.jobs import WrittenArtifacts
from finalpass.models import Report, SpecRef, Summary


def test_specs_list_contains_all_bundled() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["specs", "list"])
    assert result.exit_code == 0
    for name in ("atsc_a85", "atsc_a85_51", "ebu_r128", "netflix_stereo", "netflix_51", "streaming_-14"):
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
    html_path = tmp_path / "out" / "report.html"
    aaf_path = tmp_path / "out" / "markers.aaf"
    assert report_path.exists()
    assert html_path.exists()
    assert not aaf_path.exists()
    data = json.loads(report_path.read_text())
    assert data["schema_version"] == 3
    assert data["spec"]["name"] == "ebu_r128"
    assert data["files"][0]["role"] == "primary"
    assert "report.json" in result.output
    assert "report.html" in result.output
    assert "No exportable timed markers" in result.output
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


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
    assert data["schema_version"] == 3
    assert not (tmp_path / "out" / "report.json").exists()
    assert not (tmp_path / "out" / "report.html").exists()
    assert not (tmp_path / "out" / "markers.aaf").exists()


def test_channel_mismatch_exits_two(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "netflix_51",  # expects 5.1, got stereo
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 2
    assert "5.1" in result.output or "6 ch" in result.output


def test_loudness_v3_includes_standalone_provenance_fields(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 3
    file_report = data["files"][0]
    assert file_report["source_kind"] == "interleaved"
    assert file_report["source_paths"] == [str(pink_stereo_10s.resolve())]
    assert file_report["member_legs"] == []
    assert file_report["channel_config_actual"] == "stereo"


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


def test_version_flag_matches_package_metadata() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert importlib.metadata.version("finalpass") == __version__


def test_loudness_command_routes_through_shared_runner(monkeypatch, pink_stereo_10s: Path) -> None:
    calls: dict[str, object] = {}

    def fake_execute_loudness(*, files, spec_name, dx_file, fps):
        calls["files"] = files
        calls["spec_name"] = spec_name
        calls["dx_file"] = dx_file
        calls["fps"] = fps
        return Report(
            finalpass_version=__version__,
            schema_version=3,
            run_id="test-run",
            run_started_at="2026-04-22T00:00:00Z",
            spec=SpecRef(name="ebu_r128", display_name="EBU R128", source="bundled"),
            fps=fps,
            files=[],
            summary=Summary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=True),
        )

    monkeypatch.setattr("finalpass.cli.execute_loudness", fake_execute_loudness)

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    assert calls["files"] == (pink_stereo_10s,)
    assert calls["spec_name"] == "ebu_r128"
    assert calls["dx_file"] is None
    assert calls["fps"] == 23.976


def test_loudness_command_routes_artifact_writes_through_shared_seam(monkeypatch, pink_stereo_10s: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    report = Report(
        finalpass_version=__version__,
        schema_version=3,
        run_id="test-run",
        run_started_at="2026-04-22T00:00:00Z",
        spec=SpecRef(name="ebu_r128", display_name="EBU R128", source="bundled"),
        fps=23.976,
        files=[],
        summary=Summary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=True),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr("finalpass.cli.execute_loudness", lambda **_: report)

    def fake_persist_report_artifacts(*, report, payload, out_dir):
        calls["report"] = report
        calls["payload"] = payload
        calls["out_dir"] = out_dir
        return WrittenArtifacts(
            json_path=out_dir / "report.json",
            html_path=out_dir / "report.html",
            aaf_path=None,
        )

    monkeypatch.setattr("finalpass.cli.persist_report_artifacts", fake_persist_report_artifacts)

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    assert result.exit_code == 0, result.output
    assert calls["report"] == report
    assert calls["out_dir"] == out_dir
    assert '"schema_version": 3' in calls["payload"]


def test_loudness_repeated_runs_reserve_new_report_filenames(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "out"

    first = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])
    second = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(out_dir),
    ])

    assert first.exit_code in (0, 1), first.output
    assert second.exit_code in (0, 1), second.output
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "report-01.json").exists()
    assert (out_dir / "report-01.html").exists()
    assert "report-01.json" in second.output
    assert "report-01.html" in second.output
    assert "markers-01.aaf" in second.output


def test_loudness_terminal_output_shows_full_input_path_on_separate_line(
    pink_stereo_10s: Path,
    tmp_path: Path,
) -> None:
    source_bytes = pink_stereo_10s.read_bytes()
    nested = tmp_path / "very" / "long" / "path" / "for" / "terminal" / "polish"
    nested.mkdir(parents=True, exist_ok=True)
    target = nested / pink_stereo_10s.name
    target.write_bytes(source_bytes)

    runner = CliRunner()
    result = runner.invoke(main, [
        "loudness",
        str(target),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code in (0, 1), result.output
    assert target.name in result.output
    assert "path:" in result.output
    assert str(target) in result.output
