"""Golden-JSON regression test.

Structural comparison with tolerances — we deliberately ignore ``run_id``,
``run_started_at``, and absolute paths, since those legitimately vary per run.
Float metrics are compared with 0.1 LU / 0.1 dBTP tolerance (the same
precision the terminal table shows).
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from tests.audio_cases import build_two_episodes

FLOAT_TOLERANCE = 0.1


def _normalize(report: dict) -> dict:
    report = dict(report)
    report.pop("run_id", None)
    report.pop("run_started_at", None)
    for f in report.get("files", []):
        f["path"] = Path(f["path"]).name  # keep only basename
        if "source_paths" in f:
            f["source_paths"] = [Path(path).name for path in f["source_paths"]]
    return report


def _assert_close(actual: dict, expected: dict, path: str = "") -> None:
    assert type(actual) is type(expected), f"{path}: type mismatch {type(actual)} vs {type(expected)}"
    if isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys()), (
            f"{path}: key mismatch actual={sorted(actual)} expected={sorted(expected)}"
        )
        for k in expected:
            _assert_close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: list len mismatch {len(actual)} vs {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert abs(actual - expected) <= FLOAT_TOLERANCE, f"{path}: {actual} not within ±{FLOAT_TOLERANCE} of {expected}"
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


def test_golden_ebu_r128(pink_stereo_10s: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "out"
    result = runner.invoke(main, [
        "loudness", str(pink_stereo_10s),
        "--spec", "ebu_r128",
        "--out", str(out),
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    actual = _normalize(json.loads(result.output))

    golden_path = Path(__file__).parent / "fixtures" / "golden_ebu_r128.json"
    if not golden_path.exists():
        # First run — write the current output as the golden. The developer
        # should commit this file once they are happy with the values.
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2) + "\n")
        return

    expected = json.loads(golden_path.read_text())
    _assert_close(actual, expected)


def _normalize_all(report: dict) -> dict:
    report = dict(report)
    report.pop("run_id", None)
    report.pop("run_started_at", None)
    report.pop("folder", None)  # absolute path varies with tmp_path
    for g in report.get("groups", []):
        _normalize_analysis_window(g.get("null_test"))
        _normalize_analysis_window(g.get("me_check"))
        _normalize_analysis_window(g.get("downmix_check"))
        for asset in g.get("channel_checks", []):
            asset["path"] = Path(asset["path"]).name
            asset["source_paths"] = [Path(path).name for path in asset.get("source_paths", [])]
        for asset in g.get("assets", []):
            asset["asset_id"] = _normalize_asset_id(asset["asset_id"])
            asset["path"] = Path(asset["path"]).name
            asset["source_paths"] = [Path(path).name for path in asset.get("source_paths", [])]
        for f in g.get("files", []):
            f["path"] = Path(f["path"]).name
            f["source_paths"] = [Path(path).name for path in f.get("source_paths", [])]
    for f in report.get("files", []):
        f["path"] = Path(f["path"]).name
        f["source_paths"] = [Path(path).name for path in f.get("source_paths", [])]
    for u in report.get("unclassified", []):
        u["path"] = Path(u["path"]).name
        u["source_paths"] = [Path(path).name for path in u.get("source_paths", [])]
    for issue in report.get("discovery_errors", []):
        issue["path"] = Path(issue["path"]).name if issue["path"] else issue["path"]
        issue["source_paths"] = [Path(path).name for path in issue.get("source_paths", [])]
    return report


def _normalize_analysis_window(result: dict | None) -> None:
    if not result:
        return
    window = result.get("analysis_window")
    if not window:
        return
    for item in window.get("inputs", []):
        item["path"] = Path(item["path"]).name


def _normalize_asset_id(value: str) -> str:
    if value.startswith("interleaved::"):
        return f"interleaved::{Path(value.split('::', 1)[1]).name}"
    if value.startswith("split_mono::"):
        prefix, body = value.split("::", 1)
        parts = body.split("|")
        if parts:
            parts[0] = Path(parts[0]).name
        return f"{prefix}::{'|'.join(parts)}"
    return value


def test_golden_all_two_episodes(tmp_path: Path) -> None:
    """Golden for `finalpass all` — two-episode exact-sum delivery, both groups pass."""
    folder = build_two_episodes(tmp_path / "delivery")

    runner = CliRunner()
    result = runner.invoke(main, [
        "all", str(folder),
        "--spec", "ebu_r128",
        "--out", str(tmp_path / "out"),
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    actual = _normalize_all(json.loads(result.output))

    golden_path = Path(__file__).parent / "fixtures" / "golden_all_two_episodes.json"
    if not golden_path.exists():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2) + "\n")
        return

    expected = json.loads(golden_path.read_text())
    _assert_close(actual, expected)
