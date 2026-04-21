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

FLOAT_TOLERANCE = 0.1


def _normalize(report: dict) -> dict:
    report = dict(report)
    report.pop("run_id", None)
    report.pop("run_started_at", None)
    for f in report.get("files", []):
        f["path"] = Path(f["path"]).name  # keep only basename
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
        for f in g.get("files", []):
            f["path"] = Path(f["path"]).name
    for f in report.get("files", []):
        f["path"] = Path(f["path"]).name
    for u in report.get("unclassified", []):
        u["path"] = Path(u["path"]).name
    return report


def test_golden_all_two_episodes(tmp_path: Path) -> None:
    """Golden for `finalpass all` — two-episode delivery, both groups pass."""
    import numpy as np
    import soundfile as sf

    SR = 48000
    SEED = 0xC0DE

    def _pink(n_samples, n_channels, seed):
        rng = np.random.default_rng(seed)
        white = rng.standard_normal((n_samples, n_channels))
        pink = np.zeros_like(white)
        pink[0] = 0.05 * white[0]
        for i in range(1, n_samples):
            pink[i] = 0.99 * pink[i - 1] + 0.05 * white[i]
        rms = float(np.sqrt(np.mean(pink ** 2)))
        if rms > 0:
            pink *= (10 ** (-23.0 / 20.0)) / rms
        return pink.astype(np.float64)

    folder = tmp_path / "delivery"
    folder.mkdir()
    for idx, name in enumerate([
        "SHOW_S01E03_PM_STEREO.wav",
        "SHOW_S01E03_DX_STEREO.wav",
        "SHOW_S01E04_PM_STEREO.wav",
        "SHOW_S01E04_DX_STEREO.wav",
    ]):
        sf.write(str(folder / name), _pink(10 * SR, 2, seed=SEED + idx), SR, subtype="PCM_24")

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
