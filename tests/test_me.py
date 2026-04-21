from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.cli import main
from tests.audio_cases import SEED, SR, me_check_components, shift_with_zeros, write_audio


def _write_me_case(
    root: Path,
    *,
    seconds: float = 10.0,
    base_seed: int = SEED,
    bleed_region: tuple[float, float] | None = None,
    bleed_gain: float = 0.08,
    dx_quiet_region: tuple[float, float] | None = None,
) -> dict[str, Path]:
    data = me_check_components(
        seconds=seconds,
        base_seed=base_seed,
        bleed_region=bleed_region,
        bleed_gain=bleed_gain,
        dx_quiet_region=dx_quiet_region,
    )
    root.mkdir(parents=True, exist_ok=True)
    return {
        "dx": write_audio(root / "SHOW_S01E03_DX_STEREO.wav", data["dx"]),
        "me": write_audio(root / "SHOW_S01E03_ME_STEREO.wav", data["me"]),
    }


def test_me_independent_content_passes(tmp_path: Path) -> None:
    files = _write_me_case(tmp_path / "case")
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 1
    assert data["command"] == "me"
    assert data["me_check"]["pass"] is True
    assert data["me_check"]["flags"] == []
    assert data["summary"]["overall_pass"] is True


def test_me_injected_bleed_fails_with_flagged_region(tmp_path: Path) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 100,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["me_check"]["pass"] is False
    assert payload["me_check"]["flags"]
    first = payload["me_check"]["flags"][0]
    assert first["code"] == "ME"
    assert first["metric"] == "dialog_bleed_score"
    assert first["start_sample"] <= int(round(5.0 * SR))
    assert first["end_sample"] >= int(round(7.0 * SR))


def test_me_sample_rate_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(base_seed=SEED + 200)
    me = write_audio(root / "ME.wav", data["me"], sr=48000)
    dx = write_audio(root / "DX.wav", data["dx"], sr=44100)

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "sample rate" in result.output.lower()


def test_me_channel_count_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    stereo = me_check_components(base_seed=SEED + 300, n_channels=2)
    mono = me_check_components(base_seed=SEED + 301, n_channels=1)
    me = write_audio(root / "ME.wav", stereo["me"])
    dx = write_audio(root / "DX.wav", mono["dx"])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "channel count" in result.output.lower()


def test_me_sample_count_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    long = me_check_components(seconds=10.0, base_seed=SEED + 400)
    short = me_check_components(seconds=8.0, base_seed=SEED + 401)
    me = write_audio(root / "ME.wav", long["me"])
    dx = write_audio(root / "DX.wav", short["dx"])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "sample count" in result.output.lower() or "samples" in result.output.lower()


def test_me_detected_global_offset_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(base_seed=SEED + 500, bleed_region=(0.0, 10.0), bleed_gain=0.10)
    shifted_me = shift_with_zeros(data["me"], 2400)
    me = write_audio(root / "ME.wav", shifted_me)
    dx = write_audio(root / "DX.wav", data["dx"])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "offset" in result.output.lower() or "align" in result.output.lower()


def test_me_dx_gate_behavior_gates_out_quiet_dx_windows(tmp_path: Path) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 600,
        dx_quiet_region=(4.0, 6.0),
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["me_check"]["pass"] is True
    assert payload["me_check"]["summary"]["windows_gated_out"] > 0
    assert payload["me_check"]["flags"] == []


def test_me_short_file_shorter_than_window_uses_single_window(tmp_path: Path) -> None:
    files = _write_me_case(tmp_path / "case", seconds=0.3, base_seed=SEED + 700)
    runner = CliRunner()
    result = runner.invoke(main, [
        "me",
        str(files["me"]),
        "--dx", str(files["dx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["me_check"]["summary"]["windows_total"] == 1
