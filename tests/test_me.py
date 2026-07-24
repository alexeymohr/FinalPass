from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import numpy as np

from finalpass.audio_io import read_wav
from finalpass.cli import main
from finalpass.me_check import describe_audio as describe_me_audio
from finalpass.timecode import tc_to_sample_start, timecode_mode
from tests.audio_cases import SEED, SR, me_check_components, shift_with_zeros, write_audio


def _write_me_case(
    root: Path,
    *,
    seconds: float = 10.0,
    base_seed: int = SEED,
    bleed_region: tuple[float, float] | None = None,
    bleed_gain: float = 0.08,
    dx_quiet_region: tuple[float, float] | None = None,
    time_reference_samples: int | None = None,
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
        "dx": write_audio(root / "SHOW_S01E03_DX_STEREO.wav", data["dx"], time_reference_samples=time_reference_samples),
        "me": write_audio(root / "SHOW_S01E03_ME_STEREO.wav", data["me"], time_reference_samples=time_reference_samples),
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
    assert data["schema_version"] == 4
    assert data["command"] == "me"
    assert data["me_file"]["source_kind"] == "interleaved"
    assert data["me_file"]["source_paths"] == [str(files["me"].resolve())]
    assert data["me_check"]["pass"] is True
    assert data["me_check"]["analysis_window"]["mode"] == "file_start"
    assert data["me_check"]["flags"] == []
    assert data["summary"]["overall_pass"] is True


def test_me_describe_audio_includes_interleaved_provenance(tmp_path: Path) -> None:
    files = _write_me_case(tmp_path / "case", seconds=1.0, base_seed=SEED + 50)

    described = describe_me_audio(read_wav(files["me"]))

    assert described.source_kind == "interleaved"
    assert described.source_paths == [str(files["me"].resolve())]
    assert described.member_legs == []
    assert described.presentation_label is None


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


def test_me_flags_use_embedded_start_timecode(tmp_path: Path) -> None:
    files = _write_me_case(
        tmp_path / "case",
        base_seed=SEED + 110,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
        time_reference_samples=168648480,
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
    first = payload["me_check"]["flags"][0]
    assert first["start_tc"].startswith("00:58:35:")
    assert first["end_tc"].startswith("00:58:37:")


def test_me_sample_rate_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(base_seed=SEED + 200)
    me = write_audio(root / "ME.wav", data["me"], sr=48000)
    dx = write_audio(root / "DX.wav", data["dx"], sr=44100)

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "sample rate" in result.output.lower()


def test_me_mono_dx_against_stereo_me_analyzes_and_passes(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(base_seed=SEED + 300, n_channels=2)
    me = write_audio(root / "ME.wav", data["me"])
    dx = write_audio(root / "DX.wav", data["dx"][:, [0]])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx), "--json-only"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["me_file"]["channel_count"] == 2
    assert payload["dx_file"]["channel_count"] == 1
    assert payload["me_check"]["pass"] is True
    assert payload["me_check"]["flags"] == []


def test_me_mono_dx_bleed_into_stereo_me_flags(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(
        base_seed=SEED + 310,
        n_channels=2,
        bleed_region=(5.0, 7.0),
        bleed_gain=0.08,
    )
    me = write_audio(root / "ME.wav", data["me"])
    dx = write_audio(root / "DX.wav", data["dx"][:, [0]])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx), "--json-only"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["me_check"]["pass"] is False
    assert payload["me_check"]["flags"]
    first = payload["me_check"]["flags"][0]
    assert first["code"] == "ME"
    assert first["start_sample"] <= int(round(5.0 * SR))
    assert first["end_sample"] >= int(round(7.0 * SR))


def test_me_sample_count_mismatch_with_signal_tail_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    long = me_check_components(seconds=10.0, base_seed=SEED + 400)
    short = me_check_components(seconds=8.0, base_seed=SEED + 401)
    me = write_audio(root / "ME.wav", long["me"])
    dx = write_audio(root / "DX.wav", short["dx"])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx)])
    assert result.exit_code == 2
    assert "sample count" in result.output.lower() or "comparable range" in result.output.lower()


def test_me_sample_count_mismatch_with_mos_tail_is_cropped(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = me_check_components(seconds=8.0, base_seed=SEED + 410)
    silent_tail = np.zeros((2 * SR, data["me"].shape[1]), dtype=np.float64)
    me = write_audio(root / "ME.wav", np.vstack([data["me"], silent_tail]))
    dx = write_audio(root / "DX.wav", data["dx"])

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx), "--json-only"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["me_check"]["pass"] is True
    window = payload["me_check"]["analysis_window"]
    assert window["duration_seconds"] == 8.0
    assert window["inputs"][0]["ignored_tail_seconds"] == 2.0
    assert window["inputs"][1]["ignored_tail_seconds"] == 0.0


def test_me_ignores_head_before_shared_whole_hour(tmp_path: Path) -> None:
    root = tmp_path / "case"
    time_reference_samples = tc_to_sample_start("00:59:55:00", SR, timecode_mode(23.976))
    data = me_check_components(
        seconds=10.0,
        base_seed=SEED + 420,
        bleed_region=(1.0, 3.0),
        bleed_gain=0.10,
    )
    me = write_audio(root / "ME.wav", data["me"], time_reference_samples=time_reference_samples)
    dx = write_audio(root / "DX.wav", data["dx"], time_reference_samples=time_reference_samples)

    runner = CliRunner()
    result = runner.invoke(main, ["me", str(me), "--dx", str(dx), "--json-only"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["me_check"]["pass"] is True
    assert payload["me_check"]["flags"] == []
    window = payload["me_check"]["analysis_window"]
    assert window["mode"] == "whole_hour_time_reference"
    assert window["start_tc"] == "01:00:00:00"
    assert 5.0 <= window["inputs"][0]["ignored_head_seconds"] <= 5.01


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


def test_coherence_mean_handles_zero_power_segments_without_warning() -> None:
    import warnings

    from finalpass.me_check import _coherence_mean

    # A constant DX window detrends to exact zeros inside scipy's coherence,
    # which used to emit a RuntimeWarning and average NaN bins.
    dx_window = np.full(24000, 0.25, dtype=np.float64)
    me_window = np.random.default_rng(SEED).standard_normal(24000)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = _coherence_mean(
            dx_window,
            me_window,
            sample_rate=SR,
            band_low_hz=200.0,
            band_high_hz=4000.0,
        )

    assert value == 0.0


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
