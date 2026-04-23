from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from finalpass.audio_io import read_wav
from finalpass.cli import main
from finalpass.null_test import describe_audio as describe_null_audio
from finalpass.timecode import tc_to_sample_start
from tests.audio_cases import SEED, SR, exact_sum_components, shift_with_zeros, write_audio


def _write_exact_sum_case(
    root: Path,
    *,
    seconds: float = 10.0,
    base_seed: int = SEED,
    time_reference_samples: int | None = None,
) -> dict[str, Path]:
    data = exact_sum_components(seconds=seconds, base_seed=base_seed)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "pm": write_audio(root / "SHOW_S01E03_PM_STEREO.wav", data["pm"], time_reference_samples=time_reference_samples),
        "dx": write_audio(root / "SHOW_S01E03_DX_STEREO.wav", data["dx"], time_reference_samples=time_reference_samples),
        "mx": write_audio(root / "SHOW_S01E03_MX_STEREO.wav", data["mx"], time_reference_samples=time_reference_samples),
        "fx": write_audio(root / "SHOW_S01E03_FX_STEREO.wav", data["fx"], time_reference_samples=time_reference_samples),
    }


def test_null_exact_sum_passes(tmp_path: Path) -> None:
    files = _write_exact_sum_case(tmp_path / "case")
    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == 2
    assert data["command"] == "null"
    assert data["printmaster"]["source_kind"] == "interleaved"
    assert data["printmaster"]["source_paths"] == [str(files["pm"].resolve())]
    assert data["null_test"]["pass"] is True
    assert data["null_test"]["flags"] == []
    assert data["summary"]["overall_pass"] is True


def test_null_describe_audio_includes_interleaved_provenance(tmp_path: Path) -> None:
    data = exact_sum_components(seconds=1.0, base_seed=SEED + 50)
    path = write_audio(tmp_path / "SHOW_PM_STEREO.wav", data["pm"])

    described = describe_null_audio(read_wav(path))

    assert described.source_kind == "interleaved"
    assert described.source_paths == [str(path.resolve())]
    assert described.member_legs == []
    assert described.presentation_label is None


def test_null_injected_gross_error_fails_with_flagged_region(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = exact_sum_components(base_seed=SEED + 100)
    start = int(round(5.0 * SR))
    end = int(round(7.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    files = {
        "pm": write_audio(root / "SHOW_S01E04_PM_STEREO.wav", data["pm"]),
        "dx": write_audio(root / "SHOW_S01E04_DX_STEREO.wav", data["dx"]),
        "mx": write_audio(root / "SHOW_S01E04_MX_STEREO.wav", data["mx"]),
        "fx": write_audio(root / "SHOW_S01E04_FX_STEREO.wav", data["fx"]),
    }

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["null_test"]["pass"] is False
    assert payload["null_test"]["flags"]
    first = payload["null_test"]["flags"][0]
    assert first["start_sample"] <= start
    assert first["end_sample"] >= end


def test_null_flags_use_embedded_start_timecode(tmp_path: Path) -> None:
    time_reference_samples = tc_to_sample_start("01:00:00:00", SR, 23.976)
    files = _write_exact_sum_case(
        tmp_path / "case",
        base_seed=SEED + 150,
        time_reference_samples=time_reference_samples,
    )
    data = exact_sum_components(base_seed=SEED + 150)
    start = int(round(5.0 * SR))
    end = int(round(7.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    write_audio(files["pm"], data["pm"], time_reference_samples=time_reference_samples)

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    first = payload["null_test"]["flags"][0]
    assert first["start_tc"].startswith("01:00:04:")
    assert first["end_tc"].startswith("01:00:07:")


def test_null_ignores_defects_before_one_hour_when_embedded_start_is_known(tmp_path: Path) -> None:
    time_reference_samples = tc_to_sample_start("00:59:55:00", SR, 23.976)
    files = _write_exact_sum_case(
        tmp_path / "case",
        base_seed=SEED + 160,
        time_reference_samples=time_reference_samples,
    )
    data = exact_sum_components(base_seed=SEED + 160)
    start = int(round(2.0 * SR))
    end = int(round(4.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    write_audio(files["pm"], data["pm"], time_reference_samples=time_reference_samples)

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["null_test"]["pass"] is True
    assert payload["null_test"]["flags"] == []
    assert payload["null_test"]["summary"]["windows_total"] == 40
    assert payload["null_test"]["summary"]["windows_flagged"] == 0


def test_null_still_flags_defects_after_one_hour_when_embedded_start_is_known(tmp_path: Path) -> None:
    time_reference_samples = tc_to_sample_start("00:59:55:00", SR, 23.976)
    files = _write_exact_sum_case(
        tmp_path / "case",
        base_seed=SEED + 170,
        time_reference_samples=time_reference_samples,
    )
    data = exact_sum_components(base_seed=SEED + 170)
    start = int(round(6.0 * SR))
    end = int(round(8.0 * SR))
    data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]
    write_audio(files["pm"], data["pm"], time_reference_samples=time_reference_samples)

    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["null_test"]["pass"] is False
    assert payload["null_test"]["summary"]["windows_total"] == 40
    assert payload["null_test"]["flags"]
    assert payload["null_test"]["flags"][0]["start_tc"] >= "01:00:00:00"


def test_null_sample_rate_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = exact_sum_components(base_seed=SEED + 200)
    pm = write_audio(root / "PM.wav", data["pm"], sr=48000)
    dx = write_audio(root / "DX.wav", data["dx"], sr=44100)
    mx = write_audio(root / "MX.wav", data["mx"], sr=48000)
    fx = write_audio(root / "FX.wav", data["fx"], sr=48000)

    runner = CliRunner()
    result = runner.invoke(main, ["null", str(pm), str(dx), str(mx), str(fx)])
    assert result.exit_code == 2
    assert "sample rate" in result.output.lower()


def test_null_channel_count_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    stereo = exact_sum_components(base_seed=SEED + 300, n_channels=2)
    mono = exact_sum_components(base_seed=SEED + 301, n_channels=1)
    pm = write_audio(root / "PM.wav", stereo["pm"])
    dx = write_audio(root / "DX.wav", mono["dx"])
    mx = write_audio(root / "MX.wav", stereo["mx"])
    fx = write_audio(root / "FX.wav", stereo["fx"])

    runner = CliRunner()
    result = runner.invoke(main, ["null", str(pm), str(dx), str(mx), str(fx)])
    assert result.exit_code == 2
    assert "channel count" in result.output.lower()


def test_null_sample_count_mismatch_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    long = exact_sum_components(seconds=10.0, base_seed=SEED + 400)
    short = exact_sum_components(seconds=8.0, base_seed=SEED + 401)
    pm = write_audio(root / "PM.wav", long["pm"])
    dx = write_audio(root / "DX.wav", short["dx"])
    mx = write_audio(root / "MX.wav", long["mx"])
    fx = write_audio(root / "FX.wav", long["fx"])

    runner = CliRunner()
    result = runner.invoke(main, ["null", str(pm), str(dx), str(mx), str(fx)])
    assert result.exit_code == 2
    assert "sample count" in result.output.lower() or "samples" in result.output.lower()


def test_null_detected_global_offset_exits_two(tmp_path: Path) -> None:
    root = tmp_path / "case"
    data = exact_sum_components(base_seed=SEED + 500)
    shifted_dx = shift_with_zeros(data["dx"], 2400)
    pm = write_audio(root / "PM.wav", data["pm"])
    dx = write_audio(root / "DX.wav", shifted_dx)
    mx = write_audio(root / "MX.wav", data["mx"])
    fx = write_audio(root / "FX.wav", data["fx"])

    runner = CliRunner()
    result = runner.invoke(main, ["null", str(pm), str(dx), str(mx), str(fx)])
    assert result.exit_code == 2
    assert "offset" in result.output.lower() or "align" in result.output.lower()


def test_null_short_file_shorter_than_window_uses_single_window(tmp_path: Path) -> None:
    files = _write_exact_sum_case(tmp_path / "case", seconds=0.5, base_seed=SEED + 600)
    runner = CliRunner()
    result = runner.invoke(main, [
        "null",
        str(files["pm"]),
        str(files["dx"]),
        str(files["mx"]),
        str(files["fx"]),
        "--json-only",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["null_test"]["summary"]["windows_total"] == 1
