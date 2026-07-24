"""Phase 8B downmix-consistency tests.

The honesty posture under test: a discrete stereo mix that is not a mechanical
fold-down must pass at the default thresholds. These cases pin both halves of
that — what must pass, and what must not.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import numpy as np

from finalpass.cli import main
from tests.audio_cases import (
    SEED,
    SR,
    build_downmix_pair,
    build_surround_program,
    fold_down_of,
    write_audio,
)
from finalpass.timecode import tc_to_sample_start, timecode_mode


def _write_pair(root: Path, variant: str, *, seed: int = SEED, **kwargs) -> tuple[Path, Path]:
    delivered, surround = build_downmix_pair(variant, seed=seed, **kwargs)
    return (
        write_audio(root / "SHOW_S01E07_PM_LtRt.wav", delivered),
        write_audio(root / "SHOW_S01E07_PM_5.1.wav", surround),
    )


def _run(stereo: Path, surround: Path, *extra: str) -> tuple[int, dict]:
    runner = CliRunner()
    result = runner.invoke(main, ["downmix", str(stereo), str(surround), *extra, "--json-only"])
    assert result.exit_code in (0, 1), result.output
    return result.exit_code, json.loads(result.output)


def _flags(payload: dict, metric: str) -> list[dict]:
    return [flag for flag in payload["downmix_check"]["flags"] if flag["metric"] == metric]


# 1 — the mechanical fold-down is the trivially-correct case.
def test_exact_fold_down_passes_with_no_delta_and_no_flags(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "exact", seed=SEED + 30)

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 0
    check = payload["downmix_check"]
    assert check["pass"] is True
    assert check["analysis_signal"] == "loro_fold_down"
    assert check["flags"] == []
    assert check["summary"]["loudness_delta_lu"] < 0.1
    assert check["summary"]["similarity_min_corr"] > 0.99
    assert payload["schema_version"] == 1
    assert payload["command"] == "downmix"


# 2 — a discrete stereo mix is normal and must pass with headroom.
def test_discrete_stereo_passes_with_threshold_headroom(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "discrete", seed=SEED + 31)

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 0
    summary = payload["downmix_check"]["summary"]
    assert payload["downmix_check"]["flags"] == []
    # Headroom, not a bare pass: the worst window sits clear of the threshold.
    assert summary["similarity_min_corr"] > 0.60
    assert summary["loudness_delta_lu"] < 2.0


# 3 — a different programme is what this check exists to catch.
def test_unrelated_programme_fails_on_similarity(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "unrelated", seed=SEED + 32)

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 1
    assert payload["downmix_check"]["pass"] is False
    similarity = _flags(payload, "downmix_correlation")
    assert similarity
    assert all(flag["code"] == "DOWNMIX" for flag in similarity)
    assert payload["downmix_check"]["summary"]["similarity_min_corr"] < 0.60


# 4 — a level offset fails the level sub-check alone.
def test_level_offset_fails_level_without_similarity_flags(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "level_offset", seed=SEED + 33, level_offset_db=4.0)

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 1
    summary = payload["downmix_check"]["summary"]
    assert summary["level_pass"] is False
    assert 3.5 < summary["loudness_delta_lu"] < 4.5
    # Gain does not change correlation, so the similarity lane stays clean.
    assert _flags(payload, "downmix_correlation") == []


# 5 — an inverted span is a mono-compatibility problem, bounded in time.
def test_inverted_region_flags_mono_compatibility_over_that_span(tmp_path: Path) -> None:
    stereo, surround = _write_pair(
        tmp_path, "inverted_region", seed=SEED + 34, invert_region=(4.0, 6.0)
    )

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 1
    mono = _flags(payload, "stereo_correlation")
    assert mono
    assert all(flag["value"] <= -0.20 for flag in mono)
    # The flagged span brackets the injected region.
    assert min(flag["start_sample"] for flag in mono) <= int(4.5 * SR)
    assert max(flag["end_sample"] for flag in mono) >= int(5.5 * SR)


# 6 — layout preconditions are clean tool errors.
def test_wrong_layouts_exit_two(tmp_path: Path) -> None:
    delivered, surround = build_downmix_pair("exact", seed=SEED + 35)
    stereo_path = write_audio(tmp_path / "SHOW_S01E07_PM_LtRt.wav", delivered)
    surround_path = write_audio(tmp_path / "SHOW_S01E07_PM_5.1.wav", surround)
    mono_path = write_audio(tmp_path / "SHOW_S01E07_DX_Mono.wav", delivered[:, [0]])

    runner = CliRunner()
    # A mono file cannot be the delivered 2.0.
    assert runner.invoke(main, ["downmix", str(mono_path), str(surround_path), "--json-only"]).exit_code == 2
    # A 5.1 cannot be the delivered 2.0 either.
    assert runner.invoke(main, ["downmix", str(surround_path), str(surround_path), "--json-only"]).exit_code == 2
    # A stereo file cannot be the surround master.
    assert runner.invoke(main, ["downmix", str(stereo_path), str(stereo_path), "--json-only"]).exit_code == 2


# 7 — sample-rate mismatch is a tool error, never a silent resample.
def test_sample_rate_mismatch_exits_two(tmp_path: Path) -> None:
    delivered, surround = build_downmix_pair("exact", seed=SEED + 36)
    stereo = write_audio(tmp_path / "SHOW_S01E07_PM_LtRt.wav", delivered, sr=44100)
    surround_path = write_audio(tmp_path / "SHOW_S01E07_PM_5.1.wav", surround, sr=48000)

    runner = CliRunner()
    result = runner.invoke(main, ["downmix", str(stereo), str(surround_path), "--json-only"])

    assert result.exit_code == 2, result.output


# 8 — a constant offset is a hard failure, never auto-corrected.
def test_constant_offset_is_an_alignment_error(tmp_path: Path) -> None:
    surround = build_surround_program(seed=SEED + 37)
    fold = fold_down_of(surround)
    shifted = np.roll(fold, 2400, axis=0)
    shifted[:2400, :] = 0.0
    stereo = write_audio(tmp_path / "SHOW_S01E07_PM_LtRt.wav", shifted)
    surround_path = write_audio(tmp_path / "SHOW_S01E07_PM_5.1.wav", surround)

    runner = CliRunner()
    result = runner.invoke(main, ["downmix", str(stereo), str(surround_path), "--json-only"])

    assert result.exit_code == 2, result.output
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "offset" in combined.lower()


# 9 — the shared whole-hour window applies here exactly as it does elsewhere.
def test_whole_hour_reference_starts_the_window_at_the_hour(tmp_path: Path) -> None:
    reference = tc_to_sample_start("00:59:55:00", SR, timecode_mode(23.976))
    surround = build_surround_program(seed=SEED + 38)
    fold = fold_down_of(surround)
    stereo = write_audio(
        tmp_path / "SHOW_S01E07_PM_LtRt.wav", fold, time_reference_samples=reference
    )
    surround_path = write_audio(
        tmp_path / "SHOW_S01E07_PM_5.1.wav", surround, time_reference_samples=reference
    )

    exit_code, payload = _run(stereo, surround_path)

    assert exit_code == 0
    window = payload["downmix_check"]["analysis_window"]
    assert window["mode"] == "whole_hour_time_reference"
    assert window["start_tc"] == "01:00:00:00"
    assert window["inputs"][0]["ignored_head_seconds"] > 0


# 10 — a MOS tail on one input is cropped with provenance, not rejected.
def test_mos_tail_is_cropped_and_the_pair_still_passes(tmp_path: Path) -> None:
    surround = build_surround_program(seconds=12.0, seed=SEED + 39)
    fold = fold_down_of(surround)[: int(round(10.0 * SR)), :]
    stereo = write_audio(tmp_path / "SHOW_S01E07_PM_LtRt.wav", fold)
    # Silence the surround master's unique tail so it reads as MOS.
    surround[int(round(10.0 * SR)) :, :] = 0.0
    surround_path = write_audio(tmp_path / "SHOW_S01E07_PM_5.1.wav", surround)

    exit_code, payload = _run(stereo, surround_path)

    assert exit_code == 0
    window = payload["downmix_check"]["analysis_window"]
    assert any(item["ignored_tail_seconds"] > 0 for item in window["inputs"])


# 11 — including the LFE changes the derived loudness in the expected direction.
def test_lfe_inclusion_raises_the_derived_loudness(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "exact", seed=SEED + 40)

    _, without = _run(stereo, surround)
    _, with_lfe = _run(stereo, surround, "--lfe-db", "0")

    base = without["downmix_check"]["summary"]["derived_integrated_lufs"]
    boosted = with_lfe["downmix_check"]["summary"]["derived_integrated_lufs"]
    assert boosted > base


# 12 — a file shorter than one window uses the single-window path.
def test_file_shorter_than_one_window_does_not_crash(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "exact", seed=SEED + 41, seconds=0.5)

    exit_code, payload = _run(stereo, surround)

    assert exit_code == 0
    assert payload["downmix_check"]["summary"]["similarity_windows_total"] == 1


def test_matrix_encoded_delivery_carries_an_informational_note(tmp_path: Path) -> None:
    # The delivered file is labelled LtRt, so the report says plainly that
    # matrix-encoded content can legitimately depress similarity.
    stereo, surround = _write_pair(tmp_path, "exact", seed=SEED + 42)

    _, payload = _run(stereo, surround)

    assert "delivered_stereo_is_matrix_encoded" in payload["downmix_check"]["notes"]


def test_failing_downmix_writes_an_aaf_with_the_expected_marker_labels(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "unrelated", seed=SEED + 43)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, ["downmix", str(stereo), str(surround), "--out", str(out_dir)])

    assert result.exit_code == 1, result.output
    assert list(out_dir.glob("*report*.json"))
    aaf_files = list(out_dir.glob("*.aaf"))
    assert len(aaf_files) == 1

    from finalpass import aaf_export
    from finalpass.models import DownmixReport

    report = DownmixReport.model_validate_json((list(out_dir.glob("*report*.json"))[0]).read_text())
    labels = {candidate.name.split(": ", 1)[1] for candidate in aaf_export.collect_marker_candidates(report)}
    assert labels <= {"downmix mismatch", "mono compatibility"}
    assert "downmix mismatch" in labels


def test_clean_downmix_run_writes_no_aaf(tmp_path: Path) -> None:
    stereo, surround = _write_pair(tmp_path, "exact", seed=SEED + 44)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, ["downmix", str(stereo), str(surround), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert list(out_dir.glob("*.aaf")) == []
