"""Phase 8A channel-integrity tests.

Most cases pass an explicit window/hop so a 10-second fixture clears the
minimum qualifying-window count; the default 5000/2500 windowing is exercised
by the insufficient-content case, which is exactly what it guards.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import numpy as np

from finalpass.cli import main
from tests.audio_cases import (
    SEED,
    SPLIT_LEG_ORDERS,
    band_limited_lfe,
    build_clean_51,
    build_dead_leg_51,
    build_dual_mono,
    build_fake_51,
    build_film_order_51,
    build_head_tone_stereo,
    build_polarity_flip,
    write_audio,
    write_split_role,
)

# 10 s at 1000/500 gives 19 windows — comfortably past the 8-window minimum.
WINDOW_ARGS = ["--window-ms", "1000", "--hop-ms", "500"]


def _run(paths: list[Path], *extra: str) -> tuple[int, dict]:
    runner = CliRunner()
    result = runner.invoke(main, [
        "channels",
        *[str(path) for path in paths],
        *WINDOW_ARGS,
        *extra,
        "--json-only",
    ])
    assert result.exit_code in (0, 1), result.output
    return result.exit_code, json.loads(result.output)


def _kinds(asset: dict, *, include_skipped: bool = False) -> set[str]:
    return {
        finding["kind"]
        for finding in asset["findings"]
        if include_skipped or not finding["skipped"]
    }


def _findings(asset: dict, kind: str) -> list[dict]:
    return [finding for finding in asset["findings"] if finding["kind"] == kind]


# 1 — clean asset passes with nothing to report.
def test_clean_51_passes_with_no_findings(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_clean_51(seed=SEED + 1))

    exit_code, payload = _run([path])

    assert exit_code == 0
    asset = payload["assets"][0]
    assert asset["pass"] is True
    assert asset["findings"] == []
    assert payload["schema_version"] == 1
    assert payload["command"] == "channels"
    assert payload["summary"] == {
        "total_checks": 1, "passed": 1, "failed": 0, "skipped": 0, "overall_pass": True,
    }


# 2 — a dead full-range leg fails; a silent LFE alone never does.
def test_dead_surround_leg_fails_and_names_the_leg(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_MX_5.1.wav", build_dead_leg_51(leg="Ls", seed=SEED + 2))

    exit_code, payload = _run([path])

    assert exit_code == 1
    asset = payload["assets"][0]
    assert asset["pass"] is False
    silent = _findings(asset, "silent_leg")
    assert len(silent) == 1
    assert silent[0]["severity"] == "fail"
    assert silent[0]["channels"] == ["Ls"]


def test_silent_lfe_alone_is_informational_and_passes(tmp_path: Path) -> None:
    data = build_clean_51(seed=SEED + 3)
    data[:, SPLIT_LEG_ORDERS["5.1"].index("LFE")] = 0.0
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", data)

    exit_code, payload = _run([path])

    assert exit_code == 0
    asset = payload["assets"][0]
    assert asset["pass"] is True
    silent = _findings(asset, "silent_leg")
    assert len(silent) == 1
    assert silent[0]["severity"] == "info"
    assert silent[0]["channels"] == ["LFE"]


# 3 — dual-mono stereo: informational by default, escalatable, quantified.
def test_dual_mono_stereo_is_informational_by_default(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_LtRt.wav", build_dual_mono(seed=SEED + 4))

    exit_code, payload = _run([path])

    assert exit_code == 0
    asset = payload["assets"][0]
    assert asset["pass"] is True
    duplicates = _findings(asset, "duplicate_channels")
    assert len(duplicates) == 1
    assert duplicates[0]["severity"] == "info"
    assert duplicates[0]["measured"] >= 40.0


def test_fail_dual_mono_escalates_the_same_asset(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_LtRt.wav", build_dual_mono(seed=SEED + 5))

    exit_code, payload = _run([path], "--fail-dual-mono")

    assert exit_code == 1
    asset = payload["assets"][0]
    assert asset["pass"] is False
    assert asset["fail_dual_mono"] is True
    assert _findings(asset, "duplicate_channels")[0]["severity"] == "fail"


def test_level_scaled_copy_is_detected_and_the_offset_is_reported(tmp_path: Path) -> None:
    path = write_audio(
        tmp_path / "SHOW_S01E06_PM_LtRt.wav",
        build_dual_mono(seed=SEED + 6, right_gain=0.5),
    )

    _, payload = _run([path])

    duplicate = _findings(payload["assets"][0], "duplicate_channels")[0]
    assert duplicate["measured"] >= 40.0
    assert "6.0 dB" in duplicate["detail"]


def test_dithered_near_copy_still_nulls_past_the_default_threshold(tmp_path: Path) -> None:
    # Two channels a 24-bit dither apart are the same channel for QC purposes.
    path = write_audio(
        tmp_path / "SHOW_S01E06_PM_LtRt.wav",
        build_dual_mono(seed=SEED + 7, dither_lsb=2.0 ** -23),
    )

    _, payload = _run([path])

    assert _findings(payload["assets"][0], "duplicate_channels")[0]["measured"] >= 40.0


def test_decorrelated_stereo_sits_far_below_the_duplicate_threshold(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED + 8)
    data = rng.standard_normal((480000, 2)) * 0.1
    path = write_audio(tmp_path / "SHOW_S01E06_PM_LtRt.wav", data)

    exit_code, payload = _run([path])

    assert exit_code == 0
    assert payload["assets"][0]["findings"] == []


# 4 — fake surround fails.
def test_fake_51_fails_as_duplicated_channels(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_fake_51(seed=SEED + 9))

    exit_code, payload = _run([path])

    assert exit_code == 1
    asset = payload["assets"][0]
    assert asset["pass"] is False
    duplicates = _findings(asset, "duplicate_channels")
    assert duplicates
    assert all(finding["severity"] == "fail" for finding in duplicates)


# 5 — polarity, including the inverted-duplicate disambiguation.
def test_inverted_pair_reports_polarity_not_duplicate(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_LtRt.wav", build_polarity_flip(seed=SEED + 10))

    exit_code, payload = _run([path])

    assert exit_code == 1
    asset = payload["assets"][0]
    assert _kinds(asset) == {"polarity_inversion"}
    polarity = _findings(asset, "polarity_inversion")[0]
    assert polarity["severity"] == "fail"
    assert polarity["measured"] <= -0.95


def test_inverted_surround_pair_fails(tmp_path: Path) -> None:
    path = write_audio(
        tmp_path / "SHOW_S01E06_PM_5.1.wav",
        build_polarity_flip(layout="5.1", seed=SEED + 11),
    )

    exit_code, payload = _run([path])

    assert exit_code == 1
    polarity = _findings(payload["assets"][0], "polarity_inversion")
    assert polarity
    assert {"Ls", "Rs"} == set(polarity[0]["channels"])


# 6 / 7 — the LFE slot.
def test_film_order_interleave_fails_as_broadband_lfe(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_film_order_51(seed=SEED + 12))

    exit_code, payload = _run([path])

    assert exit_code == 1
    asset = payload["assets"][0]
    broadband = _findings(asset, "broadband_lfe")
    assert len(broadband) == 1
    assert broadband[0]["severity"] == "fail"
    assert broadband[0]["measured"] >= 0.25
    assert "film order" in broadband[0]["detail"]


def test_band_limited_lfe_produces_no_lfe_finding(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_clean_51(seed=SEED + 13))

    _, payload = _run([path])

    assert _findings(payload["assets"][0], "broadband_lfe") == []


# 8 — a 5.0 split family carries a synthesized LFE; its findings are suppressed.
def test_five_point_zero_family_suppresses_lfe_findings(tmp_path: Path) -> None:
    data = build_clean_51(seed=SEED + 14)
    five_oh = data[:, [0, 1, 2, 4, 5]]
    legs = write_split_role(tmp_path, "S01E06", "pm", "5.0", five_oh)

    exit_code, payload = _run([legs["L"]])

    assert exit_code == 0
    asset = payload["assets"][0]
    assert asset["channel_config_actual"] == "5.1"
    assert "LFE" not in asset["member_legs"]
    assert "lfe_synthesized_from_5_0_source" in asset["notes"]
    assert _findings(asset, "broadband_lfe") == []
    assert _findings(asset, "silent_leg") == []


# 9 — identical head tone must not manufacture a duplicate verdict.
def test_identical_head_tone_does_not_false_positive_as_duplicate(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_LtRt.wav", build_head_tone_stereo(seed=SEED + 15))

    exit_code, payload = _run([path])

    assert exit_code == 0
    assert _findings(payload["assets"][0], "duplicate_channels") == []


# 10 — too little gated content for a pairwise verdict is a skip, not a pass.
def test_short_file_skips_pairwise_kinds_with_default_windowing(tmp_path: Path) -> None:
    data = build_dead_leg_51(leg="Rs", seed=SEED + 16)
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", data)

    runner = CliRunner()
    result = runner.invoke(main, ["channels", str(path), "--json-only"])

    assert result.exit_code == 1, result.output
    asset = json.loads(result.output)["assets"][0]
    skipped = {
        finding["kind"]: finding["reason"]
        for finding in asset["findings"]
        if finding["skipped"]
    }
    assert skipped == {
        "duplicate_channels": "insufficient_active_content",
        "polarity_inversion": "insufficient_active_content",
    }
    # The full-span silence check still ran and still failed the asset.
    assert _findings(asset, "silent_leg")[0]["severity"] == "fail"


# 11 — mono is structurally not-applicable, never a crash.
def test_mono_asset_reports_not_applicable_and_passes(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED + 17)
    path = write_audio(tmp_path / "SHOW_S01E06_DX_Mono.wav", rng.standard_normal((480000, 1)) * 0.1)

    exit_code, payload = _run([path])

    assert exit_code == 0
    asset = payload["assets"][0]
    assert asset["pass"] is True
    reasons = {
        finding["kind"]: finding["reason"]
        for finding in asset["findings"]
        if finding["skipped"]
    }
    assert reasons == {
        "duplicate_channels": "not_applicable_mono",
        "polarity_inversion": "not_applicable_mono",
    }


# 12 — split seeds resolve, and findings name the split legs.
def test_split_seed_resolves_and_findings_name_legs(tmp_path: Path) -> None:
    data = build_clean_51(seed=SEED + 18)
    data[:, SPLIT_LEG_ORDERS["5.1"].index("C")] = 0.0
    legs = write_split_role(tmp_path, "S01E06", "pm", "5.1", data)

    exit_code, payload = _run([legs["L"]])

    assert exit_code == 1
    asset = payload["assets"][0]
    assert asset["source_kind"] == "split_mono"
    assert asset["member_legs"] == SPLIT_LEG_ORDERS["5.1"]
    assert _findings(asset, "silent_leg")[0]["channels"] == ["C"]


# 13 — unsupported layouts stay clean exit-2 tool errors.
def test_unsupported_channel_count_exits_two(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED + 19)
    path = write_audio(tmp_path / "SHOW_S01E06_PM_QUAD.wav", rng.standard_normal((48000, 4)) * 0.1)

    runner = CliRunner()
    result = runner.invoke(main, ["channels", str(path), "--json-only"])

    assert result.exit_code == 2, result.output


# Artifact contract: channels never writes an AAF, even when it fails.
def test_failing_channels_run_writes_no_aaf(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_fake_51(seed=SEED + 20))
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(main, ["channels", str(path), *WINDOW_ARGS, "--out", str(out_dir)])

    assert result.exit_code == 1, result.output
    assert list(out_dir.glob("*report*.json"))
    assert list(out_dir.glob("*report*.html"))
    assert list(out_dir.glob("*.aaf")) == []


def test_multiple_assets_each_contribute_one_check(tmp_path: Path) -> None:
    clean = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", build_clean_51(seed=SEED + 21))
    broken = write_audio(tmp_path / "SHOW_S01E06_MX_5.1.wav", build_dead_leg_51(seed=SEED + 22))

    exit_code, payload = _run([clean, broken])

    assert exit_code == 1
    assert payload["summary"]["total_checks"] == 2
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["failed"] == 1


def test_lfe_content_is_reported_when_the_lfe_is_genuinely_broadband(tmp_path: Path) -> None:
    # A real LFE leg plus a full-range leak into the same slot.
    data = build_clean_51(seed=SEED + 23)
    rng = np.random.default_rng(SEED + 24)
    n = data.shape[0]
    data[:, 3] = band_limited_lfe(n, SEED + 25) + rng.standard_normal(n) * 0.05
    path = write_audio(tmp_path / "SHOW_S01E06_PM_5.1.wav", data)

    exit_code, payload = _run([path])

    assert exit_code == 1
    assert _findings(payload["assets"][0], "broadband_lfe")[0]["severity"] == "fail"
