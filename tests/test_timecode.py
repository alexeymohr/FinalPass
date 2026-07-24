"""Timecode math tests — Phase 8T drop-frame counting per SMPTE ST 12-1.

Landmark values (minute skips, ten-minute exemptions, frames-per-hour totals)
come straight from the ST 12-1 counting rules: frame numbers 00 and 01
(00-03 for the 60-family) are skipped at the start of every minute except
minutes divisible by ten.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from finalpass.timecode import (
    frames_to_tc,
    samples_to_tc,
    tc_to_frames,
    tc_to_sample_start,
    timecode_mode,
)

SR = 48000

DF_FRAMES_PER_HOUR_2997 = 107_892
DF_FRAMES_PER_HOUR_5994 = 215_784


# --- ST 12-1 landmark conversions -----------------------------------------


def test_df_minute_boundary_skips_two_frames_at_2997() -> None:
    before = tc_to_frames("00:00:59;29", timecode_mode(29.97, drop_frame=True))
    assert frames_to_tc(before + 1, timecode_mode(29.97, drop_frame=True)) == "00:01:00;02"
    assert tc_to_frames("00:01:00;02", timecode_mode(29.97, drop_frame=True)) == before + 1


def test_df_ten_minute_exemption_keeps_frame_zero_at_2997() -> None:
    before = tc_to_frames("00:09:59;29", timecode_mode(29.97, drop_frame=True))
    assert frames_to_tc(before + 1, timecode_mode(29.97, drop_frame=True)) == "00:10:00;00"


def test_df_minute_boundary_skips_four_frames_at_5994() -> None:
    before = tc_to_frames("00:00:59;59", timecode_mode(59.94, drop_frame=True))
    assert frames_to_tc(before + 1, timecode_mode(59.94, drop_frame=True)) == "00:01:00;04"


def test_df_frames_per_hour_totals() -> None:
    assert tc_to_frames("01:00:00;00", timecode_mode(29.97, drop_frame=True)) == DF_FRAMES_PER_HOUR_2997
    assert tc_to_frames("01:00:00;00", timecode_mode(59.94, drop_frame=True)) == DF_FRAMES_PER_HOUR_5994
    assert frames_to_tc(DF_FRAMES_PER_HOUR_2997, timecode_mode(29.97, drop_frame=True)) == "01:00:00;00"
    assert frames_to_tc(DF_FRAMES_PER_HOUR_5994, timecode_mode(59.94, drop_frame=True)) == "01:00:00;00"


def test_df_round_trip_across_minute_and_ten_minute_boundaries() -> None:
    for fps in (29.97, 59.94):
        mode = timecode_mode(fps, drop_frame=True)
        two_hours = 2 * tc_to_frames("01:00:00;00", mode)
        for frames in range(0, two_hours, 373):
            label = frames_to_tc(frames, mode)
            assert tc_to_frames(label, mode) == frames


def test_df_nonexistent_labels_are_rejected() -> None:
    for label in ("00:01:00;00", "00:01:00;01", "00:59:00;01"):
        with pytest.raises(ValueError):
            tc_to_frames(label, timecode_mode(29.97, drop_frame=True))
    for label in ("00:01:00;00", "00:01:00;03"):
        with pytest.raises(ValueError):
            tc_to_frames(label, timecode_mode(59.94, drop_frame=True))
    # Ten-minute boundaries keep every frame number.
    assert tc_to_frames("00:10:00;00", timecode_mode(29.97, drop_frame=True)) >= 0
    assert tc_to_frames("00:10:00;00", timecode_mode(59.94, drop_frame=True)) >= 0


# --- samples <-> DF timecode -----------------------------------------------


def test_samples_to_tc_df_maps_wall_clock_hour_to_hour_label() -> None:
    assert samples_to_tc(3600 * SR, SR, timecode_mode(29.97, drop_frame=True)) == "01:00:00;00"
    assert samples_to_tc(3600 * SR, SR, timecode_mode(59.94, drop_frame=True)) == "01:00:00;00"
    # Non-drop keeps its documented ~3.6s/hour lag behind wall clock.
    assert samples_to_tc(3600 * SR, SR, timecode_mode(29.97)) == "00:59:56:12"


def test_tc_to_sample_start_df_hour_lands_at_wall_clock_hour() -> None:
    df_hour = tc_to_sample_start("01:00:00;00", SR, timecode_mode(29.97, drop_frame=True))
    nd_hour = tc_to_sample_start("01:00:00:00", SR, timecode_mode(29.97))
    assert abs(df_hour / SR - 3600.0) < 0.01
    assert abs(nd_hour / SR - 3603.6) < 0.01


def test_df_formatting_uses_semicolon_and_nd_keeps_colons() -> None:
    assert frames_to_tc(0, timecode_mode(29.97, drop_frame=True)) == "00:00:00;00"
    assert frames_to_tc(0, timecode_mode(29.97)) == "00:00:00:00"


# --- validation -------------------------------------------------------------


def test_drop_frame_rejected_for_non_1001_family_rates() -> None:
    for fps in (23.976, 24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 119.88, 120.0):
        with pytest.raises(ValueError):
            timecode_mode(fps, drop_frame=True)


def test_drop_frame_accepted_for_2997_and_5994() -> None:
    for fps, nominal in ((29.97, 30), (59.94, 60)):
        info = timecode_mode(fps, drop_frame=True)
        assert info.drop_frame is True
        assert info.nominal_fps == nominal
        assert info.edit_rate.denominator == 1001


def test_timecode_mode_is_the_only_way_in_and_is_immutable() -> None:
    # Every conversion takes a validated mode, so an impossible combination
    # cannot be expressed at a call site at all — it fails at construction.
    mode = timecode_mode(29.97, drop_frame=True)
    assert mode.fps == 29.97
    with pytest.raises(Exception):
        mode.drop_frame = False  # type: ignore[misc]
    with pytest.raises(ValueError):
        timecode_mode(24.0, drop_frame=True)
    with pytest.raises(ValueError):
        timecode_mode(0.0)


# --- frame-rate table -------------------------------------------------------


def test_frame_rate_table_includes_11988_and_120() -> None:
    info_11988 = timecode_mode(119.88)
    assert info_11988.edit_rate == Fraction(120000, 1001)
    assert info_11988.nominal_fps == 120
    assert info_11988.drop_frame is False

    info_120 = timecode_mode(120.0)
    assert info_120.edit_rate == Fraction(120, 1)
    assert info_120.nominal_fps == 120


def test_non_drop_behavior_unchanged_for_existing_rates() -> None:
    assert frames_to_tc(107_892, timecode_mode(29.97)) == "00:59:56:12"
    assert tc_to_frames("01:00:00:00", timecode_mode(29.97)) == 108_000
    assert samples_to_tc(0, SR, timecode_mode(23.976), start_time_reference_samples=0) == "00:00:00:00"
