"""Runtime-value tests: settings defaults and metric-display formatting."""

from __future__ import annotations

import pydantic
import pytest

from finalpass.me_check import (
    DEFAULT_ME_BAND_HIGH_HZ,
    DEFAULT_ME_BAND_LOW_HZ,
    DEFAULT_ME_COHERENCE_THRESHOLD,
    DEFAULT_ME_CORR_THRESHOLD,
    DEFAULT_ME_DX_GATE_DBFS,
    DEFAULT_ME_HOP_MS,
    DEFAULT_ME_ME_FLOOR_DBFS,
    DEFAULT_ME_WINDOW_MS,
    METunables,
)
from finalpass.null_test import (
    DEFAULT_NULL_HOP_MS,
    DEFAULT_NULL_THRESHOLD_DBFS,
    DEFAULT_NULL_WINDOW_MS,
    NullTunables,
)
from finalpass.presentation import format_metric_value, format_target_limit, metric_value_header
from finalpass.run_settings import AllSettings


def test_default_settings_match_check_module_constants() -> None:
    settings = AllSettings()
    assert settings.include_unclassified is False
    assert settings.null == NullTunables(
        window_ms=DEFAULT_NULL_WINDOW_MS,
        hop_ms=DEFAULT_NULL_HOP_MS,
        threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
    )
    assert settings.me == METunables(
        window_ms=DEFAULT_ME_WINDOW_MS,
        hop_ms=DEFAULT_ME_HOP_MS,
        band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
        band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
        corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
        coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
        dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
        me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
    )


def test_settings_values_are_frozen_and_reject_unknown_knobs() -> None:
    tunables = NullTunables()
    with pytest.raises(pydantic.ValidationError):
        tunables.window_ms = 2000.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        NullTunables(window_msec=2000.0)  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError):
        METunables(corr=0.5)  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError):
        AllSettings(nulls=NullTunables())  # type: ignore[call-arg]


def test_metric_display_is_one_truth_for_every_flag_metric() -> None:
    assert format_metric_value(-24.03, metric="residual_rms_dbfs") == "-24.0 dBFS"
    assert format_metric_value(0.857, metric="dialog_bleed_score") == "0.86"
    assert format_metric_value(-0.94, metric="true_peak_dbtp") == "-0.9 dBTP"
    assert format_metric_value(None, metric="residual_rms_dbfs") == "—"
    # An unmapped metric still renders rather than raising.
    assert format_metric_value(-12.0, metric="not_a_metric") == "-12.0 dBFS"

    assert metric_value_header("residual_rms_dbfs") == "peak residual"
    assert metric_value_header("dialog_bleed_score") == "bleed score"
    assert metric_value_header("true_peak_dbtp") == "peak true peak"


def test_format_target_limit_covers_target_limit_and_missing() -> None:
    class _Check:
        def __init__(self, target=None, tolerance=None, limit=None):
            self.target = target
            self.tolerance = tolerance
            self.limit = limit

    assert format_target_limit(_Check(target=-23.0, tolerance=2.0)) == "-23.0 ±2.0"
    assert format_target_limit(_Check(limit=-1.0)) == "≤ -1.0"
    assert format_target_limit(_Check()) == "—"
