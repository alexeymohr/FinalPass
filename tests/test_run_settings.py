"""Settings-value tests: defaults stay sourced from the check modules."""

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
