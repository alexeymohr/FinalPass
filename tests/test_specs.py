from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finalpass.errors import SpecError
from finalpass.specs import Spec, list_bundled, list_bundled_names, load_spec


EXPECTED_BUNDLED = {"atsc_a85", "ebu_r128", "netflix_stereo", "netflix_51", "streaming_-14"}


def test_list_bundled_names_contains_expected() -> None:
    assert EXPECTED_BUNDLED <= set(list_bundled_names())


def test_each_bundled_spec_loads() -> None:
    for spec in list_bundled():
        assert spec.name.strip()
        assert spec.display_name.strip()
        assert spec.integrated_lufs.tolerance >= 0.0
        assert spec.true_peak_max_dbtp <= 0.0
        assert spec.channel_config in {"mono", "stereo", "5.1", "7.1"}


def test_ebu_r128_values() -> None:
    spec, source, _ = load_spec("ebu_r128")
    assert source == "bundled"
    assert spec.integrated_lufs.target == -23.0
    assert spec.integrated_lufs.tolerance == 0.5
    assert spec.true_peak_max_dbtp == -1.0
    assert spec.lra_max == 18.0
    assert spec.channel_config == "stereo"
    assert spec.dialog_lufs is None


def test_netflix_51_is_dialog_anchored() -> None:
    spec, _, _ = load_spec("netflix_51")
    assert spec.channel_config == "5.1"
    assert spec.dialog_lufs is not None
    assert spec.dialog_lufs.target == -27.0


def test_atsc_a85_values() -> None:
    spec, source, _ = load_spec("atsc_a85")
    assert source == "bundled"
    assert spec.integrated_lufs.target == -24.0
    assert spec.integrated_lufs.tolerance == 2.0
    assert spec.true_peak_max_dbtp == -2.0
    assert spec.lra_max == 18.0
    assert spec.channel_config == "stereo"
    assert spec.dialog_lufs is None


def test_explicit_path_wins(tmp_path: Path) -> None:
    path = tmp_path / "my_spec.yaml"
    path.write_text(yaml.safe_dump({
        "name": "my_spec",
        "display_name": "custom",
        "integrated_lufs": {"target": -20.0, "tolerance": 1.0},
        "true_peak_max_dbtp": -1.0,
        "channel_config": "stereo",
    }))
    spec, source, resolved = load_spec(path)
    assert source == "explicit"
    assert spec.name == "my_spec"
    assert resolved == path.resolve()


def test_missing_spec_raises_spec_error() -> None:
    with pytest.raises(SpecError):
        load_spec("definitely_not_a_spec")


def test_negative_tolerance_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "name": "bad",
        "display_name": "bad",
        "integrated_lufs": {"target": -23.0, "tolerance": -1.0},
        "true_peak_max_dbtp": -1.0,
        "channel_config": "stereo",
    }))
    with pytest.raises(SpecError):
        load_spec(path)


def test_positive_true_peak_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "name": "bad",
        "display_name": "bad",
        "integrated_lufs": {"target": -23.0, "tolerance": 0.5},
        "true_peak_max_dbtp": 1.0,
        "channel_config": "stereo",
    }))
    with pytest.raises(SpecError):
        load_spec(path)


def test_name_must_match_filename_stem(tmp_path: Path) -> None:
    path = tmp_path / "named_one_thing.yaml"
    path.write_text(yaml.safe_dump({
        "name": "but_called_another",
        "display_name": "x",
        "integrated_lufs": {"target": -23.0, "tolerance": 0.5},
        "true_peak_max_dbtp": -1.0,
        "channel_config": "stereo",
    }))
    with pytest.raises(SpecError):
        load_spec(path)
