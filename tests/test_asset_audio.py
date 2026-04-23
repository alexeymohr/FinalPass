from __future__ import annotations

from pathlib import Path

import numpy as np

from finalpass.assets import discover_logical_assets_in_folder, read_logical_asset
from tests.asset_helpers import write_interleaved, write_split_family


def _discover_one(root: Path):
    result = discover_logical_assets_in_folder(root)
    assert result.errors == []
    assert len(result.assets) == 1
    return result.assets[0]


def test_read_logical_asset_assembles_stereo_split_in_lr_order(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo", values=[0.1, 0.2])

    asset = _discover_one(tmp_path)
    audio = read_logical_asset(asset)

    assert audio.channel_count == 2
    assert audio.channel_config_actual == "stereo"
    np.testing.assert_allclose(audio.samples[:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(audio.samples[:, 1], 0.2, atol=1e-6)


def test_read_logical_asset_assembles_51_split_in_smpte_order(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 5.1", "5.1", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    asset = _discover_one(tmp_path)
    audio = read_logical_asset(asset)

    assert audio.channel_count == 6
    assert audio.channel_config_actual == "5.1"
    for index, expected in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        np.testing.assert_allclose(audio.samples[:, index], float(expected), atol=1e-6)


def test_read_logical_asset_pads_50_split_lfe_as_silence(tmp_path: Path) -> None:
    write_split_family(tmp_path, "DX 5.0", "5.0", values=[0.1, 0.2, 0.3, 0.5, 0.6])

    asset = _discover_one(tmp_path)
    audio = read_logical_asset(asset)

    assert asset.member_legs == ["L", "R", "C", "Ls", "Rs"]
    assert audio.channel_count == 6
    assert audio.channel_config_actual == "5.1"
    for index, expected in enumerate([0.1, 0.2, 0.3, 0.0, 0.5, 0.6]):
        np.testing.assert_allclose(audio.samples[:, index], float(expected), atol=1e-6)


def test_read_logical_asset_assembles_71_split_in_smpte_order(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 7.1", "7.1", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    asset = _discover_one(tmp_path)
    audio = read_logical_asset(asset)

    assert audio.channel_count == 8
    assert audio.channel_config_actual == "7.1"
    for index, expected in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        np.testing.assert_allclose(audio.samples[:, index], float(expected), atol=1e-6)


def test_read_logical_asset_preserves_interleaved_behavior(tmp_path: Path) -> None:
    write_interleaved(tmp_path / "SHOW_PM_STEREO.wav", 2)

    asset = _discover_one(tmp_path)
    audio = read_logical_asset(asset)

    assert audio.channel_count == 2
    assert audio.source_kind == "interleaved"
    np.testing.assert_allclose(audio.samples[:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(audio.samples[:, 1], 0.2, atol=1e-6)
