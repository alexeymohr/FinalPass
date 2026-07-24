from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from finalpass.assets import (
    AssetResolutionError,
    discover_logical_assets,
    discover_logical_assets_in_folder,
    resolve_logical_asset,
)
from tests.asset_helpers import mono_signal, write_interleaved, write_split_family
from tests.audio_cases import SR, write_audio


def _errors_by_type(result) -> dict[str, list]:
    out: dict[str, list] = {}
    for error in result.errors:
        out.setdefault(error.type, []).append(error)
    return out


def test_valid_stereo_split_family_discovers_one_asset(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo")

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.source_kind == "split_mono"
    assert asset.role_hint == "pm"
    assert asset.presentation_label == "LtRt"
    assert asset.channel_config_actual == "stereo"
    assert asset.member_legs == ["L", "R"]


def test_interleaved_discovery_regression_safety(tmp_path: Path) -> None:
    mono = write_interleaved(tmp_path / "SHOW_PM_MONO.wav", 1)
    stereo = write_interleaved(tmp_path / "SHOW_PM_STEREO.wav", 2)
    surround = write_interleaved(tmp_path / "SHOW_PM_51.wav", 6)

    result = discover_logical_assets([mono, stereo, surround])

    assert result.errors == []
    assert sorted((asset.source_kind, asset.channel_config_actual) for asset in result.assets) == [
        ("interleaved", "5.1"),
        ("interleaved", "mono"),
        ("interleaved", "stereo"),
    ]


def test_seed_path_resolution_from_split_member_resolves_full_family(tmp_path: Path) -> None:
    members = write_split_family(tmp_path, "Comp 5.1", "5.1")

    asset = resolve_logical_asset(members["Rs"])

    assert asset.source_kind == "split_mono"
    assert asset.role_hint == "pm"
    assert asset.channel_config_actual == "5.1"
    assert asset.member_legs == ["L", "R", "C", "LFE", "Ls", "Rs"]
    assert asset.canonical_path == members["L"]


def test_50_split_family_discovers_as_51_with_missing_lfe_provenance(tmp_path: Path) -> None:
    members = write_split_family(tmp_path, "DX 5.0", "5.0")

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.channel_config_actual == "5.1"
    assert asset.channel_config_hint == "5.0"
    assert asset.presentation_label == "5.0"
    assert asset.member_legs == ["L", "R", "C", "Ls", "Rs"]
    assert asset.source_paths == [members[leg] for leg in ["L", "R", "C", "Ls", "Rs"]]


def test_51_split_family_missing_only_lfe_discovers_as_50_source(tmp_path: Path) -> None:
    members = write_split_family(tmp_path, "DX 5.1", "5.1")
    members["LFE"].unlink()

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.channel_config_actual == "5.1"
    assert asset.presentation_label == "5.0"
    assert asset.member_legs == ["L", "R", "C", "Ls", "Rs"]


def test_missing_leg_produces_clean_discovery_error(tmp_path: Path) -> None:
    members = write_split_family(tmp_path, "Comp 5.1", "5.1")
    members["Rs"].unlink()

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "MissingLeg" in errors
    assert "Rs" in errors["MissingLeg"][0].message
    assert result.assets == []


def test_strict_unlabeled_single_split_member_produces_missing_leg(tmp_path: Path) -> None:
    member = write_audio(tmp_path / "Comp.L.wav", np.zeros((16, 1), dtype=np.float64), sr=SR)

    result = discover_logical_assets([member], strict_explicit_split_members=True)

    errors = _errors_by_type(result)
    assert "MissingLeg" in errors
    assert "explicit split-mono member" in errors["MissingLeg"][0].message
    assert result.assets == []


def test_duplicate_leg_produces_clean_discovery_error(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 5.1", "5.1")
    write_audio(tmp_path / "Comp-5.1-L.wav", np.full((16, 1), 9.0, dtype=np.float64), sr=SR)

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "DuplicateLeg" in errors
    assert "L" in errors["DuplicateLeg"][0].message


def test_mixed_sample_rates_produce_clean_discovery_error(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo", frames=16)
    write_audio(tmp_path / "Comp LtRt.R.wav", np.full((16, 1), 2.0, dtype=np.float64), sr=44100)

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "MixedSampleRate" in errors


def test_mixed_sample_counts_produce_clean_discovery_error(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo", frames=16)
    write_audio(tmp_path / "Comp LtRt.R.wav", np.full((3, 1), 2.0, dtype=np.float64), sr=SR)

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "MixedSampleCount" in errors


def test_non_mono_member_inside_claimed_split_family_errors(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo")
    write_interleaved(tmp_path / "Comp-LtRt-R.wav", 2)

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "NonMonoMember" in errors


def test_incompatible_presentation_hint_vs_layout_errors(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 5.1", "stereo")

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "MissingLeg" in errors
    assert "5.1" in errors["MissingLeg"][0].message


def test_distinct_presentations_do_not_merge_into_one_family(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 5.1", "5.1")
    write_split_family(tmp_path, "Comp LtRt", "stereo")

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 2
    assert sorted(asset.channel_config_actual for asset in result.assets) == ["5.1", "stereo"]


def test_mixed_explicit_and_missing_presentations_error_as_ambiguous_family_key(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp 5.1", "5.1")
    write_audio(tmp_path / "Comp.L.wav", np.full((16, 1), 1.0, dtype=np.float64), sr=SR)
    write_audio(tmp_path / "Comp.R.wav", np.full((16, 1), 2.0, dtype=np.float64), sr=SR)

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "AmbiguousFamilyKey" in errors


def test_mixed_subtypes_in_family_are_hard_error(tmp_path: Path) -> None:
    write_split_family(tmp_path, "Comp LtRt", "stereo", subtype="PCM_24")
    write_audio(tmp_path / "Comp LtRt.R.wav", np.full((16, 1), 2.0, dtype=np.float64), sr=SR, subtype="PCM_16")

    result = discover_logical_assets_in_folder(tmp_path)

    errors = _errors_by_type(result)
    assert "SubtypeMismatch" in errors


_PT_51_BOUNCE_NAMES = {
    "L": "11 SHOW 5.1 Mix, Left Front.wav",
    "R": "12 SHOW 5.1 Mix, Right Front.wav",
    "C": "13 SHOW 5.1 Mix, Center.wav",
    "LFE": "14 SHOW 5.1 Mix,_LFE.wav",
    "Ls": "15 SHOW 5.1 Mix, Left Surround.wav",
    "Rs": "16 SHOW 5.1 Mix, Right Surround.wav",
}


def _write_pt_stereo_pair(root: Path) -> dict[str, Path]:
    return {
        "L": write_audio(root / "1 SHOW Stereo Mix Left.wav", mono_signal(0.1), sr=SR),
        "R": write_audio(root / "2 SHOW Stereo Mix Right.wav", mono_signal(0.2), sr=SR),
    }


def _write_pt_51_family(root: Path) -> dict[str, Path]:
    return {
        leg: write_audio(root / name, mono_signal(0.1 * (index + 1)), sr=SR)
        for index, (leg, name) in enumerate(_PT_51_BOUNCE_NAMES.items())
    }


def test_pt_spelled_out_stereo_bounce_pair_assembles(tmp_path: Path) -> None:
    members = _write_pt_stereo_pair(tmp_path)

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.source_kind == "split_mono"
    assert asset.role_hint == "pm"
    assert asset.channel_config_actual == "stereo"
    assert asset.presentation_label == "Stereo"
    assert asset.member_legs == ["L", "R"]
    assert asset.source_paths == [members["L"], members["R"]]
    assert asset.group_id == "SHOW"


def test_pt_comma_and_spelled_out_51_bounce_family_assembles(tmp_path: Path) -> None:
    members = _write_pt_51_family(tmp_path)

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 1
    asset = result.assets[0]
    assert asset.source_kind == "split_mono"
    assert asset.role_hint == "pm"
    assert asset.channel_config_actual == "5.1"
    assert asset.presentation_label == "5.1"
    assert asset.member_legs == ["L", "R", "C", "LFE", "Ls", "Rs"]
    assert asset.source_paths == [members[leg] for leg in ["L", "R", "C", "LFE", "Ls", "Rs"]]
    assert asset.group_id == "SHOW"


def test_pt_stereo_and_51_bounces_partition_by_presentation(tmp_path: Path) -> None:
    _write_pt_stereo_pair(tmp_path)
    _write_pt_51_family(tmp_path)

    result = discover_logical_assets_in_folder(tmp_path)

    assert result.errors == []
    assert len(result.assets) == 2
    assert sorted(asset.channel_config_actual for asset in result.assets) == ["5.1", "stereo"]
    assert {asset.role_hint for asset in result.assets} == {"pm"}
    assert {asset.group_id for asset in result.assets} == {"SHOW"}


def test_pt_seed_path_resolution_from_spelled_out_member(tmp_path: Path) -> None:
    members = _write_pt_51_family(tmp_path)

    asset = resolve_logical_asset(members["Ls"])

    assert asset.source_kind == "split_mono"
    assert asset.channel_config_actual == "5.1"
    assert asset.member_legs == ["L", "R", "C", "LFE", "Ls", "Rs"]
    assert asset.canonical_path == members["L"]


def test_resolve_seed_path_fails_cleanly_for_incomplete_family(tmp_path: Path) -> None:
    members = write_split_family(tmp_path, "Comp 5.1", "5.1")
    members["Rs"].unlink()

    with pytest.raises(AssetResolutionError) as exc:
        resolve_logical_asset(members["L"])

    assert exc.value.error.type == "MissingLeg"
