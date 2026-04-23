from __future__ import annotations

from pathlib import Path

from finalpass.artifact_naming import derive_program_name_from_paths


def test_derives_show_and_tv_episode_from_stem_names(tmp_path: Path) -> None:
    paths = [
        tmp_path / "Mystery Show_S01E03_PM_STEREO.wav",
        tmp_path / "Mystery Show_S01E03_DX_STEREO.wav",
        tmp_path / "Mystery Show_S01E03_MX_STEREO.wav",
        tmp_path / "Mystery Show_S01E03_FX_STEREO.wav",
    ]

    name = derive_program_name_from_paths(paths)

    assert name is not None
    assert name.slug == "mystery-show-s01e03"
    assert name.episodes == ("S01E03",)


def test_derives_multi_episode_show_name_with_episode_range(tmp_path: Path) -> None:
    paths = [
        tmp_path / "Mystery Show_S01E03_PM_STEREO.wav",
        tmp_path / "Mystery Show_S01E04_PM_STEREO.wav",
    ]

    name = derive_program_name_from_paths(paths)

    assert name is not None
    assert name.slug == "mystery-show-s01e03-s01e04"
    assert name.episodes == ("S01E03", "S01E04")


def test_derives_show_from_role_tokens_without_episode(tmp_path: Path) -> None:
    paths = [
        tmp_path / "Mystery Show_PM_STEREO.wav",
        tmp_path / "Mystery Show_DX_STEREO.wav",
    ]

    name = derive_program_name_from_paths(paths)

    assert name is not None
    assert name.slug == "mystery-show"
    assert name.episodes == ()


def test_derives_show_code_from_role_specific_split_stems(tmp_path: Path) -> None:
    paths = [
        tmp_path / "NLA1120_262_51_PM_2398_48_20200408.L.wav",
        tmp_path / "NLA1120_262_51_PM_2398_48_20200408.R.wav",
        tmp_path / "NLA1120_262_51_DXSM_2398_48_20200408.L.wav",
        tmp_path / "NLA1120_262_51_MXSM_2398_48_20200408.L.wav",
        tmp_path / "NLA1120_262_51_FXSM_2398_48_20200408.L.wav",
    ]

    name = derive_program_name_from_paths(paths)

    assert name is not None
    assert name.slug == "nla-1120"
    assert name.episodes == ()


def test_derives_show_code_from_delivery_folder_when_stems_are_generic(tmp_path: Path) -> None:
    folder = tmp_path / "NLA1120_262_Stems_Deliverables" / "Audio Files" / "FinalPass Prep" / "5.1 Printmaster"
    paths = [
        folder / "PM.L.wav",
        folder / "PM.R.wav",
    ]

    name = derive_program_name_from_paths(paths)

    assert name is not None
    assert name.slug == "nla-1120"
    assert name.episodes == ()


def test_does_not_name_generic_single_loudness_fixture(tmp_path: Path) -> None:
    name = derive_program_name_from_paths([tmp_path / "pink_stereo_10s.wav"])

    assert name is None
