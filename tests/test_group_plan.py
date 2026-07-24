"""Group-plan queries: one answer for the runner and the wizard alike."""

from __future__ import annotations

from pathlib import Path

from finalpass.all_assets import ClassifiedLogicalAsset
from finalpass.assets import LogicalAsset
from finalpass.group_plan import (
    available_auto_null_plans,
    preferred_auto_null_plan,
    same_layout_companions,
    supports_loudness_job,
    supports_me_job,
    supports_null_job,
)


def _asset(role: str, layout: str, name: str) -> ClassifiedLogicalAsset:
    logical_asset = LogicalAsset(
        asset_id=f"interleaved::{name}",
        role_hint=None,
        group_id="S01E01",
        group_hint=None,
        source_kind="interleaved",
        channel_config_actual=layout,
        channel_config_hint=None,
        presentation_label=None,
        canonical_path=Path(f"/delivery/{name}.wav"),
        source_paths=[Path(f"/delivery/{name}.wav")],
        member_legs=[],
        sample_rate=48000,
        sample_count=48000,
        time_reference_samples=None,
        bit_depth=24,
    )
    return ClassifiedLogicalAsset(
        logical_asset=logical_asset,
        role=role,
        channel_config_hint=None,
    )


def test_full_stem_set_offers_both_strategies_preferring_the_full_one() -> None:
    by_role = {
        "dx": _asset("dx", "stereo", "dx"),
        "mx": _asset("mx", "stereo", "mx"),
        "fx": _asset("fx", "stereo", "fx"),
        "me": _asset("me", "stereo", "me"),
    }

    plans = available_auto_null_plans(by_role)

    assert [plan.strategy for plan in plans] == ["dx_mx_fx", "dx_me"]
    assert preferred_auto_null_plan(by_role).strategy == "dx_mx_fx"
    assert [asset.role for asset in plans[0].stems] == ["dx", "mx", "fx"]
    assert [asset.role for asset in plans[1].stems] == ["dx", "me"]


def test_partial_stem_sets_offer_only_what_they_support() -> None:
    dx_me_only = {"dx": _asset("dx", "stereo", "dx"), "me": _asset("me", "stereo", "me")}
    assert [plan.strategy for plan in available_auto_null_plans(dx_me_only)] == ["dx_me"]

    dx_only = {"dx": _asset("dx", "stereo", "dx")}
    assert available_auto_null_plans(dx_only) == []
    assert preferred_auto_null_plan(dx_only) is None


def test_same_layout_companions_excludes_the_anchor_and_other_layouts() -> None:
    pm = _asset("pm", "5.1", "pm51")
    group = [
        pm,
        _asset("dx", "5.1", "dx51"),
        _asset("dx", "stereo", "dx20"),
        _asset("pm", "stereo", "pm20"),
    ]

    companions = same_layout_companions(group, anchor=pm)

    assert [asset.logical_asset.asset_id for asset in companions] == ["interleaved::dx51"]


def test_job_availability_matches_what_each_job_needs() -> None:
    pm_only = [_asset("pm", "stereo", "pm")]
    assert supports_loudness_job(pm_only) is True
    # A printmaster with nothing to null against cannot run the null job, even
    # though an interactive run may pick stems by hand.
    assert supports_null_job(pm_only) is False
    assert supports_me_job(pm_only) is False

    with_companion = [*pm_only, _asset("dx", "stereo", "dx")]
    assert supports_null_job(with_companion) is True
    assert supports_me_job(with_companion) is False

    with_me = [*with_companion, _asset("me", "stereo", "me")]
    assert supports_me_job(with_me) is True

    # A companion in another layout is not something the null job can use.
    mismatched = [_asset("pm", "5.1", "pm51"), _asset("dx", "stereo", "dx20")]
    assert supports_null_job(mismatched) is False
