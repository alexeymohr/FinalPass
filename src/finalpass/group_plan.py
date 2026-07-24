"""What a group's assets can support, and which assets each check would use.

Both the folder runner and the wizard need to answer the same questions about
a group: which checks its assets support, and — for auto-null — which stems a
strategy would select. Keeping those answers here means the wizard offers
exactly the jobs the runner can actually perform, instead of re-deriving the
rules from the same asset lists.

The runner asks for the single preferred plan; the wizard lists every plan so
the user can pick. Neither owns the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .all_assets import ClassifiedLogicalAsset
from .models import FileRole

NullStemStrategy = Literal["dx_mx_fx", "dx_me"]

# Auto-null strategies in preference order: full stem set first, then the
# dialogue-versus-M&E fallback. The persisted `stem_strategy` values in
# models.AutoNullTestResult are exactly these keys.
_AUTO_NULL_STRATEGIES: tuple[tuple[NullStemStrategy, tuple[FileRole, ...]], ...] = (
    ("dx_mx_fx", ("dx", "mx", "fx")),
    ("dx_me", ("dx", "me")),
)


@dataclass(frozen=True)
class AutoNullPlan:
    """One runnable auto-null strategy plus the stems it would use."""

    strategy: NullStemStrategy
    roles: list[FileRole]
    stems: list[ClassifiedLogicalAsset]


def same_layout_companions(
    assets: list[ClassifiedLogicalAsset],
    *,
    anchor: ClassifiedLogicalAsset,
) -> list[ClassifiedLogicalAsset]:
    """Assets sharing the anchor's channel layout, excluding the anchor itself."""
    return [
        asset for asset in assets
        if asset.logical_asset.channel_config_actual == anchor.logical_asset.channel_config_actual
        and asset.logical_asset.asset_id != anchor.logical_asset.asset_id
    ]


def available_auto_null_plans(
    assets_by_role: dict[FileRole, ClassifiedLogicalAsset],
) -> list[AutoNullPlan]:
    """Every auto-null strategy this role set supports, preferred first."""
    plans: list[AutoNullPlan] = []
    for strategy, roles in _AUTO_NULL_STRATEGIES:
        if all(assets_by_role.get(role) is not None for role in roles):
            plans.append(AutoNullPlan(
                strategy=strategy,
                roles=list(roles),
                stems=[assets_by_role[role] for role in roles],
            ))
    return plans


def preferred_auto_null_plan(
    assets_by_role: dict[FileRole, ClassifiedLogicalAsset],
) -> AutoNullPlan | None:
    """The strategy an unattended run would choose, or None if no set qualifies."""
    plans = available_auto_null_plans(assets_by_role)
    return plans[0] if plans else None


def supports_null_job(group_assets: list[ClassifiedLogicalAsset]) -> bool:
    """True when some printmaster has same-layout companions to null against.

    Deliberately broader than :func:`preferred_auto_null_plan`: an interactive
    run may pick stems by hand, so a group with any same-layout companion can
    run the job even when no auto strategy qualifies.
    """
    for pm_asset in group_assets:
        if pm_asset.role != "pm":
            continue
        if same_layout_companions(group_assets, anchor=pm_asset):
            return True
    return False


def supports_me_job(group_assets: list[ClassifiedLogicalAsset]) -> bool:
    """True when the group carries both an M&E and a DX asset."""
    has_me = any(asset.role == "me" for asset in group_assets)
    has_dx = any(asset.role == "dx" for asset in group_assets)
    return has_me and has_dx


def supports_loudness_job(group_assets: list[ClassifiedLogicalAsset]) -> bool:
    """True when the group carries a printmaster to measure."""
    return any(asset.role == "pm" for asset in group_assets)
