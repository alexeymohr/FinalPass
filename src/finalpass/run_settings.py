"""Runtime settings for the shared execution seam.

These bundle the analysis tunables that previously threaded through every
runner signature as loose parameters. They are runtime-only values: the report
schemas persist their own flat copies of the effective knobs, and nothing here
is written to disk. ``fps`` / ``drop_frame`` deliberately stay explicit runner
parameters — they are the timecode mode, not analysis knobs.

Phase 8C extends :class:`AllSettings` with the channels/downmix tunables.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .me_check import METunables
from .null_test import NullTunables


class AllSettings(BaseModel):
    """Settings for a folder-level ``all`` run: scope plus per-check tunables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_unclassified: bool = False
    null: NullTunables = Field(default_factory=NullTunables)
    me: METunables = Field(default_factory=METunables)
