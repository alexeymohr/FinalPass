"""Runtime settings for the shared execution seam.

These bundle the analysis tunables that previously threaded through every
runner signature as loose parameters. They are runtime-only values: the report
schemas persist their own flat copies of the effective knobs, and nothing here
is written to disk. ``fps`` / ``drop_frame`` deliberately stay explicit runner
parameters — they are the timecode mode, not analysis knobs.

Phase 8C added the channels/downmix tunables to :class:`AllSettings`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .channel_check import ChannelsTunables
from .downmix_check import DownmixTunables
from .me_check import METunables
from .null_test import NullTunables


class AllSettings(BaseModel):
    """Settings for a folder-level ``all`` run: scope plus per-check tunables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    include_unclassified: bool = False
    # Channels runs by default; it costs one full read per classified asset,
    # which is the only reason an opt-out exists.
    skip_channels: bool = False
    null: NullTunables = Field(default_factory=NullTunables)
    me: METunables = Field(default_factory=METunables)
    channels: ChannelsTunables = Field(default_factory=ChannelsTunables)
    downmix: DownmixTunables = Field(default_factory=DownmixTunables)
