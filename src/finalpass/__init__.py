from __future__ import annotations

__version__ = "0.1.0"

# Four schema versions because `loudness`, `null`, `me`, and `all` emit
# different shapes. Phase 8T bumps all four: every envelope gains
# `drop_frame: bool` beside `fps`.
LOUDNESS_SCHEMA_VERSION = 4
NULL_SCHEMA_VERSION = 4
ME_SCHEMA_VERSION = 4
ALL_SCHEMA_VERSION = 9
# Phase 8A `channels` is a new report shape, born drop-frame aware.
CHANNELS_SCHEMA_VERSION = 1
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
