from __future__ import annotations

__version__ = "0.1.0"

# Two schema versions because the `loudness` and `all` JSON reports have
# different shapes. The `loudness` subcommand stays at v1 forever — any future
# folder-level features add to v2+ without retrofitting v1. See TODO.md.
LOUDNESS_SCHEMA_VERSION = 1
ALL_SCHEMA_VERSION = 3
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
