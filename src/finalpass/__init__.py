from __future__ import annotations

__version__ = "0.1.0"

# Four schema versions because `loudness`, `null`, `me`, and `all` emit
# different shapes. The `loudness` subcommand stays at v1 forever — future
# folder-level features add to the other schemas without retrofitting v1. See
# TODO.md.
LOUDNESS_SCHEMA_VERSION = 1
NULL_SCHEMA_VERSION = 1
ME_SCHEMA_VERSION = 1
ALL_SCHEMA_VERSION = 5
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
