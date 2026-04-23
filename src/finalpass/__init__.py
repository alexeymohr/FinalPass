from __future__ import annotations

__version__ = "0.1.0"

# Four schema versions because `loudness`, `null`, `me`, and `all` emit
# different shapes. v3 adds persisted timed true-peak-over flags to the
# loudness file reports, and `all` bumps in lockstep because it reuses the same
# file-report shape.
LOUDNESS_SCHEMA_VERSION = 3
NULL_SCHEMA_VERSION = 2
ME_SCHEMA_VERSION = 2
ALL_SCHEMA_VERSION = 7
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
