from __future__ import annotations

__version__ = "0.1.0"

# Four schema versions because `loudness`, `null`, `me`, and `all` emit
# different shapes. Current null/M&E schemas persist comparison-window
# provenance, and `all` bumps because it embeds those result shapes.
LOUDNESS_SCHEMA_VERSION = 3
NULL_SCHEMA_VERSION = 3
ME_SCHEMA_VERSION = 3
ALL_SCHEMA_VERSION = 8
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
