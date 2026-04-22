from __future__ import annotations

__version__ = "0.1.0"

# Four schema versions because `loudness`, `null`, `me`, and `all` emit
# different shapes. SM-2 bumps the standalone schemas to v2 for split-mono
# provenance while SM-3 redesigns `all` around logical assets. See TODO.md.
LOUDNESS_SCHEMA_VERSION = 2
NULL_SCHEMA_VERSION = 2
ME_SCHEMA_VERSION = 2
ALL_SCHEMA_VERSION = 6
# Back-compat alias used by the Phase 1 loudness path. Do not bump.
SCHEMA_VERSION = LOUDNESS_SCHEMA_VERSION
