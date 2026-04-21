"""Generate a tiny Phase 4-ready delivery folder for manual FinalPass runs.

Run from the repo root::

    uv run python examples/_generate.py

Outputs go to ``examples/delivery_two_episodes/`` which is gitignored — the
script is deterministic (fixed seeds), so the audio regenerates identically on
every run. E03 passes loudness, null, and M&E. E04 keeps the "bad" loudness
story via a hot printmaster, injects a deterministic null defect, and injects
deterministic dialogue bleed into the M&E stem.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.audio_cases import build_two_episodes


def main() -> None:
    root = Path(__file__).parent / "delivery_two_episodes"
    build_two_episodes(
        root,
        hot_e04_pm=True,
        e04_null_defect=True,
        e04_me_bleed_defect=True,
    )
    print(f"Wrote Phase 4 example delivery to {root}")


if __name__ == "__main__":
    main()
