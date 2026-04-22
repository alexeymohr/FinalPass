"""Generate a deterministic example delivery for manual FinalPass runs.

Run from the repo root::

    uv run python examples/_generate.py

Outputs go to ``examples/delivery_two_episodes/`` which is gitignored — the
script is deterministic (fixed seeds), so the audio regenerates identically on
every run. E03 passes loudness, null, and M&E. E04 keeps the "bad" loudness
story via a hot printmaster, injects a deterministic null defect, and injects
deterministic dialogue bleed into the M&E stem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.audio_cases import build_two_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "delivery_two_episodes",
        help="Directory to populate with the deterministic example delivery.",
    )
    args = parser.parse_args()

    root = args.output
    build_two_episodes(
        root,
        hot_e04_pm=True,
        e04_null_defect=True,
        e04_me_bleed_defect=True,
    )
    print(f"Wrote deterministic example delivery to {root}")


if __name__ == "__main__":
    main()
