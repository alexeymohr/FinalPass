"""Generate a deterministic example delivery for manual FinalPass runs.

Run from the repo root::

    uv run python examples/_generate.py

Outputs go to ``examples/delivery_two_episodes/`` and
``examples/delivery_split_assets/`` which are gitignored. The script is
deterministic (fixed seeds), so the audio regenerates identically on every
run. The interleaved delivery keeps the older two-episode story; the split
delivery covers a passing 5.1 family set, a mixed-presentation group, and one
explicitly broken incomplete family for discovery-error verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.audio_cases import SEED, build_split_group, build_two_episodes


def _build_split_examples(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    build_split_group(
        root,
        "S01E03",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 2000,
    )
    build_split_group(
        root,
        "S01E04",
        layout="5.1",
        write_roles=("pm", "dx", "mx", "fx"),
        base_seed=SEED + 2010,
    )
    build_split_group(
        root,
        "S01E04",
        layout="stereo",
        write_roles=("pm", "dx"),
        base_seed=SEED + 2011,
    )
    broken = build_split_group(
        root,
        "S01E05",
        layout="5.1",
        write_roles=("pm",),
        base_seed=SEED + 2020,
    )["pm"]
    for leg in ("C", "LFE", "Ls", "Rs"):
        broken[leg].unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "delivery_two_episodes",
        help="Directory to populate with the deterministic interleaved example delivery.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path(__file__).parent / "delivery_split_assets",
        help="Directory to populate with the deterministic split-mono example delivery.",
    )
    args = parser.parse_args()

    build_two_episodes(
        args.output,
        hot_e04_pm=True,
        e04_null_defect=True,
        e04_me_bleed_defect=True,
    )
    _build_split_examples(args.split_output)
    print(f"Wrote interleaved example delivery to {args.output}")
    print(f"Wrote split-mono example delivery to {args.split_output}")


if __name__ == "__main__":
    main()
