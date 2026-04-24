"""Generate the example delivery and run a representative smoke flow.

Run from the repo root::

    uv run python examples/_smoke.py

This regenerates the deterministic two-episode delivery, then runs:

- a failing standalone ``me`` command (timed flags + AAF)
- the integrated ``all`` command (JSON + HTML + conditional AAF)

Outputs land under ``examples/out/`` by default.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DELIVERY_ROOT = Path(__file__).parent / "delivery_two_episodes"
DEFAULT_OUT_ROOT = Path(__file__).parent / "out"


def _run(argv: list[str], *, expected_exit: int) -> None:
    result = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(f"$ {' '.join(argv)}")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != expected_exit:
        raise SystemExit(
            f"Command exited {result.returncode}; expected {expected_exit}: {' '.join(argv)}"
        )


def _single_artifact(out_dir: Path, pattern: str) -> Path:
    matches = sorted(out_dir.glob(pattern))
    if not matches:
        raise SystemExit(f"Missing smoke artifact matching {out_dir / pattern}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise SystemExit(
            f"Expected one smoke artifact matching {out_dir / pattern}; found {rendered}"
        )
    return matches[0]


def _assert_artifacts(out_dir: Path, *, expect_aaf: bool) -> None:
    _single_artifact(out_dir, "*report.json")
    _single_artifact(out_dir, "*report.html")
    aaf_matches = sorted(out_dir.glob("*markers.aaf"))
    if expect_aaf and len(aaf_matches) != 1:
        raise SystemExit(
            f"Expected one smoke artifact matching {out_dir / '*markers.aaf'}"
        )
    if not expect_aaf and aaf_matches:
        rendered = ", ".join(str(path) for path in aaf_matches)
        raise SystemExit(f"Unexpected smoke artifact: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=DEFAULT_DELIVERY_ROOT,
        help="Directory for the deterministic example delivery.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Directory for disposable smoke-run artifacts.",
    )
    args = parser.parse_args()

    delivery_root = args.delivery_root
    out_root = args.out_root

    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(Path("examples") / "_generate.py"),
            "--output",
            str(delivery_root),
        ],
        expected_exit=0,
    )

    me_out = out_root / "me_failing"
    _run(
        [
            sys.executable,
            "-m",
            "finalpass.cli",
            "me",
            str(delivery_root / "SHOW_S01E04_ME_STEREO.wav"),
            "--dx",
            str(delivery_root / "SHOW_S01E04_DX_STEREO.wav"),
            "--out",
            str(me_out),
        ],
        expected_exit=1,
    )
    _assert_artifacts(me_out, expect_aaf=True)

    all_out = out_root / "all"
    _run(
        [
            sys.executable,
            "-m",
            "finalpass.cli",
            "all",
            str(delivery_root),
            "--spec",
            "ebu_r128",
            "--out",
            str(all_out),
        ],
        expected_exit=1,
    )
    _assert_artifacts(all_out, expect_aaf=True)

    print()
    print(f"Smoke artifacts written under {out_root}")


if __name__ == "__main__":
    main()
