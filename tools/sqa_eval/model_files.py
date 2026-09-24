"""Fail-closed local model provisioning.

There is deliberately no download path in this module. A missing or altered
file is an error, not a reason to reach for the network: the client-audio phase
must be provably offline, and a lazy fetch is exactly how that guarantee is
lost. Every file is pinned by SHA-256 and verified before use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ModelFileRejected(RuntimeError):
    """A required model file is missing, altered, or not allowlisted."""


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_manifest(root: Path, manifest: dict[str, str]) -> dict[str, Path]:
    """Resolve every manifest entry under `root`, or fail closed.

    `manifest` maps a path relative to `root` to its expected SHA-256. Returns
    the resolved paths only when every single one is present and matches.
    """
    root = Path(root)
    if not root.is_dir():
        raise ModelFileRejected(f"model directory does not exist: {root}")

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    altered: list[str] = []
    for rel, expected in sorted(manifest.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        actual = sha256_file(path)
        if actual != expected:
            altered.append(f"{rel} (expected {expected[:12]}…, got {actual[:12]}…)")
            continue
        resolved[rel] = path

    if missing:
        raise ModelFileRejected(
            f"missing model files under {root}: {', '.join(missing)}. "
            "This build performs no downloads; provision them locally."
        )
    if altered:
        raise ModelFileRejected(
            f"model files do not match their pinned hashes: {'; '.join(altered)}"
        )
    return resolved


def load_manifest(path: Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    files = data["files"] if isinstance(data, dict) and "files" in data else data
    if not isinstance(files, dict) or not files:
        raise ModelFileRejected(f"{path}: manifest carries no file hashes")
    for rel, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ModelFileRejected(f"{path}: {rel!r} has no usable sha256")
    return files


# Pickle globals the Paderborn training checkpoint needs beyond torch's own
# defaults. Audited individually against the checkpoint's actual opcode stream:
# the file contains exactly one numpy scalar, of dtype 'f8', whose buffer is a
# `_codecs.encode(..., 'latin1')` byte string. None of these can execute
# arbitrary code, and no object dtype appears anywhere in the pickle.
#
# This list exists to document what an audited load may allow. The evaluation
# does not use it at scan time: the audited load happens once, and the tensors
# are re-serialised into an inference-only artifact that loads under a plain
# `weights_only=True` with no allowlist at all.
AUDITED_CHECKPOINT_GLOBALS = (
    "numpy.core.multiarray.scalar",
    "numpy.dtype",
    "numpy.dtypes.Float64DType",
    "_codecs.encode",
)

FORBIDDEN_LOAD_KWARGS = ("weights_only=False", "trust_remote_code")
