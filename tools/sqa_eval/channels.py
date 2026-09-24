"""Stereo policy for single-stream speech-quality scoring.

The model takes one mono 16 kHz stream. Deciding how to get there from a
delivered file is a correctness question, not a convenience one, so the rule is
explicit and it fails closed:

* mono -> used directly;
* stereo whose legs null to at least ``DUAL_MONO_NULL_DB`` -> genuinely
  dual-mono, so the left leg IS the programme and is used;
* anything else -> refused. FinalPass has no documented downmix policy for
  scoring a discrete-stereo narration file as one stream, and silently taking
  channel 1 (as the upstream toolkit does) would analyse half the delivery.

``DUAL_MONO_NULL_DB`` is FinalPass's own duplicate-channel threshold
(`channel_check.DEFAULT_CHANNELS_DUPLICATE_NULL_DB`), reused rather than
reinvented so the two agree about what "dual mono" means.
"""
from __future__ import annotations

import numpy as np

DUAL_MONO_NULL_DB = 40.0
MAX_NULL_DEPTH_DB = 200.0


class ChannelPolicyRefused(ValueError):
    """The file's channel layout has no safe single-stream reduction."""


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _dbfs(value: float) -> float:
    return -np.inf if value <= 0.0 else 20.0 * float(np.log10(value))


def null_depth_db(a: np.ndarray, b: np.ndarray) -> float:
    """How far the L-R residual sits below the content, capped like FinalPass."""
    reference = _dbfs(_rms(a))
    if not np.isfinite(reference):
        return MAX_NULL_DEPTH_DB
    residual = _dbfs(_rms(a - b))
    if not np.isfinite(residual):
        return MAX_NULL_DEPTH_DB
    return float(min(reference - residual, MAX_NULL_DEPTH_DB))


def describe_layout(block: np.ndarray) -> dict:
    """Numeric facts about a whole-file (or probe) block. No decision yet."""
    if block.ndim == 1:
        return {"channels": 1, "null_depth_db": None, "dual_mono": True}
    channels = int(block.shape[1])
    if channels == 1:
        return {"channels": 1, "null_depth_db": None, "dual_mono": True}
    if channels != 2:
        return {"channels": channels, "null_depth_db": None, "dual_mono": False}
    depth = null_depth_db(block[:, 0].astype(np.float64), block[:, 1].astype(np.float64))
    return {"channels": 2, "null_depth_db": round(depth, 3),
            "dual_mono": depth >= DUAL_MONO_NULL_DB}


def to_mono(block: np.ndarray, layout: dict, source_id: str = "?") -> np.ndarray:
    """Reduce a decoded block to one stream under the decided policy."""
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    if layout.get("channels") == 2 and layout.get("dual_mono"):
        return block[:, 0]
    if layout.get("channels") == 2:
        raise ChannelPolicyRefused(
            f"{source_id}: discrete stereo (L/R null depth "
            f"{layout.get('null_depth_db')} dB < {DUAL_MONO_NULL_DB} dB). FinalPass has "
            "no documented single-stream downmix policy for scoring this; refusing "
            "rather than analysing one leg."
        )
    raise ChannelPolicyRefused(
        f"{source_id}: {layout.get('channels')} channels is out of scope for this evaluation"
    )
