"""Frozen evaluation parameters.

Everything in this module is fixed BEFORE any client audio is opened. Changing
a value here after seeing client results invalidates the evaluation, so the
manifest written by `calibrate.py` records these values alongside the
calibration threshold and both are re-checked at scan time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

MODEL_SR = 16_000

# Wav2Vec2 BASE feature-extractor geometry, read off the released model rather
# than assumed: conv kernels (10,3,3,3,3,2,2) with strides (5,2,2,2,2,2,2) give
# a 400-sample receptive field advancing 320 samples per frame.
FRAME_STRIDE = 320          # 20 ms at 16 kHz
RECEPTIVE_FIELD = 400       # 25 ms at 16 kHz

# Chunking. Expressed in frames so chunk edges always land on frame edges and
# a frame is never split across two chunks.
CHUNK_FRAMES = 1500         # 30.0 s
EDGE_MARGIN_FRAMES = 100    # 2.0 s discarded from each interior chunk edge
BATCH_SIZE = 1              # one chunk per forward pass: no padding, no masks

# Event post-processing (mission section 8.3).
SMOOTHING_FRAMES = 10       # 200 ms majority vote
SMOOTHING_THRESHOLD = 5     # >= 5 of 10 frames below threshold; ties detect
MIN_EVENT_FRAMES = 5        # 100 ms; strictly shorter detections are discarded

# Calibration.
CALIBRATION_PERCENTILE = 1.0

# Audition context added around each candidate in the listening list only.
AUDITION_CONTEXT_SECONDS = 2.0

# Speech-content annotation. Predeclared as reported metadata, not as a filter
# on the primary candidate set: the retired LAION evaluation showed a detector
# can score well on workload while firing only on silence, so every candidate
# carries the speech content of its own region and a speech-gated view is
# reported beside the primary one.
SPEECH_FRAME_MS = 20.0
SPEECH_FLOOR_DBFS = -45.0
SPEECH_RICH_FRACTION = 0.5


@dataclass(frozen=True)
class FrozenConfig:
    model_sr: int = MODEL_SR
    frame_stride: int = FRAME_STRIDE
    receptive_field: int = RECEPTIVE_FIELD
    chunk_frames: int = CHUNK_FRAMES
    edge_margin_frames: int = EDGE_MARGIN_FRAMES
    batch_size: int = BATCH_SIZE
    smoothing_frames: int = SMOOTHING_FRAMES
    smoothing_threshold: int = SMOOTHING_THRESHOLD
    min_event_frames: int = MIN_EVENT_FRAMES
    calibration_percentile: float = CALIBRATION_PERCENTILE
    speech_frame_ms: float = SPEECH_FRAME_MS
    speech_floor_dbfs: float = SPEECH_FLOOR_DBFS

    @property
    def frame_seconds(self) -> float:
        return self.frame_stride / self.model_sr

    @property
    def hop_frames(self) -> int:
        return self.chunk_frames - 2 * self.edge_margin_frames

    def as_dict(self) -> dict:
        d = asdict(self)
        d["frame_seconds"] = self.frame_seconds
        d["hop_frames"] = self.hop_frames
        d["chunk_seconds"] = self.chunk_frames * self.frame_seconds
        d["edge_margin_seconds"] = self.edge_margin_frames * self.frame_seconds
        d["smoothing_ms"] = self.smoothing_frames * self.frame_seconds * 1000
        d["min_event_ms"] = self.min_event_frames * self.frame_seconds * 1000
        return d


FROZEN = FrozenConfig()
