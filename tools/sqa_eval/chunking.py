"""Deterministic chunk schedule and exact frame <-> source-sample mapping.

The model scores 20 ms frames at 16 kHz, but the source is a multi-hour file at
its own sample rate. Everything here is planned in whole frames so a frame is
never split across a chunk edge, and every reported position is an exact
integer source-sample index: no float round-trip, no resampled-domain rounding.

Two facts pin the mapping, both read off the released model rather than assumed
(see `docs/../resources/paderborn_sqa_eval` audit):

* the encoder emits ``ceil(n_samples / 320)`` frames for an ``n_samples`` input
  at 16 kHz, with frame ``f`` advancing exactly 320 samples;
* padding is symmetric, so frame ``f`` is centred on 16 kHz sample
  ``320 f + 160`` and the frames tile the signal without holes or overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import FROZEN, FRAME_STRIDE, MODEL_SR


class SampleRateUnsupported(ValueError):
    """The source rate cannot carry an exact whole-sample frame grid."""


def frame_source_samples(source_sr: int) -> int:
    """Source samples per model frame, exactly.

    A frame is 320 samples at 16 kHz. For the mapping to stay in integers the
    source rate must make ``320 * source_sr / 16000`` whole — true for every
    rate that is a multiple of 50 Hz, which covers 16/22.05/32/44.1/48 kHz.
    Anything else is refused rather than silently rounded.
    """
    numerator = FRAME_STRIDE * source_sr
    if numerator % MODEL_SR:
        raise SampleRateUnsupported(
            f"{source_sr} Hz does not divide into whole model frames "
            f"({FRAME_STRIDE} samples at {MODEL_SR} Hz); refusing to round."
        )
    return numerator // MODEL_SR


def total_frames(total_samples: int, source_sr: int) -> int:
    """Frames needed to cover the whole source, with no hole at the end."""
    if total_samples <= 0:
        return 0
    step = frame_source_samples(source_sr)
    return -(-total_samples // step)  # ceil


def frame_to_source_span(
    frame_index: int, source_sr: int, total_samples: int
) -> tuple[int, int]:
    """Exact ``[start, end)`` source-sample span covered by one frame."""
    step = frame_source_samples(source_sr)
    start = frame_index * step
    return start, min(start + step, total_samples)


def frame_center_source_sample(frame_index: int, source_sr: int) -> int:
    """Source sample at the centre of a frame's 16 kHz receptive stride."""
    step = frame_source_samples(source_sr)
    return frame_index * step + step // 2


@dataclass(frozen=True)
class Chunk:
    """One inference chunk plus the frame range whose scores it owns.

    ``start_frame``/``end_frame`` are what is fed to the model;
    ``keep_start``/``keep_end`` are the frames whose scores are retained. The
    difference is the discarded edge margin — context the model needs but whose
    scores are contaminated by the artificial chunk edge.
    """

    index: int
    start_frame: int
    end_frame: int
    keep_start: int
    keep_end: int
    source_sr: int
    total_samples: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_sample(self) -> int:
        return self.start_frame * frame_source_samples(self.source_sr)

    @property
    def end_sample(self) -> int:
        end = self.end_frame * frame_source_samples(self.source_sr)
        return min(end, self.total_samples)

    @property
    def keep_offset(self) -> int:
        """Index of the first kept frame within this chunk's own output."""
        return self.keep_start - self.start_frame

    @property
    def keep_count(self) -> int:
        return self.keep_end - self.keep_start


def plan_chunks(total_samples: int, source_sr: int, config=FROZEN) -> list[Chunk]:
    """Chunks covering the whole source exactly once, with context margins.

    Kept ranges tile ``[0, total_frames)`` back to back; the analysed range of
    each chunk extends one margin further in each direction where the source
    allows, so no kept frame is ever produced at a chunk edge unless the file
    itself ends there.
    """
    n_frames = total_frames(total_samples, source_sr)
    if n_frames == 0:
        return []
    hop = config.hop_frames
    if hop <= 0:
        raise ValueError("chunk_frames must exceed twice the edge margin")

    chunks: list[Chunk] = []
    index = 0
    keep_start = 0
    while keep_start < n_frames:
        keep_end = min(keep_start + hop, n_frames)
        start = max(0, keep_start - config.edge_margin_frames)
        end = min(n_frames, keep_end + config.edge_margin_frames)
        chunks.append(Chunk(
            index=index, start_frame=start, end_frame=end,
            keep_start=keep_start, keep_end=keep_end,
            source_sr=source_sr, total_samples=total_samples,
        ))
        index += 1
        keep_start = keep_end
    return chunks


def coverage(chunks: list[Chunk], total_samples: int, source_sr: int) -> dict:
    """Prove the schedule leaves nothing unscored and nothing double-counted.

    Reports over the KEPT ranges, which are what actually reach the results —
    the analysed ranges deliberately overlap.
    """
    n_frames = total_frames(total_samples, source_sr)
    if n_frames == 0:
        return {"total_frames": 0, "kept_frames": 0, "uncovered_frames": 0,
                "duplicated_frames": 0, "chunks": 0, "context_frames": 0}
    seen = [0] * n_frames
    for c in chunks:
        for f in range(c.keep_start, c.keep_end):
            seen[f] += 1
    context = sum(c.frame_count - c.keep_count for c in chunks)
    return {
        "total_frames": n_frames,
        "kept_frames": sum(1 for s in seen if s >= 1),
        "uncovered_frames": sum(1 for s in seen if s == 0),
        "duplicated_frames": sum(1 for s in seen if s > 1),
        "chunks": len(chunks),
        "context_frames": context,
    }
