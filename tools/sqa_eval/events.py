"""Frame scores -> candidate regions, deterministically.

The pipeline is fixed by the mission and frozen before the client scan:

1. frames strictly below the frozen calibration threshold become binary
   detections;
2. a 200 ms majority-vote smoothing pass;
3. detections shorter than 100 ms are discarded;
4. surviving runs become candidate regions carrying exact source-sample bounds.

The smoothing follows the Paderborn group's own binary-activity primitive
(`conv_smoothing` in `fgnt/local_sqa`): a sliding window of `window` frames,
zero-padded at the edges with `(window - 1) // 2` frames of left context and
the remainder on the right, emitting a detection where at least `threshold` of
the frames in the window are active. With a 10-frame (200 ms) window and a
threshold of 5 that is a median filter whose ties resolve towards detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .chunking import frame_source_samples
from .config import FROZEN


def smooth_binary(
    flags: Sequence[bool], window: int, threshold: int
) -> list[bool]:
    """Sliding majority vote over a binary sequence, length preserving."""
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(flags)
    if n == 0:
        return []
    left = (window - 1) // 2
    right = window - 1 - left
    padded = [0] * left + [1 if f else 0 for f in flags] + [0] * right
    # Running sum over the padded sequence; one pass, no numpy dependency.
    out: list[bool] = []
    running = sum(padded[:window])
    out.append(running >= threshold)
    for i in range(1, n):
        running += padded[i + window - 1] - padded[i - 1]
        out.append(running >= threshold)
    return out


def binary_runs(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


@dataclass(frozen=True)
class Candidate:
    """One contiguous low-quality region, in original source coordinates."""

    source_id: str
    start_sample: int
    end_sample: int
    sample_rate: int
    start_frame: int
    end_frame: int
    min_score: float
    mean_score: float
    threshold: float
    frames_below_threshold: int
    speech_content: float | None = None
    overlaps_boundary_gap: bool = False
    distance_to_clip_boundary_samples: int | None = None

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.duration_samples / self.sample_rate

    @property
    def interior(self) -> bool:
        """True when the region is not adjacent to a known assembly boundary.

        A region touching a digital-black gap, or whose nearest clip boundary
        lies within one frame, is boundary-adjacent; everything else is
        interior. Boundary-adjacent candidates are reported, never suppressed.
        """
        if self.overlaps_boundary_gap:
            return False
        d = self.distance_to_clip_boundary_samples
        if d is None:
            return True
        return d > frame_source_samples(self.sample_rate)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            duration_samples=self.duration_samples,
            start_seconds=round(self.start_seconds, 6),
            end_seconds=round(self.end_seconds, 6),
            duration_seconds=round(self.duration_seconds, 6),
            interior=self.interior,
        )
        return d


def _boundary_facts(
    start_sample: int, end_sample: int, gap_spans: Sequence[tuple[int, int]]
) -> tuple[bool, int | None]:
    """Overlap with a digital-black gap, and distance to the nearest edge."""
    if not gap_spans:
        return False, None
    overlaps = any(gs < end_sample and ge > start_sample for gs, ge in gap_spans)
    best: int | None = None
    for gs, ge in gap_spans:
        for edge in (gs, ge):
            if start_sample <= edge <= end_sample:
                d = 0
            else:
                d = min(abs(edge - start_sample), abs(edge - end_sample))
            best = d if best is None else min(best, d)
    return overlaps, best


def build_candidates(
    scores: Sequence[float],
    *,
    threshold: float,
    source_id: str,
    sample_rate: int,
    total_samples: int,
    gap_spans: Sequence[tuple[int, int]] = (),
    speech_content: Sequence[float] | None = None,
    config=FROZEN,
) -> list[Candidate]:
    """Apply the frozen threshold/smoothing/duration pipeline to one file."""
    flags = [s < threshold for s in scores]
    smoothed = smooth_binary(flags, config.smoothing_frames, config.smoothing_threshold)
    step = frame_source_samples(sample_rate)

    out: list[Candidate] = []
    for start_f, end_f in binary_runs(smoothed):
        if end_f - start_f < config.min_event_frames:
            continue
        region = list(scores[start_f:end_f])
        start_sample = start_f * step
        end_sample = min(end_f * step, total_samples)
        overlaps, distance = _boundary_facts(start_sample, end_sample, gap_spans)
        speech = None
        if speech_content is not None:
            window = list(speech_content[start_f:end_f])
            speech = sum(window) / len(window) if window else 0.0
        out.append(Candidate(
            source_id=source_id,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
            start_frame=start_f,
            end_frame=end_f,
            min_score=min(region),
            mean_score=sum(region) / len(region),
            threshold=threshold,
            frames_below_threshold=sum(1 for s in region if s < threshold),
            speech_content=speech,
            overlaps_boundary_gap=overlaps,
            distance_to_clip_boundary_samples=distance,
        ))
    return out


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Worst first, fully tie-broken so two runs order identically.

    Mission section 13: lowest minimum frame score, then lowest mean, then
    longer duration, then earlier source position. Source id is the final
    tie-break so ordering is stable across files.
    """
    return sorted(
        candidates,
        key=lambda c: (
            c.min_score,
            c.mean_score,
            -c.duration_samples,
            c.source_id,
            c.start_sample,
        ),
    )
