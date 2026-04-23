"""Shared program-window preflight for comparison-style analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio_io import AudioFile
from .errors import ProgramWindowError
from .models import AnalysisWindow, AnalysisWindowInput
from .timecode import samples_to_tc, tc_to_sample_start

TAIL_MOS_THRESHOLD_DBFS = -80.0
TAIL_PEAK_THRESHOLD_DBFS = -60.0
TAIL_MOS_WINDOW_MS = 100.0
MAX_WHOLE_HOUR_HEAD_SECONDS = 20 * 60
_DBFS_FLOOR = -300.0
_LINEAR_FLOOR = 10 ** (_DBFS_FLOOR / 20.0)


@dataclass(frozen=True)
class WindowedInput:
    label: str
    audio: AudioFile
    start_sample: int
    end_sample: int

    @property
    def data(self) -> np.ndarray:
        source_end = min(self.end_sample, self.audio.sample_count)
        data = self.audio.data[self.start_sample : source_end, :]
        pad_samples = self.end_sample - source_end
        if pad_samples <= 0:
            return data
        padding = np.zeros((pad_samples, self.audio.channel_count), dtype=data.dtype)
        return np.vstack([data, padding])


@dataclass(frozen=True)
class PreparedAnalysisWindow:
    inputs: tuple[WindowedInput, ...]
    analysis_window: AnalysisWindow


def prepare_analysis_window(
    inputs: list[tuple[str, AudioFile]],
    *,
    fps: float,
    anchor_index: int = 0,
    tail_mos_threshold_dbfs: float = TAIL_MOS_THRESHOLD_DBFS,
    tail_peak_threshold_dbfs: float = TAIL_PEAK_THRESHOLD_DBFS,
) -> PreparedAnalysisWindow:
    """Return same-length slices suitable for null/M&E comparison.

    If all inputs carry BWF time references and a common whole-hour boundary is
    present, analysis starts at that boundary. Otherwise analysis starts at file
    sample 0. A uniquely longer overrun is accepted only when that overrun is
    MOS; shorter inputs are padded with silence inside the comparable range.
    """
    if not inputs:
        raise ProgramWindowError("Analysis window requires at least one input.")
    if anchor_index < 0 or anchor_index >= len(inputs):
        raise ProgramWindowError(f"Invalid analysis window anchor index {anchor_index}.")

    sample_rate = inputs[anchor_index][1].sample_rate
    for _, audio in inputs:
        if audio.sample_rate != sample_rate:
            raise ProgramWindowError("Analysis window inputs must share one sample rate.")

    starts, ends, mode = _window_bounds(
        inputs,
        sample_rate=sample_rate,
        fps=fps,
        tail_mos_threshold_dbfs=tail_mos_threshold_dbfs,
        tail_peak_threshold_dbfs=tail_peak_threshold_dbfs,
    )
    if min(end - start for start, end in zip(starts, ends, strict=True)) <= 0:
        raise ProgramWindowError("Analysis window has no overlapping sample range.")

    windowed_inputs = tuple(
        WindowedInput(label=label, audio=audio, start_sample=start, end_sample=end)
        for (label, audio), start, end in zip(inputs, starts, ends, strict=True)
    )
    anchor = windowed_inputs[anchor_index]
    start_tc = samples_to_tc(
        anchor.start_sample,
        anchor.audio.sample_rate,
        fps,
        start_time_reference_samples=anchor.audio.time_reference_samples,
    )
    end_tc = samples_to_tc(
        anchor.end_sample,
        anchor.audio.sample_rate,
        fps,
        start_time_reference_samples=anchor.audio.time_reference_samples,
    )
    analysis_window = AnalysisWindow(
        mode=mode,
        start_sample=anchor.start_sample,
        end_sample=anchor.end_sample,
        start_tc=start_tc,
        end_tc=end_tc,
        duration_seconds=round((anchor.end_sample - anchor.start_sample) / float(anchor.audio.sample_rate), 3),
        tail_mos_threshold_dbfs=tail_mos_threshold_dbfs,
        tail_peak_threshold_dbfs=tail_peak_threshold_dbfs,
        inputs=[
            AnalysisWindowInput(
                label=windowed.label,
                path=str(windowed.audio.path),
                start_sample=windowed.start_sample,
                end_sample=windowed.end_sample,
                ignored_head_samples=windowed.start_sample,
                ignored_tail_samples=max(0, windowed.audio.sample_count - windowed.end_sample),
                padded_tail_samples=max(0, windowed.end_sample - windowed.audio.sample_count),
                ignored_head_seconds=round(windowed.start_sample / float(windowed.audio.sample_rate), 3),
                ignored_tail_seconds=round(
                    max(0, windowed.audio.sample_count - windowed.end_sample) / float(windowed.audio.sample_rate),
                    3,
                ),
                padded_tail_seconds=round(
                    max(0, windowed.end_sample - windowed.audio.sample_count) / float(windowed.audio.sample_rate),
                    3,
                ),
            )
            for windowed in windowed_inputs
        ],
    )
    return PreparedAnalysisWindow(inputs=windowed_inputs, analysis_window=analysis_window)


def _window_bounds(
    inputs: list[tuple[str, AudioFile]],
    *,
    sample_rate: int,
    fps: float,
    tail_mos_threshold_dbfs: float,
    tail_peak_threshold_dbfs: float,
) -> tuple[list[int], list[int], str]:
    refs = [audio.time_reference_samples for _, audio in inputs]
    if all(ref is not None for ref in refs):
        concrete_refs = [int(ref) for ref in refs if ref is not None]
        common_start = max(concrete_refs)
        common_end = min(ref + audio.sample_count for ref, (_, audio) in zip(concrete_refs, inputs, strict=True))
        boundary = _common_whole_hour_boundary(common_start, common_end, sample_rate=sample_rate, fps=fps)
        if boundary is not None:
            raw_ends = [ref + audio.sample_count for ref, (_, audio) in zip(concrete_refs, inputs, strict=True)]
            analysis_end = _analysis_end_after_unique_mos_tail(
                inputs,
                raw_ends=raw_ends,
                refs=concrete_refs,
                threshold_dbfs=tail_mos_threshold_dbfs,
                peak_threshold_dbfs=tail_peak_threshold_dbfs,
            )
            starts = [boundary - ref for ref in concrete_refs]
            ends = [analysis_end - ref for ref in concrete_refs]
            return starts, ends, "whole_hour_time_reference"
        if len(set(concrete_refs)) != 1:
            detail = ", ".join(f"{audio.path.name}={_abs_tc(ref, sample_rate, fps)}" for ref, (_, audio) in zip(concrete_refs, inputs, strict=True))
            raise ProgramWindowError(
                "No shared whole-hour program boundary was found across differently timed inputs: "
                f"{detail}."
            )
        starts, ends = _file_start_bounds(inputs)
        return starts, ends, "file_start_no_whole_hour"

    starts, ends = _file_start_bounds(
        inputs,
        tail_mos_threshold_dbfs=tail_mos_threshold_dbfs,
        tail_peak_threshold_dbfs=tail_peak_threshold_dbfs,
    )
    return starts, ends, "file_start"


def _file_start_bounds(
    inputs: list[tuple[str, AudioFile]],
    *,
    tail_mos_threshold_dbfs: float = TAIL_MOS_THRESHOLD_DBFS,
    tail_peak_threshold_dbfs: float = TAIL_PEAK_THRESHOLD_DBFS,
) -> tuple[list[int], list[int]]:
    raw_ends = [audio.sample_count for _, audio in inputs]
    end = _analysis_end_after_unique_mos_tail(
        inputs,
        raw_ends=raw_ends,
        refs=None,
        threshold_dbfs=tail_mos_threshold_dbfs,
        peak_threshold_dbfs=tail_peak_threshold_dbfs,
    )
    return [0 for _ in inputs], [end for _ in inputs]


def _common_whole_hour_boundary(
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
    fps: float,
) -> int | None:
    if end_sample <= start_sample:
        return None
    start_tc = _abs_tc(start_sample, sample_rate, fps)
    start_hour = int(start_tc.split(":", 1)[0])
    for hour in range(max(0, start_hour - 1), start_hour + 3):
        boundary = tc_to_sample_start(f"{hour:02d}:00:00:00", sample_rate, fps)
        if start_sample <= boundary < end_sample:
            if boundary - start_sample > int(round(MAX_WHOLE_HOUR_HEAD_SECONDS * sample_rate)):
                return None
            return boundary
    return None


def _abs_tc(sample_index: int, sample_rate: int, fps: float) -> str:
    return samples_to_tc(0, sample_rate, fps, start_time_reference_samples=sample_index)


def _analysis_end_after_unique_mos_tail(
    inputs: list[tuple[str, AudioFile]],
    *,
    raw_ends: list[int],
    refs: list[int] | None,
    threshold_dbfs: float,
    peak_threshold_dbfs: float,
) -> int:
    if len(raw_ends) < 2:
        return raw_ends[0]
    max_end = max(raw_ends)
    longest = [index for index, end in enumerate(raw_ends) if end == max_end]
    if len(longest) != 1:
        return max_end

    longest_index = longest[0]
    other_max = max(end for index, end in enumerate(raw_ends) if index != longest_index)
    if max_end <= other_max:
        return max_end

    _, audio = inputs[longest_index]
    ref = refs[longest_index] if refs is not None else 0
    tail_start = max(0, other_max - ref)
    tail_end = max(0, max_end - ref)
    tail = audio.data[tail_start:tail_end, :]
    max_rms_dbfs, max_peak_dbfs = _tail_levels(tail, sample_rate=audio.sample_rate)
    if max_rms_dbfs > threshold_dbfs or max_peak_dbfs > peak_threshold_dbfs:
        raise ProgramWindowError(
            "Sample count mismatch cannot be safely cropped: "
            f"{Path(audio.path).name} has signal beyond the maximum duration of the other inputs "
            f"(tail max RMS {max_rms_dbfs:.1f} dBFS, peak {max_peak_dbfs:.1f} dBFS)."
        )
    return other_max


def _tail_levels(tail: np.ndarray, *, sample_rate: int) -> tuple[float, float]:
    signal = _non_lfe_channels(tail)
    if signal.size == 0:
        return _DBFS_FLOOR, _DBFS_FLOOR
    window_samples = max(1, int(round(TAIL_MOS_WINDOW_MS * sample_rate / 1000.0)))
    max_rms = 0.0
    for start in range(0, signal.shape[0], window_samples):
        block = signal[start : start + window_samples, :]
        if block.size:
            max_rms = max(max_rms, float(np.sqrt(np.mean(block ** 2))))
    max_peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return _to_dbfs(max_rms), _to_dbfs(max_peak)


def _non_lfe_channels(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        return np.atleast_2d(data)
    channel_count = data.shape[1]
    if channel_count == 6:
        return data[:, [0, 1, 2, 4, 5]]
    if channel_count == 8:
        return data[:, [0, 1, 2, 4, 5, 6, 7]]
    return data


def _to_dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, _LINEAR_FLOOR)))
