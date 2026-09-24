"""Chunked local scoring of a long audio file. Numbers out, never audio.

Streams one chunk at a time so a multi-hour file never enters RAM whole,
resamples locally to the model rate, and keeps every reported position in the
ORIGINAL source-sample domain. Nothing in this module writes, returns or logs
audio samples.

Needs the isolated evaluation venv (torch/torchaudio/soundfile).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .channels import describe_layout, to_mono
from .chunking import Chunk, frame_source_samples, plan_chunks, total_frames
from .config import FROZEN, MODEL_SR
from .paderborn_model import num_frames_for, prepare_waveform
from .speech_content import frame_rms_dbfs

PROBE_WINDOWS = 24
PROBE_SECONDS = 2.0


class ScanRefused(RuntimeError):
    """The source cannot be scored under the frozen policy."""


def source_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "source_id": Path(path).name,
        "sample_rate": int(info.samplerate),
        "total_samples": int(info.frames),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "format": info.format,
        "duration_seconds": round(info.frames / info.samplerate, 6),
    }


def probe_layout(path: Path, info: dict) -> dict:
    """Decide the channel policy from windows spread across the whole file.

    A single probe at the head would miss a file that is dual mono only where
    it happens to be quiet, so the decision takes the WORST null depth over
    evenly spaced windows and ignores windows with no content in them.
    """
    if info["channels"] == 1:
        return {"channels": 1, "null_depth_db": None, "dual_mono": True,
                "windows_used": 0}
    win = int(PROBE_SECONDS * info["sample_rate"])
    total = info["total_samples"]
    starts = [max(0, min(total - win, (total * i) // PROBE_WINDOWS))
              for i in range(PROBE_WINDOWS)]
    depths: list[float] = []
    for start in sorted(set(starts)):
        block, _ = sf.read(str(path), start=start, frames=min(win, total - start),
                           dtype="float64", always_2d=True)
        if block.size == 0 or float(np.max(np.abs(block))) <= 1e-6:
            continue
        layout = describe_layout(block)
        if layout["null_depth_db"] is not None:
            depths.append(layout["null_depth_db"])
    if not depths:
        return {"channels": info["channels"], "null_depth_db": None,
                "dual_mono": info["channels"] == 1, "windows_used": 0}
    worst = min(depths)
    from .channels import DUAL_MONO_NULL_DB
    return {"channels": info["channels"], "null_depth_db": round(worst, 3),
            "dual_mono": worst >= DUAL_MONO_NULL_DB, "windows_used": len(depths)}


def _read_mono(path: Path, start: int, frames: int, layout: dict) -> np.ndarray:
    block, _ = sf.read(str(path), start=start, frames=frames,
                       dtype="float32", always_2d=False)
    return to_mono(block, layout, Path(path).name)


def speech_levels(mono: np.ndarray, source_sr: int, n_frames: int) -> list[float]:
    """Per-model-frame RMS level of the ORIGINAL audio, in dBFS."""
    step = frame_source_samples(source_sr)
    out: list[float] = []
    for f in range(n_frames):
        block = mono[f * step:(f + 1) * step]
        out.append(frame_rms_dbfs(block))
    return out


def score_chunk(model, mono: np.ndarray, source_sr: int, expected_frames: int) -> np.ndarray:
    """Resample one chunk to 16 kHz, prepare it, and score every frame."""
    tensor = torch.from_numpy(np.ascontiguousarray(mono)).float()
    if source_sr != MODEL_SR:
        tensor = torchaudio.functional.resample(tensor, orig_freq=source_sr,
                                                new_freq=MODEL_SR)
    prepared = prepare_waveform(tensor.numpy(), MODEL_SR)
    got = num_frames_for(len(prepared))
    if got != expected_frames:
        raise ScanRefused(
            f"chunk produced {got} frames, schedule expected {expected_frames}; "
            "frame mapping would be wrong"
        )
    scores = model.frame_scores(torch.from_numpy(np.ascontiguousarray(prepared)).float())
    return scores.detach().numpy().astype(np.float64)


def score_file(model, path: Path, *, config=FROZEN, progress=None) -> dict:
    """Score a whole file. Returns frame scores plus numeric metadata only."""
    info = source_info(path)
    layout = probe_layout(path, info)
    if layout["channels"] > 1 and not layout["dual_mono"]:
        # Surface the refusal through the documented channel policy.
        to_mono(np.zeros((1, layout["channels"]), dtype=np.float32), layout, info["source_id"])

    n_frames = total_frames(info["total_samples"], info["sample_rate"])
    chunks = plan_chunks(info["total_samples"], info["sample_rate"], config)
    scores = np.full(n_frames, np.nan, dtype=np.float64)
    levels = np.full(n_frames, -np.inf, dtype=np.float64)

    for chunk in chunks:
        start, end = chunk.start_sample, chunk.end_sample
        mono = _read_mono(path, start, end - start, layout)
        chunk_scores = score_chunk(model, mono, info["sample_rate"], chunk.frame_count)
        offset, count = chunk.keep_offset, chunk.keep_count
        scores[chunk.keep_start:chunk.keep_end] = chunk_scores[offset:offset + count]
        chunk_levels = speech_levels(mono, info["sample_rate"], chunk.frame_count)
        levels[chunk.keep_start:chunk.keep_end] = chunk_levels[offset:offset + count]
        if progress is not None:
            progress(chunk, len(chunks))

    unscored = int(np.isnan(scores).sum())
    if unscored:
        raise ScanRefused(f"{info['source_id']}: {unscored} frames left unscored")
    return {
        "info": info,
        "layout": layout,
        "frames": n_frames,
        "chunks": len(chunks),
        "scores": scores,
        "levels_dbfs": levels,
    }
