"""Phase A diagnosis: what actually makes a frame's score depend on its chunk?

The first chunk-stability run showed kept-frame scores moving by a mean of 0.09
MOS (p99 0.67) when the same passage is scored in a different chunk position,
and showed NO extra sensitivity near chunk seams. That rules out a chunk-edge
artefact and points at the model's two whole-input normalisation stages:

* `standardize_audio` divides the waveform by the statistics of whatever block
  is handed to it;
* `l2_normalization` divides every frame embedding by the norm of the block's
  time-averaged embedding.

Both are utterance-level operations in the released model, which was evaluated
on ~5 s utterances. This sweep measures position sensitivity against chunk
length and against normalisation scope (per-chunk vs whole-file), on public
audio, so the scan configuration can be chosen on evidence.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqa_eval.chunking import frame_source_samples, total_frames  # noqa: E402
from sqa_eval.config import FrozenConfig  # noqa: E402
from sqa_eval.paderborn_model import (  # noqa: E402
    MODEL_SR, load_model, normalize_loudness, num_frames_for,
)


def plan(n_frames, chunk_frames, margin, first_span=None):
    hop = chunk_frames - 2 * margin
    out, keep = [], 0
    while keep < n_frames:
        span = first_span if (keep == 0 and first_span) else hop
        end = min(keep + span, n_frames)
        out.append((max(0, keep - margin), min(n_frames, end + margin), keep, end))
        keep = end
    return out


def score_stream(model, wav16: np.ndarray, chunk_frames: int, margin: int,
                 scope: str, first_span=None) -> np.ndarray:
    n_frames = num_frames_for(len(wav16))
    if scope == "file":
        prepared_all = normalize_loudness(wav16, MODEL_SR)
        mean = float(np.mean(prepared_all)); std = float(np.std(prepared_all))
        prepared_all = (prepared_all - mean) / (std + 1e-7)
    scores = np.full(n_frames, np.nan)
    for start, stop, keep_start, keep_end in plan(n_frames, chunk_frames, margin, first_span):
        lo, hi = start * 320, min(stop * 320, len(wav16))
        if scope == "file":
            block = prepared_all[lo:hi]
        else:
            raw = wav16[lo:hi]
            block = normalize_loudness(raw, MODEL_SR)
            block = (block - np.mean(block)) / (np.std(block) + 1e-7)
        s = model.frame_scores(torch.from_numpy(np.ascontiguousarray(block)).float()).numpy()
        off = keep_start - start
        scores[keep_start:keep_end] = s[off:off + (keep_end - keep_start)]
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    x, sr = sf.read(a.stream, dtype="float32", always_2d=False)
    if sr != MODEL_SR:
        x = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(x)), sr, MODEL_SR).numpy()
    model = load_model(Path(a.state))
    print(f"stream: {len(x)/MODEL_SR:.1f}s at {MODEL_SR} Hz, "
          f"{num_frames_for(len(x))} frames", flush=True)

    rows = []
    for chunk_seconds in (10, 20, 30, 60):
        chunk_frames = chunk_seconds * 50
        margin = 100
        if chunk_frames <= 2 * margin:
            margin = chunk_frames // 5
        hop = chunk_frames - 2 * margin
        for scope in ("chunk", "file"):
            base = score_stream(model, x, chunk_frames, margin, scope)
            shifted = score_stream(model, x, chunk_frames, margin, scope,
                                   first_span=max(1, hop // 2))
            d = np.abs(base - shifted)
            rows.append({
                "chunk_seconds": chunk_seconds, "margin_frames": margin,
                "normalisation_scope": scope,
                "mean_abs_diff": round(float(d.mean()), 5),
                "p99_abs_diff": round(float(np.percentile(d, 99)), 5),
                "max_abs_diff": round(float(d.max()), 5),
                "frames_over_0.05": int((d > 0.05).sum()),
                "p1_base": round(float(np.percentile(base, 1)), 5),
                "median_base": round(float(np.median(base)), 5),
            })
            print(f"  chunk={chunk_seconds:>3}s scope={scope:<5} "
                  f"mean|d|={rows[-1]['mean_abs_diff']:.4f} "
                  f"p99={rows[-1]['p99_abs_diff']:.4f} max={rows[-1]['max_abs_diff']:.4f} "
                  f"p1={rows[-1]['p1_base']:.3f} median={rows[-1]['median_base']:.3f}",
                  flush=True)
    Path(a.out).write_text(json.dumps({"sweep": rows}, indent=1))


if __name__ == "__main__":
    main()
