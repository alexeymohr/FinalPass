"""Phase A: does the chunk grid change the CANDIDATE LIST, not just scores?

Frame scores move by a mean of ~0.09 MOS when the same passage is scored in a
different chunk position. That is a property of the model, not of the harness.
What matters operationally is whether the events the operator is asked to
audition move with it. This scores a public stream on two chunk grids, runs the
frozen event pipeline over both, and reports how many candidates agree.

Public audio only, run after the threshold is frozen and before any client audio.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqa_eval.config import FROZEN  # noqa: E402
from sqa_eval.events import build_candidates  # noqa: E402
from sqa_eval.paderborn_model import MODEL_SR, load_model, normalize_loudness, num_frames_for  # noqa: E402


def plan(n, chunk_frames, margin, first_span=None):
    hop = chunk_frames - 2 * margin
    out, keep = [], 0
    while keep < n:
        span = first_span if (keep == 0 and first_span) else hop
        end = min(keep + span, n)
        out.append((max(0, keep - margin), min(n, end + margin), keep, end))
        keep = end
    return out


def score(model, wav16, first_span=None):
    n = num_frames_for(len(wav16))
    scores = np.full(n, np.nan)
    for start, stop, ks, ke in plan(n, FROZEN.chunk_frames, FROZEN.edge_margin_frames, first_span):
        block = wav16[start * 320:min(stop * 320, len(wav16))]
        block = normalize_loudness(block, MODEL_SR)
        block = (block - np.mean(block)) / (np.std(block) + 1e-7)
        s = model.frame_scores(torch.from_numpy(np.ascontiguousarray(block)).float()).numpy()
        scores[ks:ke] = s[ks - start:ks - start + (ke - ks)]
    return scores


def overlap_fraction(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0, hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    threshold = json.loads(Path(a.manifest).read_text())["threshold"]["value"]
    x, sr = sf.read(a.stream, dtype="float32", always_2d=False)
    if sr != MODEL_SR:
        x = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(x)), sr, MODEL_SR).numpy()
    model = load_model(Path(a.state))

    base = score(model, x)
    shifted = score(model, x, first_span=FROZEN.hop_frames // 2)
    total = len(x)

    def cands(s):
        return build_candidates(list(s), threshold=threshold, source_id="public",
                                sample_rate=MODEL_SR, total_samples=total)

    A, B = cands(base), cands(shifted)
    spans_a = [(c.start_frame, c.end_frame) for c in A]
    spans_b = [(c.start_frame, c.end_frame) for c in B]
    matched = sum(1 for sa in spans_a if any(overlap_fraction(sa, sb) > 0 for sb in spans_b))
    matched_b = sum(1 for sb in spans_b if any(overlap_fraction(sb, sa) > 0 for sa in spans_a))

    frames_a = np.zeros(len(base), dtype=bool); frames_b = np.zeros(len(base), dtype=bool)
    for s, e in spans_a: frames_a[s:e] = True
    for s, e in spans_b: frames_b[s:e] = True
    inter = int((frames_a & frames_b).sum()); union = int((frames_a | frames_b).sum())

    result = {
        "threshold": threshold,
        "frames": int(len(base)),
        "candidates_base": len(A), "candidates_shifted": len(B),
        "base_candidates_with_an_overlapping_match": matched,
        "shifted_candidates_with_an_overlapping_match": matched_b,
        "flagged_frames_base": int(frames_a.sum()),
        "flagged_frames_shifted": int(frames_b.sum()),
        "flagged_frame_jaccard": round(inter / union, 4) if union else None,
        "frame_score_mean_abs_diff": round(float(np.abs(base - shifted).mean()), 5),
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
