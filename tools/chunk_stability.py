"""Phase A experiment: how stable are frame scores across chunk positions?

Builds a continuous public narration stream (consecutive LibriTTS-R utterances
from one chapter, in order), scores it with the frozen schedule, then scores it
again with the chunk grid deliberately offset by half a hop so every passage
lands in a different position. Compares:

* KEPT frames (what the scan actually reports) between the two grids;
* frames a chunk would have produced inside its discarded edge margin.

Public audio only. Run during provisioning, before the calibration threshold is
frozen, so the chunk length / overlap / discard margin are justified rather
than assumed.
"""
from __future__ import annotations

import argparse, json, sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqa_eval.chunking import frame_source_samples, plan_chunks, total_frames  # noqa: E402
from sqa_eval.config import FROZEN  # noqa: E402
from sqa_eval.paderborn_model import load_model  # noqa: E402
from sqa_eval.scoring import score_chunk  # noqa: E402


def build_stream(root: Path, out: Path, minutes: float) -> dict:
    """Concatenate consecutive utterances of whole chapters into one stream."""
    chapters: dict[tuple[str, str], list[Path]] = {}
    for p in sorted(root.rglob("*.wav")):
        chapters.setdefault((p.parts[-3], p.parts[-2]), []).append(p)
    picked: list[Path] = []
    used: list[tuple[str, str]] = []
    total = 0.0
    for key in sorted(chapters):
        for p in chapters[key]:
            info = sf.info(str(p))
            picked.append(p)
            total += info.frames / info.samplerate
        used.append(key)
        if total >= minutes * 60:
            break
    blocks, sr = [], None
    for p in picked:
        x, s = sf.read(str(p), dtype="float32", always_2d=False)
        sr = s if sr is None else sr
        assert s == sr, (s, sr)
        blocks.append(x)
    stream = np.concatenate(blocks)
    sf.write(str(out), stream, sr, subtype="PCM_16")
    return {"utterances": len(picked), "chapters": ["/".join(k) for k in used],
            "sample_rate": int(sr), "samples": int(stream.shape[0]),
            "seconds": round(stream.shape[0] / sr, 3)}


def score_with_grid(model, path: Path, sr: int, total_samples: int, shift_frames: int):
    """Score the stream with the first kept block shortened by `shift_frames`."""
    n_frames = total_frames(total_samples, sr)
    chunks = plan_chunks(total_samples, sr)
    if shift_frames:
        template = chunks[0]
        shifted, keep = [], 0
        while keep < n_frames:
            span = shift_frames if keep == 0 else FROZEN.hop_frames
            end = min(keep + span, n_frames)
            start = max(0, keep - FROZEN.edge_margin_frames)
            stop = min(n_frames, end + FROZEN.edge_margin_frames)
            shifted.append(replace(template, index=len(shifted), start_frame=start,
                                   end_frame=stop, keep_start=keep, keep_end=end))
            keep = end
        chunks = shifted

    scores = np.full(n_frames, np.nan)
    for c in chunks:
        block, _ = sf.read(str(path), start=c.start_sample,
                           frames=c.end_sample - c.start_sample,
                           dtype="float32", always_2d=False)
        s = score_chunk(model, block, sr, c.frame_count)
        scores[c.keep_start:c.keep_end] = s[c.keep_offset:c.keep_offset + c.keep_count]
    return scores, chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    stream_path = work / "public_stream.wav"
    meta = build_stream(Path(a.corpus), stream_path, a.minutes)
    print(f"public stream: {meta['seconds']}s from {meta['utterances']} utterances "
          f"of {len(meta['chapters'])} chapter(s) @ {meta['sample_rate']} Hz", flush=True)

    model = load_model(Path(a.state))
    sr, total = meta["sample_rate"], meta["samples"]

    base, base_chunks = score_with_grid(model, stream_path, sr, total, 0)
    print(f"base grid scored: {len(base_chunks)} chunks", flush=True)
    shifted, shift_chunks = score_with_grid(model, stream_path, sr, total,
                                            FROZEN.hop_frames // 2)
    print(f"shifted grid scored: {len(shift_chunks)} chunks", flush=True)
    assert not np.isnan(base).any() and not np.isnan(shifted).any()

    diff = np.abs(base - shifted)
    seam = np.zeros(len(base), dtype=bool)
    for c in base_chunks[1:]:
        lo = max(0, c.keep_start - FROZEN.edge_margin_frames)
        hi = min(len(base), c.keep_start + FROZEN.edge_margin_frames)
        seam[lo:hi] = True

    # What a raw, unprotected chunk edge would have reported.
    raw_edge = []
    for c in base_chunks:
        block, _ = sf.read(str(stream_path), start=c.start_sample,
                           frames=c.end_sample - c.start_sample,
                           dtype="float32", always_2d=False)
        s = score_chunk(model, block, sr, c.frame_count)
        for local in list(range(0, c.keep_offset)) + list(
                range(c.keep_offset + c.keep_count, c.frame_count)):
            raw_edge.append(abs(s[local] - base[c.start_frame + local]))
    raw_edge = np.array(raw_edge) if raw_edge else np.zeros(0)

    result = {
        "stream": meta,
        "frames": int(len(base)),
        "chunks_base": len(base_chunks), "chunks_shifted": len(shift_chunks),
        "kept_frame_agreement": {
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "p99_abs_diff": float(np.percentile(diff, 99)),
            "frames_over_0.01": int((diff > 0.01).sum()),
            "frames_over_0.05": int((diff > 0.05).sum()),
        },
        "near_seam_vs_interior": {
            "seam_frames": int(seam.sum()),
            "seam_mean_abs_diff": float(diff[seam].mean()) if seam.any() else None,
            "interior_mean_abs_diff": float(diff[~seam].mean()) if (~seam).any() else None,
        },
        "discarded_edge_frames": {
            "count": int(len(raw_edge)),
            "max_abs_diff_vs_kept": float(raw_edge.max()) if len(raw_edge) else None,
            "mean_abs_diff_vs_kept": float(raw_edge.mean()) if len(raw_edge) else None,
            "p99_abs_diff_vs_kept": float(np.percentile(raw_edge, 99)) if len(raw_edge) else None,
        },
        "score_distribution_base": {
            "min": float(base.min()), "p1": float(np.percentile(base, 1)),
            "median": float(np.median(base)), "mean": float(base.mean()),
            "max": float(base.max()),
        },
        "frozen_config": FROZEN.as_dict(),
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "frozen_config"}, indent=1))


if __name__ == "__main__":
    main()
