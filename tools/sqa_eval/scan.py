"""Scan a delivery with the frozen detector. Strictly offline, numbers only.

Nothing about the detector is decided here: the threshold comes from the frozen
calibration manifest and the chunking/event parameters come from
`sqa_eval.config`, which the manifest also records. If either has drifted since
calibration the scan refuses to run rather than quietly reporting numbers that
mean something else.

Output is frame scores, sample positions, durations and counts. No audio, no
spectrogram, no sample array, ever leaves this process.
"""
from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.config import FROZEN  # noqa: E402
from sqa_eval.events import build_candidates, rank_candidates  # noqa: E402
from sqa_eval.model_files import load_manifest, verify_manifest  # noqa: E402
from sqa_eval.netguard import NetworkGuard  # noqa: E402
from sqa_eval.speech_content import speech_fraction  # noqa: E402


class FrozenConfigDrift(RuntimeError):
    """The scan configuration no longer matches the one the threshold was set under."""


def _check_frozen(manifest: dict) -> None:
    recorded = manifest.get("frozen_config") or {}
    current = FROZEN.as_dict()
    drifted = {k: (recorded.get(k), current.get(k))
               for k in current if recorded.get(k) != current.get(k)}
    if drifted:
        raise FrozenConfigDrift(
            "scan parameters differ from the calibrated ones: "
            + "; ".join(f"{k}: calibrated={a!r} now={b!r}" for k, (a, b) in drifted.items())
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-manifest", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--boundaries", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="", help="comma-separated subset (smoke runs)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frame_scores").mkdir(exist_ok=True)

    calibration = json.loads(Path(a.calibration).read_text())
    _check_frozen(calibration)
    threshold = float(calibration["threshold"]["value"])

    # Guard installed BEFORE any client audio is opened. Everything network
    # dependent had to happen during provisioning.
    guard = NetworkGuard().install()
    try:
        model_manifest = json.loads(Path(a.model_manifest).read_text())
        resolved = verify_manifest(Path(a.model_dir), load_manifest(Path(a.model_manifest)))

        import torch
        import torchaudio
        import soundfile as sf
        from sqa_eval.paderborn_model import load_model
        from sqa_eval.scoring import score_file

        model = load_model(resolved[model_manifest["inference_file"]])
        print(f"model loaded, network attempts so far: {len(guard.attempts)}", flush=True)

        boundaries = {}
        if a.boundaries:
            boundaries = json.loads(Path(a.boundaries).read_text())

        files = sorted(Path(a.audio_dir).glob("*.wav"))
        if a.files:
            wanted = {s.strip() for s in a.files.split(",") if s.strip()}
            missing = wanted - {p.name for p in files}
            if missing:
                raise SystemExit(f"requested files not present: {sorted(missing)}")
            files = [p for p in files if p.name in wanted]
        if a.limit:
            files = files[:a.limit]

        per_file, all_candidates = [], []
        t0 = time.time()
        for path in files:
            ft = time.time()
            res = score_file(model, path)
            scores, levels = res["scores"], res["levels_dbfs"]
            info = res["info"]
            np.save(out_dir / "frame_scores" / f"{path.stem}.npy", scores.astype(np.float32))

            speech_flags = (levels > FROZEN.speech_floor_dbfs).astype(np.float64)
            gaps = [tuple(g) for g in boundaries.get(path.name, {}).get("gap_spans", [])]
            cands = build_candidates(
                scores.tolist(), threshold=threshold, source_id=path.name,
                sample_rate=info["sample_rate"], total_samples=info["total_samples"],
                gap_spans=gaps, speech_content=speech_flags.tolist(),
            )
            all_candidates.extend(cands)
            below = scores < threshold
            per_file.append({
                **info,
                "layout": res["layout"],
                "frames": res["frames"], "chunks": res["chunks"],
                "frames_below_threshold": int(below.sum()),
                "fraction_below_threshold": float(below.mean()),
                "speech_frame_fraction": float(speech_flags.mean()),
                "score_min": float(scores.min()), "score_p1": float(np.percentile(scores, 1)),
                "score_median": float(np.median(scores)), "score_mean": float(scores.mean()),
                "score_max": float(scores.max()),
                "candidates": len(cands),
                "clip_count": boundaries.get(path.name, {}).get("clip_count"),
                "gap_spans": len(gaps),
                "seconds": info["duration_seconds"],
                "elapsed_seconds": round(time.time() - ft, 1),
            })
            print(f"  {path.name}: {res['frames']} frames, {len(cands)} candidates, "
                  f"{per_file[-1]['elapsed_seconds']}s", flush=True)

        ranked = rank_candidates(all_candidates)
        elapsed = time.time() - t0
        total_seconds = sum(f["seconds"] for f in per_file)

        (out_dir / "candidates.json").write_text(json.dumps(
            [{"rank": i + 1, **c.as_dict()} for i, c in enumerate(ranked)], indent=1))
        (out_dir / "per_file.json").write_text(json.dumps(per_file, indent=1))
        (out_dir / "run_meta.json").write_text(json.dumps({
            "model": {k: model_manifest[k] for k in
                      ("model_repo", "model_revision", "model_license",
                       "upstream_repo", "upstream_revision", "upstream_license",
                       "padertorch_revision", "paderbox_revision")},
            "model_files": model_manifest["files"],
            "threshold": threshold,
            "threshold_source": str(Path(a.calibration).name),
            "calibration_frozen_at": calibration.get("frozen_at"),
            "frozen_config": FROZEN.as_dict(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torchaudio": torchaudio.__version__, "numpy": np.__version__,
            "soundfile": sf.__version__, "device": "cpu",
            "machine": platform.machine(),
            "files_scanned": len(files),
            "files_selected": [p.name for p in files],
            "total_audio_seconds": round(total_seconds, 3),
            "total_audio_hours": round(total_seconds / 3600, 4),
            "total_frames": sum(f["frames"] for f in per_file),
            "candidates": len(ranked),
            "elapsed_seconds": round(elapsed, 1),
            "real_time_factor": round(total_seconds / elapsed, 2) if elapsed else None,
            "network_attempts": len(guard.attempts),
            "attempted_addresses": guard.attempts,
        }, indent=1))
        print(f"\nfiles={len(files)} frames={sum(f['frames'] for f in per_file)} "
              f"candidates={len(ranked)} network_attempts={len(guard.attempts)} "
              f"rtf={total_seconds/elapsed:.1f}x")
    finally:
        guard.uninstall()


if __name__ == "__main__":
    main()
