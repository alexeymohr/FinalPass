"""Re-score a deterministic subset and require bit-identical agreement.

Two runs of the same file through the same frozen configuration must produce
the same frame scores, or nothing downstream — candidates, ranking, markers —
can be reproduced. Also records peak resident memory and throughput for the
performance section of the report.
"""
from __future__ import annotations

import argparse, json, resource, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.model_files import load_manifest, verify_manifest  # noqa: E402
from sqa_eval.netguard import NetworkGuard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-manifest", required=True)
    ap.add_argument("--reference", required=True, help="frame_scores dir from the scan")
    ap.add_argument("--files", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    guard = NetworkGuard().install()
    try:
        manifest = json.loads(Path(a.model_manifest).read_text())
        resolved = verify_manifest(Path(a.model_dir), load_manifest(Path(a.model_manifest)))
        from sqa_eval.paderborn_model import load_model
        from sqa_eval.scoring import score_file

        model = load_model(resolved[manifest["inference_file"]])
        rows, audio_seconds, elapsed = [], 0.0, 0.0
        for name in a.files.split(","):
            name = name.strip()
            if not name:
                continue
            path = Path(a.audio_dir) / name
            t0 = time.time()
            res = score_file(model, path)
            dt = time.time() - t0
            elapsed += dt
            audio_seconds += res["info"]["duration_seconds"]
            ref = np.load(Path(a.reference) / f"{path.stem}.npy").astype(np.float64)
            got = res["scores"]
            diff = np.abs(ref - got)
            rows.append({
                "file": name, "frames": int(len(got)),
                "max_abs_diff_vs_scan": float(diff.max()),
                "bit_identical": bool(np.array_equal(ref.astype(np.float32),
                                                     got.astype(np.float32))),
                "seconds": res["info"]["duration_seconds"],
                "elapsed_seconds": round(dt, 2),
            })
            print(f"  {name}: max|diff|={diff.max():.3e} "
                  f"identical={rows[-1]['bit_identical']}", flush=True)
        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        guard.uninstall()

    result = {
        "files": rows,
        "all_bit_identical": all(r["bit_identical"] for r in rows),
        "worst_abs_diff": max(r["max_abs_diff_vs_scan"] for r in rows) if rows else None,
        "peak_rss_mb": round(peak_bytes / (1024 * 1024), 1),
        "audio_seconds": round(audio_seconds, 2),
        "elapsed_seconds": round(elapsed, 2),
        "real_time_factor": round(audio_seconds / elapsed, 2) if elapsed else None,
        "frames_per_second": round(sum(r["frames"] for r in rows) / elapsed, 1) if elapsed else None,
        "network_attempts": len(guard.attempts),
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=1))
    sys.exit(0 if result["all_bit_identical"] else 1)


if __name__ == "__main__":
    main()
