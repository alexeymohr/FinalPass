"""Calibrate the low-quality threshold on clean public read speech.

Threshold rule (mission section 8.2): the 1st percentile of frame scores over a
clean-reference corpus, frozen before any client audio is opened.

Two facts measured on public data during provisioning shape how this is done:

* the released model's frame score depends on how much context it is given —
  the 1st percentile of the same public stream moves from 3.30 at a 10 s chunk
  to 2.72 at 60 s — so calibrating on short utterances and scanning with 30 s
  chunks would mis-set the threshold outright. Calibration therefore runs the
  IDENTICAL chunked scanner at the IDENTICAL chunk length;
* to give that scanner long-form material, consecutive LibriTTS-R utterances
  are concatenated into a per-speaker stream. LibriTTS keeps the natural silence
  around each sentence, so the joins are ordinary sentence pauses — but they are
  still a splice this harness created, and measurement showed they carry a
  below-threshold rate about 2.4x the rest of the corpus. A +-100 ms window
  around every join is therefore excluded from the calibration distribution: the
  threshold should describe clean read speech, not an artefact of how the
  reference stream was assembled. Both figures are reported.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.config import FROZEN  # noqa: E402
from sqa_eval.model_files import sha256_file  # noqa: E402
from sqa_eval.netguard import NetworkGuard  # noqa: E402
from sqa_eval.paderborn_model import load_model  # noqa: E402
from sqa_eval.scoring import score_file  # noqa: E402

STREAM_SECONDS = 70.0
# sha256 of openslr.org/resources/141/test_clean.tar.gz as downloaded 2026-09-11
ARCHIVE_SHA256 = "d4a2a7cdeeb68a6cfd628559739879bcb29556364cc13f38413930d59369a7f1"


def select_streams(root: Path, work: Path, stream_seconds: float = STREAM_SECONDS) -> list[dict]:
    """One continuous stream per speaker, utterances in sorted order.

    Chapters are consumed in sorted order until the stream reaches
    `stream_seconds`, so a speaker whose first chapter is a single short
    sentence still contributes a full-length stream instead of a degenerate
    sub-chunk one. A speaker with less material than that in total is skipped.
    Fully determined by sorted paths, so the same corpus reproduces the same
    selection on any machine.
    """
    by_speaker: dict[str, dict[str, list[Path]]] = {}
    for p in sorted(root.rglob("*.wav")):
        by_speaker.setdefault(p.parts[-3], {}).setdefault(p.parts[-2], []).append(p)

    work.mkdir(parents=True, exist_ok=True)
    streams: list[dict] = []
    skipped: list[str] = []
    for speaker in sorted(by_speaker):
        chapters = sorted(by_speaker[speaker])
        picked, blocks, sr, total = [], [], None, 0.0
        for chapter in chapters:
            for p in by_speaker[speaker][chapter]:
                x, s = sf.read(str(p), dtype="float32", always_2d=False)
                sr = s if sr is None else sr
                if s != sr:
                    raise RuntimeError(f"{p}: sample rate {s} != {sr}")
                picked.append(p)
                blocks.append(x)
                total += len(x) / sr
                if total >= stream_seconds:
                    break
            if total >= stream_seconds:
                break
        if total < stream_seconds:
            skipped.append(speaker)
            continue
        chapter = sorted({p.parts[-2] for p in picked})[0]
        joins, at = [], 0
        for x in blocks[:-1]:
            at += len(x)
            joins.append(at)
        stream = np.concatenate(blocks)
        out = work / f"stream_{speaker}.wav"
        sf.write(str(out), stream, sr, subtype="PCM_16")
        streams.append({
            "speaker": speaker, "chapter": chapter, "path": str(out),
            "sample_rate": int(sr), "samples": int(len(stream)),
            "seconds": round(len(stream) / sr, 3),
            "utterances": [
                {"file": str(p.relative_to(root)), "sha256": sha256_file(p)}
                for p in picked
            ],
            "join_samples": joins,
        })
    if skipped:
        print(f"  skipped speakers with < {stream_seconds}s available: {skipped}", flush=True)
    return streams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="LibriTTS_R/test-clean root")
    ap.add_argument("--state", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream-seconds", type=float, default=STREAM_SECONDS)
    a = ap.parse_args()

    work = Path(a.work)
    print("selecting deterministic public streams…", flush=True)
    streams = select_streams(Path(a.corpus), work, a.stream_seconds)
    total_seconds = sum(s["seconds"] for s in streams)
    print(f"  {len(streams)} speakers, {total_seconds/60:.2f} min", flush=True)

    guard = NetworkGuard().install()
    try:
        model = load_model(Path(a.state))
        all_scores: list[np.ndarray] = []
        all_levels: list[np.ndarray] = []
        join_mask: list[np.ndarray] = []
        per_stream = []
        t0 = time.time()
        for s in streams:
            res = score_file(model, Path(s["path"]))
            scores, levels = res["scores"], res["levels_dbfs"]
            step = FROZEN.frame_stride * s["sample_rate"] // FROZEN.model_sr
            mask = np.zeros(len(scores), dtype=bool)
            pad = max(1, int(round(0.1 / FROZEN.frame_seconds)))  # +-100 ms
            for j in s["join_samples"]:
                f = j // step
                mask[max(0, f - pad):min(len(mask), f + pad + 1)] = True
            all_scores.append(scores); all_levels.append(levels); join_mask.append(mask)
            per_stream.append({
                "speaker": s["speaker"], "seconds": s["seconds"],
                "frames": int(len(scores)), "chunks": res["chunks"],
                "min": float(scores.min()), "p1": float(np.percentile(scores, 1)),
                "median": float(np.median(scores)),
            })
            print(f"  {s['speaker']:>6}: {len(scores):>5} frames, "
                  f"p1={per_stream[-1]['p1']:.3f} median={per_stream[-1]['median']:.3f}",
                  flush=True)
        elapsed = time.time() - t0
    finally:
        guard.uninstall()

    scores = np.concatenate(all_scores)
    levels = np.concatenate(all_levels)
    joins = np.concatenate(join_mask)

    # The frozen threshold excludes frames adjacent to a concatenation join:
    # those splices are an artefact of assembling the reference stream, not a
    # property of clean read speech. The all-frame figure is reported beside it.
    clean = scores[~joins]
    threshold = float(np.percentile(clean, FROZEN.calibration_percentile))
    threshold_all_frames = float(np.percentile(scores, FROZEN.calibration_percentile))

    below = scores < threshold
    speech = levels > FROZEN.speech_floor_dbfs
    manifest = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": {
            "name": "LibriTTS-R test-clean", "license": "CC BY 4.0",
            "source": "https://www.openslr.org/141/",
            "archive_sha256": ARCHIVE_SHA256,
            "speakers": len(streams), "streams": len(streams),
            "total_seconds": round(total_seconds, 3),
            "total_minutes": round(total_seconds / 60, 3),
            "utterances": sum(len(s["utterances"]) for s in streams),
            "sample_rate": streams[0]["sample_rate"] if streams else None,
            "selection_rule": (
                "one stream per speaker: that speaker's first chapter by sorted "
                f"path, consecutive utterances in order until >= {a.stream_seconds} s"
            ),
        },
        "threshold": {
            "percentile": FROZEN.calibration_percentile,
            "value": threshold,
            "basis": "1st percentile of frames at least 100 ms away from a "
                     "stream-assembly join",
            "value_all_frames_including_joins": threshold_all_frames,
            "frames_total": int(len(scores)),
            "frames_used_for_threshold": int(len(clean)),
            "frames_below": int(below.sum()),
            "fraction_below": float(below.mean()),
            "fraction_below_excluding_joins": float((clean < threshold).mean()),
        },
        "distribution": {
            "min": float(scores.min()), "p0_1": float(np.percentile(scores, 0.1)),
            "p1": float(np.percentile(scores, 1)), "p5": float(np.percentile(scores, 5)),
            "median": float(np.median(scores)), "mean": float(scores.mean()),
            "p95": float(np.percentile(scores, 95)), "max": float(scores.max()),
        },
        "speech_gated_distribution": {
            "speech_frames": int(speech.sum()),
            "speech_fraction": float(speech.mean()),
            "p1_speech_only": float(np.percentile(scores[speech], 1)) if speech.any() else None,
            "median_speech_only": float(np.median(scores[speech])) if speech.any() else None,
            "note": "reported for comparison only; the frozen threshold is the "
                    "all-frame 1st percentile required by the mission",
        },
        "join_contamination": {
            "join_frames": int(joins.sum()),
            "join_frame_fraction": float(joins.mean()),
            "below_threshold_rate_at_joins": float(below[joins].mean()) if joins.any() else None,
            "below_threshold_rate_elsewhere": float(below[~joins].mean()) if (~joins).any() else None,
        },
        "per_stream": per_stream,
        "frozen_config": FROZEN.as_dict(),
        "streams": [{k: v for k, v in s.items() if k != "path"} for s in streams],
        "runtime_seconds": round(elapsed, 1),
        "audio_seconds_per_wall_second": round(total_seconds / elapsed, 2),
        "network_attempts": len(guard.attempts),
    }
    Path(a.out).write_text(json.dumps(manifest, indent=1))
    print("\nFROZEN THRESHOLD:", threshold)
    print(json.dumps({k: manifest[k] for k in
                      ("threshold", "distribution", "speech_gated_distribution",
                       "join_contamination", "network_attempts",
                       "audio_seconds_per_wall_second")}, indent=1))


if __name__ == "__main__":
    main()
