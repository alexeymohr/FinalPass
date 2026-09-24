"""Prove the detector runs with no network, and fails closed if it tries.

Everything here runs in a FRESH process with the network guard installed before
torch is imported, so any lazy download attempt — Torch Hub, Hugging Face,
telemetry — raises instead of succeeding quietly.

Checks, in order:

1. model files resolve from explicit local paths and match their pinned hashes;
2. the model loads under `weights_only=True` with no allowlisted globals;
3. it scores synthetic audio and the frame count matches the derived mapping;
4. a deliberate outbound connection is refused by the guard;
5. the torchaudio pretrained-bundle path — the one that would fetch
   `wav2vec2_fairseq_base_ls960.pth` over Torch Hub — is refused too;
6. nothing the load or the scoring did attempted a connection.
"""
from __future__ import annotations

import argparse, json, socket, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqa_eval.netguard import NetworkAccessDenied, NetworkGuard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    checks: dict[str, object] = {}
    guard = NetworkGuard().install()
    try:
        from sqa_eval.model_files import load_manifest, verify_manifest

        manifest = json.loads(Path(a.manifest).read_text())
        resolved = verify_manifest(Path(a.model_dir), load_manifest(Path(a.manifest)))
        checks["model_files_verified"] = sorted(resolved)
        checks["attempts_after_file_verification"] = len(guard.attempts)

        import numpy as np
        import torch
        from sqa_eval.paderborn_model import load_model, num_frames_for, prepare_waveform

        checks["torch_version"] = torch.__version__
        t0 = time.time()
        model = load_model(resolved[manifest["inference_file"]])
        checks["model_loaded_weights_only_no_allowlist"] = True
        checks["model_load_seconds"] = round(time.time() - t0, 2)
        checks["attempts_after_model_load"] = len(guard.attempts)

        rng = np.random.default_rng(20260911)
        wav = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
        scores = model.frame_scores(
            torch.from_numpy(np.ascontiguousarray(prepare_waveform(wav))).float()
        )
        checks["frames_expected"] = num_frames_for(len(wav))
        checks["frames_returned"] = int(len(scores))
        checks["frame_count_matches"] = checks["frames_expected"] == checks["frames_returned"]
        checks["score_range"] = [round(float(scores.min()), 4), round(float(scores.max()), 4)]
        checks["utterance_level_mean"] = round(float(scores.mean()), 4)
        checks["attempts_after_scoring"] = len(guard.attempts)

        # 4. deliberate outbound attempt must raise
        try:
            socket.create_connection(("huggingface.co", 443), timeout=2)
            checks["deliberate_connection_refused"] = False
        except NetworkAccessDenied:
            checks["deliberate_connection_refused"] = True
        except Exception as exc:  # pragma: no cover - guard should fire first
            checks["deliberate_connection_refused"] = f"other error: {type(exc).__name__}"

        # 5. the Torch Hub download trap must also be refused
        try:
            import torchaudio
            torchaudio.pipelines.WAV2VEC2_BASE.get_model()
            checks["torch_hub_download_refused"] = False
        except NetworkAccessDenied:
            checks["torch_hub_download_refused"] = True
        except Exception as exc:
            name = type(exc).__name__
            cause = exc
            refused = False
            while cause is not None:
                if isinstance(cause, NetworkAccessDenied):
                    refused = True
                    break
                cause = cause.__cause__ or cause.__context__
            checks["torch_hub_download_refused"] = True if refused else f"other error: {name}"

        checks["total_network_attempts"] = len(guard.attempts)
        checks["attempted_addresses"] = guard.attempts
    finally:
        guard.uninstall()

    checks["pass"] = bool(
        checks.get("frame_count_matches")
        and checks.get("model_loaded_weights_only_no_allowlist")
        and checks.get("deliberate_connection_refused") is True
        and checks.get("torch_hub_download_refused") is True
        and checks.get("attempts_after_scoring") == 0
    )
    Path(a.out).write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    sys.exit(0 if checks["pass"] else 1)


if __name__ == "__main__":
    main()
