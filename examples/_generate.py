"""Generate a tiny two-episode delivery folder for manual `finalpass all` runs.

Run from the repo root::

    uv run python examples/_generate.py

Outputs go to ``examples/delivery_two_episodes/`` which is gitignored — the
script is deterministic (fixed seeds), so the audio regenerates identically
on every run. E03 is calibrated to pass ``ebu_r128``; E04's PM is hot-printed
to fail true-peak, which is exactly what the verification checklist exercises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
SEED = 0xC0DE
SECONDS = 10.0


def _pink(n_samples: int, n_channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_samples, n_channels))
    pink = np.zeros_like(white)
    pink[0] = 0.05 * white[0]
    for i in range(1, n_samples):
        pink[i] = 0.99 * pink[i - 1] + 0.05 * white[i]
    rms = float(np.sqrt(np.mean(pink ** 2)))
    if rms > 0:
        pink *= (10 ** (-23.0 / 20.0)) / rms
    return pink.astype(np.float64)


def _write(path: Path, seed: int, gain_db: float = 0.0) -> None:
    n = int(round(SECONDS * SR))
    data = _pink(n, 2, seed) * (10 ** (gain_db / 20.0))
    data = np.clip(data, -0.999, 0.999)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SR, subtype="PCM_24")


def main() -> None:
    root = Path(__file__).parent / "delivery_two_episodes"
    files = [
        ("SHOW_S01E03_PM_STEREO.wav", SEED + 0, 0.0),
        ("SHOW_S01E03_DX_STEREO.wav", SEED + 1, 0.0),
        ("SHOW_S01E03_MX_STEREO.wav", SEED + 2, 0.0),
        ("SHOW_S01E03_FX_STEREO.wav", SEED + 3, 0.0),
        # E04 PM is printed 15 dB hot → fails EBU R128 integrated + TP.
        ("SHOW_S01E04_PM_STEREO.wav", SEED + 4, 15.0),
        ("SHOW_S01E04_DX_STEREO.wav", SEED + 5, 0.0),
        ("SHOW_S01E04_MX_STEREO.wav", SEED + 6, 0.0),
        ("SHOW_S01E04_FX_STEREO.wav", SEED + 7, 0.0),
    ]
    for name, seed, gain in files:
        _write(root / name, seed=seed, gain_db=gain)
    print(f"Wrote {len(files)} files to {root}")


if __name__ == "__main__":
    main()
