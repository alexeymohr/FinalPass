"""Fixture synthesis. No committed audio — every WAV is generated per session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 48000
RNG_SEED = 0xFA57


def _write(path: Path, data: np.ndarray, sr: int = SR, subtype: str = "PCM_24") -> Path:
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def _pink_noise(n_samples: int, n_channels: int, seed: int = RNG_SEED) -> np.ndarray:
    """Approximate pink noise via Voss-McCartney-style 1/f filtering of white."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_samples, n_channels))
    # Simple 1st-order 1/f IIR: y[n] = 0.99 * y[n-1] + 0.05 * x[n]
    pink = np.zeros_like(white)
    pink[0] = 0.05 * white[0]
    for i in range(1, n_samples):
        pink[i] = 0.99 * pink[i - 1] + 0.05 * white[i]
    # Normalize to target RMS ~ -23 dBFS (gives roughly -23 LUFS after K-weighting
    # for this spectrum — we re-calibrate in the test).
    rms = float(np.sqrt(np.mean(pink ** 2)))
    if rms > 0:
        target = 10 ** (-23.0 / 20.0)
        pink = pink * (target / rms)
    return pink.astype(np.float64)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("finalpass_fixtures")


@pytest.fixture(scope="session")
def pink_stereo_10s(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "pink_stereo_10s.wav"
    data = _pink_noise(10 * SR, 2)
    return _write(path, data)


@pytest.fixture(scope="session")
def pink_stereo_10s_hot(fixtures_dir: Path) -> Path:
    """Same pink noise, scaled up 15 dB so it fails Netflix/EBU loudness + TP."""
    path = fixtures_dir / "pink_stereo_10s_hot.wav"
    data = _pink_noise(10 * SR, 2) * (10 ** (15.0 / 20.0))
    data = np.clip(data, -0.999, 0.999)
    return _write(path, data)


@pytest.fixture(scope="session")
def silent_stereo_5s(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "silent_stereo_5s.wav"
    data = np.zeros((5 * SR, 2), dtype=np.float64)
    return _write(path, data)


@pytest.fixture(scope="session")
def short_stereo_1s(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "short_stereo_1s.wav"
    t = np.arange(SR) / SR
    sine = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    data = np.stack([sine, sine], axis=1).astype(np.float64)
    return _write(path, data)


@pytest.fixture(scope="session")
def clipped_stereo_5s(fixtures_dir: Path) -> Path:
    """Full-scale square-ish wave at 997 Hz — sample peak == 1.0, TP > 0 dBTP."""
    path = fixtures_dir / "clipped_stereo_5s.wav"
    t = np.arange(5 * SR) / SR
    sine = np.sin(2 * np.pi * 997.0 * t)
    clipped = np.clip(sine * 2.0, -1.0, 1.0)
    data = np.stack([clipped, clipped], axis=1).astype(np.float64)
    return _write(path, data)


@pytest.fixture(scope="session")
def mono_10s(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "mono_10s.wav"
    data = _pink_noise(10 * SR, 1)
    return _write(path, data)
