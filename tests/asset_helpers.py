from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.audio_cases import SR, write_audio

LEG_ORDERS = {
    "stereo": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Ls", "Rs", "Lss", "Rss"],
}


def mono_signal(value: float, *, frames: int = 16) -> np.ndarray:
    return np.full((frames, 1), value, dtype=np.float64)


def write_split_family(
    root: Path,
    stem_prefix: str,
    layout: str,
    *,
    values: list[float] | None = None,
    sr: int = SR,
    frames: int = 16,
    subtype: str = "PCM_24",
) -> dict[str, Path]:
    order = LEG_ORDERS[layout]
    values = values or [0.1 * (index + 1) for index in range(len(order))]
    assert len(values) == len(order)
    out: dict[str, Path] = {}
    for leg, value in zip(order, values, strict=True):
        out[leg] = write_audio(
            root / f"{stem_prefix}.{leg}.wav",
            mono_signal(float(value), frames=frames),
            sr=sr,
            subtype=subtype,
        )
    return out


def write_interleaved(
    path: Path,
    n_channels: int,
    *,
    frames: int = 16,
    sr: int = SR,
    subtype: str = "PCM_24",
) -> Path:
    cols = [np.full(frames, 0.1 * float(index + 1), dtype=np.float64) for index in range(n_channels)]
    data = np.column_stack(cols)
    return write_audio(path, data, sr=sr, subtype=subtype)


def write_split_from_array(
    root: Path,
    stem_prefix: str,
    layout: str,
    data: np.ndarray,
    *,
    sr: int = SR,
    subtype: str = "PCM_24",
) -> dict[str, Path]:
    order = LEG_ORDERS[layout]
    assert data.ndim == 2
    assert data.shape[1] == len(order)
    out: dict[str, Path] = {}
    for index, leg in enumerate(order):
        out[leg] = write_audio(
            root / f"{stem_prefix}.{leg}.wav",
            data[:, [index]],
            sr=sr,
            subtype=subtype,
        )
    return out
