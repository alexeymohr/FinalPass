"""Thin wrapper around ``soundfile`` that returns a typed ``AudioFile``.

Library code reads with ``dtype='float64'`` and ``always_2d=True`` so downstream
callers never have to branch on mono-vs-multichannel or integer-vs-float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict

from .errors import AudioFormatError

_SUBTYPE_BIT_DEPTH: dict[str, int] = {
    "PCM_U8": 8,
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}

_ALLOWED_EXTENSIONS = {".wav", ".bwf"}


class AudioFile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    path: Path
    data: np.ndarray  # shape (n_samples, n_channels), dtype float64
    sample_rate: int
    bit_depth: int
    channel_count: int
    duration_seconds: float


def read_wav(path: Path) -> AudioFile:
    path = Path(path)
    if not path.exists():
        raise AudioFormatError(f"{path} does not exist.")
    if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise AudioFormatError(
            f"{path} has extension '{path.suffix}'; FinalPass accepts "
            f"{sorted(_ALLOWED_EXTENSIONS)}."
        )

    try:
        with sf.SoundFile(str(path)) as f:
            subtype = f.subtype
            sr = f.samplerate
            channels = f.channels
        data, sr_check = sf.read(str(path), dtype="float64", always_2d=True)
    except RuntimeError as exc:
        raise AudioFormatError(f"Could not read {path}: {exc}") from exc

    if sr_check != sr:
        raise AudioFormatError(
            f"{path}: samplerate mismatch between header ({sr}) and read ({sr_check})."
        )

    bit_depth = _SUBTYPE_BIT_DEPTH.get(subtype, 0)
    duration = data.shape[0] / float(sr) if sr > 0 else 0.0

    return AudioFile(
        path=path.resolve(),
        data=data,
        sample_rate=sr,
        bit_depth=bit_depth,
        channel_count=channels,
        duration_seconds=duration,
    )
