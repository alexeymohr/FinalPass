"""Thin wrapper around ``soundfile`` that returns a typed ``AudioFile``.

Library code reads with ``dtype='float64'`` and ``always_2d=True`` so downstream
callers never have to branch on mono-vs-multichannel or integer-vs-float.
"""

from __future__ import annotations

from dataclasses import dataclass
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
_CHANNEL_CONFIG_FROM_COUNT: dict[int, str] = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


class AudioFile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    path: Path
    data: np.ndarray  # shape (n_samples, n_channels), dtype float64
    sample_rate: int
    bit_depth: int
    channel_count: int
    duration_seconds: float

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True)
class AudioHeader:
    path: Path
    sample_rate: int
    sample_count: int
    bit_depth: int | None
    subtype: str | None
    channel_count: int
    duration_seconds: float


def channel_config_from_count(n_channels: int) -> str | None:
    """Map a raw channel count to the supported FinalPass channel label."""
    return _CHANNEL_CONFIG_FROM_COUNT.get(n_channels)


def probe_wav(path: Path) -> AudioHeader:
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
            frames = len(f)
            duration = frames / float(sr) if sr > 0 else 0.0
    except RuntimeError as exc:
        raise AudioFormatError(f"Could not read {path}: {exc}") from exc
    return AudioHeader(
        path=path.resolve(),
        sample_rate=sr,
        sample_count=frames,
        bit_depth=_SUBTYPE_BIT_DEPTH.get(subtype),
        subtype=subtype,
        channel_count=channels,
        duration_seconds=duration,
    )


def read_wav(path: Path) -> AudioFile:
    header = probe_wav(path)
    try:
        data, sr_check = sf.read(str(header.path), dtype="float64", always_2d=True)
    except RuntimeError as exc:
        raise AudioFormatError(f"Could not read {header.path}: {exc}") from exc

    if sr_check != header.sample_rate:
        raise AudioFormatError(
            f"{header.path}: samplerate mismatch between header ({header.sample_rate}) and read ({sr_check})."
        )

    return AudioFile(
        path=header.path,
        data=data,
        sample_rate=header.sample_rate,
        bit_depth=header.bit_depth or 0,
        channel_count=header.channel_count,
        duration_seconds=header.duration_seconds,
    )
