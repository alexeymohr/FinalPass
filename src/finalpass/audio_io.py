"""Thin wrapper around ``soundfile`` that returns a typed ``AudioFile``.

Library code reads with ``dtype='float64'`` and ``always_2d=True`` so downstream
callers never have to branch on mono-vs-multichannel or integer-vs-float.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

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
_RIFF_HEADERS = {b"RIFF", b"RF64"}
_WAVE_HEADER = b"WAVE"
_BEXT_CHUNK_ID = b"bext"
_BEXT_TIME_REFERENCE_OFFSET = 338
_BEXT_TIME_REFERENCE_BYTES = 8


class AudioFile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    path: Path
    data: np.ndarray  # shape (n_samples, n_channels), dtype float64
    sample_rate: int
    bit_depth: int
    channel_count: int
    duration_seconds: float
    time_reference_samples: int | None = None

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
    time_reference_samples: int | None = None


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
    time_reference_samples = _read_bext_time_reference_samples(path)
    return AudioHeader(
        path=path.resolve(),
        sample_rate=sr,
        sample_count=frames,
        bit_depth=_SUBTYPE_BIT_DEPTH.get(subtype),
        subtype=subtype,
        channel_count=channels,
        duration_seconds=duration,
        time_reference_samples=time_reference_samples,
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
        time_reference_samples=header.time_reference_samples,
    )


def _read_bext_time_reference_samples(path: Path) -> int | None:
    """Return the BWF bext time reference in samples, if present."""
    try:
        with Path(path).open("rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] not in _RIFF_HEADERS or header[8:12] != _WAVE_HEADER:
                return None

            while True:
                chunk_header = handle.read(8)
                if len(chunk_header) < 8:
                    return None
                chunk_id, size = struct.unpack("<4sI", chunk_header)
                if chunk_id == _BEXT_CHUNK_ID:
                    chunk_data = handle.read(size)
                    if len(chunk_data) < _BEXT_TIME_REFERENCE_OFFSET + _BEXT_TIME_REFERENCE_BYTES:
                        return None
                    low, high = struct.unpack(
                        "<II",
                        chunk_data[
                            _BEXT_TIME_REFERENCE_OFFSET :
                            _BEXT_TIME_REFERENCE_OFFSET + _BEXT_TIME_REFERENCE_BYTES
                        ],
                    )
                    return low + (high << 32)
                handle.seek(size + (size % 2), 1)
    except OSError:
        return None
