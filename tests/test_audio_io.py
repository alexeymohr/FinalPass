from __future__ import annotations

from pathlib import Path

from finalpass.audio_io import probe_wav, read_wav
from tests.audio_cases import true_peak_over_program, write_audio


def test_probe_and_read_wav_capture_bext_time_reference(tmp_path: Path) -> None:
    path = write_audio(
        tmp_path / "with_bext.wav",
        true_peak_over_program(),
        time_reference_samples=168648480,
    )

    header = probe_wav(path)
    audio = read_wav(path)

    assert header.time_reference_samples == 168648480
    assert audio.time_reference_samples == 168648480
