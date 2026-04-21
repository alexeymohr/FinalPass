from __future__ import annotations

from pathlib import Path

from finalpass.audio_io import read_wav
from finalpass.loudness import measure


def test_pink_stereo_10s_integrated_is_finite(pink_stereo_10s: Path) -> None:
    audio = read_wav(pink_stereo_10s)
    m = measure(audio)
    assert m.integrated_lufs is not None
    assert m.true_peak_dbtp is not None
    # Synthesized at ~-23 dBFS RMS; K-weighted integrated loudness for pink
    # noise lands near that. Wide tolerance — we just want it in the right zone.
    assert -30.0 < m.integrated_lufs < -15.0
    assert m.momentary_max_lufs is not None
    assert m.short_term_max_lufs is not None


def test_silent_file_reports_silent_or_below_gate(silent_stereo_5s: Path) -> None:
    audio = read_wav(silent_stereo_5s)
    m = measure(audio)
    assert m.integrated_lufs is None
    assert "silent_or_below_gate" in m.errors
    # True peak of silence is undefined; reported as None.
    assert m.true_peak_dbtp is None


def test_short_file_blocks_integrated(short_stereo_1s: Path) -> None:
    audio = read_wav(short_stereo_1s)
    m = measure(audio)
    assert m.integrated_lufs is None
    assert "file_too_short_for_integrated" in m.errors
    # A 1s signal still supports 400 ms momentary windows.
    assert m.momentary_max_lufs is not None
    # But short-term (3 s) requires more data.
    assert m.short_term_max_lufs is None


def test_clipped_sine_reports_true_peak_above_zero(clipped_stereo_5s: Path) -> None:
    audio = read_wav(clipped_stereo_5s)
    m = measure(audio)
    assert m.true_peak_dbtp is not None
    # Sample peak is exactly 0 dBFS (clipped to ±1). 4× oversampling reveals
    # inter-sample peaks: at 997 Hz a clipped square gives >0 dBTP.
    assert m.true_peak_dbtp > 0.0


def test_mono_pink_integrated_finite(mono_10s: Path) -> None:
    audio = read_wav(mono_10s)
    m = measure(audio)
    assert m.integrated_lufs is not None
    assert m.true_peak_dbtp is not None
