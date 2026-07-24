from __future__ import annotations

from pathlib import Path

from finalpass.audio_io import read_wav
from finalpass.loudness import measure, true_peak_over_flags
from finalpass.timecode import timecode_mode
from tests.audio_cases import true_peak_over_program, write_audio


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


def test_true_peak_over_flags_localize_short_burst(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "tp_over.wav", true_peak_over_program())
    audio = read_wav(path)
    flags = true_peak_over_flags(audio, threshold_dbtp=-1.0, mode=timecode_mode(23.976))
    assert len(flags) == 1
    flag = flags[0]
    assert flag.code == "LOUDNESS"
    assert flag.metric == "true_peak_dbtp"
    assert 4.99 <= flag.start_sample / audio.sample_rate <= 5.01
    assert flag.end_sample > flag.start_sample
    assert flag.value > flag.threshold
    assert "over by" in flag.detail


def test_true_peak_over_flags_use_embedded_start_timecode(tmp_path: Path) -> None:
    path = write_audio(
        tmp_path / "tp_over_bext.wav",
        true_peak_over_program(),
        time_reference_samples=168648480,
    )
    audio = read_wav(path)
    flags = true_peak_over_flags(audio, threshold_dbtp=-1.0, mode=timecode_mode(23.976))

    assert len(flags) == 1
    flag = flags[0]
    assert flag.start_tc.startswith("00:58:35:")
    assert flag.end_tc.startswith("00:58:35:")
