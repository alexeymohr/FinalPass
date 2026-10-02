"""`finalpass breaths`: detection, pause rule, grading, T-inhale flag, CLI.

Synthetic audio only.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from scipy.signal import butter, sosfilt

from finalpass.audio_io import AudioFile
from finalpass.breath_check import (
    GRADE_2_FROM_DB, GRADE_3_FROM_DB, BreathTunables, analyze_breaths, grade_for, noticeability_db, summary_line,
)
from finalpass.breath_detect import detect
from finalpass.breath_edges import click_gap, onset_features, t_inhale_score
from finalpass.breath_features import compute
from finalpass.breath_pause import breath_end
from finalpass.cli import main
from finalpass.errors import UnsupportedChannelConfigError
from finalpass.timecode import samples_to_clock

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
RNG = np.random.default_rng(11)


def _vowel(seconds: float, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return 0.2 * x / np.max(np.abs(x))


def _band_noise(seconds: float, lo: float, hi: float, level_db: float) -> np.ndarray:
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), RNG.standard_normal(int(seconds * SR)))
    ref = np.sqrt(np.mean(_vowel(0.5) ** 2))
    return y / np.sqrt(np.mean(y ** 2)) * ref * 10 ** (level_db / 20)


def _floor(seconds: float) -> np.ndarray:
    return RNG.standard_normal(int(seconds * SR)) * 1e-4


def _word() -> np.ndarray:
    """A word whose first 25 ms are voiced but soft, as real vowel onsets are."""
    return np.concatenate([_vowel(0.025) * 10 ** (-30 / 20), _vowel(1.0)])


def _narration(*breath_db: float) -> np.ndarray:
    parts = [_vowel(1.0)]
    for db in breath_db:
        parts += [_floor(0.3), _band_noise(0.35, 900, 3200, db), _floor(0.12), _vowel(1.0)]
    return np.concatenate(parts)


def _audio(x: np.ndarray, name: str = "narration.wav") -> AudioFile:
    data = x[:, None] if x.ndim == 1 else x
    return AudioFile(path=Path(name), data=data, sample_rate=SR, bit_depth=16,
                     channel_count=data.shape[1], duration_seconds=len(data) / SR)


# --- clock times and grading ---------------------------------------------


def test_clock_is_exact_and_file_relative() -> None:
    assert samples_to_clock(0, SR) == "0:00:00.000"
    assert samples_to_clock(SR * 3723 + SR // 2, SR) == "1:02:03.500"
    assert samples_to_clock(SR - 1, SR) == "0:00:00.999"   # truncated, never rounded up


def test_noticeability_adds_six_db_per_doubling_of_length() -> None:
    assert noticeability_db(-30.0, 0.25) == pytest.approx(-30.0)
    assert noticeability_db(-30.0, 0.50) == pytest.approx(-30.0 + 6.0206, abs=1e-3)


def test_grade_cut_points() -> None:
    assert grade_for(GRADE_2_FROM_DB - 0.01) == 1
    assert grade_for(GRADE_2_FROM_DB) == 2
    assert grade_for(GRADE_3_FROM_DB - 0.01) == 2
    assert grade_for(GRADE_3_FROM_DB) == 3


# --- the fitted T-inhale score --------------------------------------------


def test_embedded_t_inhale_score_reproduces_the_saved_model() -> None:
    model = json.loads((ROOT / "tools" / "breaths" / "t_inhale_model_v3.json").read_text())
    rng = np.random.default_rng(3)
    for _ in range(50):
        feats = {f: float(rng.normal(m, s)) for f, m, s in zip(model["features"], model["mean"], model["std"])}
        gap_s = float(rng.uniform(0.02, 1.0))
        vals = [np.log10(max(gap_s * 1000, 10)) if f == "gap_before_ms" else feats[f] for f in model["features"]]
        z = (np.array(vals) - np.array(model["mean"])) / np.array(model["std"])
        expected = model["intercept"] + z @ np.array(model["weights"])
        assert t_inhale_score(feats, gap_s) == pytest.approx(expected, abs=1e-4)


# --- analysis -------------------------------------------------------------


def test_breaths_are_found_and_louder_ones_grade_higher() -> None:
    result = analyze_breaths(_audio(_narration(-24, -38)))
    assert result.counts.breaths == 2
    loud, quiet = result.breaths
    assert loud.grade > quiet.grade
    assert loud.start_time == samples_to_clock(loud.start_sample, SR)


def test_short_events_are_not_reported() -> None:
    """The floor applies to the reported span — the length the reviewer heard."""
    x = np.concatenate([_vowel(1.0), _floor(0.3), _band_noise(0.09, 900, 3200, -24), _floor(0.12), _vowel(1.0)])
    assert analyze_breaths(_audio(x)).counts.breaths == 0
    for db in (-24, -30, -36):
        for e in analyze_breaths(_audio(_narration(db))).breaths:
            assert e.duration_ms >= 150


def test_dual_mono_is_accepted_and_discrete_stereo_refused() -> None:
    x = _narration(-26)
    dual = np.stack([x, x], axis=1)
    result = analyze_breaths(_audio(dual))
    assert result.counts.breaths == 1 and result.notes
    discrete = np.stack([x, np.roll(x, 100)], axis=1)
    with pytest.raises(UnsupportedChannelConfigError):
        analyze_breaths(_audio(discrete))


def test_t_inhale_flag_can_be_switched_off() -> None:
    result = analyze_breaths(_audio(_narration(-24)), BreathTunables(t_inhale=False))
    assert all(e.t_inhale is False and e.t_inhale_score is None for e in result.breaths)
    assert all(e.click_gap_samples is None and e.click_rel_db is None for e in result.breaths)


def test_every_breath_reports_its_click_gap_and_level() -> None:
    result = analyze_breaths(_audio(_narration(-24, -38)))
    assert result.breaths and all(
        e.click_gap_samples is not None and e.click_rel_db is not None for e in result.breaths)


def _click(amp: float) -> np.ndarray:
    c = np.zeros(int(0.004 * SR))
    c[: len(c) // 2], c[len(c) // 2:] = amp, -amp
    return c


def _click_gap(before: np.ndarray, amp: float = 0.08) -> dict:
    breath = _band_noise(0.30, 900, 3200, -26)
    level = 20 * np.log10(np.sqrt(np.mean(_vowel(0.5) ** 2)))
    start = len(before)
    click = _click(amp)
    x = np.concatenate([before, click, breath, _floor(0.1), _vowel(0.5)])
    return click_gap(x, SR, start, start + len(click) + len(breath), level)


def test_click_out_of_silence_has_a_gap_before_it() -> None:
    g = _click_gap(np.concatenate([_vowel(1.0), np.zeros(int(0.35 * SR))]))
    assert g["click_gap_samples"] >= int(0.029 * SR)          # the whole 30 ms search is quiet
    assert g["click_gap_ms"] == pytest.approx(g["click_gap_samples"] * 1000 / SR, abs=0.01)


def test_click_straight_out_of_a_vowel_has_no_gap() -> None:
    assert _click_gap(_vowel(1.0))["click_gap_samples"] == 0


def test_short_gap_is_measured_in_samples() -> None:
    gap = 300
    g = _click_gap(np.concatenate([_vowel(1.0), np.zeros(gap)]))
    assert abs(g["click_gap_samples"] - gap) <= 40              # 32-sample RMS smears the edges


def test_louder_click_reads_louder_against_the_narration() -> None:
    lead = np.concatenate([_vowel(1.0), np.zeros(int(0.35 * SR))])
    assert _click_gap(lead, 0.3)["click_rel_db"] > _click_gap(lead, 0.03)["click_rel_db"] + 15


def test_breath_end_needs_a_pause_and_cuts_what_follows_it() -> None:
    breath = _band_noise(0.30, 900, 3200, -26)
    pause = _band_noise(0.06, 900, 3200, -46)             # about -66 dBFS
    h = _band_noise(0.05, 900, 3200, -24)
    x = np.concatenate([breath, pause, h, _word()])
    b_end, h_end = len(breath), len(breath) + len(pause) + len(h)
    # the detected span runs through the pause into the "h": it ends at the pause
    assert abs(breath_end(x, SR, 0, h_end, h_end + int(0.03 * SR)) - b_end) < int(0.002 * SR)
    # a span that already stops before the pause is left alone
    assert breath_end(x, SR, 0, b_end, h_end + int(0.03 * SR)) == b_end
    # no stretch reaches -60 dBFS before the word: not a breath
    run_on = np.concatenate([breath, _band_noise(0.08, 900, 3200, -36), _word()])
    assert breath_end(run_on, SR, 0, int(0.38 * SR), int(0.4 * SR)) is None


def test_breath_running_straight_into_the_word_is_not_reported() -> None:
    x = np.concatenate([_vowel(1.0), _floor(0.3), _band_noise(0.30, 900, 3200, -26),
                        _band_noise(0.08, 900, 3200, -36), _word()])
    result = analyze_breaths(_audio(x))
    assert result.counts.breaths == 0 and result.counts.excluded_no_pause == 1


def test_h_after_the_pause_is_cut_off_the_breath() -> None:
    breath_end_at = int(1.6 * SR)
    x = np.concatenate([_vowel(1.0), _floor(0.3), _band_noise(0.30, 900, 3200, -26),
                        _band_noise(0.06, 900, 3200, -46), _band_noise(0.05, 900, 3200, -24), _word()])
    (raw,), _ = detect(compute(x, SR))
    assert raw.end_sample > breath_end_at + int(0.06 * SR)      # the detector alone runs on into the "h"
    (e,) = analyze_breaths(_audio(x)).breaths
    assert abs(e.end_sample - breath_end_at) < int(0.005 * SR)
    assert e.end_time == samples_to_clock(e.end_sample, SR)


def test_sibilant_as_loud_as_speech_before_a_pause_is_not_a_breath() -> None:
    """A dull "sh" left alone at a word's end, after a stop closure, passes every other rule."""
    sh = _band_noise(0.18, 2500, 3600, -10)
    closure = np.zeros(int(0.08 * SR))
    result = analyze_breaths(_audio(np.concatenate([_vowel(1.0), closure, sh, _floor(0.8), _vowel(1.0)])))
    assert result.counts.breaths == 0 and result.counts.excluded_as_loud_as_speech == 1
    assert "1 excluded (as loud as speech" in summary_line(result.counts)
    soft = sh * 10 ** (-16 / 20)                                  # the same sound at a breath's level is one
    x = np.concatenate([_vowel(1.0), closure, soft, _floor(0.8), _vowel(1.0)])
    assert analyze_breaths(_audio(x)).counts.breaths == 1


def test_breath_opening_with_a_loud_consonant_is_still_a_breath() -> None:
    x = np.concatenate([_vowel(1.0), _floor(0.3), _band_noise(0.06, 1500, 4000, -10),
                        _band_noise(0.30, 900, 3200, -27), _floor(0.3), _vowel(1.0)])
    result = analyze_breaths(_audio(x))
    assert result.counts.breaths == 1 and result.counts.excluded_as_loud_as_speech == 0


def test_low_thump_at_the_onset_is_measured() -> None:
    """A T-inhale's release is near full-band: a low thump as well as a click."""
    breath = _band_noise(0.30, 900, 3200, -26)
    click = np.zeros(int(0.004 * SR))
    click[: len(click) // 2], click[len(click) // 2:] = 0.08, -0.08
    t = np.arange(int(0.012 * SR)) / SR
    thump = 0.1 * np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.004)
    bright = sosfilt(butter(4, 3000, btype="high", fs=SR, output="sos"), click)
    level = 20 * np.log10(np.sqrt(np.mean(_vowel(0.5) ** 2)))
    lead = np.concatenate([_vowel(1.0), np.zeros(int(0.35 * SR))])
    start = len(lead)

    def feats(burst: np.ndarray) -> dict:
        burst = np.pad(burst, (0, max(0, len(thump) - len(burst))))
        x = np.concatenate([lead, burst, breath, _floor(0.1), _vowel(0.5)])
        return onset_features(x, SR, start, start + len(burst) + len(breath), level)

    with_thump, click_only = feats(np.pad(click, (0, len(thump) - len(click))) + thump), feats(bright)
    assert with_thump["burst_low_rise_db"] > click_only["burst_low_rise_db"] + 10


# --- CLI ------------------------------------------------------------------


def test_breaths_command_writes_a_text_list_and_json(tmp_path: Path) -> None:
    wav = tmp_path / "chapter.wav"
    sf.write(str(wav), _narration(-24, -38), SR, subtype="PCM_16")
    out = tmp_path / "out"
    r = CliRunner().invoke(main, ["breaths", "--out", str(out), str(wav)])
    assert r.exit_code == 0, r.output
    text = (out / "breaths.txt").read_text()
    assert "chapter.wav" in text and "grade" in text
    data = json.loads((out / "breaths-report.json").read_text())
    assert data["command"] == "breaths" and data["summary"]["breaths"] == 2


def test_breaths_json_only_emits_the_report(tmp_path: Path) -> None:
    wav = tmp_path / "chapter.wav"
    sf.write(str(wav), _narration(-24), SR, subtype="PCM_16")
    r = CliRunner().invoke(main, ["breaths", "--json-only", "--out", str(tmp_path / "o"), str(wav)])
    assert r.exit_code == 0
    assert json.loads(r.output)["summary"]["breaths"] == 1


def test_breaths_never_fails_a_run_even_with_t_inhales(tmp_path: Path) -> None:
    wav = tmp_path / "chapter.wav"
    sf.write(str(wav), _narration(-12, -12, -12), SR, subtype="PCM_16")
    r = CliRunner().invoke(main, ["breaths", "--out", str(tmp_path / "o"), str(wav)])
    assert r.exit_code == 0


def test_breath_modules_use_no_ml_stack() -> None:
    banned = {"torch", "torchaudio", "transformers", "padertorch", "paderbox", "sklearn"}
    for name in ("breath_features", "breath_detect", "breath_edges", "breath_check"):
        tree = ast.parse((ROOT / "src" / "finalpass" / f"{name}.py").read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            assert not (set(mods) & banned), (name, mods)


def test_breath_list_says_mouth_click_inhale_not_t_inhale(tmp_path: Path) -> None:
    wav = tmp_path / "chapter.wav"
    sf.write(str(wav), _narration(-24, -38), SR, subtype="PCM_16")
    for opts in ([], ["--list", "mouth-click"], ["--list", "t-inhale"], ["--no-mouth-click"], ["--no-t-inhale"]):
        r = CliRunner().invoke(main, ["breaths", "--out", str(tmp_path / "o"), *opts, str(wav)])
        assert r.exit_code == 0, r.output
        text = (tmp_path / "o" / "breaths.txt").read_text()
        assert "MOUTH-CLICK INHALE" in text and "T-INHALE" not in text and "T-inhale" not in text
        assert "T-inhale" not in r.output


def test_click_gap_measures_the_same_time_at_any_sample_rate() -> None:
    """A 7 ms gap of digital silence before the click reads about 7 ms at 22.05, 44.1 and 96 kHz."""
    from scipy.signal import resample_poly
    gaps = {}
    for sr in (22050, 44100, 96000):
        t = np.arange(int(0.6 * sr)) / sr
        word = 0.2 * np.sin(2 * np.pi * 150 * t)
        gap = np.zeros(int(0.007 * sr))
        c = np.zeros(max(2, int(0.004 * sr)))
        c[: len(c) // 2], c[len(c) // 2:] = 0.08, -0.08
        rng = np.random.default_rng(1)
        breath = sosfilt(butter(4, [900, 3200], btype="band", fs=sr, output="sos"), rng.standard_normal(int(0.3 * sr))) * 0.01
        x = np.concatenate([word, gap, c, breath, np.zeros(int(0.1 * sr))])
        start = len(word) + len(gap)
        g = click_gap(x, sr, start, start + len(c) + len(breath), -20.0)
        gaps[sr] = g["click_gap_ms"]
    assert max(gaps.values()) - min(gaps.values()) < 0.6, gaps
