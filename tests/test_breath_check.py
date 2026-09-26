"""`finalpass breaths`: detection, grading, T-inhale flag, H-sound filter, CLI.

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
    GRADE_2_FROM_DB, GRADE_3_FROM_DB, BreathTunables, analyze_breaths, grade_for, noticeability_db,
)
from finalpass.breath_edges import following_similarity, t_inhale_score
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
    model = json.loads((ROOT / "tools" / "breaths" / "t_inhale_model_v2.json").read_text())
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


def test_similarity_to_the_following_sound_separates_shared_resonances() -> None:
    shaped = butter(4, [500, 1500], btype="band", fs=SR, output="sos")
    same = sosfilt(shaped, RNG.standard_normal(int(0.4 * SR))) * 0.05
    after_same = sosfilt(shaped, RNG.standard_normal(int(0.2 * SR))) * 0.2
    after_other = _vowel(0.2, f0=220.0)
    event_len = int(0.2 * SR)
    x1 = np.concatenate([same[:event_len], after_same])
    x2 = np.concatenate([_band_noise(0.2, 2500, 4500, -26), after_other])
    s_same = following_similarity(x1, SR, 0, event_len, 0.0)
    s_other = following_similarity(x2, SR, 0, event_len, 0.0)
    assert s_same > 0.5 > s_other


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
