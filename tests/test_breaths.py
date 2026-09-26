"""Breath detector on synthetic sounds: finds a breath between phrases, ignores
word-attached sibilants, too-short bursts, and pitched speech."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from breaths.detect import BreathParams, detect  # noqa: E402
from breaths.features import compute  # noqa: E402

SR = 44100
RNG = np.random.default_rng(7)


def _vowel(seconds: float, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return 0.2 * x / np.max(np.abs(x))


def _band_noise(seconds: float, lo: float, hi: float, level_db: float) -> np.ndarray:
    n = RNG.standard_normal(int(seconds * SR))
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), n)
    ref = np.sqrt(np.mean(_vowel(0.5) ** 2))
    return y / np.sqrt(np.mean(y ** 2)) * ref * 10 ** (level_db / 20)


def _floor(seconds: float) -> np.ndarray:
    return RNG.standard_normal(int(seconds * SR)) * 1e-4


def _run(x: np.ndarray):
    return detect(compute(x, SR), BreathParams())[0]


def test_breath_between_phrases_is_found_where_it_is() -> None:
    breath = _band_noise(0.35, 900, 3200, -28)
    x = np.concatenate([_vowel(1.0), _floor(0.15), breath, _floor(0.12), _vowel(1.0)])
    found = _run(x)
    assert len(found) == 1
    start = int(1.15 * SR)
    assert abs(found[0].start_sample - start) < int(0.05 * SR)
    assert 0.25 <= found[0].duration_s <= 0.45


def test_sibilant_attached_to_a_vowel_is_not_a_breath() -> None:
    s = _band_noise(0.2, 5000, 9500, -18)
    x = np.concatenate([_vowel(1.0), s, _floor(0.3), _vowel(1.0)])
    assert _run(x) == []


def test_too_short_burst_is_ignored() -> None:
    blip = _band_noise(0.06, 900, 3200, -28)
    x = np.concatenate([_vowel(1.0), _floor(0.2), blip, _floor(0.2), _vowel(1.0)])
    assert _run(x) == []


def test_pitched_speech_alone_yields_nothing() -> None:
    assert _run(np.concatenate([_vowel(1.5), _floor(0.2), _vowel(1.5, f0=180.0)])) == []


def test_span_stops_before_the_next_word_even_when_it_follows_closely() -> None:
    """A 35 ms analysis window reaches ahead; the reported span must not."""
    breath = _band_noise(0.30, 900, 3200, -26)
    x = np.concatenate([_vowel(1.0), _floor(0.15), breath, _floor(0.03), _vowel(1.0)])
    (found,) = _run(x)
    next_word = int((1.0 + 0.15 + 0.30 + 0.03) * SR)
    assert found.end_sample <= next_word
    seg = x[found.start_sample:found.end_sample]
    # nothing inside the span is as loud as the vowel that follows it
    assert np.max(np.abs(seg)) < 0.5 * np.max(np.abs(_vowel(0.5)))


def test_quiet_lead_in_is_trimmed_from_the_span() -> None:
    lead = _band_noise(0.25, 900, 3200, -52)
    breath = _band_noise(0.30, 900, 3200, -26)
    x = np.concatenate([_vowel(1.0), _floor(0.1), lead, breath, _floor(0.15), _vowel(1.0)])
    (found,) = _run(x)
    breath_start = int((1.0 + 0.1 + 0.25) * SR)
    assert abs(found.start_sample - breath_start) < int(0.04 * SR)


def test_mouth_release_burst_scores_as_more_t_like_than_a_plain_breath() -> None:
    from breaths.onset import load_model, onset_features, t_inhale_score

    model = load_model()
    breath = _band_noise(0.30, 900, 3200, -26)
    click = np.zeros(int(0.004 * SR))
    click[: len(click) // 2] = 0.08
    click[len(click) // 2:] = -0.08               # a bright 4 ms release
    silence = np.zeros(int(0.35 * SR))
    plain = np.concatenate([_vowel(1.0), silence, breath, _floor(0.1), _vowel(0.5)])
    t_inh = np.concatenate([_vowel(1.0), silence, click, breath, _floor(0.1), _vowel(0.5)])
    start = int(1.35 * SR)
    level = 20 * np.log10(np.sqrt(np.mean(_vowel(0.5) ** 2)))
    f_plain = onset_features(plain, SR, start, start + len(breath), level)
    f_t = onset_features(t_inh, SR, start, start + len(click) + len(breath), level)
    assert f_t["burst_width_ms"] <= f_plain["burst_width_ms"]
    assert f_t["burst_rise_db"] > f_plain["burst_rise_db"]
    assert t_inhale_score(f_t, 0.35, model) > t_inhale_score(f_plain, 0.35, model)
