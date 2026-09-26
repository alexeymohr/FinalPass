"""The before/after QC diff must recover known edits to the exact sample."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from qc_diff import diff_file  # noqa: E402

RNG = np.random.default_rng(1)


def _speech(n: int) -> np.ndarray:
    return (RNG.standard_normal(n) * 3000).astype(np.int16)


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, np.int16)


def test_identical_files_have_no_edits() -> None:
    x = np.concatenate([_speech(50000), _zeros(8000), _speech(40000)])
    edits, stats = diff_file(x, x.copy())
    assert edits == []
    assert stats["matched_fraction_of_before"] == 1.0


def test_known_edits_are_recovered_exactly() -> None:
    c1, c2, c3, c4 = _speech(200000), _speech(150000), _speech(180000), _speech(120000)
    gap = _zeros(20000)
    before = np.concatenate([c1, gap, c2, gap, c3, gap, c4])
    breath = (RNG.standard_normal(3000) * 200).astype(np.int16)
    c2b = np.concatenate([c2[:60000], breath, c2[60000:]])
    before = np.concatenate([c1, gap, c2b, gap, c3, gap, c4])
    new3 = _speech(175000)
    after = np.concatenate([
        c1, gap, _zeros(15000),                         # pause added
        c2b[:60000], _zeros(3000), c2b[63000:],         # breath cut to silence
        gap, new3,                                      # block regenerated
        gap, c4,
    ])
    edits, _ = diff_file(before, after)
    kinds = [e.kind for e in edits]
    assert kinds == ["pause_added", "audio_to_silence", "audio_rerendered"]

    added, cut, regen = edits
    assert (added.before_start, added.before_end) == (220000, 220000)
    assert added.after_end - added.after_start == 15000
    assert (cut.before_start, cut.before_end) == (280000, 283000)
    assert cut.after_end - cut.after_start == 3000
    assert regen.before_start == 220000 + 153000 + 20000
    assert regen.before_end - regen.before_start == 180000


def test_diff_accounts_for_every_sample() -> None:
    a1, a2 = _speech(90000), _speech(70000)
    before = np.concatenate([a1, _zeros(10000), a2])
    after = np.concatenate([a1, _zeros(4000), a2])
    edits, stats = diff_file(before, after)
    removed = sum(e.before_end - e.before_start for e in edits)
    added = sum(e.after_end - e.after_start for e in edits)
    assert stats["matched_samples"] + removed == len(before)
    assert stats["matched_samples"] + added == len(after)
    assert [e.kind for e in edits] == ["pause_shortened"]
