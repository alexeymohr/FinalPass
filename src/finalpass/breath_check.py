"""Audiobook breath check: find, grade and flag breaths in narration.

Informational only — nothing here fails a delivery. For each breath:

* **grade** 1 very minor / 2 noticeable / 3 quite noticeable, from a
  noticeability score = the breath's median level relative to the narration
  plus 6 dB per doubling of its length (re 250 ms). The cut points were fitted
  to an experienced re-recording mixer's grades (43 breaths: leave-one-out
  exact 86 %, never more than one grade off);
* **T-inhale**: the breath opens with a mouth-release burst. Some clients' QC
  rejects these, so they are reported as a *client QC risk*, not a defect.

Events shorter than 150 ms are not reported: the reviewing mixer found them too
short to judge. A breath must also fade into a pause (about -60 dBFS for 10 ms)
before the next word; a sound that runs straight into the word — an "h", or a
consonant — is not reported, and an "h" after the pause is cut off the breath.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from pydantic import BaseModel, Field

from .audio_io import AudioFile
from .breath_detect import Breath, BreathParams, detect, frame_centre, shorten, speech_mask
from .breath_edges import T_INHALE_THRESHOLD, onset_features, t_inhale_score
from .breath_features import Frames, compute
from .breath_pause import DEFAULT_PAUSE_DBFS, DEFAULT_PAUSE_MIN_MS, breath_end
from .errors import UnsupportedChannelConfigError
from .timecode import samples_to_clock

BREATHS_SCHEMA_VERSION = 2
DEFAULT_MIN_BREATH_MS = 150.0
GRADE_2_FROM_DB = -31.6
GRADE_3_FROM_DB = -24.2
GRADE_REF_S = 0.25
GRADE_LABELS = {1: "very minor", 2: "noticeable", 3: "quite noticeable"}
T_INHALE_NOTE = "client QC risk"
DUAL_MONO_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BreathTunables:
    min_breath_ms: float = DEFAULT_MIN_BREATH_MS
    t_inhale: bool = True
    pause_dbfs: float = DEFAULT_PAUSE_DBFS
    pause_min_ms: float = DEFAULT_PAUSE_MIN_MS

    def as_dict(self) -> dict:
        return asdict(self)


class BreathEvent(BaseModel):
    start_sample: int
    end_sample: int
    start_time: str
    end_time: str
    duration_ms: int
    peak_db: float = Field(description="peak level relative to the narration, dB")
    body_db: float = Field(description="median level relative to the narration, dB")
    noticeability_db: float
    grade: int
    t_inhale: bool
    t_inhale_score: float | None
    note: str = ""


class BreathCounts(BaseModel):
    breaths: int = 0
    grade_1: int = 0
    grade_2: int = 0
    grade_3: int = 0
    t_inhale: int = 0
    excluded_no_pause: int = 0

    def add(self, other: "BreathCounts") -> None:
        for name in type(self).model_fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))


class BreathAssetResult(BaseModel):
    path: str
    sample_rate: int
    duration_seconds: float
    narration_dbfs: float | None
    counts: BreathCounts
    breaths: list[BreathEvent]
    notes: list[str] = []


class BreathsReport(BaseModel):
    finalpass_version: str
    schema_version: int
    command: str = "breaths"
    run_id: str
    run_started_at: str
    tunables: dict
    assets: list[BreathAssetResult]
    summary: BreathCounts


def noticeability_db(body_db: float, duration_s: float) -> float:
    return body_db + 20.0 * math.log10(max(duration_s, 1e-3) / GRADE_REF_S)


def grade_for(score_db: float) -> int:
    if score_db >= GRADE_3_FROM_DB:
        return 3
    return 2 if score_db >= GRADE_2_FROM_DB else 1


def _mono(audio: AudioFile) -> tuple[np.ndarray, list[str]]:
    data = audio.data
    if data.ndim == 1 or data.shape[1] == 1:
        return (data if data.ndim == 1 else data[:, 0]), []
    if data.shape[1] == 2 and float(np.max(np.abs(data[:, 0] - data[:, 1]))) <= DUAL_MONO_TOLERANCE:
        return data[:, 0], ["dual-mono stereo: analysed the left channel"]
    raise UnsupportedChannelConfigError(
        f"{audio.path.name}: breath check needs mono narration "
        f"(got {data.shape[1]} channels that differ)"
    )


def followed_by_pause(x: np.ndarray, f: Frames, level: float, found: list[Breath],
                      tunables: BreathTunables) -> tuple[list[Breath], int]:
    """Keep the breaths that fade into a pause, each ending where its pause begins."""
    speech_idx = np.flatnonzero(speech_mask(f, level))
    kept, dropped = [], 0
    for b in found:
        k = np.searchsorted(speech_idx, b.start_frame)
        nxt = int(speech_idx[k]) * f.hop + frame_centre(f) if k < len(speech_idx) else len(x)
        end = breath_end(x, f.sample_rate, b.start_sample, b.end_sample, nxt,
                         tunables.pause_dbfs, tunables.pause_min_ms)
        if end is None or (end - b.start_sample) * 1000 < tunables.min_breath_ms * f.sample_rate:
            dropped += 1
            continue
        kept.append(b if end == b.end_sample else shorten(b, end, f, level))
    return kept, dropped


def analyze_breaths(audio: AudioFile, tunables: BreathTunables = BreathTunables()) -> BreathAssetResult:
    x, notes = _mono(audio)
    sr = audio.sample_rate
    params = BreathParams(min_body_s=tunables.min_breath_ms / 1000.0)
    frames = compute(x, sr)
    found, level = detect(frames, params)
    counts = BreathCounts()
    events: list[BreathEvent] = []
    if found:
        found, counts.excluded_no_pause = followed_by_pause(x, frames, level, found, tunables)
    for b in found:
        feats = onset_features(x, sr, b.start_sample, b.end_sample, level)
        score = round(t_inhale_score(feats, b.gap_before_s), 3) if feats else None
        is_t = bool(tunables.t_inhale and score is not None and score >= T_INHALE_THRESHOLD)
        n_db = noticeability_db(b.median_rel_db, b.duration_s)
        g = grade_for(n_db)
        counts.breaths += 1
        setattr(counts, f"grade_{g}", getattr(counts, f"grade_{g}") + 1)
        counts.t_inhale += is_t
        events.append(BreathEvent(
            start_sample=b.start_sample, end_sample=b.end_sample,
            start_time=samples_to_clock(b.start_sample, sr),
            end_time=samples_to_clock(b.end_sample, sr),
            duration_ms=int(round(b.duration_s * 1000)),
            peak_db=round(b.peak_rel_db, 2), body_db=round(b.median_rel_db, 2),
            noticeability_db=round(n_db, 2), grade=g,
            t_inhale=is_t, t_inhale_score=score if tunables.t_inhale else None,
            note=T_INHALE_NOTE if is_t else "",
        ))
    return BreathAssetResult(
        path=str(audio.path), sample_rate=sr, duration_seconds=audio.duration_seconds,
        narration_dbfs=round(level, 2) if np.isfinite(level) else None,
        counts=counts, breaths=events, notes=notes,
    )


def summary_line(c: BreathCounts) -> str:
    parts = [f"{c.breaths} breaths",
             f"grades {c.grade_1} / {c.grade_2} / {c.grade_3}",
             f"{c.t_inhale} T-inhale ({T_INHALE_NOTE})"]
    if c.excluded_no_pause:
        parts.append(f"{c.excluded_no_pause} excluded (no pause before the next word)")
    return " · ".join(parts)


def render_breath_list(report: BreathsReport) -> str:
    """Plain-text list of every breath. Times are from the start of each file."""
    lines = [
        f"FinalPass {report.finalpass_version} — breath check — run {report.run_id}",
        "Grades: 1 very minor, 2 noticeable, 3 quite noticeable.",
        f"T-INHALE = breath opening with a mouth-release burst: {T_INHALE_NOTE}, not a failure.",
        "Informational only. Times are from the start of each file.",
        "A breath must fade into a pause before the next word; sounds that run",
        "straight into the word (an \"h\", a consonant) are not listed.",
        "",
        f"All files: {summary_line(report.summary)}",
    ]
    for asset in report.assets:
        name = asset.path.rsplit("/", 1)[-1]
        lines += ["", f"== {name} — {summary_line(asset.counts)}"]
        lines += [f"   note: {n}" for n in asset.notes]
        for e in asset.breaths:
            tag = "  T-INHALE (client QC risk)" if e.t_inhale else ""
            lines.append(f"   {e.start_time}  {e.duration_ms:>4} ms  grade {e.grade}{tag}")
    return "\n".join(lines) + "\n"
