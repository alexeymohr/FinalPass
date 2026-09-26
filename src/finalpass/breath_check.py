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
short to judge. Events whose spectrum carries the following vowel's resonances
are dropped as likely "h" sounds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from pydantic import BaseModel, Field

from .audio_io import AudioFile
from .breath_detect import BreathParams, detect
from .breath_edges import T_INHALE_THRESHOLD, following_similarity, onset_features, t_inhale_score
from .breath_features import compute
from .errors import UnsupportedChannelConfigError
from .timecode import samples_to_clock

BREATHS_SCHEMA_VERSION = 1
DEFAULT_MIN_BREATH_MS = 150.0
DEFAULT_H_SIMILARITY = 0.10
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
    h_similarity: float = DEFAULT_H_SIMILARITY

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
    excluded_h_sound: int = 0

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


def analyze_breaths(audio: AudioFile, tunables: BreathTunables = BreathTunables()) -> BreathAssetResult:
    x, notes = _mono(audio)
    sr = audio.sample_rate
    params = BreathParams(min_body_s=tunables.min_breath_ms / 1000.0)
    found, level = detect(compute(x, sr), params)
    counts = BreathCounts()
    events: list[BreathEvent] = []
    for b in found:
        sim = following_similarity(x, sr, b.start_sample, b.end_sample, b.gap_after_s)
        if sim is not None and sim >= tunables.h_similarity:
            counts.excluded_h_sound += 1
            continue
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
    if c.excluded_h_sound:
        parts.append(f"{c.excluded_h_sound} likely \"h\" sounds excluded")
    return " · ".join(parts)


def render_breath_list(report: BreathsReport) -> str:
    """Plain-text list of every breath. Times are from the start of each file."""
    lines = [
        f"FinalPass {report.finalpass_version} — breath check — run {report.run_id}",
        "Grades: 1 very minor, 2 noticeable, 3 quite noticeable.",
        f"T-INHALE = breath opening with a mouth-release burst: {T_INHALE_NOTE}, not a failure.",
        "Informational only. Times are from the start of each file.",
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
