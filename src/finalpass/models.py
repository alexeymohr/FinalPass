"""Pydantic v2 models for the on-disk JSON reports.

Phase 1's ``loudness`` command emits schema v1; Phase 2's ``all`` command emits
schema v3. Kept separate from :mod:`finalpass.specs` so the spec input schema
and the report output schemas can evolve independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SpecSource = Literal["bundled", "user", "explicit"]
# "primary" is the Phase 1 role used by the `loudness` command when the caller
# doesn't classify. Phase 2's `all` command uses the full classifier role set.
FileRole = Literal["primary", "dx", "pm", "mx", "fx", "me", "opt", "unknown"]


class SpecRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    source: SpecSource


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    lra: float | None = None
    short_term_max_lufs: float | None = None
    momentary_max_lufs: float | None = None


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metric: str
    target: float | None = None
    tolerance: float | None = None
    limit: float | None = None
    measured: float | None
    # `pass` is None when the check was skipped (not measurable on this input).
    # Skipped checks do not count toward the pass/fail tally in Summary.
    pass_: bool | None = Field(alias="pass")
    skipped: bool = False
    reason: str | None = None
    error: str | None = None


class FileReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    role: FileRole
    sample_rate: int
    bit_depth: int
    channel_count: int
    duration_seconds: float
    measurements: Measurements
    checks: list[CheckResult]
    errors: list[str] = Field(default_factory=list)
    # Channel-config observability. Populated by the classifier (hint, from
    # filename tokens) and from the audio header (actual, from channel count).
    # Phase 2 surfaces these for consumers and for Phase 3's hint-vs-actual
    # cross-check. They are deliberately excluded from the Phase 1 `loudness`
    # JSON output so schema_version:1 stays shape-stable.
    channel_config_hint: str | None = None
    channel_config_actual: str | None = None


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_checks: int
    passed: int
    failed: int
    skipped: int = 0
    overall_pass: bool


class Report(BaseModel):
    """Phase 1 `loudness` command report — `schema_version == 1`.

    Kept deliberately separate from :class:`AllReport` (Phase 2) so Phase 1
    JSON shape never gains Phase 2-only fields.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    finalpass_version: str
    schema_version: int
    run_id: str
    run_started_at: str
    spec: SpecRef
    fps: float
    files: list[FileReport]
    summary: Summary


class GroupError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    message: str


class GroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_checks: int
    passed: int
    failed: int
    skipped: int = 0
    overall_pass: bool


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    files: list[FileReport]
    group_summary: GroupSummary
    errors: list[GroupError] = Field(default_factory=list)


class UnclassifiedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    reason: str


class AllSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groups_total: int
    groups_passed: int
    groups_failed: int
    total_checks: int
    passed: int
    failed: int
    skipped: int = 0
    overall_pass: bool


class AllReport(BaseModel):
    """Phase 2 `all` command report — `schema_version == 3`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    finalpass_version: str
    schema_version: int
    run_id: str
    run_started_at: str
    spec: SpecRef
    fps: float
    command: Literal["all"]
    folder: str
    groups: list[Group]
    files: list[FileReport]
    unclassified: list[UnclassifiedEntry]
    summary: AllSummary
