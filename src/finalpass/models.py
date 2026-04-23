"""Pydantic v2 models for the on-disk JSON reports.

SM-2 bumps standalone reports to schema v2 so they can persist split-mono
provenance, while the integrated ``all`` command stays at schema v5. Kept
separate from :mod:`finalpass.specs` so the spec input schema and the report
output schemas can evolve independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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


class FlaggedRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal["NULL", "ME", "LOUDNESS"]
    metric: Literal["residual_rms_dbfs", "dialog_bleed_score", "true_peak_dbtp"]
    value: float
    threshold: float
    start_sample: int
    end_sample: int
    start_tc: str
    end_tc: str
    duration_seconds: float
    detail: str


class FileReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _time_reference_samples: int | None = PrivateAttr(default=None)

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
    source_kind: Literal["interleaved", "split_mono"] | None = None
    source_paths: list[str] = Field(default_factory=list)
    member_legs: list[str] = Field(default_factory=list)
    presentation_label: str | None = None
    flags: list[FlaggedRegion] = Field(default_factory=list)


class StandaloneFileReport(FileReport):
    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["interleaved", "split_mono"]
    source_paths: list[str]
    member_legs: list[str]
    presentation_label: str | None = None


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_checks: int
    passed: int
    failed: int
    skipped: int = 0
    overall_pass: bool


class Report(BaseModel):
    """SM-2 `loudness` command report — `schema_version == 3`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    finalpass_version: str
    schema_version: int
    run_id: str
    run_started_at: str
    spec: SpecRef
    fps: float
    files: list[StandaloneFileReport]
    summary: Summary


class GroupError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    message: str


class NullSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    windows_total: int
    windows_flagged: int
    flagged_regions: int
    max_residual_rms_dbfs: float
    mean_residual_rms_dbfs: float


class NullTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    _time_reference_samples: int | None = PrivateAttr(default=None)

    pass_: bool | None = Field(alias="pass")
    skipped: bool = False
    reason: str | None = None
    window_ms: float
    hop_ms: float
    threshold_dbfs: float
    summary: NullSummary | None = None
    flags: list[FlaggedRegion] = Field(default_factory=list)
    errors: list[GroupError] = Field(default_factory=list)


class AutoNullTestResult(NullTestResult):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stem_strategy: Literal["dx_mx_fx", "dx_me"] | None = None
    selected_roles: list[Literal["dx", "mx", "fx", "me"]] = Field(default_factory=list)


class AnalysisInputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _time_reference_samples: int | None = PrivateAttr(default=None)

    path: str
    sample_rate: int
    bit_depth: int
    channel_count: int
    channel_config_actual: str | None = None
    duration_seconds: float
    source_kind: Literal["interleaved", "split_mono"]
    source_paths: list[str]
    member_legs: list[str]
    presentation_label: str | None = None


# Back-compat alias for the existing standalone `null` path.
NullInputFile = AnalysisInputFile


class MECheckSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    windows_total: int
    windows_gated_out: int
    windows_analyzed: int
    windows_flagged: int
    flagged_regions: int
    max_dialog_bleed_score: float
    max_corr_abs: float
    max_coherence_mean: float


class MECheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    _time_reference_samples: int | None = PrivateAttr(default=None)

    pass_: bool | None = Field(alias="pass")
    skipped: bool = False
    reason: str | None = None
    analysis_signal: Literal["mono_downmix_excluding_lfe"]
    window_ms: float
    hop_ms: float
    band_low_hz: float
    band_high_hz: float
    corr_threshold: float
    coherence_threshold: float
    dx_gate_dbfs: float
    me_floor_dbfs: float
    summary: MECheckSummary | None = None
    flags: list[FlaggedRegion] = Field(default_factory=list)
    errors: list[GroupError] = Field(default_factory=list)


class GroupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_checks: int
    passed: int
    failed: int
    skipped: int = 0
    overall_pass: bool


class AssetInventoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    role: FileRole
    path: str
    source_kind: Literal["interleaved", "split_mono"]
    source_paths: list[str]
    member_legs: list[str]
    presentation_label: str | None = None
    channel_config_actual: str
    channel_config_hint: str | None = None
    used_by: list[str] = Field(default_factory=list)
    selection_note: str | None = None


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    assets: list[AssetInventoryEntry] = Field(default_factory=list)
    files: list[FileReport]
    null_test: AutoNullTestResult | None = None
    me_check: MECheckResult | None = None
    group_summary: GroupSummary
    errors: list[GroupError] = Field(default_factory=list)


class UnclassifiedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    source_kind: Literal["interleaved", "split_mono"]
    source_paths: list[str]
    member_legs: list[str] = Field(default_factory=list)
    presentation_label: str | None = None
    reason: str


class DiscoveryIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    source_paths: list[str]
    error_type: str
    message: str
    family_key: str | None = None
    group_hint: str | None = None


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
    """SM-3 `all` command report — `schema_version == 7`."""

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
    discovery_errors: list[DiscoveryIssue] = Field(default_factory=list)
    summary: AllSummary


class NullReport(BaseModel):
    """SM-2 standalone `null` command report — `schema_version == 2`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    finalpass_version: str
    schema_version: int
    command: Literal["null"]
    run_id: str
    run_started_at: str
    fps: float
    printmaster: AnalysisInputFile
    stems: list[AnalysisInputFile]
    null_test: NullTestResult
    summary: Summary


class MEReport(BaseModel):
    """SM-2 standalone `me` command report — `schema_version == 2`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    finalpass_version: str
    schema_version: int
    command: Literal["me"]
    run_id: str
    run_started_at: str
    fps: float
    me_file: AnalysisInputFile
    dx_file: AnalysisInputFile
    me_check: MECheckResult
    summary: Summary
