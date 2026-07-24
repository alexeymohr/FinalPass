from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import secrets
import tempfile

from . import (
    ALL_SCHEMA_VERSION,
    LOUDNESS_SCHEMA_VERSION,
    ME_SCHEMA_VERSION,
    NULL_SCHEMA_VERSION,
    __version__,
)
from . import aaf_export
from .artifact_naming import derive_program_name_from_report
from .all_assets import (
    AssetFolderScan,
    ClassifiedLogicalAsset,
    discover_assets_from_paths,
    discover_folder_assets,
    read_classified_audio,
)
from .audio_io import AudioFile
from .classify import load_config
from .errors import (
    AAFExportError,
    AlignmentError,
    AudioFormatError,
    ChannelConfigLabelMismatch,
    ChannelMismatchError,
    FinalPassError,
    NoAudioFilesError,
    ProgramWindowError,
    SampleCountMismatchError,
    SampleRateMismatchError,
    UnsupportedChannelConfigError,
)
from .loudness import check, measure, true_peak_over_flags
from .models import (
    AllReport,
    AllSummary,
    AssetInventoryEntry,
    AutoNullTestResult,
    DiscoveryIssue,
    FileReport,
    FileRole,
    Group,
    GroupError,
    GroupSummary,
    MECheckResult,
    MEReport,
    NullReport,
    NullTestResult,
    Report,
    SpecRef,
    StandaloneFileReport,
    Summary,
    UnclassifiedEntry,
)
from .me_check import (
    ANALYSIS_SIGNAL_NAME,
    analyze_me,
)
from .null_test import analyze_null
from .prep_folders import PrepBucketHint
from .report import render_report_html
from .specs import BundledSpecFamily, CHANNEL_COUNTS, Spec, load_spec
from .standalone_ingest import describe_analysis_input, resolve_standalone_asset
from .timecode import frame_rate_info


@dataclass(frozen=True)
class WrittenArtifacts:
    json_path: Path
    html_path: Path
    aaf_path: Path | None


def run_loudness(*, files: tuple[Path, ...], spec_name: str, dx_file: Path | None, fps: float, drop_frame: bool = False) -> Report:
    _validate_fps(fps, drop_frame=drop_frame)
    spec, source, _ = load_spec(spec_name)

    primary_inputs = [resolve_standalone_asset(f) for f in files]
    dx_input = resolve_standalone_asset(dx_file) if dx_file is not None else None

    primary_audio = [resolved.audio for resolved in primary_inputs]
    dx_audio = dx_input.audio if dx_input is not None else None

    _validate_channels(primary_audio, spec)
    if dx_audio is not None:
        _validate_channels([dx_audio], spec)

    _validate_homogeneous_sample_rate([*primary_audio, *([dx_audio] if dx_audio else [])])

    file_reports: list[StandaloneFileReport] = []
    for resolved in primary_inputs:
        audio = resolved.audio
        m = measure(audio)
        cs = check(m, spec, role="primary")
        flags = _timed_loudness_flags(audio=audio, checks=cs, fps=fps, drop_frame=drop_frame)
        file_reports.append(_attach_time_reference(
            StandaloneFileReport(
                path=str(resolved.logical_asset.canonical_path),
                role="primary",
                sample_rate=audio.sample_rate,
                bit_depth=audio.bit_depth,
                channel_count=audio.channel_count,
                duration_seconds=round(audio.duration_seconds, 3),
                measurements=m.as_measurements(),
                checks=cs,
                errors=m.errors,
                channel_config_hint=resolved.logical_asset.channel_config_hint,
                channel_config_actual=resolved.logical_asset.channel_config_actual,
                source_kind=resolved.logical_asset.source_kind,
                source_paths=[str(path) for path in resolved.logical_asset.source_paths],
                member_legs=list(resolved.logical_asset.member_legs),
                presentation_label=resolved.logical_asset.presentation_label,
                flags=flags,
            ),
            audio.time_reference_samples,
        ))

    if dx_input is not None:
        dx_audio = dx_input.audio
        m = measure(dx_audio)
        cs = check(m, spec, role="dx")
        flags = _timed_loudness_flags(audio=dx_audio, checks=cs, fps=fps, drop_frame=drop_frame)
        file_reports.append(_attach_time_reference(
            StandaloneFileReport(
                path=str(dx_input.logical_asset.canonical_path),
                role="dx",
                sample_rate=dx_audio.sample_rate,
                bit_depth=dx_audio.bit_depth,
                channel_count=dx_audio.channel_count,
                duration_seconds=round(dx_audio.duration_seconds, 3),
                measurements=m.as_measurements(),
                checks=cs,
                errors=m.errors,
                channel_config_hint=dx_input.logical_asset.channel_config_hint,
                channel_config_actual=dx_input.logical_asset.channel_config_actual,
                source_kind=dx_input.logical_asset.source_kind,
                source_paths=[str(path) for path in dx_input.logical_asset.source_paths],
                member_legs=list(dx_input.logical_asset.member_legs),
                presentation_label=dx_input.logical_asset.presentation_label,
                flags=flags,
            ),
            dx_audio.time_reference_samples,
        ))

    all_checks = [c for fr in file_reports for c in fr.checks]
    passed = sum(1 for c in all_checks if c.pass_ is True)
    failed = sum(1 for c in all_checks if c.pass_ is False)
    skipped = sum(1 for c in all_checks if c.skipped)
    summary = Summary(
        total_checks=len(all_checks),
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=failed == 0,
    )

    now, run_id = _run_timestamp()
    return Report(
        finalpass_version=__version__,
        schema_version=LOUDNESS_SCHEMA_VERSION,
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        spec=SpecRef(name=spec.name, display_name=spec.display_name, source=source),
        fps=fps,
        drop_frame=drop_frame,
        files=file_reports,
        summary=summary,
    )


def run_null(
    *,
    pm: Path,
    stems: tuple[Path, ...],
    fps: float,
    drop_frame: bool = False,
    window_ms: float,
    hop_ms: float,
    threshold_dbfs: float,
) -> NullReport:
    _validate_fps(fps, drop_frame=drop_frame)
    pm_input = resolve_standalone_asset(pm)
    stem_inputs = [resolve_standalone_asset(stem) for stem in stems]
    pm_audio = pm_input.audio
    stem_audios = [resolved.audio for resolved in stem_inputs]
    analysis = analyze_null(
        pm_audio,
        stem_audios,
        fps=fps,
        drop_frame=drop_frame,
        time_reference_samples=pm_audio.time_reference_samples,
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold_dbfs=threshold_dbfs,
    )
    passed = len(analysis.flags) == 0
    null_test = _attach_time_reference(
        NullTestResult(
            **{"pass": passed},
            skipped=False,
            reason=None,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
            summary=analysis.summary,
            analysis_window=analysis.analysis_window,
            flags=analysis.flags,
            errors=[],
        ),
        pm_audio.time_reference_samples,
    )
    now, run_id = _run_timestamp()
    return NullReport(
        finalpass_version=__version__,
        schema_version=NULL_SCHEMA_VERSION,
        command="null",
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        fps=fps,
        drop_frame=drop_frame,
        printmaster=describe_analysis_input(pm_input),
        stems=[describe_analysis_input(resolved) for resolved in stem_inputs],
        null_test=null_test,
        summary=Summary(
            total_checks=1,
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            skipped=0,
            overall_pass=passed,
        ),
    )


def run_me(
    *,
    me_file: Path,
    dx_file: Path,
    fps: float,
    drop_frame: bool = False,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> MEReport:
    _validate_fps(fps, drop_frame=drop_frame)
    me_input = resolve_standalone_asset(me_file)
    dx_input = resolve_standalone_asset(dx_file)
    me_audio = me_input.audio
    dx_audio = dx_input.audio
    analysis = analyze_me(
        me_audio,
        dx_audio,
        fps=fps,
        drop_frame=drop_frame,
        time_reference_samples=me_audio.time_reference_samples,
        window_ms=window_ms,
        hop_ms=hop_ms,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        corr_threshold=corr_threshold,
        coherence_threshold=coherence_threshold,
        dx_gate_dbfs=dx_gate_dbfs,
        me_floor_dbfs=me_floor_dbfs,
    )
    passed = len(analysis.flags) == 0
    me_check = _attach_time_reference(
        MECheckResult(
            **{"pass": passed},
            skipped=False,
            reason=None,
            analysis_signal=ANALYSIS_SIGNAL_NAME,
            window_ms=window_ms,
            hop_ms=hop_ms,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            corr_threshold=corr_threshold,
            coherence_threshold=coherence_threshold,
            dx_gate_dbfs=dx_gate_dbfs,
            me_floor_dbfs=me_floor_dbfs,
            summary=analysis.summary,
            analysis_window=analysis.analysis_window,
            flags=analysis.flags,
            errors=[],
        ),
        me_audio.time_reference_samples,
    )
    now, run_id = _run_timestamp()
    return MEReport(
        finalpass_version=__version__,
        schema_version=ME_SCHEMA_VERSION,
        command="me",
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        fps=fps,
        drop_frame=drop_frame,
        me_file=describe_analysis_input(me_input),
        dx_file=describe_analysis_input(dx_input),
        me_check=me_check,
        summary=Summary(
            total_checks=1,
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            skipped=0,
            overall_pass=passed,
        ),
    )


def run_all(
    *,
    folder: Path,
    spec_name: str,
    patterns_path: Path | None,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    spec, source, _ = load_spec(spec_name)
    cfg = load_config(patterns_path)
    scan: AssetFolderScan = discover_folder_assets(folder, cfg)
    return run_all_from_scan(
        folder=folder,
        scan=scan,
        spec=spec,
        source=source,
        include_unclassified=include_unclassified,
        fps=fps,
        drop_frame=drop_frame,
        null_window_ms=null_window_ms,
        null_hop_ms=null_hop_ms,
        null_threshold_dbfs=null_threshold_dbfs,
        me_window_ms=me_window_ms,
        me_hop_ms=me_hop_ms,
        me_band_low_hz=me_band_low_hz,
        me_band_high_hz=me_band_high_hz,
        me_corr_threshold=me_corr_threshold,
        me_coherence_threshold=me_coherence_threshold,
        me_dx_gate_dbfs=me_dx_gate_dbfs,
        me_me_floor_dbfs=me_me_floor_dbfs,
    )


def run_all_filtered(
    *,
    folder: Path,
    paths: tuple[Path, ...],
    path_hints: dict[Path, PrepBucketHint],
    spec_name: str,
    patterns_path: Path | None,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    spec, source, _ = load_spec(spec_name)
    cfg = load_config(patterns_path)
    scan = discover_assets_from_paths(paths, cfg, path_hints=path_hints)
    return run_all_from_scan(
        folder=folder,
        scan=scan,
        spec=spec,
        source=source,
        include_unclassified=include_unclassified,
        fps=fps,
        drop_frame=drop_frame,
        null_window_ms=null_window_ms,
        null_hop_ms=null_hop_ms,
        null_threshold_dbfs=null_threshold_dbfs,
        me_window_ms=me_window_ms,
        me_hop_ms=me_hop_ms,
        me_band_low_hz=me_band_low_hz,
        me_band_high_hz=me_band_high_hz,
        me_corr_threshold=me_corr_threshold,
        me_coherence_threshold=me_coherence_threshold,
        me_dx_gate_dbfs=me_dx_gate_dbfs,
        me_me_floor_dbfs=me_me_floor_dbfs,
    )


def run_all_with_spec_family(
    *,
    folder: Path,
    family: BundledSpecFamily,
    patterns_path: Path | None,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    cfg = load_config(patterns_path)
    scan: AssetFolderScan = discover_folder_assets(folder, cfg)
    return run_all_family_from_scan(
        folder=folder,
        scan=scan,
        family=family,
        include_unclassified=include_unclassified,
        fps=fps,
        drop_frame=drop_frame,
        null_window_ms=null_window_ms,
        null_hop_ms=null_hop_ms,
        null_threshold_dbfs=null_threshold_dbfs,
        me_window_ms=me_window_ms,
        me_hop_ms=me_hop_ms,
        me_band_low_hz=me_band_low_hz,
        me_band_high_hz=me_band_high_hz,
        me_corr_threshold=me_corr_threshold,
        me_coherence_threshold=me_coherence_threshold,
        me_dx_gate_dbfs=me_dx_gate_dbfs,
        me_me_floor_dbfs=me_me_floor_dbfs,
    )


def run_all_filtered_with_spec_family(
    *,
    folder: Path,
    paths: tuple[Path, ...],
    path_hints: dict[Path, PrepBucketHint],
    family: BundledSpecFamily,
    patterns_path: Path | None,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    cfg = load_config(patterns_path)
    scan = discover_assets_from_paths(paths, cfg, path_hints=path_hints)
    return run_all_family_from_scan(
        folder=folder,
        scan=scan,
        family=family,
        include_unclassified=include_unclassified,
        fps=fps,
        drop_frame=drop_frame,
        null_window_ms=null_window_ms,
        null_hop_ms=null_hop_ms,
        null_threshold_dbfs=null_threshold_dbfs,
        me_window_ms=me_window_ms,
        me_hop_ms=me_hop_ms,
        me_band_low_hz=me_band_low_hz,
        me_band_high_hz=me_band_high_hz,
        me_corr_threshold=me_corr_threshold,
        me_coherence_threshold=me_coherence_threshold,
        me_dx_gate_dbfs=me_dx_gate_dbfs,
        me_me_floor_dbfs=me_me_floor_dbfs,
    )


def run_all_from_scan(
    *,
    folder: Path,
    scan: AssetFolderScan,
    spec: Spec,
    source: str,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    if not scan.groups and not scan.discovery_errors and not scan.unclassified:
        raise NoAudioFilesError(
            f"No WAV/BWF files found in {folder}. "
            "Provide a folder containing at least one .wav file."
        )

    groups_out: list[Group] = []
    flat_files: list[FileReport] = []

    for group_id, classified_assets in scan.groups.items():
        group = _process_group(
            group_id=group_id,
            classified_assets=classified_assets,
            spec=spec,
            include_unclassified=include_unclassified,
            fps=fps,
            drop_frame=drop_frame,
            null_window_ms=null_window_ms,
            null_hop_ms=null_hop_ms,
            null_threshold_dbfs=null_threshold_dbfs,
            me_window_ms=me_window_ms,
            me_hop_ms=me_hop_ms,
            me_band_low_hz=me_band_low_hz,
            me_band_high_hz=me_band_high_hz,
            me_corr_threshold=me_corr_threshold,
            me_coherence_threshold=me_coherence_threshold,
            me_dx_gate_dbfs=me_dx_gate_dbfs,
            me_me_floor_dbfs=me_me_floor_dbfs,
        )
        groups_out.append(group)
        flat_files.extend(group.files)

    unclassified = [
        _unclassified_entry(asset, reason="no_role_pattern_match")
        for asset in scan.unclassified
    ]
    discovery_errors = [
        _discovery_issue(error)
        for error in scan.discovery_errors
    ]

    total_checks = sum(g.group_summary.total_checks for g in groups_out)
    passed = sum(g.group_summary.passed for g in groups_out)
    failed = sum(g.group_summary.failed for g in groups_out)
    skipped = sum(g.group_summary.skipped for g in groups_out)
    groups_passed = sum(1 for g in groups_out if g.group_summary.overall_pass)
    groups_failed = len(groups_out) - groups_passed
    overall_pass = failed == 0 and groups_failed == 0 and not discovery_errors

    summary = AllSummary(
        groups_total=len(groups_out),
        groups_passed=groups_passed,
        groups_failed=groups_failed,
        total_checks=total_checks,
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=overall_pass,
    )

    now, run_id = _run_timestamp()

    return AllReport(
        finalpass_version=__version__,
        schema_version=ALL_SCHEMA_VERSION,
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        spec=SpecRef(name=spec.name, display_name=spec.display_name, source=source),
        fps=fps,
        drop_frame=drop_frame,
        command="all",
        folder=str(folder.resolve()),
        groups=groups_out,
        files=flat_files,
        unclassified=unclassified,
        discovery_errors=discovery_errors,
        summary=summary,
    )


def run_all_family_from_scan(
    *,
    folder: Path,
    scan: AssetFolderScan,
    family: BundledSpecFamily,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> AllReport:
    _validate_fps(fps, drop_frame=drop_frame)
    if not scan.groups and not scan.discovery_errors and not scan.unclassified:
        raise NoAudioFilesError(
            f"No WAV/BWF files found in {folder}. "
            "Provide a folder containing at least one .wav file."
        )

    groups_out: list[Group] = []
    flat_files: list[FileReport] = []

    for group_id, classified_assets in scan.groups.items():
        group = _process_group_with_spec_family(
            group_id=group_id,
            classified_assets=classified_assets,
            family=family,
            include_unclassified=include_unclassified,
            fps=fps,
            drop_frame=drop_frame,
            null_window_ms=null_window_ms,
            null_hop_ms=null_hop_ms,
            null_threshold_dbfs=null_threshold_dbfs,
            me_window_ms=me_window_ms,
            me_hop_ms=me_hop_ms,
            me_band_low_hz=me_band_low_hz,
            me_band_high_hz=me_band_high_hz,
            me_corr_threshold=me_corr_threshold,
            me_coherence_threshold=me_coherence_threshold,
            me_dx_gate_dbfs=me_dx_gate_dbfs,
            me_me_floor_dbfs=me_me_floor_dbfs,
        )
        groups_out.append(group)
        flat_files.extend(group.files)

    unclassified = [
        _unclassified_entry(asset, reason="no_role_pattern_match")
        for asset in scan.unclassified
    ]
    discovery_errors = [
        _discovery_issue(error)
        for error in scan.discovery_errors
    ]

    total_checks = sum(g.group_summary.total_checks for g in groups_out)
    passed = sum(g.group_summary.passed for g in groups_out)
    failed = sum(g.group_summary.failed for g in groups_out)
    skipped = sum(g.group_summary.skipped for g in groups_out)
    groups_passed = sum(1 for g in groups_out if g.group_summary.overall_pass)
    groups_failed = len(groups_out) - groups_passed
    overall_pass = failed == 0 and groups_failed == 0 and not discovery_errors

    summary = AllSummary(
        groups_total=len(groups_out),
        groups_passed=groups_passed,
        groups_failed=groups_failed,
        total_checks=total_checks,
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=overall_pass,
    )

    now, run_id = _run_timestamp()

    return AllReport(
        finalpass_version=__version__,
        schema_version=ALL_SCHEMA_VERSION,
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        spec=SpecRef(name=family.key, display_name=family.display_name, source="bundled"),
        fps=fps,
        drop_frame=drop_frame,
        command="all",
        folder=str(folder.resolve()),
        groups=groups_out,
        files=flat_files,
        unclassified=unclassified,
        discovery_errors=discovery_errors,
        summary=summary,
    )


def write_report_artifacts(
    *,
    report: Report | NullReport | MEReport | AllReport,
    payload: str,
    out_dir: Path,
) -> WrittenArtifacts:
    candidates = aaf_export.collect_marker_candidates(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    program_name = derive_program_name_from_report(report)
    json_path, html_path, aaf_path = _reserve_artifact_paths(
        out_dir=out_dir,
        include_aaf=bool(candidates),
        name_slug=program_name.slug if program_name is not None else None,
    )
    artifacts = tuple(
        path.name
        for path in (json_path, html_path, aaf_path)
        if path is not None
    )
    html = render_report_html(report, artifacts=artifacts)
    json_path.write_text(payload + "\n", encoding="utf-8")

    if not candidates:
        html_path.write_text(html + "\n", encoding="utf-8")
        return WrittenArtifacts(json_path=json_path, html_path=html_path, aaf_path=None)

    temp_fd, temp_name = tempfile.mkstemp(
        prefix="finalpass-markers-",
        suffix=".aaf",
        dir=str(out_dir),
    )
    os.close(temp_fd)
    Path(temp_name).unlink(missing_ok=True)
    temp_path = Path(temp_name)
    try:
        aaf_export.write_markers_aaf(temp_path, candidates, fps=report.fps, drop_frame=report.drop_frame)
        temp_path.replace(aaf_path)
    except FinalPassError:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # pragma: no cover - defensive wrapper
        temp_path.unlink(missing_ok=True)
        raise AAFExportError(f"Could not write {aaf_path.name}: {exc}") from exc
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass

    html_path.write_text(html + "\n", encoding="utf-8")
    return WrittenArtifacts(json_path=json_path, html_path=html_path, aaf_path=aaf_path)


def _reserve_artifact_paths(*, out_dir: Path, include_aaf: bool, name_slug: str | None = None) -> tuple[Path, Path, Path | None]:
    report_stem = f"{name_slug}-report" if name_slug else "report"
    markers_stem = f"{name_slug}-markers" if name_slug else "markers"
    index = 0
    while True:
        suffix = "" if index == 0 else f"-{index:02d}"
        json_path = out_dir / f"{report_stem}{suffix}.json"
        html_path = out_dir / f"{report_stem}{suffix}.html"
        aaf_path = out_dir / f"{markers_stem}{suffix}.aaf" if include_aaf else None
        candidates = [json_path, html_path]
        if aaf_path is not None:
            candidates.append(aaf_path)
        if not any(path.exists() for path in candidates):
            return json_path, html_path, aaf_path
        index += 1


def _validate_channels(files, spec: Spec) -> None:
    expected = spec.expected_channel_count
    for audio in files:
        if audio.channel_count != expected:
            raise ChannelMismatchError(
                f"{audio.path}: spec '{spec.name}' expects {spec.channel_config} "
                f"({expected} ch); got {audio.channel_count} ch."
            )


def _validate_homogeneous_sample_rate(files) -> None:
    rates = {audio.sample_rate for audio in files}
    if len(rates) > 1:
        raise FinalPassError(
            f"Sample-rate mismatch across inputs: {sorted(rates)}. "
            "All files in a single run must share a sample rate."
        )


def _validate_fps(fps: float, *, drop_frame: bool = False) -> None:
    if not math.isfinite(fps) or fps <= 0:
        raise FinalPassError(f"fps must be a finite number greater than zero; got {fps!r}.")
    if drop_frame:
        try:
            frame_rate_info(fps, drop_frame=True)
        except ValueError as exc:
            raise FinalPassError(str(exc)) from exc


def _process_group(
    *,
    group_id: str,
    classified_assets: list[ClassifiedLogicalAsset],
    spec: Spec,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> Group:
    errors: list[GroupError] = []
    used_by: dict[str, list[str]] = {asset.logical_asset.asset_id: [] for asset in classified_assets}
    selection_notes: dict[str, str | None] = {asset.logical_asset.asset_id: None for asset in classified_assets}
    target_layout = _target_layout_for_spec(spec)
    role_to_assets: dict[FileRole, list[ClassifiedLogicalAsset]] = {}
    for asset in classified_assets:
        role_to_assets.setdefault(asset.role, []).append(asset)
    target_layout_matches: dict[FileRole, list[ClassifiedLogicalAsset]] = {
        role: [
            asset for asset in assets
            if asset.logical_asset.channel_config_actual == target_layout
        ]
        for role, assets in role_to_assets.items()
    }

    def has_target(role: FileRole) -> bool:
        return bool(target_layout_matches.get(role, []))

    dx_duplicate_is_blocking = (
        spec.dialog_lufs is not None
        or has_target("me")
        or (has_target("mx") and has_target("fx"))
    )
    mx_duplicate_is_blocking = has_target("dx") and has_target("fx")
    fx_duplicate_is_blocking = has_target("dx") and has_target("mx")
    me_duplicate_is_blocking = has_target("dx")

    pm_asset, _, _ = _select_target_layout_asset(
        group_id=group_id,
        role="pm",
        assets=role_to_assets.get("pm", []),
        target_layout=target_layout,
        errors=errors,
        selection_notes=selection_notes,
        required=True,
        missing_type="MissingTargetLayoutPrintmaster",
        duplicate_is_blocking=True,
    )
    dx_target, _, dx_alternates = _select_target_layout_asset(
        group_id=group_id,
        role="dx",
        assets=role_to_assets.get("dx", []),
        target_layout=target_layout,
        errors=errors,
        selection_notes=selection_notes,
        required=False,
        duplicate_is_blocking=dx_duplicate_is_blocking,
    )
    mx_target, _, _ = _select_target_layout_asset(
        group_id=group_id,
        role="mx",
        assets=role_to_assets.get("mx", []),
        target_layout=target_layout,
        errors=errors,
        selection_notes=selection_notes,
        required=False,
        duplicate_is_blocking=mx_duplicate_is_blocking,
    )
    fx_target, _, _ = _select_target_layout_asset(
        group_id=group_id,
        role="fx",
        assets=role_to_assets.get("fx", []),
        target_layout=target_layout,
        errors=errors,
        selection_notes=selection_notes,
        required=False,
        duplicate_is_blocking=fx_duplicate_is_blocking,
    )
    me_target, _, _ = _select_target_layout_asset(
        group_id=group_id,
        role="me",
        assets=role_to_assets.get("me", []),
        target_layout=target_layout,
        errors=errors,
        selection_notes=selection_notes,
        required=False,
        duplicate_is_blocking=me_duplicate_is_blocking,
    )

    selected_target_assets: dict[FileRole, ClassifiedLogicalAsset] = {}
    for role, asset in (
        ("pm", pm_asset),
        ("dx", dx_target),
        ("mx", mx_target),
        ("fx", fx_target),
        ("me", me_target),
    ):
        if asset is not None:
            selected_target_assets[role] = asset

    if pm_asset is not None:
        _mark_used(used_by, pm_asset, "pm_loudness")

    dialog_dx: ClassifiedLogicalAsset | None = None
    if spec.dialog_lufs is not None:
        if dx_target is not None:
            dialog_dx = dx_target
            _mark_used(used_by, dx_target, "dialog_loudness")
        elif len(dx_alternates) == 1:
            dialog_dx = dx_alternates[0]
            _mark_used(used_by, dialog_dx, "dialog_loudness")
            _append_selection_note(selection_notes, dialog_dx, "selected_for_dialog_fallback")
        elif len(dx_alternates) > 1:
            errors.append(GroupError(
                type="DialogFallbackAmbiguity",
                message=(
                    f"group {group_id!r} has no target-layout DX asset for {target_layout}, "
                    f"and {len(dx_alternates)} alternate-layout DX assets compete for dialog fallback."
                ),
            ))
            for asset in dx_alternates:
                _append_selection_note(selection_notes, asset, "dialog_fallback_ambiguous")

    unknown_assets = role_to_assets.get("unknown", [])
    if include_unclassified:
        for asset in unknown_assets:
            _mark_used(used_by, asset, "unknown_loudness")
            _append_selection_note(selection_notes, asset, "selected_for_loudness_only")
    else:
        for asset in unknown_assets:
            _append_selection_note(selection_notes, asset, "unknown_skipped")

    file_reports: list[FileReport] = []
    measure_targets: list[tuple[ClassifiedLogicalAsset, FileRole]] = []
    if pm_asset is not None:
        measure_targets.append((pm_asset, "pm"))
    if dialog_dx is not None:
        measure_targets.append((dialog_dx, "dx"))
    if include_unclassified:
        measure_targets.extend((asset, "unknown") for asset in unknown_assets)

    for asset, role in measure_targets:
        try:
            audio = read_classified_audio(asset)
            fr = _measure_asset_file(asset, audio, spec, role, fps=fps, drop_frame=drop_frame)
        except ChannelMismatchError as exc:
            errors.append(GroupError(type="ChannelMismatchError", message=str(exc)))
            continue
        except FinalPassError as exc:
            errors.append(GroupError(type=exc.__class__.__name__, message=str(exc)))
            continue
        file_reports.append(fr)

    null_test = _run_group_null_test(
        printmaster=pm_asset,
        selected_assets=selected_target_assets,
        used_by=used_by,
        fps=fps,
        drop_frame=drop_frame,
        window_ms=null_window_ms,
        hop_ms=null_hop_ms,
        threshold_dbfs=null_threshold_dbfs,
    )
    me_check = _run_group_me_check(
        selected_assets=selected_target_assets,
        used_by=used_by,
        fps=fps,
        drop_frame=drop_frame,
        window_ms=me_window_ms,
        hop_ms=me_hop_ms,
        band_low_hz=me_band_low_hz,
        band_high_hz=me_band_high_hz,
        corr_threshold=me_corr_threshold,
        coherence_threshold=me_coherence_threshold,
        dx_gate_dbfs=me_dx_gate_dbfs,
        me_floor_dbfs=me_me_floor_dbfs,
    )

    inventory = [
        _inventory_entry(
            asset,
            used_by=used_by[asset.logical_asset.asset_id],
            selection_note=selection_notes[asset.logical_asset.asset_id],
        )
        for asset in classified_assets
    ]
    inventory.sort(key=lambda item: (item.role, item.path.lower()))

    all_checks = [c for fr in file_reports for c in fr.checks]
    passed = sum(1 for c in all_checks if c.pass_ is True)
    failed = sum(1 for c in all_checks if c.pass_ is False)
    skipped = sum(1 for c in all_checks if c.skipped)
    if null_test.pass_ is True:
        passed += 1
    elif null_test.pass_ is False:
        failed += 1
    if null_test.skipped:
        skipped += 1
    if me_check.pass_ is True:
        passed += 1
    elif me_check.pass_ is False:
        failed += 1
    if me_check.skipped:
        skipped += 1
    summary = GroupSummary(
        total_checks=len(all_checks) + 2,
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=failed == 0 and not errors,
    )

    return Group(
        group_id=group_id,
        assets=inventory,
        files=file_reports,
        null_test=null_test,
        me_check=me_check,
        group_summary=summary,
        errors=errors,
    )


def _process_group_with_spec_family(
    *,
    group_id: str,
    classified_assets: list[ClassifiedLogicalAsset],
    family: BundledSpecFamily,
    include_unclassified: bool,
    fps: float,
    drop_frame: bool = False,
    null_window_ms: float,
    null_hop_ms: float,
    null_threshold_dbfs: float,
    me_window_ms: float,
    me_hop_ms: float,
    me_band_low_hz: float,
    me_band_high_hz: float,
    me_corr_threshold: float,
    me_coherence_threshold: float,
    me_dx_gate_dbfs: float,
    me_me_floor_dbfs: float,
) -> Group:
    errors: list[GroupError] = []
    used_by: dict[str, list[str]] = {asset.logical_asset.asset_id: [] for asset in classified_assets}
    selection_notes: dict[str, str | None] = {asset.logical_asset.asset_id: None for asset in classified_assets}
    role_to_assets: dict[FileRole, list[ClassifiedLogicalAsset]] = {}
    for asset in classified_assets:
        role_to_assets.setdefault(asset.role, []).append(asset)

    pm_layouts = {
        asset.logical_asset.channel_config_actual
        for asset in role_to_assets.get("pm", [])
        if family.supports_channel_config(asset.logical_asset.channel_config_actual)
    }
    primary_spec = family.best_spec_for_channel_configs(pm_layouts)

    file_reports: list[FileReport] = []
    selected_target_assets: dict[FileRole, ClassifiedLogicalAsset] = {}
    null_test: AutoNullTestResult
    me_check: MECheckResult

    if primary_spec is None:
        errors.append(GroupError(
            type="MissingFamilyLayoutPrintmaster",
            message=(
                f"group {group_id!r} has no printmaster asset matching any "
                f"{family.display_name} layout ({', '.join(family.supported_channel_configs)})."
            ),
        ))
        for asset in role_to_assets.get("unknown", []):
            _append_selection_note(selection_notes, asset, "unknown_skipped")
        null_test = AutoNullTestResult(
            **{"pass": None},
            skipped=True,
            reason="missing_target_layout_printmaster",
            stem_strategy=None,
            selected_roles=[],
            window_ms=null_window_ms,
            hop_ms=null_hop_ms,
            threshold_dbfs=null_threshold_dbfs,
            summary=None,
            flags=[],
            errors=[],
        )
        me_check = MECheckResult(
            **{"pass": None},
            skipped=True,
            reason="missing_dx_or_me",
            analysis_signal=ANALYSIS_SIGNAL_NAME,
            window_ms=me_window_ms,
            hop_ms=me_hop_ms,
            band_low_hz=me_band_low_hz,
            band_high_hz=me_band_high_hz,
            corr_threshold=me_corr_threshold,
            coherence_threshold=me_coherence_threshold,
            dx_gate_dbfs=me_dx_gate_dbfs,
            me_floor_dbfs=me_me_floor_dbfs,
            summary=None,
            flags=[],
            errors=[],
        )
    else:
        target_layout = _target_layout_for_spec(primary_spec)
        target_layout_matches: dict[FileRole, list[ClassifiedLogicalAsset]] = {
            role: [
                asset for asset in assets
                if asset.logical_asset.channel_config_actual == target_layout
            ]
            for role, assets in role_to_assets.items()
        }

        def has_target(role: FileRole) -> bool:
            return bool(target_layout_matches.get(role, []))

        dx_duplicate_is_blocking = (
            primary_spec.dialog_lufs is not None
            or has_target("me")
            or (has_target("mx") and has_target("fx"))
        )
        mx_duplicate_is_blocking = has_target("dx") and has_target("fx")
        fx_duplicate_is_blocking = has_target("dx") and has_target("mx")
        me_duplicate_is_blocking = has_target("dx")

        pm_asset, _, _ = _select_target_layout_asset(
            group_id=group_id,
            role="pm",
            assets=role_to_assets.get("pm", []),
            target_layout=target_layout,
            errors=errors,
            selection_notes=selection_notes,
            required=True,
            missing_type="MissingTargetLayoutPrintmaster",
            duplicate_is_blocking=True,
        )
        dx_target, _, dx_alternates = _select_target_layout_asset(
            group_id=group_id,
            role="dx",
            assets=role_to_assets.get("dx", []),
            target_layout=target_layout,
            errors=errors,
            selection_notes=selection_notes,
            required=False,
            duplicate_is_blocking=dx_duplicate_is_blocking,
        )
        mx_target, _, _ = _select_target_layout_asset(
            group_id=group_id,
            role="mx",
            assets=role_to_assets.get("mx", []),
            target_layout=target_layout,
            errors=errors,
            selection_notes=selection_notes,
            required=False,
            duplicate_is_blocking=mx_duplicate_is_blocking,
        )
        fx_target, _, _ = _select_target_layout_asset(
            group_id=group_id,
            role="fx",
            assets=role_to_assets.get("fx", []),
            target_layout=target_layout,
            errors=errors,
            selection_notes=selection_notes,
            required=False,
            duplicate_is_blocking=fx_duplicate_is_blocking,
        )
        me_target, _, _ = _select_target_layout_asset(
            group_id=group_id,
            role="me",
            assets=role_to_assets.get("me", []),
            target_layout=target_layout,
            errors=errors,
            selection_notes=selection_notes,
            required=False,
            duplicate_is_blocking=me_duplicate_is_blocking,
        )

        for role, asset in (
            ("pm", pm_asset),
            ("dx", dx_target),
            ("mx", mx_target),
            ("fx", fx_target),
            ("me", me_target),
        ):
            if asset is not None:
                selected_target_assets[role] = asset

        if pm_asset is not None:
            _mark_used(used_by, pm_asset, "pm_loudness")

        dialog_dx: ClassifiedLogicalAsset | None = None
        if primary_spec.dialog_lufs is not None:
            if dx_target is not None:
                dialog_dx = dx_target
                _mark_used(used_by, dx_target, "dialog_loudness")
            elif len(dx_alternates) == 1:
                dialog_dx = dx_alternates[0]
                _mark_used(used_by, dialog_dx, "dialog_loudness")
                _append_selection_note(selection_notes, dialog_dx, "selected_for_dialog_fallback")
            elif len(dx_alternates) > 1:
                errors.append(GroupError(
                    type="DialogFallbackAmbiguity",
                    message=(
                        f"group {group_id!r} has no target-layout DX asset for {target_layout}, "
                        f"and {len(dx_alternates)} alternate-layout DX assets compete for dialog fallback."
                    ),
                ))
                for asset in dx_alternates:
                    _append_selection_note(selection_notes, asset, "dialog_fallback_ambiguous")

        unknown_assets = role_to_assets.get("unknown", [])
        if include_unclassified:
            for asset in unknown_assets:
                _mark_used(used_by, asset, "unknown_loudness")
                _append_selection_note(selection_notes, asset, "selected_for_loudness_only")
        else:
            for asset in unknown_assets:
                _append_selection_note(selection_notes, asset, "unknown_skipped")

        measure_targets: list[tuple[ClassifiedLogicalAsset, FileRole, Spec]] = []
        if pm_asset is not None:
            measure_targets.append((pm_asset, "pm", primary_spec))
        if dialog_dx is not None:
            measure_targets.append((dialog_dx, "dx", primary_spec))
        if include_unclassified:
            measure_targets.extend((asset, "unknown", primary_spec) for asset in unknown_assets)

        for layout in sorted(pm_layouts, key=lambda value: CHANNEL_COUNTS[value]):
            if layout == target_layout:
                continue
            layout_spec = family.spec_for_channel_config(layout)
            if layout_spec is None:
                continue
            extra_pm, _, _ = _select_target_layout_asset(
                group_id=group_id,
                role="pm",
                assets=role_to_assets.get("pm", []),
                target_layout=layout,
                errors=errors,
                selection_notes=selection_notes,
                required=True,
                missing_type="MissingFamilyLayoutPrintmaster",
                duplicate_is_blocking=True,
                selection_note_for_match="selected_for_family_loudness",
                selection_note_for_alternates=None,
            )
            if extra_pm is not None:
                _mark_used(used_by, extra_pm, "pm_loudness")
                measure_targets.append((extra_pm, "pm", layout_spec))

            if layout_spec.dialog_lufs is not None:
                extra_dx, _, _ = _select_target_layout_asset(
                    group_id=group_id,
                    role="dx",
                    assets=role_to_assets.get("dx", []),
                    target_layout=layout,
                    errors=errors,
                    selection_notes=selection_notes,
                    required=False,
                    duplicate_is_blocking=True,
                    selection_note_for_match="selected_for_family_dialog_loudness",
                    selection_note_for_alternates=None,
                )
                if extra_dx is not None:
                    _mark_used(used_by, extra_dx, "dialog_loudness")
                    measure_targets.append((extra_dx, "dx", layout_spec))

        seen_measurements: set[tuple[str, FileRole]] = set()
        for asset, role, spec in measure_targets:
            measurement_key = (asset.logical_asset.asset_id, role)
            if measurement_key in seen_measurements:
                continue
            seen_measurements.add(measurement_key)
            try:
                audio = read_classified_audio(asset)
                fr = _measure_asset_file(asset, audio, spec, role, fps=fps, drop_frame=drop_frame)
            except ChannelMismatchError as exc:
                errors.append(GroupError(type="ChannelMismatchError", message=str(exc)))
                continue
            except FinalPassError as exc:
                errors.append(GroupError(type=exc.__class__.__name__, message=str(exc)))
                continue
            file_reports.append(fr)

        null_test = _run_group_null_test(
            printmaster=pm_asset,
            selected_assets=selected_target_assets,
            used_by=used_by,
            fps=fps,
            drop_frame=drop_frame,
            window_ms=null_window_ms,
            hop_ms=null_hop_ms,
            threshold_dbfs=null_threshold_dbfs,
        )
        me_check = _run_group_me_check(
            selected_assets=selected_target_assets,
            used_by=used_by,
            fps=fps,
            drop_frame=drop_frame,
            window_ms=me_window_ms,
            hop_ms=me_hop_ms,
            band_low_hz=me_band_low_hz,
            band_high_hz=me_band_high_hz,
            corr_threshold=me_corr_threshold,
            coherence_threshold=me_coherence_threshold,
            dx_gate_dbfs=me_dx_gate_dbfs,
            me_floor_dbfs=me_me_floor_dbfs,
        )

    inventory = [
        _inventory_entry(
            asset,
            used_by=used_by[asset.logical_asset.asset_id],
            selection_note=selection_notes[asset.logical_asset.asset_id],
        )
        for asset in classified_assets
    ]
    inventory.sort(key=lambda item: (item.role, item.path.lower()))
    file_reports.sort(key=lambda item: (item.role, -item.channel_count, item.path.lower()))
    summary = _group_summary_from_results(
        file_reports=file_reports,
        null_test=null_test,
        me_check=me_check,
        errors=errors,
    )

    return Group(
        group_id=group_id,
        assets=inventory,
        files=file_reports,
        null_test=null_test,
        me_check=me_check,
        group_summary=summary,
        errors=errors,
    )


def _group_summary_from_results(
    *,
    file_reports: list[FileReport],
    null_test: AutoNullTestResult,
    me_check: MECheckResult,
    errors: list[GroupError],
) -> GroupSummary:
    all_checks = [c for fr in file_reports for c in fr.checks]
    passed = sum(1 for c in all_checks if c.pass_ is True)
    failed = sum(1 for c in all_checks if c.pass_ is False)
    skipped = sum(1 for c in all_checks if c.skipped)
    if null_test.pass_ is True:
        passed += 1
    elif null_test.pass_ is False:
        failed += 1
    if null_test.skipped:
        skipped += 1
    if me_check.pass_ is True:
        passed += 1
    elif me_check.pass_ is False:
        failed += 1
    if me_check.skipped:
        skipped += 1
    return GroupSummary(
        total_checks=len(all_checks) + 2,
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=failed == 0 and not errors,
    )


def _measure_asset_file(
    asset: ClassifiedLogicalAsset,
    audio: AudioFile,
    spec: Spec,
    role: FileRole,
    *,
    fps: float,
    drop_frame: bool = False,
) -> FileReport:
    if role == "pm":
        _validate_channels([audio], spec)
        check_role = "primary"
    elif role == "dx":
        check_role = "dx"
    else:
        check_role = "primary"

    m = measure(audio)
    cs = check(m, spec, role=check_role)
    flags = _timed_loudness_flags(audio=audio, checks=cs, fps=fps, drop_frame=drop_frame)

    return _attach_time_reference(
        FileReport(
            path=str(audio.path),
            role=role,
            sample_rate=audio.sample_rate,
            bit_depth=audio.bit_depth,
            channel_count=audio.channel_count,
            duration_seconds=round(audio.duration_seconds, 3),
            measurements=m.as_measurements(),
            checks=cs,
            errors=m.errors,
            channel_config_hint=asset.channel_config_hint,
            channel_config_actual=asset.logical_asset.channel_config_actual,
            source_kind=asset.logical_asset.source_kind,
            source_paths=[str(path) for path in asset.logical_asset.source_paths],
            member_legs=list(asset.logical_asset.member_legs),
            presentation_label=asset.logical_asset.presentation_label,
            flags=flags,
        ),
        audio.time_reference_samples,
    )


def _timed_loudness_flags(
    *,
    audio: AudioFile,
    checks,
    fps: float,
    drop_frame: bool = False,
):
    tp_check = next((check for check in checks if check.metric == "true_peak_dbtp"), None)
    if tp_check is None or tp_check.pass_ is not False or tp_check.limit is None:
        return []
    return true_peak_over_flags(audio, threshold_dbtp=tp_check.limit, fps=fps, drop_frame=drop_frame)


def _run_group_null_test(
    *,
    printmaster: ClassifiedLogicalAsset | None,
    selected_assets: dict[FileRole, ClassifiedLogicalAsset],
    used_by: dict[str, list[str]],
    fps: float,
    drop_frame: bool = False,
    window_ms: float,
    hop_ms: float,
    threshold_dbfs: float,
) -> AutoNullTestResult:
    if printmaster is None:
        return AutoNullTestResult(
            **{"pass": None},
            skipped=True,
            reason="missing_target_layout_printmaster",
            stem_strategy=None,
            selected_roles=[],
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
            summary=None,
            flags=[],
            errors=[],
        )

    strategy, selected_roles, stems = _select_auto_null_stems(selected_assets)
    if strategy is None:
        return AutoNullTestResult(
            **{"pass": None},
            skipped=True,
            reason="insufficient_stems_for_auto_null",
            stem_strategy=None,
            selected_roles=[],
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
            summary=None,
            flags=[],
            errors=[],
        )

    try:
        _mark_used(used_by, printmaster, "null")
        for stem in stems:
            _mark_used(used_by, stem, "null")
        pm_audio = read_classified_audio(printmaster)
        stem_audios = [read_classified_audio(asset) for asset in stems]
        _validate_channel_label_hints([(printmaster, pm_audio), *zip(stems, stem_audios)])
        analysis = analyze_null(
            pm_audio,
            stem_audios,
            fps=fps,
            drop_frame=drop_frame,
            time_reference_samples=pm_audio.time_reference_samples,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
        )
    except FinalPassError as exc:
        return AutoNullTestResult(
            **{"pass": False},
            skipped=False,
            reason=_analysis_reason(exc),
            stem_strategy=strategy,
            selected_roles=selected_roles,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
            summary=None,
            flags=[],
            errors=[GroupError(type=exc.__class__.__name__, message=str(exc))],
        )

    passed = len(analysis.flags) == 0
    return _attach_time_reference(
        AutoNullTestResult(
            **{"pass": passed},
            skipped=False,
            reason=None,
            stem_strategy=strategy,
            selected_roles=selected_roles,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
            summary=analysis.summary,
            analysis_window=analysis.analysis_window,
            flags=analysis.flags,
            errors=[],
        ),
        pm_audio.time_reference_samples,
    )


def _run_group_me_check(
    *,
    selected_assets: dict[FileRole, ClassifiedLogicalAsset],
    used_by: dict[str, list[str]],
    fps: float,
    drop_frame: bool = False,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> MECheckResult:
    dx_asset = selected_assets.get("dx")
    me_asset = selected_assets.get("me")
    if dx_asset is None or me_asset is None:
        return MECheckResult(
            **{"pass": None},
            skipped=True,
            reason="missing_dx_or_me",
            analysis_signal=ANALYSIS_SIGNAL_NAME,
            window_ms=window_ms,
            hop_ms=hop_ms,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            corr_threshold=corr_threshold,
            coherence_threshold=coherence_threshold,
            dx_gate_dbfs=dx_gate_dbfs,
            me_floor_dbfs=me_floor_dbfs,
            summary=None,
            flags=[],
            errors=[],
        )

    try:
        _mark_used(used_by, dx_asset, "me")
        _mark_used(used_by, me_asset, "me")
        dx_audio = read_classified_audio(dx_asset)
        me_audio = read_classified_audio(me_asset)
        _validate_channel_label_hints([(dx_asset, dx_audio), (me_asset, me_audio)])
        analysis = analyze_me(
            me_audio,
            dx_audio,
            fps=fps,
            drop_frame=drop_frame,
            time_reference_samples=me_audio.time_reference_samples,
            window_ms=window_ms,
            hop_ms=hop_ms,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            corr_threshold=corr_threshold,
            coherence_threshold=coherence_threshold,
            dx_gate_dbfs=dx_gate_dbfs,
            me_floor_dbfs=me_floor_dbfs,
        )
    except FinalPassError as exc:
        return MECheckResult(
            **{"pass": False},
            skipped=False,
            reason=_analysis_reason(exc),
            analysis_signal=ANALYSIS_SIGNAL_NAME,
            window_ms=window_ms,
            hop_ms=hop_ms,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            corr_threshold=corr_threshold,
            coherence_threshold=coherence_threshold,
            dx_gate_dbfs=dx_gate_dbfs,
            me_floor_dbfs=me_floor_dbfs,
            summary=None,
            flags=[],
            errors=[GroupError(type=exc.__class__.__name__, message=str(exc))],
        )

    passed = len(analysis.flags) == 0
    result = _attach_time_reference(
        MECheckResult(
            **{"pass": passed},
            skipped=False,
            reason=None,
            analysis_signal=ANALYSIS_SIGNAL_NAME,
            window_ms=window_ms,
            hop_ms=hop_ms,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            corr_threshold=corr_threshold,
            coherence_threshold=coherence_threshold,
            dx_gate_dbfs=dx_gate_dbfs,
            me_floor_dbfs=me_floor_dbfs,
            summary=analysis.summary,
            analysis_window=analysis.analysis_window,
            flags=analysis.flags,
            errors=[],
        ),
        me_audio.time_reference_samples,
    )
    result._sample_rate = me_audio.sample_rate
    return result


def _select_auto_null_stems(
    selected_assets: dict[FileRole, ClassifiedLogicalAsset],
) -> tuple[str | None, list[str], list[ClassifiedLogicalAsset]]:
    if all(selected_assets.get(role) is not None for role in ("dx", "mx", "fx")):
        return "dx_mx_fx", ["dx", "mx", "fx"], [selected_assets["dx"], selected_assets["mx"], selected_assets["fx"]]
    if all(selected_assets.get(role) is not None for role in ("dx", "me")):
        return "dx_me", ["dx", "me"], [selected_assets["dx"], selected_assets["me"]]
    return None, [], []


def _validate_channel_label_hints(items: list[tuple[ClassifiedLogicalAsset, AudioFile]]) -> None:
    for asset, audio in items:
        hint = asset.channel_config_hint
        if hint is None:
            continue
        actual = asset.logical_asset.channel_config_actual
        if hint == actual:
            continue
        if hint == "5.0" and actual == "5.1":
            continue
        actual_label = actual if actual is not None else f"unsupported {audio.channel_count}ch"
        raise ChannelConfigLabelMismatch(
            f"{audio.path.name} hints {hint} but header is {actual_label}."
        )


def _target_layout_for_spec(spec: Spec) -> str:
    return spec.channel_config


def _select_target_layout_asset(
    *,
    group_id: str,
    role: FileRole,
    assets: list[ClassifiedLogicalAsset],
    target_layout: str,
    errors: list[GroupError],
    selection_notes: dict[str, str | None],
    required: bool,
    missing_type: str | None = None,
    duplicate_is_blocking: bool,
    selection_note_for_match: str = "selected_for_target_layout",
    selection_note_for_alternates: str | None = "alternate_presentation",
    selection_note_for_duplicate: str = "duplicate_target_layout",
) -> tuple[ClassifiedLogicalAsset | None, list[ClassifiedLogicalAsset], list[ClassifiedLogicalAsset]]:
    matching = [asset for asset in assets if asset.logical_asset.channel_config_actual == target_layout]
    alternates = [asset for asset in assets if asset.logical_asset.channel_config_actual != target_layout]

    if selection_note_for_alternates is not None:
        for asset in alternates:
            _append_selection_note(selection_notes, asset, selection_note_for_alternates)

    if len(matching) == 1:
        _append_selection_note(selection_notes, matching[0], selection_note_for_match)
        return matching[0], matching, alternates

    if len(matching) == 0:
        if required:
            errors.append(GroupError(
                type=missing_type or "MissingTargetLayoutRole",
                message=f"group {group_id!r} has no {role!r} asset matching target layout {target_layout!r}.",
            ))
        return None, matching, alternates

    names = ", ".join(asset.logical_asset.canonical_path.name for asset in matching)
    if duplicate_is_blocking:
        errors.append(GroupError(
            type="DuplicateTargetLayoutRoleError",
            message=(
                f"group {group_id!r} has {len(matching)} {role!r} assets in target layout "
                f"{target_layout!r}: {names}"
            ),
        ))
    for asset in matching:
        _append_selection_note(selection_notes, asset, selection_note_for_duplicate)
    return None, matching, alternates


def _append_selection_note(
    selection_notes: dict[str, str | None],
    asset: ClassifiedLogicalAsset,
    note: str,
) -> None:
    asset_id = asset.logical_asset.asset_id
    current = selection_notes.get(asset_id)
    if current is None:
        selection_notes[asset_id] = note
    elif note not in current.split("; "):
        selection_notes[asset_id] = f"{current}; {note}"


def _mark_used(
    used_by: dict[str, list[str]],
    asset: ClassifiedLogicalAsset,
    purpose: str,
) -> None:
    purposes = used_by[asset.logical_asset.asset_id]
    if purpose not in purposes:
        purposes.append(purpose)


def _inventory_entry(
    asset: ClassifiedLogicalAsset,
    *,
    used_by: list[str],
    selection_note: str | None,
) -> AssetInventoryEntry:
    logical_asset = asset.logical_asset
    return AssetInventoryEntry(
        asset_id=logical_asset.asset_id,
        role=asset.role,
        path=str(logical_asset.canonical_path),
        source_kind=logical_asset.source_kind,
        source_paths=[str(path) for path in logical_asset.source_paths],
        member_legs=list(logical_asset.member_legs),
        presentation_label=logical_asset.presentation_label,
        channel_config_actual=logical_asset.channel_config_actual,
        channel_config_hint=asset.channel_config_hint,
        used_by=list(used_by),
        selection_note=selection_note,
    )


def _unclassified_entry(asset: ClassifiedLogicalAsset, *, reason: str) -> UnclassifiedEntry:
    logical_asset = asset.logical_asset
    return UnclassifiedEntry(
        path=str(logical_asset.canonical_path),
        source_kind=logical_asset.source_kind,
        source_paths=[str(path) for path in logical_asset.source_paths],
        member_legs=list(logical_asset.member_legs),
        presentation_label=logical_asset.presentation_label,
        reason=reason,
    )


def _attach_time_reference(item, time_reference_samples: int | None):
    item._time_reference_samples = time_reference_samples
    return item


def _discovery_issue(error) -> DiscoveryIssue:
    paths = [str(path) for path in error.paths]
    return DiscoveryIssue(
        path=paths[0] if paths else (error.family_key or ""),
        source_paths=paths,
        error_type=error.type,
        message=error.message,
        family_key=error.family_key,
        group_hint=error.group_hint,
    )


def _analysis_reason(exc: FinalPassError) -> str:
    if isinstance(exc, ChannelConfigLabelMismatch):
        return "channel_config_label_mismatch"
    if isinstance(exc, AlignmentError):
        return "alignment_error"
    if isinstance(exc, SampleRateMismatchError):
        return "sample_rate_mismatch"
    if isinstance(exc, SampleCountMismatchError):
        return "sample_count_mismatch"
    if isinstance(exc, ProgramWindowError):
        return "program_window_error"
    if isinstance(exc, ChannelMismatchError):
        return "channel_count_mismatch"
    if isinstance(exc, UnsupportedChannelConfigError):
        return "unsupported_channel_config"
    if isinstance(exc, AudioFormatError):
        return "audio_format_error"
    return exc.__class__.__name__.lower()


def _run_timestamp() -> tuple[datetime, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + secrets.token_hex(3)
    return now, run_id
