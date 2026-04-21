"""CLI layer. Library modules return data; this layer prints, writes, and exits.

Exit codes (from PHASE_1_SPEC.md):

* 0 — all checks pass.
* 1 — at least one check failed.
* 2 — tool / validation error (unknown spec, channel mismatch, unreadable audio).
"""

from __future__ import annotations

import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import soundfile as sf
from rich.console import Console
from rich.table import Table

from . import (
    ALL_SCHEMA_VERSION,
    LOUDNESS_SCHEMA_VERSION,
    ME_SCHEMA_VERSION,
    NULL_SCHEMA_VERSION,
    __version__,
)
from .audio_io import AudioFile, channel_config_from_count, read_wav
from .classify import (
    ClassifiedFile,
    FolderScan,
    load_config,
    scan_folder,
)
from .errors import (
    AlignmentError,
    AmbiguousClassificationError,
    AudioFormatError,
    ChannelConfigLabelMismatch,
    ChannelMismatchError,
    ClassifierConfigError,
    DuplicateRoleError,
    FinalPassError,
    NoAudioFilesError,
    SampleCountMismatchError,
    SampleRateMismatchError,
    UnsupportedChannelConfigError,
)
from .loudness import check, measure
from .models import (
    AllReport,
    AllSummary,
    AutoNullTestResult,
    FileReport,
    Group,
    MECheckResult,
    MEReport,
    GroupError,
    GroupSummary,
    NullReport,
    NullTestResult,
    Report,
    SpecRef,
    Summary,
    UnclassifiedEntry,
)
from .me_check import (
    ANALYSIS_SIGNAL_NAME,
    DEFAULT_ME_BAND_HIGH_HZ,
    DEFAULT_ME_BAND_LOW_HZ,
    DEFAULT_ME_COHERENCE_THRESHOLD,
    DEFAULT_ME_CORR_THRESHOLD,
    DEFAULT_ME_DX_GATE_DBFS,
    DEFAULT_ME_HOP_MS,
    DEFAULT_ME_ME_FLOOR_DBFS,
    DEFAULT_ME_WINDOW_MS,
    analyze_me,
    describe_audio as describe_me_audio,
)
from .null_test import (
    DEFAULT_NULL_HOP_MS,
    DEFAULT_NULL_THRESHOLD_DBFS,
    DEFAULT_NULL_WINDOW_MS,
    analyze_null,
    describe_audio as describe_null_audio,
)
from .specs import Spec, list_bundled, load_spec

_err_console = Console(stderr=True)
_out_console = Console()


@click.group(help="FinalPass — delivery QC for post-production sound.")
@click.version_option(__version__, prog_name="finalpass")
def main() -> None:
    pass


@main.command("loudness")
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--spec", "spec_name", required=True, help="Name of a bundled spec or path to a YAML.")
@click.option("--dx", "dx_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Dialog stem — checked against spec's dialog_lufs target.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress the table; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate (for downstream TC display).")
def loudness_cmd(files: tuple[Path, ...], spec_name: str, dx_file: Path | None, out_dir: Path, json_only: bool, fps: float) -> None:
    try:
        report = _run_loudness(files=files, spec_name=spec_name, dx_file=dx_file, fps=fps)
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    # Schema v1 deliberately omits channel_config_hint/actual (Phase 2+ fields).
    # See models.FileReport for the rationale.
    payload = report.model_dump_json(
        indent=2,
        by_alias=True,
        exclude={
            "files": {"__all__": {"channel_config_hint", "channel_config_actual"}}
        },
    )

    if json_only:
        click.echo(payload)
    else:
        _render_table(report)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(payload + "\n", encoding="utf-8")
        _out_console.print(f"\n[dim]Wrote[/dim] {out_path}")

    sys.exit(0 if report.summary.overall_pass else 1)


@main.group("specs")
def specs_group() -> None:
    """Inspect bundled and user spec presets."""


@specs_group.command("list")
def specs_list_cmd() -> None:
    table = Table(title="Bundled specs")
    table.add_column("name", style="cyan")
    table.add_column("channel_config")
    table.add_column("display_name")
    for spec in list_bundled():
        table.add_row(spec.name, spec.channel_config, spec.display_name)
    _out_console.print(table)


@specs_group.command("show")
@click.argument("name")
def specs_show_cmd(name: str) -> None:
    try:
        spec, source, path = load_spec(name)
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)
    _out_console.print(f"[bold]{spec.display_name}[/bold]  [dim]({source})[/dim]")
    _out_console.print(f"  path: {path}")
    _out_console.print(f"  channel_config: {spec.channel_config}")
    _out_console.print(f"  integrated_lufs: target={spec.integrated_lufs.target} tolerance=±{spec.integrated_lufs.tolerance}")
    if spec.dialog_lufs is not None:
        _out_console.print(f"  dialog_lufs:     target={spec.dialog_lufs.target} tolerance=±{spec.dialog_lufs.tolerance}")
    _out_console.print(f"  true_peak_max_dbtp: {spec.true_peak_max_dbtp}")
    if spec.lra_max is not None:
        _out_console.print(f"  lra_max: {spec.lra_max}")
    if spec.notes:
        _out_console.print("  notes:")
        for line in spec.notes.rstrip().splitlines():
            _out_console.print(f"    {line}")


def _run_loudness(*, files: tuple[Path, ...], spec_name: str, dx_file: Path | None, fps: float) -> Report:
    spec, source, _ = load_spec(spec_name)

    primary_audio = [read_wav(f) for f in files]
    dx_audio = read_wav(dx_file) if dx_file is not None else None

    _validate_channels(primary_audio, spec)
    if dx_audio is not None:
        _validate_channels([dx_audio], spec)

    _validate_homogeneous_sample_rate([*primary_audio, *( [dx_audio] if dx_audio else [] )])

    file_reports: list[FileReport] = []
    for audio in primary_audio:
        m = measure(audio)
        cs = check(m, spec, role="primary")
        file_reports.append(FileReport(
            path=str(audio.path),
            role="primary",
            sample_rate=audio.sample_rate,
            bit_depth=audio.bit_depth,
            channel_count=audio.channel_count,
            duration_seconds=round(audio.duration_seconds, 3),
            measurements=m.as_measurements(),
            checks=cs,
            errors=m.errors,
        ))

    if dx_audio is not None:
        if spec.dialog_lufs is None:
            _err_console.print(
                "[yellow]warning:[/yellow] --dx provided but spec has no dialog_lufs target; ignoring."
            )
        else:
            m = measure(dx_audio)
            cs = check(m, spec, role="dx")
            file_reports.append(FileReport(
                path=str(dx_audio.path),
                role="dx",
                sample_rate=dx_audio.sample_rate,
                bit_depth=dx_audio.bit_depth,
                channel_count=dx_audio.channel_count,
                duration_seconds=round(dx_audio.duration_seconds, 3),
                measurements=m.as_measurements(),
                checks=cs,
                errors=m.errors,
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

    now = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + secrets.token_hex(3)

    return Report(
        finalpass_version=__version__,
        schema_version=LOUDNESS_SCHEMA_VERSION,
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        spec=SpecRef(name=spec.name, display_name=spec.display_name, source=source),
        fps=fps,
        files=file_reports,
        summary=summary,
    )


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


def _render_table(report: Report) -> None:
    _out_console.print(f"[bold]{report.spec.display_name}[/bold]  [dim]({report.spec.source})[/dim]")
    for fr in report.files:
        title = f"{fr.path}  [dim]{fr.role} · {fr.channel_count}ch · {fr.sample_rate}Hz · {fr.bit_depth}-bit · {fr.duration_seconds:.1f}s[/dim]"
        table = Table(title=title, show_lines=False, title_justify="left")
        table.add_column("metric", style="cyan")
        table.add_column("measured", justify="right")
        table.add_column("target / limit", justify="right")
        table.add_column("pass")
        for c in fr.checks:
            measured = "—" if c.measured is None else f"{c.measured:.1f}"
            target_limit = _format_target_limit(c)
            if c.skipped:
                status = f"[dim]— (skipped: {c.reason})[/dim]"
            elif c.pass_:
                status = "[green]PASS[/green]"
            else:
                status = "[red]FAIL[/red]"
            if c.error and not c.skipped:
                status += f" [dim]({c.error})[/dim]"
            table.add_row(c.metric, measured, target_limit, status)
        _out_console.print(table)
    color = "green" if report.summary.overall_pass else "red"
    s = report.summary
    skipped_suffix = f", {s.skipped} skipped" if s.skipped else ""
    _out_console.print(
        f"[{color}]{'PASS' if s.overall_pass else 'FAIL'}[/{color}] — "
        f"{s.passed} passed, {s.failed} failed{skipped_suffix} "
        f"(of {s.total_checks} checks)"
    )


def _format_target_limit(c) -> str:
    if c.target is not None and c.tolerance is not None:
        return f"{c.target} ±{c.tolerance}"
    if c.limit is not None:
        return f"≤ {c.limit}"
    return "—"


# ---------------------------------------------------------------------------
# Phase 3 — `finalpass null` command


@main.command("null")
@click.argument("pm", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("stems", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate for flagged-region timecode fields.")
@click.option("--window-ms", type=float, default=DEFAULT_NULL_WINDOW_MS, show_default=True, help="Residual RMS window size in milliseconds.")
@click.option("--hop-ms", type=float, default=DEFAULT_NULL_HOP_MS, show_default=True, help="Residual RMS hop size in milliseconds.")
@click.option("--threshold-dbfs", type=float, default=DEFAULT_NULL_THRESHOLD_DBFS, show_default=True, help="Flag windows whose residual RMS exceeds this dBFS threshold.")
def null_cmd(pm: Path, stems: tuple[Path, ...], out_dir: Path, json_only: bool, fps: float, window_ms: float, hop_ms: float, threshold_dbfs: float) -> None:
    try:
        report = _run_null(
            pm=pm,
            stems=stems,
            fps=fps,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold_dbfs=threshold_dbfs,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    payload = report.model_dump_json(indent=2, by_alias=True)

    if json_only:
        click.echo(payload)
    else:
        _render_null_report(report)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(payload + "\n", encoding="utf-8")
        _out_console.print(f"\n[dim]Wrote[/dim] {out_path}")

    sys.exit(0 if report.summary.overall_pass else 1)


def _run_null(
    *,
    pm: Path,
    stems: tuple[Path, ...],
    fps: float,
    window_ms: float,
    hop_ms: float,
    threshold_dbfs: float,
) -> NullReport:
    pm_audio = read_wav(pm)
    stem_audios = [read_wav(stem) for stem in stems]
    analysis = analyze_null(
        pm_audio,
        stem_audios,
        fps=fps,
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold_dbfs=threshold_dbfs,
    )
    passed = len(analysis.flags) == 0
    null_test = NullTestResult(
        **{"pass": passed},
        skipped=False,
        reason=None,
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold_dbfs=threshold_dbfs,
        summary=analysis.summary,
        flags=analysis.flags,
        errors=[],
    )
    now, run_id = _run_timestamp()
    return NullReport(
        finalpass_version=__version__,
        schema_version=NULL_SCHEMA_VERSION,
        command="null",
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        fps=fps,
        printmaster=describe_null_audio(pm_audio),
        stems=[describe_null_audio(audio) for audio in stem_audios],
        null_test=null_test,
        summary=Summary(
            total_checks=1,
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            skipped=0,
            overall_pass=passed,
        ),
    )


def _render_null_report(report: NullReport) -> None:
    meta = Table(title="Null Test", title_justify="left", show_header=False)
    meta.add_column("label", style="cyan")
    meta.add_column("value")
    meta.add_row("printmaster", report.printmaster.path)
    meta.add_row("stem count", str(len(report.stems)))
    meta.add_row("sample rate", f"{report.printmaster.sample_rate} Hz")
    meta.add_row(
        "channel config",
        report.printmaster.channel_config_actual or f"{report.printmaster.channel_count}ch",
    )
    meta.add_row(
        "window / hop / threshold",
        f"{report.null_test.window_ms:.0f} ms / {report.null_test.hop_ms:.0f} ms / {report.null_test.threshold_dbfs:.1f} dBFS",
    )
    _out_console.print(meta)

    color = "green" if report.null_test.pass_ else "red"
    verdict = "PASS" if report.null_test.pass_ else "FAIL"
    _out_console.print(f"[{color}]{verdict}[/{color}]")
    if report.null_test.flags:
        _out_console.print(
            _flagged_regions_table(
                report.null_test.flags,
                title="Flagged Regions",
                value_header="peak residual",
                value_formatter=lambda flag: f"{flag.value:.1f} dBFS",
            )
        )


# ---------------------------------------------------------------------------
# Phase 4 — `finalpass me` command


@main.command("me")
@click.argument("me_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dx", "dx_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Matching dialogue stem.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate for flagged-region timecode fields.")
@click.option("--window-ms", type=float, default=DEFAULT_ME_WINDOW_MS, show_default=True, help="Window size in milliseconds.")
@click.option("--hop-ms", type=float, default=DEFAULT_ME_HOP_MS, show_default=True, help="Hop size in milliseconds.")
@click.option("--band-low-hz", type=float, default=DEFAULT_ME_BAND_LOW_HZ, show_default=True, help="Speech-band low cutoff in Hz.")
@click.option("--band-high-hz", type=float, default=DEFAULT_ME_BAND_HIGH_HZ, show_default=True, help="Speech-band high cutoff in Hz.")
@click.option("--corr-threshold", type=float, default=DEFAULT_ME_CORR_THRESHOLD, show_default=True, help="Absolute zero-lag correlation threshold.")
@click.option("--coherence-threshold", type=float, default=DEFAULT_ME_COHERENCE_THRESHOLD, show_default=True, help="Mean coherence threshold.")
@click.option("--dx-gate-dbfs", type=float, default=DEFAULT_ME_DX_GATE_DBFS, show_default=True, help="DX gate threshold in dBFS.")
@click.option("--me-floor-dbfs", type=float, default=DEFAULT_ME_ME_FLOOR_DBFS, show_default=True, help="Minimum M&E level in dBFS.")
def me_cmd(
    me_file: Path,
    dx_file: Path,
    out_dir: Path,
    json_only: bool,
    fps: float,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> None:
    try:
        report = _run_me(
            me_file=me_file,
            dx_file=dx_file,
            fps=fps,
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
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    payload = report.model_dump_json(indent=2, by_alias=True)

    if json_only:
        click.echo(payload)
    else:
        _render_me_report(report)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(payload + "\n", encoding="utf-8")
        _out_console.print(f"\n[dim]Wrote[/dim] {out_path}")

    sys.exit(0 if report.summary.overall_pass else 1)


def _run_me(
    *,
    me_file: Path,
    dx_file: Path,
    fps: float,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> MEReport:
    me_audio = read_wav(me_file)
    dx_audio = read_wav(dx_file)
    analysis = analyze_me(
        me_audio,
        dx_audio,
        fps=fps,
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
    me_check = MECheckResult(
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
        flags=analysis.flags,
        errors=[],
    )
    now, run_id = _run_timestamp()
    return MEReport(
        finalpass_version=__version__,
        schema_version=ME_SCHEMA_VERSION,
        command="me",
        run_id=run_id,
        run_started_at=now.isoformat().replace("+00:00", "Z"),
        fps=fps,
        me_file=describe_me_audio(me_audio),
        dx_file=describe_me_audio(dx_audio),
        me_check=me_check,
        summary=Summary(
            total_checks=1,
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            skipped=0,
            overall_pass=passed,
        ),
    )


def _render_me_report(report: MEReport) -> None:
    meta = Table(title="M&E Check", title_justify="left", show_header=False)
    meta.add_column("label", style="cyan")
    meta.add_column("value")
    meta.add_row("M&E", report.me_file.path)
    meta.add_row("Dx", report.dx_file.path)
    meta.add_row("sample rate", f"{report.me_file.sample_rate} Hz")
    meta.add_row(
        "channel config",
        report.me_file.channel_config_actual or f"{report.me_file.channel_count}ch",
    )
    meta.add_row(
        "window / hop",
        f"{report.me_check.window_ms:.0f} ms / {report.me_check.hop_ms:.0f} ms",
    )
    meta.add_row(
        "band",
        f"{report.me_check.band_low_hz:.0f}-{report.me_check.band_high_hz:.0f} Hz",
    )
    meta.add_row(
        "thresholds",
        (
            f"corr {report.me_check.corr_threshold:.2f} / "
            f"coh {report.me_check.coherence_threshold:.2f} / "
            f"dx {report.me_check.dx_gate_dbfs:.1f} dBFS / "
            f"me {report.me_check.me_floor_dbfs:.1f} dBFS"
        ),
    )
    _out_console.print(meta)

    color = "green" if report.me_check.pass_ else "red"
    verdict = "PASS" if report.me_check.pass_ else "FAIL"
    _out_console.print(f"[{color}]{verdict}[/{color}]")
    if report.me_check.flags:
        _out_console.print(
            _flagged_regions_table(
                report.me_check.flags,
                title="Flagged Regions",
                value_header="bleed score",
                value_formatter=lambda flag: f"{flag.value:.2f}",
            )
        )


# ---------------------------------------------------------------------------
# Phase 4 — `finalpass all` integration


@main.command("all")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--spec", "spec_name", required=True, help="Name of a bundled spec or path to a YAML.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate (for downstream TC display).")
@click.option("--patterns", "patterns_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Override bundled classifier patterns with a user YAML.")
@click.option("--include-unclassified", is_flag=True, help="Measure unclassified files too (integrated/TP/LRA; no dialog check).")
@click.option("--null-window-ms", type=float, default=DEFAULT_NULL_WINDOW_MS, show_default=True, help="Auto-null residual RMS window size in milliseconds.")
@click.option("--null-hop-ms", type=float, default=DEFAULT_NULL_HOP_MS, show_default=True, help="Auto-null residual RMS hop size in milliseconds.")
@click.option("--null-threshold-dbfs", type=float, default=DEFAULT_NULL_THRESHOLD_DBFS, show_default=True, help="Auto-null flag threshold in dBFS.")
@click.option("--me-window-ms", type=float, default=DEFAULT_ME_WINDOW_MS, show_default=True, help="Auto-M&E window size in milliseconds.")
@click.option("--me-hop-ms", type=float, default=DEFAULT_ME_HOP_MS, show_default=True, help="Auto-M&E hop size in milliseconds.")
@click.option("--me-band-low-hz", type=float, default=DEFAULT_ME_BAND_LOW_HZ, show_default=True, help="Auto-M&E speech-band low cutoff in Hz.")
@click.option("--me-band-high-hz", type=float, default=DEFAULT_ME_BAND_HIGH_HZ, show_default=True, help="Auto-M&E speech-band high cutoff in Hz.")
@click.option("--me-corr-threshold", type=float, default=DEFAULT_ME_CORR_THRESHOLD, show_default=True, help="Auto-M&E correlation threshold.")
@click.option("--me-coherence-threshold", type=float, default=DEFAULT_ME_COHERENCE_THRESHOLD, show_default=True, help="Auto-M&E coherence threshold.")
@click.option("--me-dx-gate-dbfs", type=float, default=DEFAULT_ME_DX_GATE_DBFS, show_default=True, help="Auto-M&E DX gate in dBFS.")
@click.option("--me-me-floor-dbfs", type=float, default=DEFAULT_ME_ME_FLOOR_DBFS, show_default=True, help="Auto-M&E M&E floor in dBFS.")
def all_cmd(
    folder: Path,
    spec_name: str,
    out_dir: Path,
    json_only: bool,
    fps: float,
    patterns_path: Path | None,
    include_unclassified: bool,
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
) -> None:
    try:
        report = _run_all(
            folder=folder,
            spec_name=spec_name,
            patterns_path=patterns_path,
            include_unclassified=include_unclassified,
            fps=fps,
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
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    payload = report.model_dump_json(indent=2, by_alias=True)

    if json_only:
        click.echo(payload)
    else:
        _render_all_report(report)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.json"
        out_path.write_text(payload + "\n", encoding="utf-8")
        _out_console.print(f"\n[dim]Wrote[/dim] {out_path}")

    sys.exit(0 if report.summary.overall_pass else 1)

def _run_all(
    *,
    folder: Path,
    spec_name: str,
    patterns_path: Path | None,
    include_unclassified: bool,
    fps: float,
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
    spec, source, _ = load_spec(spec_name)
    cfg = load_config(patterns_path)

    try:
        scan: FolderScan = scan_folder(folder, cfg)
    except AmbiguousClassificationError:
        raise  # caller converts to exit 2

    if not scan.groups and not scan.unclassified:
        raise NoAudioFilesError(
            f"No WAV/BWF files found in {folder}. "
            "Provide a folder containing at least one .wav file."
        )

    groups_out: list[Group] = []
    flat_files: list[FileReport] = []

    for group_id, classified in scan.groups.items():
        group = _process_group(
            group_id=group_id,
            classified=classified,
            spec=spec,
            include_unclassified=include_unclassified,
            fps=fps,
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

    # Any filename whose role was "unknown" is always recorded in
    # `unclassified[]`, even when --include-unclassified added it to a group's
    # measured `files[]`. This preserves a single place to see what wasn't
    # classifiable.
    unclassified = [
        UnclassifiedEntry(path=str(u.path), reason=u.reason)
        for u in scan.unclassified
    ]

    total_checks = sum(g.group_summary.total_checks for g in groups_out)
    passed = sum(g.group_summary.passed for g in groups_out)
    failed = sum(g.group_summary.failed for g in groups_out)
    skipped = sum(g.group_summary.skipped for g in groups_out)
    groups_passed = sum(1 for g in groups_out if g.group_summary.overall_pass and not g.errors)
    groups_failed = len(groups_out) - groups_passed
    overall_pass = failed == 0 and groups_failed == 0

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
        command="all",
        folder=str(folder.resolve()),
        groups=groups_out,
        files=flat_files,
        unclassified=unclassified,
        summary=summary,
    )


def _process_group(
    *,
    group_id: str,
    classified: list[ClassifiedFile],
    spec: Spec,
    include_unclassified: bool,
    fps: float,
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
    """Validate group cardinality, measure the files that make sense for this
    spec, and roll the per-file results up into a :class:`Group`."""
    errors: list[GroupError] = []

    # Cardinality: at most one of each role; the PM role is special-cased below.
    role_to_files: dict[str, list[ClassifiedFile]] = {}
    for cf in classified:
        role_to_files.setdefault(cf.role, []).append(cf)

    for role, files in role_to_files.items():
        if role == "unknown":
            continue
        if len(files) > 1:
            names = ", ".join(f.path.name for f in files)
            errors.append(GroupError(
                type="DuplicateRoleError",
                message=f"group {group_id!r} has {len(files)} files classified as {role!r}: {names}",
            ))

    pm_files = role_to_files.get("pm", [])
    if len(pm_files) == 0:
        errors.append(GroupError(
            type="MissingPrintmaster",
            message=f"group {group_id!r} has no PM (printmaster) file; loudness pass has no primary target.",
        ))

    if errors:
        return Group(
            group_id=group_id,
            files=[],
            group_summary=GroupSummary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=False),
            errors=errors,
        )

    # Probe sample rates for every classified file in the group (not just the
    # ones we'll measure). SR homogeneity is a group-level property — if the
    # Dx is at 44.1 k while the PM is at 48 k, the downstream null test will
    # be nonsense regardless of whether this phase loads the Dx audio.
    rates: dict[str, int] = {}
    for cf in classified:
        try:
            rates[cf.path.name] = sf.info(str(cf.path)).samplerate
        except RuntimeError as exc:
            errors.append(GroupError(
                type="AudioFormatError",
                message=f"{cf.path.name}: could not read audio header: {exc}",
            ))
    if errors:
        return Group(
            group_id=group_id,
            files=[],
            group_summary=GroupSummary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=False),
            errors=errors,
        )
    distinct_rates = set(rates.values())
    if len(distinct_rates) > 1:
        detail = ", ".join(f"{name}={sr}Hz" for name, sr in rates.items())
        errors.append(GroupError(
            type="SampleRateMismatch",
            message=f"group {group_id!r} has mixed sample rates: {detail}",
        ))
        return Group(
            group_id=group_id,
            files=[],
            group_summary=GroupSummary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=False),
            errors=errors,
        )

    # Resolve which files actually get measured in Phase 2.
    targets: list[tuple[ClassifiedFile, str]] = []
    pm = pm_files[0]
    targets.append((pm, "pm"))

    dx_files = role_to_files.get("dx", [])
    if spec.dialog_lufs is not None and dx_files:
        targets.append((dx_files[0], "dx"))

    if include_unclassified:
        for cf in role_to_files.get("unknown", []):
            targets.append((cf, "unknown"))

    audios: list[tuple[ClassifiedFile, str, AudioFile]] = []
    try:
        for cf, role in targets:
            audios.append((cf, role, read_wav(cf.path)))
    except FinalPassError as exc:
        errors.append(GroupError(type=exc.__class__.__name__, message=str(exc)))
        return Group(
            group_id=group_id,
            files=[],
            group_summary=GroupSummary(total_checks=0, passed=0, failed=0, skipped=0, overall_pass=False),
            errors=errors,
        )

    file_reports: list[FileReport] = []
    for cf, role, audio in audios:
        try:
            fr = _measure_classified_file(audio, spec, role, hint=cf.channel_config_hint)
        except ChannelMismatchError as exc:
            errors.append(GroupError(type="ChannelMismatchError", message=str(exc)))
            continue
        file_reports.append(fr)

    null_test = _run_group_null_test(
        role_to_files=role_to_files,
        fps=fps,
        window_ms=null_window_ms,
        hop_ms=null_hop_ms,
        threshold_dbfs=null_threshold_dbfs,
    )
    me_check = _run_group_me_check(
        role_to_files=role_to_files,
        fps=fps,
        window_ms=me_window_ms,
        hop_ms=me_hop_ms,
        band_low_hz=me_band_low_hz,
        band_high_hz=me_band_high_hz,
        corr_threshold=me_corr_threshold,
        coherence_threshold=me_coherence_threshold,
        dx_gate_dbfs=me_dx_gate_dbfs,
        me_floor_dbfs=me_me_floor_dbfs,
    )

    all_checks = [c for fr in file_reports for c in fr.checks]
    passed = sum(1 for c in all_checks if c.pass_ is True)
    failed = sum(1 for c in all_checks if c.pass_ is False)
    skipped = sum(1 for c in all_checks if c.skipped)
    if null_test is not None:
        if null_test.pass_ is True:
            passed += 1
        elif null_test.pass_ is False:
            failed += 1
        if null_test.skipped:
            skipped += 1
    if me_check is not None:
        if me_check.pass_ is True:
            passed += 1
        elif me_check.pass_ is False:
            failed += 1
        if me_check.skipped:
            skipped += 1
    summary = GroupSummary(
        total_checks=len(all_checks) + (1 if null_test is not None else 0) + (1 if me_check is not None else 0),
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=failed == 0 and not errors,
    )

    return Group(
        group_id=group_id,
        files=file_reports,
        null_test=null_test,
        me_check=me_check,
        group_summary=summary,
        errors=errors,
    )


def _measure_classified_file(audio: AudioFile, spec: Spec, role: str, *, hint: str | None) -> FileReport:
    """Measure one file in a group. ``role`` drives which checks run:

    - ``"pm"`` → spec.channel_config must match; runs integrated/TP/LRA.
    - ``"dx"`` → no channel-config check; runs a single ``dialog_lufs`` check.
    - ``"unknown"`` (only reached with --include-unclassified) → no channel
      check; runs integrated/TP/LRA against the general spec targets.

    ``hint`` is the classifier's filename-derived channel_config guess; it is
    recorded on the :class:`FileReport` alongside the actual channel_config
    derived from the audio header. Phase 3 will cross-check the two.
    """
    if role == "pm":
        _validate_channels([audio], spec)
        check_role = "primary"
    elif role == "dx":
        check_role = "dx"
    else:
        check_role = "primary"

    m = measure(audio)
    cs = check(m, spec, role=check_role)

    return FileReport(
        path=str(audio.path),
        role=role,
        sample_rate=audio.sample_rate,
        bit_depth=audio.bit_depth,
        channel_count=audio.channel_count,
        duration_seconds=round(audio.duration_seconds, 3),
        measurements=m.as_measurements(),
        checks=cs,
        errors=m.errors,
        channel_config_hint=hint,
        channel_config_actual=channel_config_from_count(audio.channel_count),
    )


def _run_group_null_test(
    *,
    role_to_files: dict[str, list[ClassifiedFile]],
    fps: float,
    window_ms: float,
    hop_ms: float,
    threshold_dbfs: float,
) -> AutoNullTestResult:
    pm = role_to_files["pm"][0]
    strategy, selected_roles, stems = _select_auto_null_stems(role_to_files)
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
        pm_audio = read_wav(pm.path)
        stem_audios = [read_wav(cf.path) for cf in stems]
        _validate_channel_label_hints([(pm, pm_audio), *zip(stems, stem_audios)])
        analysis = analyze_null(
            pm_audio,
            stem_audios,
            fps=fps,
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
    return AutoNullTestResult(
        **{"pass": passed},
        skipped=False,
        reason=None,
        stem_strategy=strategy,
        selected_roles=selected_roles,
        window_ms=window_ms,
        hop_ms=hop_ms,
        threshold_dbfs=threshold_dbfs,
        summary=analysis.summary,
        flags=analysis.flags,
        errors=[],
    )


def _run_group_me_check(
    *,
    role_to_files: dict[str, list[ClassifiedFile]],
    fps: float,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> MECheckResult:
    dx_files = role_to_files.get("dx", [])
    me_files = role_to_files.get("me", [])
    if not dx_files or not me_files:
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

    dx_file = dx_files[0]
    me_file = me_files[0]
    try:
        dx_audio = read_wav(dx_file.path)
        me_audio = read_wav(me_file.path)
        _validate_channel_label_hints([(dx_file, dx_audio), (me_file, me_audio)])
        analysis = analyze_me(
            me_audio,
            dx_audio,
            fps=fps,
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
    return MECheckResult(
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
        flags=analysis.flags,
        errors=[],
    )


def _select_auto_null_stems(role_to_files: dict[str, list[ClassifiedFile]]) -> tuple[str | None, list[str], list[ClassifiedFile]]:
    if all(role_to_files.get(role) for role in ("dx", "mx", "fx")):
        return "dx_mx_fx", ["dx", "mx", "fx"], [role_to_files["dx"][0], role_to_files["mx"][0], role_to_files["fx"][0]]
    if all(role_to_files.get(role) for role in ("dx", "me")):
        return "dx_me", ["dx", "me"], [role_to_files["dx"][0], role_to_files["me"][0]]
    return None, [], []


def _validate_channel_label_hints(items: list[tuple[ClassifiedFile, AudioFile]]) -> None:
    for classified, audio in items:
        hint = classified.channel_config_hint
        if hint is None:
            continue
        actual = channel_config_from_count(audio.channel_count)
        if hint == actual:
            continue
        actual_label = actual if actual is not None else f"unsupported {audio.channel_count}ch"
        raise ChannelConfigLabelMismatch(
            f"{audio.path.name} hints {hint} but header is {actual_label}."
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


def _flagged_regions_table(flags, *, title: str, value_header: str, value_formatter) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("start", style="cyan")
    table.add_column("end", style="cyan")
    table.add_column("duration", justify="right")
    table.add_column(value_header, justify="right")
    table.add_column("detail")
    for flag in flags:
        table.add_row(
            flag.start_tc,
            flag.end_tc,
            f"{flag.duration_seconds:.2f}s",
            value_formatter(flag),
            flag.detail,
        )
    return table


def _format_stem_strategy(strategy: str | None) -> str:
    if strategy == "dx_mx_fx":
        return "dx+mx+fx"
    if strategy == "dx_me":
        return "dx+me"
    return "—"


def _render_all_report(report: AllReport) -> None:
    _out_console.print(f"[bold]{report.spec.display_name}[/bold]  [dim]({report.spec.source})[/dim]")
    _out_console.print(f"[dim]folder:[/dim] {report.folder}")

    for g in report.groups:
        status_color = "green" if g.group_summary.overall_pass else "red"
        header = f"[{status_color}]●[/{status_color}] [bold]{g.group_id}[/bold]"
        _out_console.print(f"\n{header}")

        for err in g.errors:
            _out_console.print(f"  [red]{err.type}:[/red] {err.message}")

        for fr in g.files:
            title = (
                f"{fr.path}  [dim]{fr.role} · {fr.channel_count}ch · "
                f"{fr.sample_rate}Hz · {fr.bit_depth}-bit · {fr.duration_seconds:.1f}s[/dim]"
            )
            table = Table(title=title, show_lines=False, title_justify="left")
            table.add_column("metric", style="cyan")
            table.add_column("measured", justify="right")
            table.add_column("target / limit", justify="right")
            table.add_column("pass")
            for c in fr.checks:
                measured = "—" if c.measured is None else f"{c.measured:.1f}"
                target_limit = _format_target_limit(c)
                if c.skipped:
                    status = f"[dim]— (skipped: {c.reason})[/dim]"
                elif c.pass_:
                    status = "[green]PASS[/green]"
                else:
                    status = "[red]FAIL[/red]"
                if c.error and not c.skipped:
                    status += f" [dim]({c.error})[/dim]"
                table.add_row(c.metric, measured, target_limit, status)
            _out_console.print(table)

        if g.null_test is not None:
            nt = g.null_test
            if nt.skipped:
                _out_console.print(f"  [dim]Null:[/dim] SKIPPED — {nt.reason}")
            elif nt.pass_ is True and nt.summary is not None:
                _out_console.print(
                    f"  [green]Null:[/green] PASS — {_format_stem_strategy(nt.stem_strategy)} — "
                    f"max residual {nt.summary.max_residual_rms_dbfs:.1f} dBFS"
                )
            elif nt.summary is not None:
                _out_console.print(
                    f"  [red]Null:[/red] FAIL — {nt.summary.flagged_regions} flagged region"
                    f"{'' if nt.summary.flagged_regions == 1 else 's'}"
                )
            else:
                _out_console.print(f"  [red]Null:[/red] FAIL — {nt.reason}")

            for err in nt.errors:
                _out_console.print(f"    [red]{err.type}:[/red] {err.message}")
            if nt.flags:
                _out_console.print(
                    _flagged_regions_table(
                        nt.flags,
                        title="Null Flags",
                        value_header="peak residual",
                        value_formatter=lambda flag: f"{flag.value:.1f} dBFS",
                    )
                )

        if g.me_check is not None:
            mt = g.me_check
            if mt.skipped:
                _out_console.print(f"  [dim]M&E:[/dim] SKIPPED — {mt.reason}")
            elif mt.pass_ is True and mt.summary is not None:
                _out_console.print("  [green]M&E:[/green] PASS — 0 flagged regions")
            elif mt.summary is not None:
                _out_console.print(
                    f"  [red]M&E:[/red] FAIL — {mt.summary.flagged_regions} flagged region"
                    f"{'' if mt.summary.flagged_regions == 1 else 's'}"
                )
            else:
                _out_console.print(f"  [red]M&E:[/red] FAIL — {mt.reason}")

            for err in mt.errors:
                _out_console.print(f"    [red]{err.type}:[/red] {err.message}")
            if mt.flags:
                _out_console.print(
                    _flagged_regions_table(
                        mt.flags,
                        title="M&E Flags",
                        value_header="bleed score",
                        value_formatter=lambda flag: f"{flag.value:.2f}",
                    )
                )

        gs = g.group_summary
        verdict = "PASS" if gs.overall_pass else "FAIL"
        _out_console.print(
            f"  [{status_color}]{verdict}[/{status_color}] — "
            f"{gs.passed} passed, {gs.failed} failed, {gs.skipped} skipped "
            f"(of {gs.total_checks} checks)"
        )

    if report.unclassified:
        _out_console.print("\n[dim]Unclassified (skipped):[/dim]")
        for u in report.unclassified:
            _out_console.print(f"  [dim]{u.path} — {u.reason}[/dim]")

    s = report.summary
    color = "green" if s.overall_pass else "red"
    _out_console.print(
        f"\n[bold]{s.groups_total}[/bold] groups checked — "
        f"[green]{s.groups_passed} passed[/green], [red]{s.groups_failed} failed[/red]. "
        f"Overall: [{color}]{'PASS' if s.overall_pass else 'FAIL'}[/{color}]."
    )


if __name__ == "__main__":
    main()
