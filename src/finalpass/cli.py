"""CLI layer. Library modules return data; this layer prints, writes, and exits.

Exit codes:

* 0 — all checks pass.
* 1 — at least one check failed.
* 2 — tool / validation error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .errors import FinalPassError
from .jobs import (
    WrittenArtifacts,
    run_all as execute_all,
    run_loudness as execute_loudness,
    run_me as execute_me,
    run_null as execute_null,
    write_report_artifacts as persist_report_artifacts,
)
from .me_check import (
    DEFAULT_ME_BAND_HIGH_HZ,
    DEFAULT_ME_BAND_LOW_HZ,
    DEFAULT_ME_COHERENCE_THRESHOLD,
    DEFAULT_ME_CORR_THRESHOLD,
    DEFAULT_ME_DX_GATE_DBFS,
    DEFAULT_ME_HOP_MS,
    DEFAULT_ME_ME_FLOOR_DBFS,
    DEFAULT_ME_WINDOW_MS,
)
from .models import AllReport, FileReport, MEReport, NullReport, Report
from .null_test import (
    DEFAULT_NULL_HOP_MS,
    DEFAULT_NULL_THRESHOLD_DBFS,
    DEFAULT_NULL_WINDOW_MS,
)
from .presentation import (
    blocking_issue_count,
    display_group_name,
    humanize_code,
    humanize_code_list,
    logical_asset_display_name,
    source_summary,
    verdict_explainer,
)
from .specs import list_bundled, load_spec
from .wizard import run_wizard

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
        effective_dx = dx_file
        spec, _, _ = load_spec(spec_name)
        if effective_dx is not None and spec.dialog_lufs is None:
            _err_console.print(
                "[yellow]warning:[/yellow] --dx provided but spec has no dialog_lufs target; ignoring."
            )
            effective_dx = None
        report = execute_loudness(files=files, spec_name=spec_name, dx_file=effective_dx, fps=fps)
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, out_dir=out_dir, json_only=json_only, renderer=_render_loudness_report)


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


@main.command("wizard")
@click.argument("folder", required=False, type=click.Path(path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate used as the wizard default.")
def wizard_cmd(folder: Path | None, out_dir: Path, fps: float) -> None:
    run_wizard(folder=folder, out_dir=out_dir, fps=fps)


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
        report = execute_null(
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

    _finalize_run(report, out_dir=out_dir, json_only=json_only, renderer=_render_null_report)


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
        report = execute_me(
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

    _finalize_run(report, out_dir=out_dir, json_only=json_only, renderer=_render_me_report)


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
        report = execute_all(
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

    _finalize_run(report, out_dir=out_dir, json_only=json_only, renderer=_render_all_report)


def _finalize_run(
    report: Report | NullReport | MEReport | AllReport,
    *,
    out_dir: Path,
    json_only: bool,
    renderer,
) -> None:
    payload = report.model_dump_json(indent=2, by_alias=True)

    if json_only:
        click.echo(payload)
    else:
        try:
            renderer(report)
            written = persist_report_artifacts(report=report, payload=payload, out_dir=out_dir)
            _announce_written_artifacts(written)
        except FinalPassError as exc:
            _err_console.print(f"[red]error:[/red] {exc}")
            sys.exit(2)

    sys.exit(0 if report.summary.overall_pass else 1)


def _render_loudness_report(report: Report) -> None:
    _out_console.print(f"[bold]{report.spec.display_name}[/bold]  [dim]({report.spec.source})[/dim]")
    for file_report in report.files:
        _render_file_checks(file_report)
    _print_summary(report.summary.overall_pass, report.summary.passed, report.summary.failed, report.summary.skipped, report.summary.total_checks)


def _render_null_report(report: NullReport) -> None:
    meta = Table(title="Null Test", title_justify="left", show_header=False)
    meta.add_column("label", style="cyan")
    meta.add_column("value")
    meta.add_row("printmaster", logical_asset_display_name(report.printmaster))
    meta.add_row("source", source_summary(report.printmaster))
    meta.add_row("path", report.printmaster.path)
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


def _render_me_report(report: MEReport) -> None:
    meta = Table(title="M&E Check", title_justify="left", show_header=False)
    meta.add_column("label", style="cyan")
    meta.add_column("value")
    meta.add_row("M&E", logical_asset_display_name(report.me_file))
    meta.add_row("M&E source", source_summary(report.me_file))
    meta.add_row("M&E path", report.me_file.path)
    meta.add_row("DX", logical_asset_display_name(report.dx_file))
    meta.add_row("DX source", source_summary(report.dx_file))
    meta.add_row("DX path", report.dx_file.path)
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


def _render_all_report(report: AllReport) -> None:
    _out_console.print(f"[bold]{report.spec.display_name}[/bold]  [dim]({report.spec.source})[/dim]")
    _out_console.print(f"[dim]folder:[/dim] {escape(report.folder)}", soft_wrap=True)

    if report.discovery_errors:
        _out_console.print("\n[red]Discovery issues:[/red]")
        for issue in report.discovery_errors:
            scope = f" [{display_group_name(issue.group_hint)}]" if issue.group_hint else ""
            _out_console.print(f"  [red]{humanize_code(issue.error_type)}{scope}:[/red] {issue.message}")
            if issue.source_paths:
                for path in issue.source_paths:
                    _out_console.print(f"    [dim]{path}[/dim]")

    for group in report.groups:
        status_color = "green" if group.group_summary.overall_pass else "red"
        _out_console.print(f"\n[{status_color}]●[/{status_color}] [bold]{display_group_name(group.group_id)}[/bold]")

        for error in group.errors:
            _out_console.print(f"  [red]{humanize_code(error.type)}:[/red] {error.message}")

        if group.assets:
            _out_console.print("  [dim]Assets:[/dim]")
            for asset in group.assets:
                used_by = humanize_code_list(asset.used_by)
                note = humanize_code(asset.selection_note)
                _out_console.print(
                    "    "
                    f"{humanize_code(asset.role):<12} {escape(logical_asset_display_name(asset))}  "
                    f"[dim]{escape(source_summary(asset))}[/dim]"
                )
                _out_console.print(
                    "           "
                    f"[dim]path:[/dim] {escape(asset.path)}  "
                    f"[dim]used:[/dim] {escape(used_by)}  "
                    f"[dim]note:[/dim] {escape(note)}",
                    soft_wrap=True,
                )

        for file_report in group.files:
            _render_file_checks(file_report)

        if group.null_test is not None:
            null_test = group.null_test
            if null_test.skipped:
                _out_console.print(f"  [dim]Null:[/dim] SKIPPED — {humanize_code(null_test.reason)}")
            elif null_test.pass_ is True and null_test.summary is not None:
                _out_console.print(
                    f"  [green]Null:[/green] PASS — {_format_stem_strategy(null_test.stem_strategy)} — "
                    f"max residual {null_test.summary.max_residual_rms_dbfs:.1f} dBFS"
                )
            elif null_test.summary is not None:
                flagged = null_test.summary.flagged_regions
                noun = "region" if flagged == 1 else "regions"
                _out_console.print(f"  [red]Null:[/red] FAIL — {flagged} flagged {noun}")
            else:
                _out_console.print(f"  [red]Null:[/red] FAIL — {humanize_code(null_test.reason)}")

            for error in null_test.errors:
                _out_console.print(f"    [red]{humanize_code(error.type)}:[/red] {error.message}")
            if null_test.flags:
                _out_console.print(
                    _flagged_regions_table(
                        null_test.flags,
                        title="Null Flags",
                        value_header="peak residual",
                        value_formatter=lambda flag: f"{flag.value:.1f} dBFS",
                    )
                )

        if group.me_check is not None:
            me_check = group.me_check
            if me_check.skipped:
                _out_console.print(f"  [dim]M&E:[/dim] SKIPPED — {humanize_code(me_check.reason)}")
            elif me_check.pass_ is True and me_check.summary is not None:
                _out_console.print("  [green]M&E:[/green] PASS — 0 flagged regions")
            elif me_check.summary is not None:
                flagged = me_check.summary.flagged_regions
                noun = "region" if flagged == 1 else "regions"
                _out_console.print(f"  [red]M&E:[/red] FAIL — {flagged} flagged {noun}")
            else:
                _out_console.print(f"  [red]M&E:[/red] FAIL — {humanize_code(me_check.reason)}")

            for error in me_check.errors:
                _out_console.print(f"    [red]{humanize_code(error.type)}:[/red] {error.message}")
            if me_check.flags:
                _out_console.print(
                    _flagged_regions_table(
                        me_check.flags,
                        title="M&E Flags",
                        value_header="bleed score",
                        value_formatter=lambda flag: f"{flag.value:.2f}",
                    )
                )

        _print_summary(
            group.group_summary.overall_pass,
            group.group_summary.passed,
            group.group_summary.failed,
            group.group_summary.skipped,
            group.group_summary.total_checks,
            blocking_issues=blocking_issue_count(group),
            prefix="  ",
        )

    if report.unclassified:
        _out_console.print("\n[dim]Unclassified logical assets:[/dim]")
        for item in report.unclassified:
            _out_console.print(
                "  "
                f"{escape(logical_asset_display_name(item))}  "
                f"[dim]{escape(source_summary(item))}[/dim]"
            )
            _out_console.print(
                f"    [dim]path:[/dim] {escape(item.path)}  [dim]reason:[/dim] {escape(humanize_code(item.reason))}",
                soft_wrap=True,
            )

    summary = report.summary
    color = "green" if summary.overall_pass else "red"
    _out_console.print(
        f"\n[bold]{summary.groups_total}[/bold] groups checked — "
        f"[green]{summary.groups_passed} passed[/green], [red]{summary.groups_failed} failed[/red]. "
        f"Overall: [{color}]{verdict_explainer(overall_pass=summary.overall_pass, failed=summary.failed, blocking_issues=blocking_issue_count(report))}[/{color}]."
    )


def _render_file_checks(file_report: FileReport) -> None:
    _out_console.print(_file_summary_heading(file_report))
    _out_console.print(_file_source_line(file_report))
    _out_console.print(_file_path_line(file_report), soft_wrap=True)
    _out_console.print(_checks_table(file_report))


def _file_summary_heading(file_report: FileReport) -> str:
    name = escape(logical_asset_display_name(file_report))
    meta = escape(
        f"{file_report.role} · {file_report.channel_count}ch · {file_report.sample_rate}Hz · "
        f"{file_report.bit_depth}-bit · {file_report.duration_seconds:.1f}s"
    )
    return f"[bold]{name}[/bold]  [dim]{meta}[/dim]"


def _file_source_line(file_report: FileReport) -> str:
    return f"[dim]source:[/dim] {escape(source_summary(file_report))}"


def _file_path_line(file_report: FileReport) -> str:
    return f"[dim]path:[/dim] {escape(file_report.path)}"


def _checks_table(file_report: FileReport) -> Table:
    table = Table(show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("measured", justify="right")
    table.add_column("target / limit", justify="right")
    table.add_column("pass")
    for check in file_report.checks:
        measured = "—" if check.measured is None else f"{check.measured:.1f}"
        target_limit = _format_target_limit(check)
        if check.skipped:
            status = f"[dim]— (skipped: {check.reason})[/dim]"
        elif check.pass_:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
        if check.error and not check.skipped:
            status += f" [dim]({check.error})[/dim]"
        table.add_row(check.metric, measured, target_limit, status)
    return table


def _announce_written_artifacts(written: WrittenArtifacts) -> None:
    _out_console.print(f"\n[dim]Wrote[/dim] {written.json_path}")
    _out_console.print(f"[dim]Wrote[/dim] {written.html_path}")
    if written.aaf_path is None:
        _out_console.print(
            f"[dim]No exportable timed markers; did not write[/dim] {written.html_path.parent / 'markers.aaf'}"
        )
    else:
        _out_console.print(f"[dim]Wrote[/dim] {written.aaf_path}")


def _format_target_limit(check) -> str:
    if check.target is not None and check.tolerance is not None:
        return f"{check.target} ±{check.tolerance}"
    if check.limit is not None:
        return f"≤ {check.limit}"
    return "—"


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


def _print_summary(
    overall_pass: bool,
    passed: int,
    failed: int,
    skipped: int,
    total_checks: int,
    *,
    blocking_issues: int = 0,
    prefix: str = "",
) -> None:
    color = "green" if overall_pass else "red"
    skipped_suffix = f", {skipped} skipped" if skipped else ""
    _out_console.print(
        f"{prefix}[{color}]{'PASS' if overall_pass else 'FAIL'}[/{color}] — "
        f"{passed} passed, {failed} failed{skipped_suffix}, {blocking_issues} blocking "
        f"{'issue' if blocking_issues == 1 else 'issues'} (of {total_checks} checks)"
    )


if __name__ == "__main__":
    main()
