"""CLI layer. Library modules return data; this layer prints, writes, and exits.

Exit codes:

* 0 — all checks pass.
* 1 — at least one check failed.
* 2 — tool / validation error.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .channel_check import (
    DEFAULT_CHANNELS_ACTIVITY_DBFS,
    DEFAULT_CHANNELS_DUPLICATE_NULL_DB,
    DEFAULT_CHANNELS_HOP_MS,
    DEFAULT_CHANNELS_IMBALANCE_DB,
    DEFAULT_CHANNELS_LFE_CUTOFF_HZ,
    DEFAULT_CHANNELS_LFE_ENERGY_RATIO,
    DEFAULT_CHANNELS_POLARITY_CORR,
    DEFAULT_CHANNELS_SILENCE_DBFS,
    DEFAULT_CHANNELS_WINDOW_MS,
    ChannelsTunables,
)
from .downmix_check import (
    DEFAULT_DOWNMIX_ACTIVITY_DBFS,
    DEFAULT_DOWNMIX_CENTER_DB,
    DEFAULT_DOWNMIX_HOP_MS,
    DEFAULT_DOWNMIX_LFE_LOWPASS_HZ,
    DEFAULT_DOWNMIX_LOUDNESS_DELTA_LU,
    DEFAULT_DOWNMIX_MONO_CORR,
    DEFAULT_DOWNMIX_SIMILARITY_CORR,
    DEFAULT_DOWNMIX_SURROUND_DB,
    DEFAULT_DOWNMIX_WINDOW_MS,
    DownmixTunables,
)
from .errors import FinalPassError
from .jobs import (
    WrittenArtifacts,
    run_all as execute_all,
    run_channels as execute_channels,
    run_downmix as execute_downmix,
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
    METunables,
)
from .models import AllReport, ChannelAssetResult, ChannelsReport, DownmixReport, FileReport, MEReport, NullReport, Report
from .null_test import (
    DEFAULT_NULL_HOP_MS,
    DEFAULT_NULL_THRESHOLD_DBFS,
    DEFAULT_NULL_WINDOW_MS,
    NullTunables,
)
from .presentation import (
    blocking_issue_count,
    display_group_name,
    format_metric_value,
    format_target_limit,
    humanize_code,
    humanize_code_list,
    logical_asset_display_name,
    metric_value_header,
    source_summary,
    stem_strategy_label,
    verdict_explainer,
)
from .run_settings import AllSettings
from .specs import list_bundled, load_spec
from .terminal_spinner import processing_spinner
from .timecode import timecode_mode
from .wizard import run_wizard

_err_console = Console(stderr=True)
_out_console = Console()

# Terminal readability cap only; the JSON and HTML reports always carry the
# full flagged-region list.
_FLAG_TABLE_MAX_ROWS = 20


def _warn_if_drop_frame_rate(fps: float, drop_frame: bool) -> None:
    """One-line honesty note at DF-capable rates when the run is non-drop."""
    if drop_frame:
        return
    mode = timecode_mode(fps)
    if mode.edit_rate.denominator == 1001 and mode.nominal_fps in (30, 60):
        _err_console.print(
            "[yellow]note:[/yellow] timecode strings are non-drop; if this is a "
            "drop-frame show, rerun with --drop-frame (a drop-frame session "
            "counter runs ~3.6s per hour ahead of non-drop labels)."
        )


def _validate_fps_option(_ctx: click.Context, _param: click.Parameter, value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise click.BadParameter("must be a finite number greater than zero")
    return value


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
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate (for downstream TC display).")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
def loudness_cmd(files: tuple[Path, ...], spec_name: str, dx_file: Path | None, out_dir: Path, json_only: bool, fps: float, drop_frame: bool) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        effective_dx = dx_file
        spec, _, _ = load_spec(spec_name)
        if effective_dx is not None and spec.dialog_lufs is None:
            _err_console.print(
                "[yellow]warning:[/yellow] --dx provided but spec has no dialog_lufs target; ignoring."
            )
            effective_dx = None
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_loudness(files=files, spec_name=spec_name, dx_file=effective_dx, fps=fps, drop_frame=drop_frame),
            message="Running loudness...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_loudness_report)


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
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate used as the wizard default.")
def wizard_cmd(folder: Path | None, out_dir: Path, fps: float) -> None:
    run_wizard(folder=folder, out_dir=out_dir, fps=fps)


@main.command("null")
@click.argument("pm", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("stems", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate for flagged-region timecode fields.")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
@click.option("--window-ms", type=float, default=DEFAULT_NULL_WINDOW_MS, show_default=True, help="Residual RMS window size in milliseconds.")
@click.option("--hop-ms", type=float, default=DEFAULT_NULL_HOP_MS, show_default=True, help="Residual RMS hop size in milliseconds.")
@click.option("--threshold-dbfs", type=float, default=DEFAULT_NULL_THRESHOLD_DBFS, show_default=True, help="Flag windows whose residual RMS exceeds this dBFS threshold.")
def null_cmd(pm: Path, stems: tuple[Path, ...], out_dir: Path, json_only: bool, fps: float, drop_frame: bool, window_ms: float, hop_ms: float, threshold_dbfs: float) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_null(
                pm=pm,
                stems=stems,
                fps=fps,
                drop_frame=drop_frame,
                tunables=NullTunables(
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                    threshold_dbfs=threshold_dbfs,
                ),
            ),
            message="Running null check...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_null_report)


@main.command("me")
@click.argument("me_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dx", "dx_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Matching dialogue stem.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate for flagged-region timecode fields.")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
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
    drop_frame: bool,
    window_ms: float,
    hop_ms: float,
    band_low_hz: float,
    band_high_hz: float,
    corr_threshold: float,
    coherence_threshold: float,
    dx_gate_dbfs: float,
    me_floor_dbfs: float,
) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_me(
                me_file=me_file,
                dx_file=dx_file,
                fps=fps,
                drop_frame=drop_frame,
                tunables=METunables(
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                    band_low_hz=band_low_hz,
                    band_high_hz=band_high_hz,
                    corr_threshold=corr_threshold,
                    coherence_threshold=coherence_threshold,
                    dx_gate_dbfs=dx_gate_dbfs,
                    me_floor_dbfs=me_floor_dbfs,
                ),
            ),
            message="Running M&E check...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_me_report)


@main.command("channels")
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate (recorded in the report; channel findings are untimed).")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
@click.option("--window-ms", type=float, default=DEFAULT_CHANNELS_WINDOW_MS, show_default=True, help="Pairwise statistics window size in milliseconds.")
@click.option("--hop-ms", type=float, default=DEFAULT_CHANNELS_HOP_MS, show_default=True, help="Pairwise statistics hop size in milliseconds.")
@click.option("--activity-dbfs", type=float, default=DEFAULT_CHANNELS_ACTIVITY_DBFS, show_default=True, help="Both channels must exceed this window RMS to be compared.")
@click.option("--silence-dbfs", type=float, default=DEFAULT_CHANNELS_SILENCE_DBFS, show_default=True, help="Full-span RMS below this marks a leg silent.")
@click.option("--duplicate-null-db", type=float, default=DEFAULT_CHANNELS_DUPLICATE_NULL_DB, show_default=True, help="Median phase-invert null depth marking a duplicated pair.")
@click.option("--polarity-corr", type=float, default=DEFAULT_CHANNELS_POLARITY_CORR, show_default=True, help="Median signed correlation marking a polarity inversion.")
@click.option("--lfe-cutoff-hz", type=float, default=DEFAULT_CHANNELS_LFE_CUTOFF_HZ, show_default=True, help="LFE energy above this frequency counts as broadband.")
@click.option("--lfe-energy-ratio", type=float, default=DEFAULT_CHANNELS_LFE_ENERGY_RATIO, show_default=True, help="Fraction of LFE energy above the cutoff that fails.")
@click.option("--imbalance-db", type=float, default=DEFAULT_CHANNELS_IMBALANCE_DB, show_default=True, help="L/R full-span RMS delta that produces an informational note.")
@click.option("--fail-dual-mono", is_flag=True, help="Treat a dual-mono stereo asset as a failure instead of a notification.")
def channels_cmd(
    files: tuple[Path, ...],
    out_dir: Path,
    json_only: bool,
    fps: float,
    drop_frame: bool,
    window_ms: float,
    hop_ms: float,
    activity_dbfs: float,
    silence_dbfs: float,
    duplicate_null_db: float,
    polarity_corr: float,
    lfe_cutoff_hz: float,
    lfe_energy_ratio: float,
    imbalance_db: float,
    fail_dual_mono: bool,
) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_channels(
                files=files,
                fps=fps,
                drop_frame=drop_frame,
                tunables=ChannelsTunables(
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                    activity_dbfs=activity_dbfs,
                    silence_dbfs=silence_dbfs,
                    duplicate_null_db=duplicate_null_db,
                    polarity_corr=polarity_corr,
                    lfe_cutoff_hz=lfe_cutoff_hz,
                    lfe_energy_ratio=lfe_energy_ratio,
                    imbalance_db=imbalance_db,
                    fail_dual_mono=fail_dual_mono,
                ),
            ),
            message="Running channel integrity...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_channels_report)


@main.command("breaths")
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--list", "list_mode", type=click.Choice(["mouth-click", "t-inhale", "grade-3", "all"]), default="mouth-click", show_default=True, help="Which breaths to list in the terminal (t-inhale is the old name for mouth-click). The text file always lists every breath.")
@click.option("--min-ms", type=float, default=150.0, show_default=True, help="Ignore breaths shorter than this; shorter ones are too brief to judge.")
@click.option("--no-mouth-click", "--no-t-inhale", "no_t_inhale", is_flag=True, help="Do not flag mouth-click inhales (breaths opening with a mouth click) as a client QC risk.")
@click.option("--pause-db", type=float, default=-60.0, show_default=True, help="A breath must fade into a pause at or below this level (dBFS) before the next word.")
@click.option("--pause-ms", type=float, default=10.0, show_default=True, help="Shortest pause that counts, in milliseconds.")
def breaths_cmd(files: tuple[Path, ...], out_dir: Path, json_only: bool, list_mode: str, min_ms: float, no_t_inhale: bool, pause_db: float, pause_ms: float) -> None:
    """Audiobook breath check: find, grade (1-3) and flag breaths. Informational only."""
    from .breath_check import BreathTunables
    from .jobs import run_breaths, write_breath_artifacts

    if min_ms <= 0 or pause_ms <= 0:
        _err_console.print("[red]error:[/red] --min-ms and --pause-ms must be positive.")
        sys.exit(2)
    tunables = BreathTunables(min_breath_ms=min_ms, t_inhale=not no_t_inhale,
                              pause_dbfs=pause_db, pause_min_ms=pause_ms)
    try:
        if json_only:
            report = run_breaths(files=files, tunables=tunables)
        else:
            with _err_console.status("Running breath check..."):
                report = run_breaths(files=files, tunables=tunables)
        json_path, text_path = write_breath_artifacts(report, out_dir)
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)
    if json_only:
        click.echo(report.model_dump_json(indent=2))
        sys.exit(0)
    _render_breaths_report(report, list_mode=list_mode)
    _out_console.print(f"\nWrote {text_path}\nWrote {json_path}")
    sys.exit(0)


def _render_breaths_report(report, *, list_mode: str) -> None:
    from .breath_check import summary_line

    for asset in report.assets:
        _out_console.print(f"\n[bold]{Path(asset.path).name}[/bold]  [dim]{asset.duration_seconds:.1f}s[/dim]")
        _out_console.print(f"  {summary_line(asset.counts)}")
        for note in asset.notes:
            _out_console.print(f"  [dim]note: {note}[/dim]")
        rows = [e for e in asset.breaths
                if list_mode == "all" or e.t_inhale or (list_mode == "grade-3" and e.grade == 3)]
        if not rows:
            continue
        table = Table(show_header=True, header_style="bold")
        table.add_column("time")
        table.add_column("length", justify="right")
        table.add_column("grade", justify="center")
        table.add_column("flag")
        for e in rows:
            table.add_row(e.start_time, f"{e.duration_ms} ms", str(e.grade),
                          "[yellow]mouth-click inhale — client QC risk[/yellow]" if e.t_inhale else "")
        _out_console.print(table)
    _out_console.print(f"\n[bold]All files:[/bold] {summary_line(report.summary)}")
    _out_console.print("[dim]Informational only — nothing here fails a delivery.[/dim]")


@main.command("downmix")
@click.argument("stereo", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("surround", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate for flagged-region timecode fields.")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
@click.option("--center-db", type=float, default=DEFAULT_DOWNMIX_CENTER_DB, show_default=True, help="Center gain into each derived channel.")
@click.option("--surround-db", type=float, default=DEFAULT_DOWNMIX_SURROUND_DB, show_default=True, help="Gain per surround leg into its side.")
@click.option("--lfe-db", type=float, default=None, help="Include the LFE at this gain (omitted by default).")
@click.option("--lfe-lowpass-hz", type=float, default=DEFAULT_DOWNMIX_LFE_LOWPASS_HZ, show_default=True, help="LFE low-pass cutoff when --lfe-db is given.")
@click.option("--window-ms", type=float, default=DEFAULT_DOWNMIX_WINDOW_MS, show_default=True, help="Similarity / mono-compatibility window size in milliseconds.")
@click.option("--hop-ms", type=float, default=DEFAULT_DOWNMIX_HOP_MS, show_default=True, help="Similarity / mono-compatibility hop size in milliseconds.")
@click.option("--activity-dbfs", type=float, default=DEFAULT_DOWNMIX_ACTIVITY_DBFS, show_default=True, help="Windows quieter than this are not compared.")
@click.option("--similarity-corr", type=float, default=DEFAULT_DOWNMIX_SIMILARITY_CORR, show_default=True, help="Flag windows whose derived-vs-delivered correlation falls below this.")
@click.option("--mono-corr", type=float, default=DEFAULT_DOWNMIX_MONO_CORR, show_default=True, help="Flag windows whose delivered L/R correlation falls below this.")
@click.option("--loudness-delta-lu", type=float, default=DEFAULT_DOWNMIX_LOUDNESS_DELTA_LU, show_default=True, help="Fail when the integrated-loudness difference exceeds this.")
def downmix_cmd(
    stereo: Path,
    surround: Path,
    out_dir: Path,
    json_only: bool,
    fps: float,
    drop_frame: bool,
    center_db: float,
    surround_db: float,
    lfe_db: float | None,
    lfe_lowpass_hz: float,
    window_ms: float,
    hop_ms: float,
    activity_dbfs: float,
    similarity_corr: float,
    mono_corr: float,
    loudness_delta_lu: float,
) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_downmix(
                stereo=stereo,
                surround=surround,
                fps=fps,
                drop_frame=drop_frame,
                tunables=DownmixTunables(
                    center_db=center_db,
                    surround_db=surround_db,
                    lfe_db=lfe_db,
                    lfe_lowpass_hz=lfe_lowpass_hz,
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                    activity_dbfs=activity_dbfs,
                    similarity_corr=similarity_corr,
                    mono_corr=mono_corr,
                    loudness_delta_lu=loudness_delta_lu,
                ),
            ),
            message="Running downmix check...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_downmix_report)


@main.command("all")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--spec", "spec_name", required=True, help="Name of a bundled spec or path to a YAML.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, callback=_validate_fps_option, help="Frame rate (for downstream TC display).")
@click.option("--drop-frame", "drop_frame", is_flag=True, help="Count timecode as SMPTE drop-frame (29.97/59.94 only; HH:MM:SS;FF).")
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
@click.option("--skip-channels", is_flag=True, help="Skip the per-asset channel-integrity pass (it reads every classified asset).")
@click.option("--ch-window-ms", type=float, default=DEFAULT_CHANNELS_WINDOW_MS, show_default=True, help="Channels pairwise window size in milliseconds.")
@click.option("--ch-hop-ms", type=float, default=DEFAULT_CHANNELS_HOP_MS, show_default=True, help="Channels pairwise hop size in milliseconds.")
@click.option("--ch-activity-dbfs", type=float, default=DEFAULT_CHANNELS_ACTIVITY_DBFS, show_default=True, help="Channels window activity floor in dBFS.")
@click.option("--ch-silence-dbfs", type=float, default=DEFAULT_CHANNELS_SILENCE_DBFS, show_default=True, help="Channels silent-leg threshold in dBFS.")
@click.option("--ch-duplicate-null-db", type=float, default=DEFAULT_CHANNELS_DUPLICATE_NULL_DB, show_default=True, help="Channels duplicate null-depth threshold in dB.")
@click.option("--ch-polarity-corr", type=float, default=DEFAULT_CHANNELS_POLARITY_CORR, show_default=True, help="Channels polarity correlation threshold.")
@click.option("--ch-lfe-cutoff-hz", type=float, default=DEFAULT_CHANNELS_LFE_CUTOFF_HZ, show_default=True, help="Channels LFE broadband cutoff in Hz.")
@click.option("--ch-lfe-energy-ratio", type=float, default=DEFAULT_CHANNELS_LFE_ENERGY_RATIO, show_default=True, help="Channels LFE broadband energy ratio.")
@click.option("--ch-imbalance-db", type=float, default=DEFAULT_CHANNELS_IMBALANCE_DB, show_default=True, help="Channels L/R imbalance note threshold in dB.")
@click.option("--ch-fail-dual-mono", is_flag=True, help="Treat dual-mono stereo assets as failures during the channels pass.")
@click.option("--dm-center-db", type=float, default=DEFAULT_DOWNMIX_CENTER_DB, show_default=True, help="Downmix center fold gain.")
@click.option("--dm-surround-db", type=float, default=DEFAULT_DOWNMIX_SURROUND_DB, show_default=True, help="Downmix surround fold gain.")
@click.option("--dm-lfe-db", type=float, default=None, help="Include the LFE in the derived fold-down at this gain.")
@click.option("--dm-lfe-lowpass-hz", type=float, default=DEFAULT_DOWNMIX_LFE_LOWPASS_HZ, show_default=True, help="Downmix LFE low-pass cutoff in Hz.")
@click.option("--dm-window-ms", type=float, default=DEFAULT_DOWNMIX_WINDOW_MS, show_default=True, help="Downmix window size in milliseconds.")
@click.option("--dm-hop-ms", type=float, default=DEFAULT_DOWNMIX_HOP_MS, show_default=True, help="Downmix hop size in milliseconds.")
@click.option("--dm-activity-dbfs", type=float, default=DEFAULT_DOWNMIX_ACTIVITY_DBFS, show_default=True, help="Downmix window activity floor in dBFS.")
@click.option("--dm-similarity-corr", type=float, default=DEFAULT_DOWNMIX_SIMILARITY_CORR, show_default=True, help="Downmix similarity correlation threshold.")
@click.option("--dm-mono-corr", type=float, default=DEFAULT_DOWNMIX_MONO_CORR, show_default=True, help="Downmix mono-compatibility correlation threshold.")
@click.option("--dm-loudness-delta-lu", type=float, default=DEFAULT_DOWNMIX_LOUDNESS_DELTA_LU, show_default=True, help="Downmix integrated-loudness delta threshold in LU.")
def all_cmd(
    folder: Path,
    spec_name: str,
    out_dir: Path,
    json_only: bool,
    fps: float,
    drop_frame: bool,
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
    skip_channels: bool,
    ch_window_ms: float,
    ch_hop_ms: float,
    ch_activity_dbfs: float,
    ch_silence_dbfs: float,
    ch_duplicate_null_db: float,
    ch_polarity_corr: float,
    ch_lfe_cutoff_hz: float,
    ch_lfe_energy_ratio: float,
    ch_imbalance_db: float,
    ch_fail_dual_mono: bool,
    dm_center_db: float,
    dm_surround_db: float,
    dm_lfe_db: float | None,
    dm_lfe_lowpass_hz: float,
    dm_window_ms: float,
    dm_hop_ms: float,
    dm_activity_dbfs: float,
    dm_similarity_corr: float,
    dm_mono_corr: float,
    dm_loudness_delta_lu: float,
) -> None:
    _warn_if_drop_frame_rate(fps, drop_frame)
    try:
        report, payload, written = _execute_direct_job(
            runner=lambda: execute_all(
                folder=folder,
                spec_name=spec_name,
                patterns_path=patterns_path,
                fps=fps,
                drop_frame=drop_frame,
                settings=AllSettings(
                    include_unclassified=include_unclassified,
                    skip_channels=skip_channels,
                    channels=ChannelsTunables(
                        window_ms=ch_window_ms,
                        hop_ms=ch_hop_ms,
                        activity_dbfs=ch_activity_dbfs,
                        silence_dbfs=ch_silence_dbfs,
                        duplicate_null_db=ch_duplicate_null_db,
                        polarity_corr=ch_polarity_corr,
                        lfe_cutoff_hz=ch_lfe_cutoff_hz,
                        lfe_energy_ratio=ch_lfe_energy_ratio,
                        imbalance_db=ch_imbalance_db,
                        fail_dual_mono=ch_fail_dual_mono,
                    ),
                    downmix=DownmixTunables(
                        center_db=dm_center_db,
                        surround_db=dm_surround_db,
                        lfe_db=dm_lfe_db,
                        lfe_lowpass_hz=dm_lfe_lowpass_hz,
                        window_ms=dm_window_ms,
                        hop_ms=dm_hop_ms,
                        activity_dbfs=dm_activity_dbfs,
                        similarity_corr=dm_similarity_corr,
                        mono_corr=dm_mono_corr,
                        loudness_delta_lu=dm_loudness_delta_lu,
                    ),
                    null=NullTunables(
                        window_ms=null_window_ms,
                        hop_ms=null_hop_ms,
                        threshold_dbfs=null_threshold_dbfs,
                    ),
                    me=METunables(
                        window_ms=me_window_ms,
                        hop_ms=me_hop_ms,
                        band_low_hz=me_band_low_hz,
                        band_high_hz=me_band_high_hz,
                        corr_threshold=me_corr_threshold,
                        coherence_threshold=me_coherence_threshold,
                        dx_gate_dbfs=me_dx_gate_dbfs,
                        me_floor_dbfs=me_me_floor_dbfs,
                    ),
                ),
            ),
            message="Running folder analysis...",
            out_dir=out_dir,
            json_only=json_only,
        )
    except FinalPassError as exc:
        _err_console.print(f"[red]error:[/red] {exc}")
        sys.exit(2)

    _finalize_run(report, payload=payload, written=written, json_only=json_only, renderer=_render_all_report)


def _execute_direct_job(
    *,
    runner,
    message: str,
    out_dir: Path,
    json_only: bool,
) -> tuple[Report | NullReport | MEReport | AllReport | ChannelsReport | DownmixReport, str, WrittenArtifacts | None]:
    with processing_spinner(message, stream=_err_console.file, enabled=not json_only):
        report = runner()
        payload = report.model_dump_json(indent=2, by_alias=True)
        written = None
        if not json_only:
            written = persist_report_artifacts(report=report, payload=payload, out_dir=out_dir)
    return report, payload, written


def _finalize_run(
    report: Report | NullReport | MEReport | AllReport | ChannelsReport | DownmixReport,
    *,
    payload: str,
    written: WrittenArtifacts | None,
    json_only: bool,
    renderer,
) -> None:
    if json_only:
        click.echo(payload)
    else:
        renderer(report)
        if written is not None:
            _announce_written_artifacts(written)

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
            )
        )


def _render_channels_report(report: ChannelsReport) -> None:
    for asset in report.assets:
        _out_console.print(_channels_asset_heading(asset))
        _out_console.print(f"[dim]source:[/dim] {escape(source_summary(asset))}")
        _out_console.print(f"[dim]path:[/dim] {escape(asset.path)}", soft_wrap=True)
        for note in asset.notes:
            _out_console.print(f"  [dim]note:[/dim] {escape(humanize_code(note))}")
        for error in asset.errors:
            _out_console.print(f"  [red]{humanize_code(error.type)}:[/red] {error.message}")

        if asset.findings:
            _out_console.print(_channel_findings_table(asset.findings))
        else:
            _out_console.print("[dim]No channel integrity findings.[/dim]")

        if asset.skipped:
            _out_console.print(f"[dim]SKIPPED[/dim] — {humanize_code(asset.reason)}")
        else:
            color = "green" if asset.pass_ else "red"
            _out_console.print(f"[{color}]{'PASS' if asset.pass_ else 'FAIL'}[/{color}]")

    _print_summary(
        report.summary.overall_pass,
        report.summary.passed,
        report.summary.failed,
        report.summary.skipped,
        report.summary.total_checks,
    )


def _channels_asset_heading(asset: ChannelAssetResult) -> str:
    name = escape(logical_asset_display_name(asset))
    meta = escape(
        f"{asset.channel_config_actual or f'{asset.channel_count}ch'} · "
        f"{asset.sample_rate}Hz · {asset.bit_depth}-bit · {asset.duration_seconds:.1f}s"
    )
    return f"[bold]{name}[/bold]  [dim]{meta}[/dim]"


def _channel_findings_table(findings) -> Table:
    table = Table(title="Channel Findings", title_justify="left")
    table.add_column("finding", style="cyan")
    table.add_column("severity")
    table.add_column("channels")
    table.add_column("measured vs threshold", justify="right")
    table.add_column("detail")
    for finding in findings:
        if finding.skipped:
            severity = f"[dim]skipped: {humanize_code(finding.reason)}[/dim]"
            measured = "[dim]—[/dim]"
        elif finding.severity == "fail":
            severity = "[red]FAIL[/red]"
            measured = _channel_measured_label(finding)
        else:
            severity = "[dim]info[/dim]"
            measured = f"[dim]{_channel_measured_label(finding)}[/dim]"
        detail = escape(finding.detail)
        table.add_row(
            humanize_code(finding.kind),
            severity,
            ", ".join(finding.channels) or "—",
            measured,
            detail if finding.severity == "fail" and not finding.skipped else f"[dim]{detail}[/dim]",
        )
    return table


def _channel_measured_label(finding) -> str:
    if finding.measured is None:
        return "—"
    if finding.threshold is None:
        return f"{finding.measured:g}"
    return f"{finding.measured:g} vs {finding.threshold:g}"


def _render_downmix_report(report: DownmixReport) -> None:
    check = report.downmix_check
    meta = Table(title="Downmix Consistency", title_justify="left", show_header=False)
    meta.add_column("label", style="cyan")
    meta.add_column("value")
    meta.add_row("delivered 2.0", logical_asset_display_name(report.stereo_file))
    meta.add_row("delivered source", source_summary(report.stereo_file))
    meta.add_row("delivered path", report.stereo_file.path)
    meta.add_row("surround master", logical_asset_display_name(report.surround_file))
    meta.add_row("surround source", source_summary(report.surround_file))
    meta.add_row("surround path", report.surround_file.path)
    meta.add_row("sample rate", f"{report.stereo_file.sample_rate} Hz")
    meta.add_row("analysis signal", humanize_code(check.analysis_signal))
    lfe_label = "omitted" if check.lfe_db is None else f"{check.lfe_db:+.1f} dB below {check.lfe_lowpass_hz:.0f} Hz"
    meta.add_row(
        "fold-down gains",
        f"center {check.center_db:+.1f} dB / surround {check.surround_db:+.1f} dB / LFE {lfe_label}",
    )
    meta.add_row("window / hop", f"{check.window_ms:.0f} ms / {check.hop_ms:.0f} ms")
    meta.add_row(
        "thresholds",
        (
            f"similarity {check.similarity_corr:.2f} / mono {check.mono_corr:.2f} / "
            f"level {check.loudness_delta_lu:.1f} LU"
        ),
    )
    _out_console.print(meta)

    for note in check.notes:
        _out_console.print(f"[dim]note:[/dim] {escape(humanize_code(note))}")

    if check.summary is not None:
        _out_console.print(_downmix_loudness_line(check.summary))

    color = "green" if check.pass_ else "red"
    verdict = "PASS" if check.pass_ else "FAIL"
    _out_console.print(f"[{color}]{verdict}[/{color}]")

    for error in check.errors:
        _out_console.print(f"  [red]{humanize_code(error.type)}:[/red] {error.message}")

    for metric, title in (
        ("downmix_correlation", "Downmix Similarity Flags"),
        ("stereo_correlation", "Mono Compatibility Flags"),
    ):
        lane = [flag for flag in check.flags if flag.metric == metric]
        if lane:
            _out_console.print(_flagged_regions_table(lane, title=title))


def _downmix_loudness_line(summary) -> str:
    delivered = summary.delivered_integrated_lufs
    derived = summary.derived_integrated_lufs
    if delivered is None or derived is None or summary.loudness_delta_lu is None:
        return "[dim]integrated loudness unavailable on this material[/dim]"
    color = "green" if summary.level_pass else "red"
    return (
        f"delivered {delivered:.1f} LUFS / derived {derived:.1f} LUFS / "
        f"[{color}]\u0394 {summary.loudness_delta_lu:.1f} LU[/{color}]"
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
                    f"  [green]Null:[/green] PASS — {stem_strategy_label(null_test.stem_strategy)} — "
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
                    )
                )

        _render_group_channel_checks(group.channel_checks)
        _render_group_downmix_check(group.downmix_check)

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


def _render_group_channel_checks(channel_checks: list[ChannelAssetResult]) -> None:
    if not channel_checks:
        return

    failing = [result for result in channel_checks if result.pass_ is False]
    skipped = [result for result in channel_checks if result.skipped]
    if not failing and not skipped:
        count = len(channel_checks)
        noun = "asset" if count == 1 else "assets"
        _out_console.print(f"  [green]Channels:[/green] PASS — {count} {noun} clean")
        return

    if failing:
        noun = "asset" if len(failing) == 1 else "assets"
        _out_console.print(f"  [red]Channels:[/red] FAIL — {len(failing)} {noun} with findings")
    else:
        _out_console.print(f"  [dim]Channels:[/dim] SKIPPED — {len(skipped)} unreadable")

    for result in channel_checks:
        if result.skipped:
            _out_console.print(
                f"    [dim]{escape(logical_asset_display_name(result))}:[/dim] "
                f"skipped — {humanize_code(result.reason)}"
            )
            continue
        if result.pass_ is False:
            kinds = humanize_code_list(sorted({
                finding.kind for finding in result.findings
                if finding.severity == "fail" and not finding.skipped
            }))
            _out_console.print(
                f"    [red]{escape(logical_asset_display_name(result))}:[/red] {escape(kinds)}"
            )


def _render_group_downmix_check(downmix_check) -> None:
    if downmix_check is None:
        return

    if downmix_check.skipped:
        _out_console.print(f"  [dim]Downmix:[/dim] SKIPPED — {humanize_code(downmix_check.reason)}")
    elif downmix_check.pass_ is True and downmix_check.summary is not None:
        delta = downmix_check.summary.loudness_delta_lu
        delta_label = "—" if delta is None else f"Δ{delta:.1f} LU"
        _out_console.print(f"  [green]Downmix:[/green] PASS — {delta_label}")
    elif downmix_check.summary is not None and downmix_check.flags:
        flagged = downmix_check.summary.flagged_regions
        noun = "region" if flagged == 1 else "regions"
        _out_console.print(f"  [red]Downmix:[/red] FAIL — {flagged} flagged {noun}")
    elif downmix_check.summary is not None:
        delta = downmix_check.summary.loudness_delta_lu
        delta_label = "—" if delta is None else f"Δ{delta:.1f} LU"
        _out_console.print(f"  [red]Downmix:[/red] FAIL — level delta {delta_label}")
    else:
        _out_console.print(f"  [red]Downmix:[/red] FAIL — {humanize_code(downmix_check.reason)}")

    for error in downmix_check.errors:
        _out_console.print(f"    [red]{humanize_code(error.type)}:[/red] {error.message}")
    for metric, title in (
        ("downmix_correlation", "Downmix Similarity Flags"),
        ("stereo_correlation", "Mono Compatibility Flags"),
    ):
        lane = [flag for flag in downmix_check.flags if flag.metric == metric]
        if lane:
            _out_console.print(_flagged_regions_table(lane, title=title))


def _render_file_checks(file_report: FileReport) -> None:
    _out_console.print(_file_summary_heading(file_report))
    _out_console.print(_file_source_line(file_report))
    _out_console.print(_file_path_line(file_report), soft_wrap=True)
    _out_console.print(_checks_table(file_report))
    if file_report.flags:
        _out_console.print(
            _flagged_regions_table(
                file_report.flags,
                title="Loudness Flags",
            )
        )


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
        target_limit = format_target_limit(check)
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
    _announce_artifact_path("Wrote", written.json_path, leading_blank=True)
    _announce_artifact_path("Wrote", written.html_path)
    if written.aaf_path is None:
        _announce_artifact_path(
            "No exportable timed markers; did not write",
            _expected_aaf_path(written),
        )
    else:
        _announce_artifact_path("Wrote", written.aaf_path)


def _announce_artifact_path(label: str, path: Path, *, leading_blank: bool = False) -> None:
    prefix = "\n" if leading_blank else ""
    _out_console.print(f"{prefix}[dim]{escape(label)}[/dim] {escape(path.name)}")
    _out_console.print(f"[dim]path:[/dim] {escape(str(path))}", soft_wrap=True)


def _expected_aaf_path(written: WrittenArtifacts) -> Path:
    json_stem = written.json_path.stem
    suffix = ""
    base = json_stem
    if len(json_stem) > 3 and json_stem[-3] == "-" and json_stem[-2:].isdigit():
        suffix = json_stem[-3:]
        base = json_stem[:-3]
    if base == "report":
        marker_stem = "markers"
    elif base.endswith("-report"):
        marker_stem = f"{base.removesuffix('-report')}-markers"
    else:
        marker_stem = f"{base}-markers"
    return written.json_path.with_name(f"{marker_stem}{suffix}.aaf")


def _flagged_regions_table(flags, *, title: str) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("start", style="cyan")
    table.add_column("end", style="cyan")
    table.add_column("duration", justify="right")
    table.add_column(metric_value_header(flags[0].metric) if flags else "value", justify="right")
    table.add_column("detail")
    for flag in flags[:_FLAG_TABLE_MAX_ROWS]:
        table.add_row(
            flag.start_tc,
            flag.end_tc,
            f"{flag.duration_seconds:.2f}s",
            format_metric_value(flag.value, metric=flag.metric),
            flag.detail,
        )
    if len(flags) > _FLAG_TABLE_MAX_ROWS:
        table.caption = (
            f"Showing first {_FLAG_TABLE_MAX_ROWS} of {len(flags)} flagged regions; "
            "the full list is in the JSON and HTML reports."
        )
        table.caption_justify = "left"
    return table


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
