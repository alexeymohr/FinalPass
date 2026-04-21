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

from . import ALL_SCHEMA_VERSION, LOUDNESS_SCHEMA_VERSION, __version__
from .audio_io import AudioFile, read_wav
from .classify import (
    ClassifiedFile,
    FolderScan,
    load_config,
    scan_folder,
)
from .errors import (
    AmbiguousClassificationError,
    ChannelMismatchError,
    ClassifierConfigError,
    DuplicateRoleError,
    FinalPassError,
    NoAudioFilesError,
)
from .loudness import check, measure
from .models import (
    AllReport,
    AllSummary,
    FileReport,
    Group,
    GroupError,
    GroupSummary,
    Report,
    SpecRef,
    Summary,
    UnclassifiedEntry,
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
# Phase 2 — `finalpass all` command


@main.command("all")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--spec", "spec_name", required=True, help="Name of a bundled spec or path to a YAML.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=Path("./finalpass-report"), show_default=True)
@click.option("--json-only", is_flag=True, help="Suppress terminal output; emit JSON to stdout.")
@click.option("--fps", type=float, default=23.976, show_default=True, help="Frame rate (for downstream TC display).")
@click.option("--patterns", "patterns_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Override bundled classifier patterns with a user YAML.")
@click.option("--include-unclassified", is_flag=True, help="Measure unclassified files too (integrated/TP/LRA; no dialog check).")
def all_cmd(folder: Path, spec_name: str, out_dir: Path, json_only: bool, fps: float, patterns_path: Path | None, include_unclassified: bool) -> None:
    try:
        report = _run_all(
            folder=folder,
            spec_name=spec_name,
            patterns_path=patterns_path,
            include_unclassified=include_unclassified,
            fps=fps,
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


def _run_all(*, folder: Path, spec_name: str, patterns_path: Path | None, include_unclassified: bool, fps: float) -> AllReport:
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

    now = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + secrets.token_hex(3)

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

    all_checks = [c for fr in file_reports for c in fr.checks]
    passed = sum(1 for c in all_checks if c.pass_ is True)
    failed = sum(1 for c in all_checks if c.pass_ is False)
    skipped = sum(1 for c in all_checks if c.skipped)
    summary = GroupSummary(
        total_checks=len(all_checks),
        passed=passed,
        failed=failed,
        skipped=skipped,
        overall_pass=failed == 0 and not errors,
    )

    return Group(
        group_id=group_id,
        files=file_reports,
        group_summary=summary,
        errors=errors,
    )


_CHANNEL_CONFIG_FROM_COUNT: dict[int, str] = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


def _channel_config_from_count(n: int) -> str | None:
    """Map a raw channel count to a spec channel_config name, or None if the
    layout isn't one we understand."""
    return _CHANNEL_CONFIG_FROM_COUNT.get(n)


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
        channel_config_actual=_channel_config_from_count(audio.channel_count),
    )


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
