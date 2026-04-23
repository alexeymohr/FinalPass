from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Literal

import click

from .all_assets import AssetFolderScan, ClassifiedLogicalAsset, discover_assets_from_paths, discover_folder_assets
from .classify import load_config
from .errors import FinalPassError
from .jobs import (
    WrittenArtifacts,
    run_all,
    run_all_filtered,
    run_all_filtered_with_spec_family,
    run_all_with_spec_family,
    run_loudness,
    run_me,
    run_null,
    write_report_artifacts,
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
from .models import AllReport, MEReport, NullReport, Report
from .null_test import (
    DEFAULT_NULL_HOP_MS,
    DEFAULT_NULL_THRESHOLD_DBFS,
    DEFAULT_NULL_WINDOW_MS,
)
from .presentation import asset_menu_label, logical_asset_display_name, source_summary
from .prep_folders import PrepLayout, PrepScanResult, create_prep_layout, detect_prep_layout, scan_prep_layout
from .specs import BundledSpecFamily, CHANNEL_COUNTS, Spec, list_bundled_families, load_spec
from .wizard_io import Choice, WizardBack, WizardQuit, choose_many, choose_one, print_section, prompt_text

WizardAction = Literal["job_menu", "choose_folder", "exit"]
WizardStatus = Literal["PASS", "FAIL", "TOOL ERROR"]
WizardMode = Literal["folder", "prep"]

DEFAULT_OUTPUT_DIR = Path("./finalpass-report")
COMMON_FPS_OPTIONS = [23.976, 24.0, 25.0, 29.97, 30.0]
_SPINNER_INTERVAL_SECONDS = 0.2
_SPINNER_MESSAGE = "Running job"


@dataclass(frozen=True)
class FolderContext:
    folder: Path
    scan: AssetFolderScan
    mode: WizardMode = "folder"
    prep_scan: PrepScanResult | None = None
    prep_root: Path | None = None

    @property
    def asset_count(self) -> int:
        return sum(len(assets) for assets in self.scan.groups.values())


@dataclass(frozen=True)
class FlowResult:
    action: WizardAction
    out_dir: Path
    fps: float


@dataclass(frozen=True)
class JobExecutionResult:
    job_name: str
    status: WizardStatus
    summary_line: str
    discovery_issue_count: int
    artifacts: list[Path]
    error_message: str | None = None


@dataclass(frozen=True)
class SpecSelection:
    spec_name: str | None
    spec: Spec | None
    review_label: str
    family: BundledSpecFamily | None = None


def run_wizard(*, folder: Path | None, out_dir: Path, fps: float) -> None:
    current_folder = folder.expanduser() if folder is not None else None
    current_out_dir = out_dir
    current_fps = fps

    try:
        while True:
            folder_context = _choose_folder_context(current_folder)
            if folder_context is None:
                return

            current_folder = folder_context.folder
            while True:
                action = _job_menu(folder_context)
                if action == "choose_folder":
                    current_folder = None
                    break
                if action == "exit":
                    return

                if action == "all":
                    flow = _all_flow(folder_context, default_out_dir=current_out_dir, default_fps=current_fps)
                elif action == "loudness":
                    flow = _loudness_flow(folder_context, default_out_dir=current_out_dir, default_fps=current_fps)
                elif action == "null":
                    flow = _null_flow(folder_context, default_out_dir=current_out_dir, default_fps=current_fps)
                else:
                    flow = _me_flow(folder_context, default_out_dir=current_out_dir, default_fps=current_fps)

                current_out_dir = flow.out_dir
                current_fps = flow.fps
                if flow.action == "choose_folder":
                    current_folder = None
                    break
                if flow.action == "exit":
                    return
    except WizardQuit:
        return


def _choose_folder_context(initial_folder: Path | None) -> FolderContext | None:
    candidate = initial_folder
    while True:
        if candidate is None:
            try:
                raw = prompt_text("Enter the delivery folder path.", allow_back=False)
            except WizardQuit:
                return None
            candidate = Path(raw).expanduser()

        resolved_folder, error_message = _resolve_folder(candidate)
        if resolved_folder is None:
            click.echo(f"Folder error: {error_message}")
            next_action = choose_one(
                "Choose an option:",
                [
                    Choice("Choose a different folder", "choose_folder"),
                    Choice("Exit", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if next_action == "exit":
                return None
            candidate = None
            continue

        prep_layout = detect_prep_layout(resolved_folder)
        if prep_layout is not None:
            click.echo("I found an existing FinalPass prep layout.")
            click.echo("I'll scan the populated prep folders and ignore everything else.")
            prep_result = _handle_prep_layout(prep_layout)
            if isinstance(prep_result, FolderContext):
                return prep_result
            if prep_result == "exit":
                return None
            candidate = None
            continue

        next_action = choose_one(
            "Choose an option:",
            [
                Choice("Analyze folder as-is", "analyze"),
                Choice("Create FinalPass prep folders", "create_prep"),
                Choice("Choose a different folder", "choose_folder"),
                Choice("Quit", "exit"),
            ],
            allow_back=False,
            default_index=1,
        )
        if next_action == "exit":
            return None
        if next_action == "choose_folder":
            candidate = None
            continue
        if next_action == "create_prep":
            prep_layout = create_prep_layout(resolved_folder)
            _print_prep_created(prep_layout)
            follow_up = choose_one(
                "Choose an option:",
                [
                    Choice("Re-scan prep folders now", "rescan"),
                    Choice("Quit for now", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if follow_up == "exit":
                return None
            prep_result = _handle_prep_layout(prep_layout)
            if isinstance(prep_result, FolderContext):
                return prep_result
            if prep_result == "exit":
                return None
            candidate = None
            continue

        context, error_message = _scan_folder_as_is(resolved_folder)
        if context is None:
            click.echo(f"Folder error: {error_message}")
            candidate = None
            continue

        _print_folder_summary(context)

        if context.asset_count == 0:
            click.echo("No valid logical assets were found in this folder.")
            next_action = choose_one(
                "Choose an option:",
                [
                    Choice("Choose a different folder", "choose_folder"),
                    Choice("Exit", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if next_action == "exit":
                return None
            candidate = None
            continue

        if context.scan.discovery_errors:
            while True:
                next_action = choose_one(
                    "Discovery errors were found. What do you want to do?",
                    [
                        Choice("Continue anyway", "continue"),
                        Choice("View discovery errors", "view"),
                        Choice("Choose a different folder", "choose_folder"),
                        Choice("Quit", "exit"),
                    ],
                    allow_back=False,
                    default_index=1,
                )
                if next_action == "continue":
                    return context
                if next_action == "view":
                    _print_discovery_errors(context)
                    continue
                if next_action == "exit":
                    return None
                candidate = None
                break
            continue

        return context


def _job_menu(folder_context: FolderContext) -> Literal["all", "loudness", "null", "me", "choose_folder", "exit"]:
    print_section("Choose a job:")
    if folder_context.mode == "prep":
        options = _prep_job_options(folder_context)
    else:
        options = [
            Choice("Analyze delivery folder", "all"),
            Choice("Measure loudness on one asset", "loudness"),
            Choice("Run printmaster null", "null"),
            Choice("Run M&E bleed check", "me"),
            Choice("Choose a different folder", "choose_folder"),
            Choice("Exit", "exit"),
        ]
    return choose_one(
        "Select a job type.",
        options,
        allow_back=False,
    )


def _all_flow(folder_context: FolderContext, *, default_out_dir: Path, default_fps: float) -> FlowResult:
    while True:
        try:
            spec_selection = _choose_spec_for_all(folder_context)
        except WizardBack:
            return FlowResult(action="job_menu", out_dir=default_out_dir, fps=default_fps)

        while True:
            try:
                out_dir = _choose_output_dir(default_out_dir)
            except WizardBack:
                break

            while True:
                try:
                    fps = _choose_fps(default_fps)
                except WizardBack:
                    break

                try:
                    _review_and_confirm([
                        "Job: all",
                        *_context_review_lines(folder_context),
                        f"Loudness standard: {spec_selection.review_label}",
                        f"Output: {out_dir}",
                        f"FPS: {fps:g}",
                    ])
                except WizardBack:
                    continue

                result = _execute_report_job(
                    job_name="all",
                    out_dir=out_dir,
                    summary_builder=lambda report: (
                        f"{report.summary.groups_passed} groups passed, "
                        f"{report.summary.groups_failed} failed, "
                        f"{len(report.discovery_errors)} discovery issue(s)."
                    ),
                    runner=lambda: _run_all_for_context(
                        folder_context,
                        spec_name=spec_selection.spec_name,
                        family=spec_selection.family,
                        fps=fps,
                    ),
                )
                return _result_screen(result, out_dir=out_dir, fps=fps)


def _loudness_flow(folder_context: FolderContext, *, default_out_dir: Path, default_fps: float) -> FlowResult:
    while True:
        try:
            group_id = _choose_group(folder_context, title="Choose a group for loudness.")
        except WizardBack:
            return FlowResult(action="job_menu", out_dir=default_out_dir, fps=default_fps)

        group_assets = folder_context.scan.groups[group_id]
        while True:
            try:
                asset = _choose_asset(
                    group_assets,
                    title=f"Choose a logical asset from {group_id}.",
                    auto_select_notice="Auto-selected the only logical asset in this group.",
                )
            except WizardBack:
                break

            while True:
                try:
                    spec_selection = _choose_spec_for_loudness(asset)
                except WizardBack:
                    break

                while True:
                    try:
                        dx_asset = _choose_optional_dx_asset(group_id, group_assets, spec_selection.spec)
                    except WizardBack:
                        break

                    while True:
                        try:
                            out_dir = _choose_output_dir(default_out_dir)
                        except WizardBack:
                            break

                        while True:
                            try:
                                fps = _choose_fps(default_fps)
                            except WizardBack:
                                break

                            try:
                                review_lines = [
                                    "Job: loudness",
                                    *_context_review_lines(folder_context),
                                    f"Group: {group_id}",
                                    _review_asset_line("Asset", asset),
                                    f"Loudness standard: {spec_selection.review_label}",
                                    _review_asset_line("DX asset", dx_asset),
                                    f"Output: {out_dir}",
                                    f"FPS: {fps:g}",
                                ]
                                _review_and_confirm(review_lines)
                            except WizardBack:
                                continue

                            result = _execute_report_job(
                                job_name="loudness",
                                out_dir=out_dir,
                                summary_builder=lambda report: (
                                    f"{report.summary.passed} passed, {report.summary.failed} failed, "
                                    f"{report.summary.skipped} skipped across {len(report.files)} file(s)."
                                ),
                                runner=lambda: run_loudness(
                                    files=(asset.logical_asset.canonical_path,),
                                    spec_name=spec_selection.spec_name,
                                    dx_file=dx_asset.logical_asset.canonical_path if dx_asset is not None else None,
                                    fps=fps,
                                ),
                            )
                            return _result_screen(result, out_dir=out_dir, fps=fps)


def _null_flow(folder_context: FolderContext, *, default_out_dir: Path, default_fps: float) -> FlowResult:
    while True:
        try:
            group_id = _choose_group(folder_context, title="Choose a group for null.")
        except WizardBack:
            return FlowResult(action="job_menu", out_dir=default_out_dir, fps=default_fps)

        group_assets = folder_context.scan.groups[group_id]
        pm_candidates = [asset for asset in group_assets if asset.role == "pm"]
        if not pm_candidates:
            click.echo(f"No printmaster assets are available in {group_id}.")
            continue

        while True:
            try:
                pm_asset = _choose_asset(
                    pm_candidates,
                    title=f"Choose the printmaster asset from {group_id}.",
                    auto_select_notice="Auto-selected the only printmaster asset in this group.",
                )
            except WizardBack:
                break

            while True:
                try:
                    stem_assets = _choose_null_stem_plan(group_id, group_assets, pm_asset)
                except WizardBack:
                    break

                while True:
                    try:
                        out_dir = _choose_output_dir(default_out_dir)
                    except WizardBack:
                        break

                    while True:
                        try:
                            fps = _choose_fps(default_fps)
                        except WizardBack:
                            break

                        try:
                            _review_and_confirm([
                                "Job: null",
                                *_context_review_lines(folder_context),
                                f"Group: {group_id}",
                                _review_asset_line("Printmaster", pm_asset),
                                f"Stems: {', '.join(logical_asset_display_name(asset) for asset in stem_assets)}",
                                f"Output: {out_dir}",
                                f"FPS: {fps:g}",
                            ])
                        except WizardBack:
                            continue

                        result = _execute_report_job(
                            job_name="null",
                            out_dir=out_dir,
                            summary_builder=_null_summary_line,
                            runner=lambda: run_null(
                                pm=pm_asset.logical_asset.canonical_path,
                                stems=tuple(asset.logical_asset.canonical_path for asset in stem_assets),
                                fps=fps,
                                window_ms=DEFAULT_NULL_WINDOW_MS,
                                hop_ms=DEFAULT_NULL_HOP_MS,
                                threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
                            ),
                        )
                        return _result_screen(result, out_dir=out_dir, fps=fps)


def _me_flow(folder_context: FolderContext, *, default_out_dir: Path, default_fps: float) -> FlowResult:
    while True:
        try:
            group_id = _choose_group(folder_context, title="Choose a group for M&E.")
        except WizardBack:
            return FlowResult(action="job_menu", out_dir=default_out_dir, fps=default_fps)

        group_assets = folder_context.scan.groups[group_id]
        me_candidates = [asset for asset in group_assets if asset.role == "me"]
        if not me_candidates:
            click.echo(f"No M&E assets are available in {group_id}.")
            continue

        while True:
            try:
                me_asset = _choose_asset(
                    me_candidates,
                    title=f"Choose the M&E asset from {group_id}.",
                    auto_select_notice="Auto-selected the only M&E asset in this group.",
                )
            except WizardBack:
                break

            while True:
                try:
                    dx_asset = _choose_dx_for_me(group_id, group_assets, me_asset)
                except WizardBack:
                    break

                while True:
                    try:
                        out_dir = _choose_output_dir(default_out_dir)
                    except WizardBack:
                        break

                    while True:
                        try:
                            fps = _choose_fps(default_fps)
                        except WizardBack:
                            break

                        try:
                            _review_and_confirm([
                                "Job: me",
                                *_context_review_lines(folder_context),
                                f"Group: {group_id}",
                                _review_asset_line("M&E asset", me_asset),
                                _review_asset_line("DX asset", dx_asset),
                                f"Output: {out_dir}",
                                f"FPS: {fps:g}",
                            ])
                        except WizardBack:
                            continue

                        result = _execute_report_job(
                            job_name="me",
                            out_dir=out_dir,
                            summary_builder=_me_summary_line,
                            runner=lambda: run_me(
                                me_file=me_asset.logical_asset.canonical_path,
                                dx_file=dx_asset.logical_asset.canonical_path,
                                fps=fps,
                                window_ms=DEFAULT_ME_WINDOW_MS,
                                hop_ms=DEFAULT_ME_HOP_MS,
                                band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
                                band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
                                corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
                                coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
                                dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
                                me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
                            ),
                        )
                        return _result_screen(result, out_dir=out_dir, fps=fps)


def _resolve_folder(folder: Path) -> tuple[Path | None, str | None]:
    folder = folder.expanduser()
    if not folder.exists():
        return None, f"{folder} does not exist."
    if not folder.is_dir():
        return None, f"{folder} is not a directory."
    return folder.resolve(), None


def _scan_folder_as_is(folder: Path) -> tuple[FolderContext | None, str | None]:
    folder = folder.resolve()

    try:
        scan = discover_folder_assets(folder, load_config(None))
    except FinalPassError as exc:
        return None, str(exc)
    return FolderContext(folder=folder, scan=scan, mode="folder"), None


def _scan_prep_folder(folder: Path, prep_scan: PrepScanResult) -> tuple[FolderContext | None, str | None]:
    try:
        scan = discover_assets_from_paths(
            prep_scan.analyzable_paths,
            load_config(None),
            path_hints=prep_scan.path_hints,
        )
    except FinalPassError as exc:
        return None, str(exc)
    return FolderContext(
        folder=folder.resolve(),
        scan=scan,
        mode="prep",
        prep_scan=prep_scan,
        prep_root=prep_scan.layout.prep_root,
    ), None


def _handle_prep_layout(prep_layout: PrepLayout) -> FolderContext | WizardAction:
    while True:
        prep_scan = scan_prep_layout(prep_layout)
        if prep_scan.empty:
            click.echo("I found the FinalPass prep folders, but none of the analysis buckets contain files yet.")
            action = choose_one(
                "Choose an option:",
                [
                    Choice("Re-scan prep folders", "rescan"),
                    Choice("Choose a different folder", "choose_folder"),
                    Choice("Quit", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if action == "rescan":
                continue
            return action

        context, error_message = _scan_prep_folder(prep_layout.folder, prep_scan)
        if context is None:
            click.echo(f"Prep scan error: {error_message}")
            action = choose_one(
                "Choose an option:",
                [
                    Choice("Re-scan prep folders", "rescan"),
                    Choice("Choose a different folder", "choose_folder"),
                    Choice("Quit", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if action == "rescan":
                continue
            return action

        _print_folder_summary(context)

        if context.asset_count == 0:
            click.echo("I scanned the populated prep folders, but no valid logical assets were found.")
            action = choose_one(
                "Choose an option:",
                [
                    Choice("Re-scan prep folders", "rescan"),
                    Choice("Choose a different folder", "choose_folder"),
                    Choice("Quit", "exit"),
                ],
                allow_back=False,
                default_index=1,
            )
            if action == "rescan":
                continue
            return action

        if context.scan.discovery_errors:
            while True:
                action = choose_one(
                    "Discovery errors were found. What do you want to do?",
                    [
                        Choice("Continue anyway", "continue"),
                        Choice("View discovery errors", "view"),
                        Choice("Re-scan prep folders", "rescan"),
                        Choice("Choose a different folder", "choose_folder"),
                        Choice("Quit", "exit"),
                    ],
                    allow_back=False,
                    default_index=1,
                )
                if action == "continue":
                    return context
                if action == "view":
                    _print_discovery_errors(context)
                    continue
                if action == "rescan":
                    break
                return action
            continue

        return context


def _prep_job_options(folder_context: FolderContext) -> list[Choice[str]]:
    options: list[Choice[str]] = []
    if _all_available(folder_context):
        options.append(Choice("Analyze populated prep folders", "all"))
    if folder_context.asset_count:
        options.append(Choice("Measure loudness on one asset", "loudness"))
    if _null_available(folder_context):
        options.append(Choice("Run printmaster null", "null"))
    if _me_available(folder_context):
        options.append(Choice("Run M&E bleed check", "me"))
    options.extend([
        Choice("Choose a different folder", "choose_folder"),
        Choice("Exit", "exit"),
    ])
    return options


def _all_available(folder_context: FolderContext) -> bool:
    for assets in folder_context.scan.groups.values():
        if any(asset.role == "pm" for asset in assets):
            return True
    return False


def _null_available(folder_context: FolderContext) -> bool:
    for assets in folder_context.scan.groups.values():
        for pm_asset in assets:
            if pm_asset.role != "pm":
                continue
            same_layout_assets = [
                asset for asset in assets
                if asset.logical_asset.channel_config_actual == pm_asset.logical_asset.channel_config_actual
                and asset.logical_asset.asset_id != pm_asset.logical_asset.asset_id
            ]
            if same_layout_assets:
                return True
    return False


def _me_available(folder_context: FolderContext) -> bool:
    for assets in folder_context.scan.groups.values():
        has_me = any(asset.role == "me" for asset in assets)
        has_dx = any(asset.role == "dx" for asset in assets)
        if has_me and has_dx:
            return True
    return False


def _run_all_for_context(
    folder_context: FolderContext,
    *,
    spec_name: str | None,
    family: BundledSpecFamily | None,
    fps: float,
) -> AllReport:
    if family is not None:
        if folder_context.mode == "prep":
            prep_scan = folder_context.prep_scan
            if prep_scan is None:
                raise FinalPassError("Prep mode is active, but no prep scan is available.")
            return run_all_filtered_with_spec_family(
                folder=folder_context.folder,
                paths=prep_scan.analyzable_paths,
                path_hints=prep_scan.path_hints,
                family=family,
                patterns_path=None,
                include_unclassified=False,
                fps=fps,
                null_window_ms=DEFAULT_NULL_WINDOW_MS,
                null_hop_ms=DEFAULT_NULL_HOP_MS,
                null_threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
                me_window_ms=DEFAULT_ME_WINDOW_MS,
                me_hop_ms=DEFAULT_ME_HOP_MS,
                me_band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
                me_band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
                me_corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
                me_coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
                me_dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
                me_me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
            )
        return run_all_with_spec_family(
            folder=folder_context.folder,
            family=family,
            patterns_path=None,
            include_unclassified=False,
            fps=fps,
            null_window_ms=DEFAULT_NULL_WINDOW_MS,
            null_hop_ms=DEFAULT_NULL_HOP_MS,
            null_threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
            me_window_ms=DEFAULT_ME_WINDOW_MS,
            me_hop_ms=DEFAULT_ME_HOP_MS,
            me_band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
            me_band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
            me_corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
            me_coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
            me_dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
            me_me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
        )

    if spec_name is None:
        raise FinalPassError("Wizard all flow requires either a concrete spec or a bundled spec family.")

    if folder_context.mode == "prep":
        prep_scan = folder_context.prep_scan
        if prep_scan is None:
            raise FinalPassError("Prep mode is active, but no prep scan is available.")
        return run_all_filtered(
            folder=folder_context.folder,
            paths=prep_scan.analyzable_paths,
            path_hints=prep_scan.path_hints,
            spec_name=spec_name,
            patterns_path=None,
            include_unclassified=False,
            fps=fps,
            null_window_ms=DEFAULT_NULL_WINDOW_MS,
            null_hop_ms=DEFAULT_NULL_HOP_MS,
            null_threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
            me_window_ms=DEFAULT_ME_WINDOW_MS,
            me_hop_ms=DEFAULT_ME_HOP_MS,
            me_band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
            me_band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
            me_corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
            me_coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
            me_dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
            me_me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
        )
    return run_all(
        folder=folder_context.folder,
        spec_name=spec_name,
        patterns_path=None,
        include_unclassified=False,
        fps=fps,
        null_window_ms=DEFAULT_NULL_WINDOW_MS,
        null_hop_ms=DEFAULT_NULL_HOP_MS,
        null_threshold_dbfs=DEFAULT_NULL_THRESHOLD_DBFS,
        me_window_ms=DEFAULT_ME_WINDOW_MS,
        me_hop_ms=DEFAULT_ME_HOP_MS,
        me_band_low_hz=DEFAULT_ME_BAND_LOW_HZ,
        me_band_high_hz=DEFAULT_ME_BAND_HIGH_HZ,
        me_corr_threshold=DEFAULT_ME_CORR_THRESHOLD,
        me_coherence_threshold=DEFAULT_ME_COHERENCE_THRESHOLD,
        me_dx_gate_dbfs=DEFAULT_ME_DX_GATE_DBFS,
        me_me_floor_dbfs=DEFAULT_ME_ME_FLOOR_DBFS,
    )


def _print_prep_created(layout: PrepLayout) -> None:
    print_section("Prep folders created:")
    click.echo(f"Prep root: {layout.prep_root}")
    click.echo("Buckets created:")
    for bucket in layout.buckets:
        click.echo(f"- {bucket.name}")
    click.echo("Move or copy the stems you want analyzed into the appropriate prep buckets.")


def _choose_group(folder_context: FolderContext, *, title: str) -> str:
    options = [
        Choice(_group_label(group_id, assets), group_id)
        for group_id, assets in folder_context.scan.groups.items()
    ]
    return choose_one(
        title,
        options,
        auto_select_notice="Auto-selected the only group in this folder.",
    )


def _choose_asset(
    assets: list[ClassifiedLogicalAsset],
    *,
    title: str,
    auto_select_notice: str,
) -> ClassifiedLogicalAsset:
    options = [Choice(asset_menu_label(asset), asset) for asset in assets]
    return choose_one(title, options, auto_select_notice=auto_select_notice)


def _choose_spec_for_all(folder_context: FolderContext) -> SpecSelection:
    detected_layouts = _detected_printmaster_layouts(folder_context)
    families = [
        family for family in list_bundled_families()
        if family.best_spec_for_channel_configs(detected_layouts) is not None
    ]
    options = [
        Choice(_family_menu_label_for_context(family, detected_layouts), family)
        for family in families
    ]
    options.append(Choice("Enter a custom spec path", "__custom__"))
    choice = choose_one(
        "Choose a loudness standard. FinalPass will auto-match the correct spec layout to the detected printmaster.",
        options,
    )
    return _resolve_spec_selection(
        choice,
        auto_channel_configs=detected_layouts,
        auto_scope="printmaster layout",
        preserve_family=True,
    )


def _choose_spec_for_loudness(asset: ClassifiedLogicalAsset) -> SpecSelection:
    asset_layout = asset.logical_asset.channel_config_actual
    families = [
        family for family in list_bundled_families()
        if family.supports_channel_config(asset_layout)
    ]
    options = [
        Choice(_family_menu_label_for_asset(family), family)
        for family in families
    ]
    options.append(Choice("Enter a custom spec path", "__custom__"))
    choice = choose_one(
        "Choose a loudness standard. FinalPass will auto-match the correct spec layout to the selected asset.",
        options,
    )
    return _resolve_spec_selection(choice, auto_channel_configs={asset_layout}, auto_scope="asset layout")


def _resolve_spec_selection(
    choice,
    *,
    auto_channel_configs: set[str] | None = None,
    auto_scope: str | None = None,
    preserve_family: bool = False,
) -> SpecSelection:
    if isinstance(choice, BundledSpecFamily):
        if not auto_channel_configs:
            raise FinalPassError("Wizard spec-family selection requires a detected channel layout.")
        spec = choice.best_spec_for_channel_configs(auto_channel_configs)
        if spec is None:
            detected = ", ".join(_sort_channel_configs(auto_channel_configs))
            raise FinalPassError(
                f"{choice.display_name} has no bundled preset matching the detected {auto_scope or 'layout'} "
                f"({detected})."
            )
        scope = auto_scope or "layout"
        if preserve_family:
            return SpecSelection(
                spec_name=None,
                spec=None,
                review_label=f"{choice.display_name} (auto per {scope})",
                family=choice,
            )
        return SpecSelection(
            spec_name=spec.name,
            spec=spec,
            review_label=f"{choice.display_name} (auto -> {spec.name} for {spec.channel_config} {scope})",
        )

    if choice != "__custom__":
        spec, source, path = load_spec(choice)
        return SpecSelection(spec_name=choice, spec=spec, review_label=f"{spec.name} ({source}, {path})")

    while True:
        try:
            raw = prompt_text("Enter the custom spec path.")
        except WizardBack:
            raise
        try:
            spec, source, path = load_spec(raw)
        except FinalPassError as exc:
            click.echo(f"Spec error: {exc}")
            continue
        return SpecSelection(spec_name=str(path), spec=spec, review_label=f"{spec.name} ({source}, {path})")


def _detected_printmaster_layouts(folder_context: FolderContext) -> set[str]:
    return {
        asset.logical_asset.channel_config_actual
        for assets in folder_context.scan.groups.values()
        for asset in assets
        if asset.role == "pm"
    }


def _family_menu_label_for_context(family: BundledSpecFamily, detected_layouts: set[str]) -> str:
    resolved = family.best_spec_for_channel_configs(detected_layouts)
    if resolved is None:
        return family.display_name
    if len(family.supported_channel_configs) == 1:
        return f"{family.display_name} — {resolved.channel_config} only"
    return f"{family.display_name} — auto {resolved.channel_config}"


def _family_menu_label_for_asset(family: BundledSpecFamily) -> str:
    if len(family.supported_channel_configs) == 1:
        return f"{family.display_name} — {family.supported_channel_configs[0]} only"
    return family.display_name


def _sort_channel_configs(channel_configs: set[str]) -> list[str]:
    return sorted(channel_configs, key=lambda layout: CHANNEL_COUNTS[layout])


def _choose_optional_dx_asset(
    group_id: str,
    group_assets: list[ClassifiedLogicalAsset],
    spec: Spec,
) -> ClassifiedLogicalAsset | None:
    if spec.dialog_lufs is None:
        return None

    candidates = [
        asset for asset in group_assets
        if asset.role == "dx" and asset.logical_asset.channel_config_actual == spec.channel_config
    ]
    if not candidates:
        click.echo(f"No DX assets in {group_id} match the {spec.channel_config} target layout. Continuing without DX.")
        return None

    include_dx = choose_one(
        "This spec includes dialog loudness. Do you want to include a DX asset?",
        [
            Choice("Include a DX asset", True),
            Choice("No DX asset", False),
        ],
        default_index=1,
    )
    if not include_dx:
        return None

    if len(candidates) == 1:
        click.echo("Auto-selected the only valid DX asset for dialog loudness.")
        return candidates[0]

    return _choose_asset(
        candidates,
        title="Choose the DX asset for dialog loudness.",
        auto_select_notice="Auto-selected the only valid DX asset for dialog loudness.",
    )


def _choose_null_stem_plan(
    group_id: str,
    group_assets: list[ClassifiedLogicalAsset],
    pm_asset: ClassifiedLogicalAsset,
) -> list[ClassifiedLogicalAsset]:
    same_layout_assets = [
        asset for asset in group_assets
        if asset.logical_asset.channel_config_actual == pm_asset.logical_asset.channel_config_actual
        and asset.logical_asset.asset_id != pm_asset.logical_asset.asset_id
    ]
    by_role = {asset.role: asset for asset in same_layout_assets}

    options: list[Choice[str]] = []
    if all(role in by_role for role in ("dx", "mx", "fx")):
        options.append(Choice("Auto-select DX + MX + FX", "dx_mx_fx"))
    if all(role in by_role for role in ("dx", "me")):
        options.append(Choice("Auto-select DX + ME", "dx_me"))
    if same_layout_assets:
        options.append(Choice("Choose stems manually", "manual"))

    if not options:
        click.echo(f"No same-layout non-PM assets are available for null in {group_id}.")
        raise WizardBack

    plan = choose_one("Choose a stem plan.", options)
    if plan == "dx_mx_fx":
        return [by_role["dx"], by_role["mx"], by_role["fx"]]
    if plan == "dx_me":
        return [by_role["dx"], by_role["me"]]

    while True:
        selected = choose_many(
            "Choose one or more stems.",
            [Choice(asset_menu_label(asset), asset) for asset in same_layout_assets],
        )
        if selected:
            return selected
        click.echo("Choose at least one stem.")


def _choose_dx_for_me(
    group_id: str,
    group_assets: list[ClassifiedLogicalAsset],
    me_asset: ClassifiedLogicalAsset,
) -> ClassifiedLogicalAsset:
    same_layout = [
        asset for asset in group_assets
        if asset.role == "dx" and asset.logical_asset.channel_config_actual == me_asset.logical_asset.channel_config_actual
    ]
    fallback = [asset for asset in group_assets if asset.role == "dx"]
    candidates = same_layout or fallback
    if not candidates:
        click.echo(f"No DX assets are available in {group_id}.")
        raise WizardBack

    notice = "Auto-selected the only valid DX asset."
    if same_layout and len(same_layout) == 1:
        return choose_one(
            "Choose the DX asset.",
            [Choice(asset_menu_label(same_layout[0]), same_layout[0])],
            auto_select_notice="Auto-selected the only same-layout DX asset.",
        )
    return _choose_asset(candidates, title="Choose the DX asset.", auto_select_notice=notice)


def _choose_output_dir(default_out_dir: Path) -> Path:
    choice = choose_one(
        "Choose an output directory.",
        [
            Choice(f"Keep current default output directory ({default_out_dir})", "default"),
            Choice("Enter a custom output directory", "custom"),
        ],
        default_index=1,
    )
    if choice == "default":
        return default_out_dir

    while True:
        try:
            raw = prompt_text("Enter the output directory path.")
        except WizardBack:
            raise
        path = Path(raw).expanduser()
        if raw.strip():
            return path
        click.echo("Enter a valid output directory path.")


def _choose_fps(default_fps: float) -> float:
    default_index = None
    for index, option in enumerate(COMMON_FPS_OPTIONS, start=1):
        if option == default_fps:
            default_index = index
            break
    if default_index is None:
        default_index = len(COMMON_FPS_OPTIONS) + 1

    choice = choose_one(
        "Choose the frame rate.",
        [
            Choice("23.976 fps", 23.976),
            Choice("24 fps", 24.0),
            Choice("25 fps", 25.0),
            Choice("29.97 fps", 29.97),
            Choice("30 fps", 30.0),
            Choice("Enter a custom value", "__custom__"),
        ],
        default_index=default_index,
    )
    if choice != "__custom__":
        return float(choice)

    while True:
        try:
            raw = prompt_text("Enter the FPS value.", default=f"{default_fps:g}")
        except WizardBack:
            raise
        try:
            value = float(raw)
        except ValueError:
            click.echo("Enter a numeric FPS value.")
            continue
        if value <= 0:
            click.echo("FPS must be greater than zero.")
            continue
        return value


def _review_and_confirm(lines: list[str]) -> None:
    print_section("Review:")
    for line in lines:
        click.echo(line)
    choose_one(
        "Ready to run?",
        [Choice("Run job", "run")],
        default_index=1,
        auto_select=False,
        auto_select_notice=None,
    )


def _execute_report_job(*, job_name: str, out_dir: Path, summary_builder, runner) -> JobExecutionResult:
    try:
        with _job_spinner():
            report = runner()
            payload = report.model_dump_json(indent=2, by_alias=True)
            written = write_report_artifacts(report=report, payload=payload, out_dir=out_dir)
    except FinalPassError as exc:
        return JobExecutionResult(
            job_name=job_name,
            status="TOOL ERROR",
            summary_line="The job could not be completed.",
            discovery_issue_count=0,
            artifacts=[],
            error_message=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive wizard wrapper
        return JobExecutionResult(
            job_name=job_name,
            status="TOOL ERROR",
            summary_line="The job could not be completed.",
            discovery_issue_count=0,
            artifacts=[],
            error_message=str(exc),
        )

    status: WizardStatus = "PASS" if report.summary.overall_pass else "FAIL"
    return JobExecutionResult(
        job_name=job_name,
        status=status,
        summary_line=summary_builder(report),
        discovery_issue_count=len(report.discovery_errors) if isinstance(report, AllReport) else 0,
        artifacts=_written_artifacts(written),
        error_message=None,
    )


def _result_screen(result: JobExecutionResult, *, out_dir: Path, fps: float) -> FlowResult:
    print_section("Result:")
    click.echo(f"Job: {result.job_name}")
    click.echo(f"Status: {result.status}")
    if result.discovery_issue_count:
        click.echo(f"Discovery issues: {result.discovery_issue_count}")
    click.echo(result.summary_line)
    if result.error_message:
        click.echo(f"Error: {result.error_message}")
    if result.artifacts:
        click.echo(f"Artifacts written to: {out_dir}")
        click.echo("Produced artifacts:")
        for path in result.artifacts:
            click.echo(f"- {path}")
    else:
        click.echo(f"Artifacts written to: {out_dir}")
        click.echo("Produced artifacts: none")

    action = choose_one(
        "What do you want to do next?",
        [
            Choice("Run another job on this folder", "job_menu"),
            Choice("Choose a different folder", "choose_folder"),
            Choice("Exit", "exit"),
        ],
        allow_back=False,
        default_index=3,
    )
    return FlowResult(action=action, out_dir=out_dir, fps=fps)


def _print_folder_summary(context: FolderContext) -> None:
    print_section("Folder summary:")
    click.echo(f"Folder: {context.folder}")
    if context.mode == "prep" and context.prep_root is not None and context.prep_scan is not None:
        click.echo("Mode: prep")
        click.echo(f"Prep root: {context.prep_root}")
        click.echo(f"Populated prep buckets: {len(context.prep_scan.analyzable_buckets)}")
    click.echo(f"Groups: {len(context.scan.groups)}")
    click.echo(f"Logical assets: {context.asset_count}")
    click.echo(f"Discovery errors: {len(context.scan.discovery_errors)}")


def _print_discovery_errors(context: FolderContext) -> None:
    print_section("Discovery errors:")
    for error in context.scan.discovery_errors:
        click.echo(f"- {error.type}: {error.message}")
        for path in error.paths:
            click.echo(f"  {path}")


def _group_label(group_id: str, assets: list[ClassifiedLogicalAsset]) -> str:
    layouts = ", ".join(sorted({asset.logical_asset.channel_config_actual for asset in assets}))
    presentations = sorted({
        asset.logical_asset.presentation_label or asset.logical_asset.channel_config_actual
        for asset in assets
    })
    return (
        f"{group_id} — {len(assets)} assets — layouts: {layouts} — "
        f"presentations: {', '.join(presentations)}"
    )


def _review_asset_line(label: str, asset: ClassifiedLogicalAsset | None) -> str:
    if asset is None:
        return f"{label}: none"
    return f"{label}: {logical_asset_display_name(asset)} ({source_summary(asset)})"


def _context_review_lines(folder_context: FolderContext) -> list[str]:
    lines = [f"Folder: {folder_context.folder}"]
    if folder_context.mode == "prep" and folder_context.prep_root is not None:
        lines.append(f"Prep root: {folder_context.prep_root}")
    return lines


def _null_summary_line(report: NullReport) -> str:
    if report.null_test.summary is None:
        return f"Null {report.summary.overall_pass and 'PASS' or 'FAIL'} — {report.null_test.reason or 'no summary'}."
    flagged = report.null_test.summary.flagged_regions
    verdict = "PASS" if report.null_test.pass_ else "FAIL"
    return f"Null {verdict} — {flagged} flagged region{'' if flagged == 1 else 's'}."


def _me_summary_line(report: MEReport) -> str:
    if report.me_check.summary is None:
        return f"M&E {report.summary.overall_pass and 'PASS' or 'FAIL'} — {report.me_check.reason or 'no summary'}."
    flagged = report.me_check.summary.flagged_regions
    verdict = "PASS" if report.me_check.pass_ else "FAIL"
    return f"M&E {verdict} — {flagged} flagged region{'' if flagged == 1 else 's'}."


def _written_artifacts(written: WrittenArtifacts) -> list[Path]:
    out = [written.json_path, written.html_path]
    if written.aaf_path is not None:
        out.append(written.aaf_path)
    return out


@contextmanager
def _job_spinner(*, message: str = _SPINNER_MESSAGE):
    stream = click.get_text_stream("stderr")
    if not _spinner_enabled(stream):
        yield
        return

    stop_event = threading.Event()
    width = len(message) + 3

    def render(frame: int) -> None:
        dots = "." * ((frame % 3) + 1)
        padding = " " * (3 - len(dots))
        stream.write(f"\r{message}{dots}{padding}")
        stream.flush()

    def clear() -> None:
        stream.write("\r" + (" " * width) + "\r")
        stream.flush()

    def spin() -> None:
        frame = 0
        render(frame)
        while not stop_event.wait(_SPINNER_INTERVAL_SECONDS):
            frame += 1
            render(frame)

    thread = threading.Thread(target=spin, name="finalpass-wizard-spinner", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        clear()


def _spinner_enabled(stream) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())
