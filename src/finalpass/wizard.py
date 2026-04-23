from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

from .all_assets import AssetFolderScan, ClassifiedLogicalAsset, discover_folder_assets
from .classify import load_config
from .errors import FinalPassError
from .jobs import (
    WrittenArtifacts,
    run_all,
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
from .specs import Spec, list_bundled, load_spec
from .wizard_io import Choice, WizardBack, WizardQuit, choose_many, choose_one, print_section, prompt_text

WizardAction = Literal["job_menu", "choose_folder", "exit"]
WizardStatus = Literal["PASS", "FAIL", "TOOL ERROR"]

DEFAULT_OUTPUT_DIR = Path("./finalpass-report")
COMMON_FPS_OPTIONS = [23.976, 24.0, 25.0, 29.97, 30.0]


@dataclass(frozen=True)
class FolderContext:
    folder: Path
    scan: AssetFolderScan

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
    status: WizardStatus
    summary_line: str
    artifacts: list[Path]
    error_message: str | None = None


@dataclass(frozen=True)
class SpecSelection:
    spec_name: str
    spec: Spec
    source_label: str


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
                action = _job_menu()
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

        context, error_message = _scan_folder(candidate)
        if context is None:
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


def _job_menu() -> Literal["all", "loudness", "null", "me", "choose_folder", "exit"]:
    print_section("Choose a job:")
    return choose_one(
        "Select a job type.",
        [
            Choice("Analyze delivery folder", "all"),
            Choice("Measure loudness on one asset", "loudness"),
            Choice("Run printmaster null", "null"),
            Choice("Run M&E bleed check", "me"),
            Choice("Choose a different folder", "choose_folder"),
            Choice("Exit", "exit"),
        ],
        allow_back=False,
    )


def _all_flow(folder_context: FolderContext, *, default_out_dir: Path, default_fps: float) -> FlowResult:
    while True:
        try:
            spec_selection = _choose_spec_for_all()
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
                        f"Folder: {folder_context.folder}",
                        "Job: all",
                        f"Spec: {spec_selection.source_label}",
                        f"Output dir: {out_dir}",
                        f"FPS: {fps:g}",
                    ])
                except WizardBack:
                    continue

                result = _execute_report_job(
                    out_dir=out_dir,
                    summary_builder=lambda report: (
                        f"{report.summary.groups_passed} groups passed, "
                        f"{report.summary.groups_failed} failed."
                    ),
                    runner=lambda: run_all(
                        folder=folder_context.folder,
                        spec_name=spec_selection.spec_name,
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
                                    f"Folder: {folder_context.folder}",
                                    "Job: loudness",
                                    f"Group: {group_id}",
                                    f"Asset: {_asset_label(asset)}",
                                    f"Spec: {spec_selection.source_label}",
                                    f"DX asset: {_asset_label(dx_asset) if dx_asset is not None else 'No DX asset'}",
                                    f"Output dir: {out_dir}",
                                    f"FPS: {fps:g}",
                                ]
                                _review_and_confirm(review_lines)
                            except WizardBack:
                                continue

                            result = _execute_report_job(
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
                                f"Folder: {folder_context.folder}",
                                "Job: null",
                                f"Group: {group_id}",
                                f"Printmaster: {_asset_label(pm_asset)}",
                                f"Stems: {', '.join(_asset_label(asset) for asset in stem_assets)}",
                                f"Output dir: {out_dir}",
                                f"FPS: {fps:g}",
                            ])
                        except WizardBack:
                            continue

                        result = _execute_report_job(
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
                                f"Folder: {folder_context.folder}",
                                "Job: me",
                                f"Group: {group_id}",
                                f"M&E asset: {_asset_label(me_asset)}",
                                f"DX asset: {_asset_label(dx_asset)}",
                                f"Output dir: {out_dir}",
                                f"FPS: {fps:g}",
                            ])
                        except WizardBack:
                            continue

                        result = _execute_report_job(
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


def _scan_folder(folder: Path) -> tuple[FolderContext | None, str | None]:
    folder = folder.expanduser()
    if not folder.exists():
        return None, f"{folder} does not exist."
    if not folder.is_dir():
        return None, f"{folder} is not a directory."

    try:
        scan = discover_folder_assets(folder.resolve(), load_config(None))
    except FinalPassError as exc:
        return None, str(exc)
    return FolderContext(folder=folder.resolve(), scan=scan), None


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
    options = [Choice(_asset_label(asset), asset) for asset in assets]
    return choose_one(title, options, auto_select_notice=auto_select_notice)


def _choose_spec_for_all() -> SpecSelection:
    bundled = list_bundled()
    options = [
        Choice(f"{spec.name} — {spec.display_name} — {spec.channel_config}", spec.name)
        for spec in bundled
    ]
    options.append(Choice("Enter a custom spec path", "__custom__"))
    choice = choose_one("Choose a spec.", options)
    return _resolve_spec_selection(choice)


def _choose_spec_for_loudness(asset: ClassifiedLogicalAsset) -> SpecSelection:
    bundled = [spec for spec in list_bundled() if spec.channel_config == asset.logical_asset.channel_config_actual]
    options = [
        Choice(f"{spec.name} — {spec.display_name} — {spec.channel_config}", spec.name)
        for spec in bundled
    ]
    options.append(Choice("Enter a custom spec path", "__custom__"))
    choice = choose_one("Choose a spec.", options)
    return _resolve_spec_selection(choice)


def _resolve_spec_selection(choice: str) -> SpecSelection:
    if choice != "__custom__":
        spec, source, path = load_spec(choice)
        return SpecSelection(spec_name=choice, spec=spec, source_label=f"{spec.name} ({source}, {path})")

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
        return SpecSelection(spec_name=str(path), spec=spec, source_label=f"{spec.name} ({source}, {path})")


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
            [Choice(_asset_label(asset), asset) for asset in same_layout_assets],
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
            [Choice(_asset_label(same_layout[0]), same_layout[0])],
            auto_select_notice="Auto-selected the only same-layout DX asset.",
        )
    return _choose_asset(candidates, title="Choose the DX asset.", auto_select_notice=notice)


def _choose_output_dir(default_out_dir: Path) -> Path:
    choice = choose_one(
        "Choose an output directory.",
        [
            Choice(f"Use current default output directory ({default_out_dir})", "default"),
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
            Choice("23.976", 23.976),
            Choice("24", 24.0),
            Choice("25", 25.0),
            Choice("29.97", 29.97),
            Choice("30", 30.0),
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
        click.echo(f"- {line}")
    choose_one(
        "Ready to run?",
        [Choice("Run job", "run")],
        default_index=1,
        auto_select=False,
        auto_select_notice=None,
    )


def _execute_report_job(*, out_dir: Path, summary_builder, runner) -> JobExecutionResult:
    try:
        report = runner()
        payload = report.model_dump_json(indent=2, by_alias=True)
        written = write_report_artifacts(report=report, payload=payload, out_dir=out_dir)
    except FinalPassError as exc:
        return JobExecutionResult(
            status="TOOL ERROR",
            summary_line="The job could not be completed.",
            artifacts=_existing_artifacts(out_dir),
            error_message=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive wizard wrapper
        return JobExecutionResult(
            status="TOOL ERROR",
            summary_line="The job could not be completed.",
            artifacts=_existing_artifacts(out_dir),
            error_message=str(exc),
        )

    status: WizardStatus = "PASS" if report.summary.overall_pass else "FAIL"
    return JobExecutionResult(
        status=status,
        summary_line=summary_builder(report),
        artifacts=_written_artifacts(written),
        error_message=None,
    )


def _result_screen(result: JobExecutionResult, *, out_dir: Path, fps: float) -> FlowResult:
    print_section("Result:")
    click.echo(f"Status: {result.status}")
    click.echo(result.summary_line)
    if result.error_message:
        click.echo(f"Error: {result.error_message}")
    if result.artifacts:
        click.echo("Artifacts:")
        for path in result.artifacts:
            click.echo(f"- {path}")
    else:
        click.echo("Artifacts: none")

    action = choose_one(
        "What do you want to do next?",
        [
            Choice("Run another job on this folder", "job_menu"),
            Choice("Choose a different folder", "choose_folder"),
            Choice("Exit", "exit"),
        ],
        allow_back=False,
        default_index=1,
    )
    return FlowResult(action=action, out_dir=out_dir, fps=fps)


def _print_folder_summary(context: FolderContext) -> None:
    print_section("Folder summary:")
    click.echo(f"Folder: {context.folder}")
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


def _asset_label(asset: ClassifiedLogicalAsset | None) -> str:
    if asset is None:
        return "None"
    logical_asset = asset.logical_asset
    member_count = len(logical_asset.source_paths)
    presentation = logical_asset.presentation_label or logical_asset.channel_config_actual
    file_word = "file" if member_count == 1 else "files"
    name = _logical_asset_name(asset)
    return (
        f"{asset.role} · {logical_asset.channel_config_actual} · {presentation} · "
        f"{logical_asset.source_kind} · {member_count} {file_word} · "
        f"{name}"
    )


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


def _existing_artifacts(out_dir: Path) -> list[Path]:
    candidates = [
        out_dir / "report.json",
        out_dir / "report.html",
        out_dir / "markers.aaf",
    ]
    return [path for path in candidates if path.exists()]


def _logical_asset_name(asset: ClassifiedLogicalAsset) -> str:
    logical_asset = asset.logical_asset
    stem = logical_asset.canonical_path.stem
    if logical_asset.source_kind == "split_mono" and logical_asset.member_legs:
        suffix = f".{logical_asset.member_legs[0]}"
        if stem.upper().endswith(suffix.upper()):
            return stem[:-len(suffix)]
    return logical_asset.canonical_path.name
