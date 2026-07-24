# FinalPass

FinalPass is a local command-line QC tool for post-production sound deliverables.
It measures WAV/BWF printmasters and stems, produces a machine-readable JSON
report, renders a self-contained HTML report for review, and optionally writes
an AAF marker session for timed failures that need to be inspected in a DAW.

FinalPass is built for the practical delivery-QC pass before handoff: loudness,
true peak, stem-sum null checks, M&E dialogue-bleed checks, split-mono handling,
prep-folder curation, and readable reporting. It is not a certification service
and does not replace current delivery paperwork or platform-specific tools.

## Current Status

This repository is a `0.1.0` release candidate. The implemented commands are:

- `finalpass loudness` - measure one or more program files against a bundled or
  user loudness spec.
- `finalpass null` - compare a printmaster against a supplied stem sum.
- `finalpass me` - check an M&E stem against its matching DX stem for likely
  dialogue bleed.
- `finalpass all` - analyze a folder of deliverables, classify logical assets,
  select the correct presentation for the chosen spec, then run the applicable
  loudness, null, and M&E checks.
- `finalpass channels` - per-channel integrity diagnostics on one or more
  logical assets.
- `finalpass downmix` - compare a delivered 2.0 against a fold-down derived
  from the matching 5.1/7.1 master.
- `finalpass wizard` - guide a user through folder-first analysis and optional
  `FinalPass Prep/` curation.
- `finalpass specs list` and `finalpass specs show` - inspect bundled spec
  presets.

Normal runs write report artifacts under `./finalpass-report/` unless `--out` is
provided. `--json-only` suppresses file writing and prints JSON to stdout.

## Capabilities

FinalPass currently supports:

- WAV/BWF input with local, offline analysis.
- PCM WAV/BWF at common bit depths and sample rates.
- BWF `bext` time references for timecode-aware flag strings and marker
  placement.
- All common delivery frame rates (23.976 through 120), with SMPTE ST 12-1
  drop-frame counting at 29.97/59.94 via `--drop-frame`.
- Interleaved mono, stereo, 5.1, and 7.1 sources in SMPTE channel order.
- Canonical split-mono stereo, 5.0, 5.1, and 7.1 sources.
- Special handling for common 5.0 split stems, especially dialog stems: the real
  five source legs are reported, and analysis uses a temporary silent LFE channel
  where a 5.1-compatible comparison is needed.
- Folder-level logical-asset discovery, including split-mono family assembly.
- Conservative filename classification for printmaster, DX, MX, FX, M&E, and
  optional stems.
- Prep-folder guided filtering for messy deliveries.
- ITU-R BS.1770-4 loudness measurement, 4x oversampled true peak, and LRA.
- Standalone and automatic stem-sum null checks.
- Standalone and automatic M&E dialogue-bleed checks.
- Standalone channel-integrity diagnostics: dead legs, duplicated channels,
  polarity inversion, broadband LFE content, and channel-imbalance notes.
- Standalone downmix-consistency checks between a delivered 2.0 and its
  surround master: loudness delta, windowed similarity, mono compatibility.
- Program-window comparison logic for null and M&E checks when files differ only
  by pre-program or tail MOS.
- Self-contained HTML reports rendered from the persisted report model.
- Optional AAF marker export for timed loudness, null, and M&E flags.
- Deterministic examples and a GitHub Actions test/smoke workflow.

FinalPass intentionally does not make network calls during analysis. There is no
telemetry, cloud upload, or LLM/API dependency in the runtime path.

## Installation

FinalPass is currently intended to run from source.

Requirements:

- Python 3.11 or newer.
- `uv` for dependency sync and command execution.

From the repository root:

```bash
uv sync
uv run finalpass --version
uv run finalpass --help
```

For development and CI parity:

```bash
uv sync --dev --frozen
uv run pytest
uv run python examples/_smoke.py
```

## Quick Start

Inspect the bundled specs:

```bash
uv run finalpass specs list
uv run finalpass specs show netflix_51
```

Analyze a prepared delivery folder:

```bash
uv run finalpass all /path/to/delivery --spec netflix_51 --out /path/to/qc-report
```

Use the interactive wizard for a real-world delivery that may contain extra
files, alternates, exports, or non-analysis material:

```bash
uv run finalpass wizard /path/to/delivery --out /path/to/qc-report
```

Run standalone loudness:

```bash
uv run finalpass loudness /path/to/SHOW_S01E03_COMP_5.1.wav --spec netflix_51
```

Run a standalone null check:

```bash
uv run finalpass null \
  /path/to/SHOW_S01E03_COMP_5.1.wav \
  /path/to/SHOW_S01E03_DX_5.1.wav \
  /path/to/SHOW_S01E03_MX_5.1.wav \
  /path/to/SHOW_S01E03_FX_5.1.wav
```

Run a standalone M&E dialogue-bleed check:

```bash
uv run finalpass me \
  /path/to/SHOW_S01E03_ME_5.1.wav \
  --dx /path/to/SHOW_S01E03_DX_5.1.wav
```

## Command Reference

```bash
uv run finalpass --version
uv run finalpass specs list
uv run finalpass specs show <name>
uv run finalpass wizard [folder] [--out <dir>] [--fps <rate>]
uv run finalpass loudness <file>... --spec <name-or-yaml> [--dx <file>] [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
uv run finalpass null <pm> <stems>... [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
                                      [--window-ms <ms>] [--hop-ms <ms>] [--threshold-dbfs <dbfs>]
uv run finalpass me <me_file> --dx <dx_file> [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
                                             [--window-ms <ms>] [--hop-ms <ms>]
                                             [--band-low-hz <hz>] [--band-high-hz <hz>]
                                             [--corr-threshold <value>]
                                             [--coherence-threshold <value>]
                                             [--dx-gate-dbfs <dbfs>] [--me-floor-dbfs <dbfs>]
uv run finalpass channels <file-or-seed>... [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
                                           [--window-ms <ms>] [--hop-ms <ms>]
                                           [--activity-dbfs <dbfs>] [--silence-dbfs <dbfs>]
                                           [--duplicate-null-db <db>] [--polarity-corr <value>]
                                           [--lfe-cutoff-hz <hz>] [--lfe-energy-ratio <value>]
                                           [--imbalance-db <db>] [--fail-dual-mono]
uv run finalpass downmix <stereo> <surround> [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
                                            [--center-db <db>] [--surround-db <db>]
                                            [--lfe-db <db>] [--lfe-lowpass-hz <hz>]
                                            [--window-ms <ms>] [--hop-ms <ms>]
                                            [--activity-dbfs <dbfs>]
                                            [--similarity-corr <value>] [--mono-corr <value>]
                                            [--loudness-delta-lu <lu>]
uv run finalpass all <folder> --spec <name-or-yaml> [--out <dir>] [--json-only] [--fps <rate>] [--drop-frame]
                                                    [--patterns <config.yaml>] [--include-unclassified]
                                                    [--null-window-ms <ms>] [--null-hop-ms <ms>]
                                                    [--null-threshold-dbfs <dbfs>]
                                                    [--me-window-ms <ms>] [--me-hop-ms <ms>]
                                                    [--me-band-low-hz <hz>] [--me-band-high-hz <hz>]
                                                    [--me-corr-threshold <value>]
                                                    [--me-coherence-threshold <value>]
                                                    [--me-dx-gate-dbfs <dbfs>]
                                                    [--me-me-floor-dbfs <dbfs>]
```

Exit codes:

- `0` - all checks passed.
- `1` - FinalPass completed and found one or more failures or blocking
  discovery issues.
- `2` - tool, validation, input, or runtime error.

`--fps` is the frame rate used for timecode strings and AAF marker placement.
FinalPass does not infer FPS or drop-frame mode from WAV/BWF metadata.

Supported frame rates: 23.976, 24, 25, 29.97, 30, 47.952, 48, 50, 59.94, 60,
119.88, and 120, all counted non-drop by default. `--drop-frame` switches
29.97 or 59.94 to SMPTE ST 12-1 drop-frame counting; drop-frame timecode
renders with the semicolon convention (`HH:MM:SS;FF`) everywhere a timecode
string appears — terminal tables, JSON, HTML, and AAF marker fields — and the
report envelope records `drop_frame: true`. Combining `--drop-frame` with any
other rate is a validation error (SMPTE ST 12-1 defines drop-frame counting
only for the 30000/1001 and 60000/1001 families). At 29.97 or 59.94 without
the flag, the CLI prints a note recommending `--drop-frame` for drop-frame
shows, since a drop-frame session counter runs about 3.6 seconds per hour
ahead of non-drop labels.

## Main Workflows

### Folder QC with `all`

`finalpass all` is the main non-interactive workflow. It scans a folder for
WAV/BWF files, assembles split-mono families into logical assets, classifies
each logical asset by role, groups related deliverables by episode/reel/fallback
identifier, selects the layout required by the spec, and runs the applicable
checks.

The automatic null strategy is intentionally narrow:

- Prefer `DX + MX + FX`.
- Fall back to `DX + M&E`.
- Otherwise record `Null: SKIPPED` with an honest reason.

Automatic M&E analysis runs only when both `DX` and `M&E` are present. If either
role is missing, the group records `M&E: SKIPPED`.

Use standalone `finalpass null` for unusual or manual stem combinations. Use
standalone `finalpass me` when you need to tune band, gate, hop, window, or
threshold settings directly.

### Wizard and Prep Folders

The wizard is a minimal terminal flow for real delivery folders. It presents
logical assets rather than raw mono legs wherever practical and uses the same
job/report pipeline as the direct commands.

For messy deliveries, use prep mode:

```bash
uv run finalpass wizard /path/to/delivery
```

Typical prep flow:

1. Choose `Create FinalPass prep folders`.
2. Move or copy only the files that should be analyzed into `FinalPass Prep/`.
3. Put explicit non-analysis material into `FinalPass Prep/Ignore/`, or leave it
   outside the prep root.
4. Re-run the wizard or choose `Re-scan prep folders now`.
5. Run `all`, `loudness`, `null`, or `me` from the curated prep assets.

If a valid `FinalPass Prep/` layout already exists, the wizard auto-resumes prep
mode and ignores files outside that prep root for wizard analysis.

### Standalone Split-Mono Inputs

The standalone `loudness`, `null`, and `me` commands accept either an
interleaved file path or a seed path to one member of a canonical split-mono
family in the same directory.

Supported split layouts:

- Stereo: `L`, `R`
- 5.0: `L`, `R`, `C`, `Ls`, `Rs`
- 5.1: `L`, `R`, `C`, `LFE`, `Ls`, `Rs`
- 7.1: `L`, `R`, `C`, `LFE`, `Ls`, `Rs`, `Lss`, `Rss`

Leg tokens are matched at the end of the filename stem. In addition to the
abbreviations above, FinalPass accepts the spelled-out channel words that
Pro Tools writes in bounce filenames: `Left`, `Right`, `Center`/`Centre`,
`Left Front`, `Right Front`, `Left Surround`, and `Right Surround`. Commas
count as token separators, and a leading track-number prefix (for example
`"11 SHOW 5.1 Mix, Left Front.wav"`) is ignored when family membership and
group identity are derived, so a raw Pro Tools bounce folder can assemble
without renaming.

For a 5.0 split source, FinalPass analyzes the source as 5.1 with a temporary
silent LFE channel where needed. Reports still preserve the real five source
legs so reviewers can see what was actually delivered.

### Examples and Smoke Runs

Generate deterministic example deliveries:

```bash
uv run python examples/_generate.py
```

Run the smoke flow used by CI:

```bash
uv run python examples/_smoke.py
```

See [examples/README.md](examples/README.md) for representative interleaved,
split-mono, and prep-folder commands.

## Outputs and Artifact Naming

Normal runs write artifacts to `--out`, defaulting to `./finalpass-report/`.

FinalPass tries to derive a program slug from source filenames or delivery
folder names. For TV-style names it tries to preserve show and episode
information. Examples:

- `NLA1120_262_51_DXSM_2398_48_20200408.L.wav` -> `nla-1120-report.html`
- `SHOW_S01E03_COMP_5.1.wav` -> `show-s01e03-report.html`
- multi-episode folder input can become `show-s01e03-s01e04-report.html`

If no stable program name can be inferred, FinalPass uses the historical names:

- `report.json`
- `report.html`
- `markers.aaf`, only when exportable timed markers exist.

When an output filename already exists, FinalPass writes a numbered sibling such
as `show-s01e03-report-01.html` instead of overwriting the previous run.

Artifacts:

- `<program>-report.json` - stable machine-readable output.
- `<program>-report.html` - self-contained HTML report rendered only from the
  persisted report model.
- `<program>-markers.aaf` - optional AAF marker session written only when the
  report contains exportable timed flags.

`--json-only` writes no files and emits the JSON report to stdout.

## HTML Report Sections

The HTML report is designed for local review and handoff. It does not reopen
audio or recompute measurements; it renders the same model persisted to JSON.

Common sections:

- **Header and verdict** - identifies the report type and overall pass/fail
  state.
- **Run metadata** - command, FinalPass version, schema version, run ID, run
  timestamp, and artifact names.
- **Source files** - a registry of every unique source file with full paths.
  Later sections use filenames only so long split-mono paths do not dominate the
  page.
- **Run summary** - group/check totals, skipped checks, failures, and blocking
  issue count.
- **Discovery issues** - incomplete split families, ambiguous classification,
  missing legs, unsupported layouts, and other folder-level blockers.
- **Unclassified logical assets** - valid WAV/BWF assets that did not match a
  role pattern.
- **Group sections** - per-episode or per-reel analysis blocks for `all`.
- **Logical assets** - classified assets, source kind, layout, source members,
  how each asset was used, and selection notes.
- **Measured assets** - sample rate, bit depth, channel configuration, duration,
  source provenance, measurements, and per-metric checks.
- **Null check** - auto or standalone null strategy, selected roles, analysis
  summary, comparison window, timeline, flagged regions, and errors.
- **M&E check** - analysis signal, gate/band/threshold summary, comparison
  window, timeline, flagged regions, and errors.
- **Comparison window** - the exact sample/timecode span used for null and M&E
  comparison, plus ignored head, ignored tail, and padded tail per input.

The report uses compact tables, basename-only file references after the source
registry, and fixed-width source chips for split-mono members.

## AAF Marker Session

FinalPass writes an AAF marker session only when the report contains timed
flags. It never creates synthetic markers for skipped checks, group-level
errors, discovery errors, or untimed loudness failures.

Exportable timed flags:

- Standalone `loudness`: `files[].flags[]` for true-peak-over regions.
- Standalone `null`: `null_test.flags[]`.
- Standalone `me`: `me_check.flags[]`.
- `all`: `groups[].files[].flags[]`, `groups[].null_test.flags[]`, and
  `groups[].me_check.flags[]`.

- Standalone `downmix`: `downmix_check.flags[]` for both timed lanes, with
  marker labels `downmix mismatch` and `mono compatibility`.

Channel-integrity findings are never exported: they describe a channel, not a
moment on the timeline, so `finalpass channels` writes `report.json` and
`report.html` only — even when it fails.

The exported AAF contains:

- A top-level composition named `FinalPass Markers`.
- A timecode timeline at the edit rate derived from `--fps`, with the
  drop-frame flag set when the run used `--drop-frame`.
- A marker guide audio slot with filler, so common DAW/NLE imports have a real
  timeline container.
- A marker event slot containing `CommentMarker` events.
- Marker names like `[1] FAIL: null mismatch`, `[2] FAIL: dialog bleed`, or
  `[3] FAIL: true peak over`.
- User comments/tags containing label, comment, detail, failure code, metric,
  measured value, threshold, group context, and source context where available.

Marker placement is based on the same sample positions and timecode strings in
the JSON report. If the source carries a BWF time reference, FinalPass anchors
the flag timecode and AAF edit-unit position to that reference. If there is no
BWF time reference, placement starts from file sample zero. The frame rate and
drop-frame mode still come from `--fps` and `--drop-frame`; placement itself
uses exact rational edit rates and is identical in both counting modes — only
the timecode labels differ.

For loudness true-peak regions, markers are throttled to at most one marker per
second per source so a clipped passage does not flood the marker lane.

## How Analysis Works

### Loudness

`finalpass loudness` and the loudness portions of `finalpass all` measure:

- Integrated loudness.
- True peak.
- Loudness range.
- Short-term maximum.
- Momentary maximum.

The implementation follows the current open FinalPass measurement path:
ITU-R BS.1770-4 style gating through `pyloudnorm`, 4x oversampled true peak, and
LRA per the EBU Tech 3342 approach. Dialog-anchored bundled specs can measure a
classified DX file or a standalone `--dx` file against `dialog_lufs`.

### Null

The null check sums supplied stems and compares the result against the
printmaster. It reports residual RMS windows and flags regions whose residual
exceeds the configured threshold.

Automatic null in `all` uses conservative known stem sets. The standalone
command exists for manual or unusual deliverables.

### M&E Dialogue-Bleed

The M&E check compares an M&E stem against its matching DX stem. It downmixes for
analysis while excluding LFE, gates windows by DX level, band-limits to a
speech-oriented range, and flags windows that exceed the configured correlation
and coherence thresholds.

The two inputs may carry different channel layouts — a mono DX against a
stereo or 5.1 M&E is a normal delivery shape. Both signals are reduced to the
same mono analysis downmix before comparison, so only per-file layout support
is validated. The null check still requires matching channel counts because it
subtracts in the sample domain.

This is an honest heuristic for likely dialogue bleed. It is not speech
recognition, transcription, diarization, or Dolby Dialogue Intelligence.

### Channel Integrity

`finalpass channels` inspects one logical asset at a time and reports what its
individual channels actually contain. Every detector is a deterministic
statistic; nothing here is auto-repaired, and FinalPass never remaps channels.

- **Silent leg** - a channel whose full-span RMS falls below `--silence-dbfs`
  while the asset is otherwise active. Every delivered full-range channel is
  expected to carry at least minimal content, so a dead leg fails regardless of
  the asset's role. The LFE is exempt and reported as an informational note:
  programs that never feed the LFE are legitimate deliveries.
- **Duplicate channels** - a phase-invert null test. For each qualifying window
  FinalPass fits the gain that best cancels one channel with the other and
  measures how far the residual sits below the content. A median null depth at
  or beyond `--duplicate-null-db` with a positive fitted gain means the pair is
  the same signal, and the reported level offset quantifies a scaled copy. On a
  5.1 or 7.1 asset this fails - that is fake surround. On a stereo asset it is
  an informational notification by default, because dual-mono 2.0 is low-rent
  but real; `--fail-dual-mono` escalates it.
- **Polarity inversion** - a canonical pair (L/R, Ls/Rs, Lss/Rss) whose median
  windowed signed correlation falls at or below `--polarity-corr`. A pair that
  nulls deeply against a *negative* fitted gain is an inverted copy and is
  reported here rather than as a duplicate.
- **Broadband LFE** - with the LFE above the silence floor, the fraction of its
  energy surviving a 4th-order zero-phase high-pass at `--lfe-cutoff-hz`. At or
  above `--lfe-energy-ratio` this fails and names the likely cause: a film-order
  interleave (L C R Ls Rs LFE) read as SMPTE (L R C LFE Ls Rs).
- **Channel imbalance** - an L/R full-span RMS delta beyond `--imbalance-db`.
  Always informational, never a failure.

Pairwise statistics are windowed and reduced by median rather than measured
across the whole file. Identical head tone, sync pops, and silence would
otherwise dominate a whole-file correlation and manufacture duplicate findings
on legitimately decorrelated content. A pair needs at least eight windows where
both channels exceed `--activity-dbfs`; below that the duplicate and polarity
findings are recorded as skipped with `insufficient_active_content` rather than
presented as a pass. Silence, LFE, and imbalance statistics use the whole file,
because head and tail material are part of the delivered channel truth.

When a 5.0 split family is assembled into a 5.1 container, its LFE is
FinalPass-synthesized silence. All LFE findings are suppressed for that asset
and the report records `lfe_synthesized_from_5_0_source`. A mono asset reports
its pairwise checks as structurally not applicable and passes.

### Downmix Consistency

`finalpass downmix` takes a delivered 2.0 and the matching 5.1/7.1 master,
derives a fold-down from the surround in memory, and compares the two:

```text
L' = L + g_c*C + g_s*Ls [+ g_s*Lss]
R' = R + g_c*C + g_s*Rs [+ g_s*Rss]
g_c = --center-db    (default -3 dB)
g_s = --surround-db  (default -3 dB)
LFE omitted unless --lfe-db is supplied, in which case it is low-passed first
```

The default gains are the published derivation values used across the
industry. There is no clipping, normalization, limiting, or phase
manipulation, and the derived signal is never written to disk.

**This is a plain in-phase fold-down. It is not an Lt/Rt matrix encode, and
FinalPass does not emulate any matrix decoder.** When the delivered stereo is
labelled `LtRt`, the report adds a note saying that matrix-encoded surround
content can legitimately depress similarity in surround-heavy passages.

Three sub-checks contribute to one verdict:

- **Level** (untimed) - integrated loudness of both signals over the
  comparison window; fails when the difference exceeds `--loudness-delta-lu`.
- **Similarity** (timed) - per-window correlation between the two mono sums;
  windows below `--similarity-corr` merge into flagged regions.
- **Mono compatibility** (timed) - per-window L/R correlation of the
  *delivered* stereo alone; windows below `--mono-corr` merge into their own
  flagged regions.

A discrete stereo mix that is not a mechanical fold-down is normal and passes
at the default thresholds. This check is tuned to catch the wrong episode, a
gross level offset, sync drift between layouts, missing elements, and
phase-hostile stereo - not to demand fold-down identity. A constant offset
between the two inputs is a hard alignment failure, never auto-corrected.

### Program-Window Comparison

Null and M&E comparisons often encounter files with different head builds or MOS
tail lengths. FinalPass handles that by deriving a comparison window before
analysis:

- If all comparison inputs have BWF time references and share a whole-hour
  program boundary, analysis starts at that boundary, such as `01:00:00:00` or
  `10:00:00:00`. The boundary is found in the selected timecode mode: with
  `--drop-frame`, `01:00:00;00` sits at the ST 12-1 drop-frame frame count
  (107,892 frames per hour at 29.97) — the wall-clock hour — instead of the
  non-drop hour roughly 3.6 seconds later.
- If files share the same start reference but no whole-hour boundary is present,
  analysis starts at file start and records that mode.
- If time references are unavailable, analysis starts at file sample zero.
- A uniquely longer file is accepted only when the overrun after every other
  input ends is MOS under the configured RMS and peak gates.
- Shorter inputs are padded with silence inside the comparable range.
- The report records ignored head, ignored tail, and padded tail for each input.

This lets FinalPass compare the actual program region when one delivery has
extra pre-program material or post-program dead air while still rejecting a file
whose unique overrun contains signal.

## Filename Conventions

The bundled classifier looks for boundary-delimited role tokens in each filename
stem. Tokens are case-insensitive and must be bracketed by start-of-string or
one of `_`, `-`, `.`, or a space.

| Role | Accepted tokens |
| --- | --- |
| `pm` printmaster | `PM`, `PRINTMASTER`, `MIX`, `FINAL`, `COMP` |
| `dx` dialog | `DX`, `DIA`, `DIALOG`, `DIALOGUE` |
| `mx` music | `MX`, `MUS`, `MUSIC` |
| `fx` effects | `FX`, `SFX`, `EFFECTS` |
| `me` M&E | `ME`, `M&E`, `M_AND_E`, `MANDE` |
| `opt` optional | `OPT`, `OPTIONAL`, `NARR`, `NARRATION` |

Group IDs come from the first matching pattern:

- `S##E##`
- `EP##`
- `R##`
- fallback stem after role and channel tokens are stripped.

Examples:

- `SHOW_S01E03_COMP_5.1.wav` -> role `pm`, group `S01E03`.
- `SHOW_S01E03_DX_5.1.L.wav` -> role `dx`, group `S01E03`, split member `L`.
- `MYSHOW_PM_STEREO.wav` and `MYSHOW_DX_STEREO.wav` group together under
  `MYSHOW`.

A filename that matches multiple roles fails with an ambiguous-classification
error. Rename it or provide a custom classifier pattern file rather than letting
FinalPass guess.

Override the bundled classifier with:

```bash
uv run finalpass all /path/to/delivery --spec netflix_51 --patterns path/to/patterns.yaml
```

Use the same top-level shape as
[src/finalpass/_bundled_patterns.yaml](src/finalpass/_bundled_patterns.yaml).

## Bundled Specs

Bundled presets:

- `atsc_a85` - stereo, -24 LKFS +/-2, true peak <= -2 dBTP, LRA <= 18.
- `atsc_a85_51` - 5.1, -24 LKFS +/-2, true peak <= -2 dBTP, LRA <= 18.
- `atsc_a85_mono` - mono, -24 LKFS +/-2, true peak <= -2 dBTP, LRA <= 18.
  Also convenient for spot-checking a mono dialog or VO stem against the
  -24 LKFS anchor.
- `ebu_r128` - stereo, -23 LUFS +/-0.5, true peak <= -1 dBTP, LRA <= 18.
- `netflix_stereo` - stereo, -27 LKFS +/-2, true peak <= -2 dBTP,
  dialog-anchored, LRA <= 18.
- `netflix_51` - 5.1, -27 LKFS +/-2, true peak <= -2 dBTP,
  dialog-anchored, LRA <= 18.
- `streaming_-14` - stereo, -14 LUFS +/-1, true peak <= -1 dBTP, LRA <= 18.

Each YAML spec includes notes and source URLs. The values are best-public
knowledge presets and FinalPass conventions, not platform certification. Always
check current delivery paperwork.

User specs can be supplied by passing a YAML path to `--spec`.

## JSON Schemas

The JSON reports are Pydantic v2 models with `extra="forbid"` to keep the
on-disk contract explicit.

Current schema versions:

- `loudness`: `schema_version: 4`
- `null`: `schema_version: 4`
- `me`: `schema_version: 4`
- `all`: `schema_version: 9`
- `channels`: `schema_version: 1`
- `downmix`: `schema_version: 1`

The version bumps from 3/3/3/8 add one envelope field: `drop_frame: bool`,
recorded beside `fps`. Timecode strings persist in the selected counting mode
(`HH:MM:SS:FF` non-drop, `HH:MM:SS;FF` drop-frame).

Important persisted fields include:

- `finalpass_version`, `schema_version`, `run_id`, `run_started_at`, `fps`,
  and `drop_frame`.
- `spec` references for loudness and folder analysis.
- Source provenance: `source_kind`, `source_paths`, `member_legs`, and
  `presentation_label`.
- `flags[]` for timed loudness, null, and M&E failures.
- `analysis_window` for null and M&E comparisons.
- `groups[].assets[]`, `groups[].files[]`, `groups[].null_test`,
  `groups[].me_check`, `unclassified[]`, and `discovery_errors[]` for `all`.

## Channel Order

FinalPass uses SMPTE order throughout:

```text
L R C LFE Ls Rs [Lss Rss]
```

A 5.0 split source is the accepted exception. FinalPass inserts a temporary
silent LFE channel for analysis compatibility and reports the true 5.0 source
provenance. It does not otherwise remap channels. If a file uses a different
order, fix or relabel it before analysis.

## Codebase Overview

The repository is intentionally small and module-oriented.

Core package:

- [src/finalpass/cli.py](src/finalpass/cli.py) - Click command definitions,
  option validation, terminal rendering, exit-code behavior, and artifact
  announcements.
- [src/finalpass/jobs.py](src/finalpass/jobs.py) - orchestration for direct and
  folder jobs, report model construction, auto-null/auto-M&E selection, and
  artifact writing.
- [src/finalpass/models.py](src/finalpass/models.py) - Pydantic report models
  and schema-versioned output contracts.
- [src/finalpass/errors.py](src/finalpass/errors.py) - user-facing error types.
- [src/finalpass/specs.py](src/finalpass/specs.py) and
  [src/finalpass/_bundled_specs](src/finalpass/_bundled_specs) - bundled and
  user loudness specs.
- [src/finalpass/audio_io.py](src/finalpass/audio_io.py) - WAV/BWF loading,
  audio metadata, channel handling, and BWF time-reference extraction.
- [src/finalpass/loudness.py](src/finalpass/loudness.py) - loudness,
  true-peak, LRA, and timed true-peak flag generation.
- [src/finalpass/null_test.py](src/finalpass/null_test.py) - stem-sum null
  analysis and residual flagging.
- [src/finalpass/me_check.py](src/finalpass/me_check.py) - M&E dialogue-bleed
  heuristic, speech-band filtering, gating, correlation, and coherence.
- [src/finalpass/analysis_window.py](src/finalpass/analysis_window.py) - shared
  comparison-window preflight for null and M&E checks.
- [src/finalpass/assets.py](src/finalpass/assets.py),
  [src/finalpass/all_assets.py](src/finalpass/all_assets.py), and
  [src/finalpass/standalone_ingest.py](src/finalpass/standalone_ingest.py) -
  interleaved and split-mono discovery, logical-asset assembly, strict family
  validation, and standalone seed-path resolution.
- [src/finalpass/classify.py](src/finalpass/classify.py) and
  [src/finalpass/_bundled_patterns.yaml](src/finalpass/_bundled_patterns.yaml) -
  role, group, and channel-hint classification.
- [src/finalpass/artifact_naming.py](src/finalpass/artifact_naming.py) - derived
  show/episode slugs for report and marker artifact names.
- [src/finalpass/prep_folders.py](src/finalpass/prep_folders.py),
  [src/finalpass/wizard.py](src/finalpass/wizard.py), and
  [src/finalpass/wizard_io.py](src/finalpass/wizard_io.py) - terminal wizard
  and prep-folder workflow.
- [src/finalpass/report.py](src/finalpass/report.py) and
  [src/finalpass/templates](src/finalpass/templates) - self-contained HTML
  rendering from persisted models.
- [src/finalpass/aaf_export.py](src/finalpass/aaf_export.py) - AAF marker
  candidate collection and `pyaaf2` export.
- [src/finalpass/timecode.py](src/finalpass/timecode.py) - sample/timecode and
  edit-unit conversions for reports and markers.
- [src/finalpass/presentation.py](src/finalpass/presentation.py) - shared
  human-readable labels, verdict descriptions, and source summaries.

Supporting files:

- [tests](tests) - unit and integration coverage for specs, CLI behavior,
  reports, examples, split-mono discovery, artifact naming, AAF export, null,
  M&E, and wizard flows.
- [examples](examples) - deterministic synthetic deliveries and smoke commands.
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - CI on Python 3.11 and
  3.12, including the example smoke flow.
- [specs](specs) and [docs](docs) - implementation notes, phase specs, and
  handoff/reference documents from development.

## Development

Set up the environment:

```bash
uv sync --dev --frozen
```

Run tests:

```bash
uv run pytest
```

Run the example smoke flow:

```bash
uv run python examples/_smoke.py
```

Check CLI help:

```bash
uv run finalpass --help
uv run finalpass all --help
uv run finalpass me --help
```

CI runs the test suite and smoke helper on Python 3.11 and 3.12.

## Scope and Non-Goals

In scope for this release candidate:

- Local WAV/BWF delivery QC.
- Interleaved and canonical split-mono source handling.
- Folder-first and prep-folder workflows.
- Loudness, true peak, LRA, null, and M&E heuristic checks.
- JSON, HTML, terminal summary, and conditional AAF marker output.

Out of scope:

- Atmos, ADM BWF, MXF, DCP, or video-container audio.
- Automatic time alignment of genuinely misaligned stems.
- Speech recognition, transcription, diarization, or ML/VAD.
- Dolby Dialogue Intelligence or other closed proprietary algorithms.
- Watch folders, background services, cloud processing, telemetry, or GUI.
- DAW editing. FinalPass reports problems; it does not modify audio.

## Privacy and Security

FinalPass analyzes local files in place and writes local report artifacts. The
runtime path does not call network services, upload media, or collect telemetry.

Development dependencies are pinned through `uv.lock`. This repository follows a
conservative dependency policy during development: newly released package
versions should age before adoption, and install scripts should not be enabled
without explicit review.

## Contributing

This is an early release candidate. Useful contributions should include:

- A clear description of the delivery layout or failure case.
- Small reproducible fixtures when possible.
- Tests for classifier, split-mono, report, or analysis behavior.
- Notes about whether the issue affects interleaved input, split-mono input,
  prep-folder mode, direct commands, `all`, or the wizard.

Generated media and local report outputs should stay out of commits unless they
are intentionally added as small deterministic fixtures.

## License

MIT. See [LICENSE](LICENSE).
