# FinalPass

Delivery QC CLI for post-production sound. The v0.1 release candidate ships loudness, standalone stem-sum null, standalone M&E dialogue-bleed checks, split-mono support for the standalone commands and for `all`, folder-level auto-null / auto-M&E against a named spec, a self-contained HTML report, conditional AAF marker export for timed null/M&E flags, example smoke flows, and CI.

**Status: Phase 7 release candidate plus SM-3 logical-asset `all` support.** `loudness`, standalone `null`, standalone `me`, and `all` with auto-null / auto-M&E are implemented. Standalone `loudness` / `null` / `me` accept either a normal interleaved WAV/BWF path or a seed path to one member of a canonical split stereo / 5.1 / 7.1 family. `all` now discovers logical assets at the folder level, so one split stereo/5.1/7.1 family is treated as one analyzable asset, alternate presentations are preserved in inventory, and invalid split families surface as discovery errors. Normal runs always write `report.json` and `report.html`, and write `markers.aaf` only when the run contains exportable timed `null` / `me` flagged regions.

## Install (dev)

```
uv sync
uv run finalpass --version
uv run finalpass --help
```

Python 3.11+. WAV/BWF input.

## Commands

```
uv run finalpass --version
uv run finalpass specs list
uv run finalpass specs show <name>
uv run finalpass loudness <file>... --spec <name> [--dx <file>] [--out <dir>] [--json-only] [--fps <rate>]
uv run finalpass null <pm> <stems>... [--out <dir>] [--json-only] [--fps <rate>]
                                      [--window-ms <ms>] [--hop-ms <ms>] [--threshold-dbfs <dbfs>]
uv run finalpass me <me_file> --dx <dx_file> [--out <dir>] [--json-only] [--fps <rate>]
                                             [--window-ms <ms>] [--hop-ms <ms>]
                                             [--band-low-hz <hz>] [--band-high-hz <hz>]
                                             [--corr-threshold <value>]
                                             [--coherence-threshold <value>]
                                             [--dx-gate-dbfs <dbfs>] [--me-floor-dbfs <dbfs>]
uv run finalpass all <folder> --spec <name> [--out <dir>] [--json-only] [--fps <rate>]
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

Exit codes: `0` all pass, `1` failures found, `2` tool error.

Standalone input note:
- `finalpass loudness`, `finalpass null`, and `finalpass me` accept either an interleaved WAV/BWF path or a seed path to one member of a canonical split-mono family in the same directory.
- Supported split layouts are stereo `L/R`, 5.1 `L/R/C/LFE/Ls/Rs`, and 7.1 `L/R/C/LFE/Ls/Rs/Lss/Rss`.
- `finalpass all` now does folder-level logical-asset discovery too. Interleaved files remain supported, but explicit split-mono families are assembled in memory, grouped by logical asset, and selected by actual layout for the chosen spec.

## Examples / smoke flow

See [examples/README.md](examples/README.md) for the deterministic delivery
generator, disposable smoke outputs under `examples/out/`, and representative
`loudness`, `null`, `me`, and `all` commands.

## Artifacts

Normal runs always write these files to `--out` (default `./finalpass-report/`):

- `report.json` — the stable machine-readable contract.
- `report.html` — a self-contained local HTML report rendered from the persisted report model only.

Conditional artifact:

- `markers.aaf` — written only when the run contains one or more timed `null` / `me` flagged regions.

`--json-only` writes no files and emits JSON to stdout only.

Still not implemented:

- `summary.txt` — not a current output artifact.

AAF export stays intentionally narrow in v0.1:

- standalone `null` exports `null_test.flags[]`
- standalone `me` exports `me_check.flags[]`
- `all` exports only `groups[].null_test.flags[]` and `groups[].me_check.flags[]`
- no synthetic markers are created for loudness-only failures, group-level errors, or skipped checks

### `finalpass all` quickstart

Point it at a folder. FinalPass first discovers logical assets, so one
interleaved multichannel file is one asset and one valid split-mono family is
also one asset. It then classifies those logical assets by role (PM, DX, MX,
FX, M&E, optional), groups them by episode/reel/fallback identifier, selects
the correct presentation for the chosen spec, runs the loudness pass on each
group, auto-runs the null pass when a group has a reconstructable stem set,
and auto-runs the M&E bleed heuristic when a group has both `dx` and `me`.

```
uv run python examples/_generate.py          # synthesize a two-episode delivery
uv run finalpass all examples/delivery_two_episodes --spec ebu_r128
```

With a dialog-anchored spec (`netflix_stereo`, `netflix_51`), the classified
DX file is automatically measured against `dialog_lufs` — no `--dx` flag.

Auto-null stem selection is intentionally narrow:
- prefer `dx + mx + fx`
- fallback to `dx + me`
- otherwise the group records `Null: SKIPPED — insufficient_stems_for_auto_null`

Auto-M&E runs only on `dx + me`. If either role is missing, the group records
`M&E: SKIPPED — missing_dx_or_me`.

Use standalone `finalpass null` for any unusual or manual stem combination.
Use standalone `finalpass me` when you want to tune the speech-band, gate, or
threshold settings directly.

Schema versions:
- `finalpass loudness` emits `schema_version: 2`
- `finalpass null` emits `schema_version: 2`
- `finalpass me` emits `schema_version: 2`
- `finalpass all` emits `schema_version: 6` with `groups[].assets[]`,
  `groups[].null_test`, `groups[].me_check`, `unclassified[]`,
  `discovery_errors[]`, `command: "all"`, `folder`, and honest split-mono
  provenance on measured file reports

### Filename conventions

The classifier looks for boundary-delimited role tokens in the filename stem.
Tokens are case-insensitive and must be bracketed by start-of-string or one
of `_`, `-`, `.`.

| Role | Accepted tokens |
|---|---|
| `pm` (printmaster) | `PM`, `PRINTMASTER`, `MIX`, `FINAL` |
| `dx` (dialog)      | `DX`, `DIA`, `DIALOG`, `DIALOGUE` |
| `mx` (music)       | `MX`, `MUS`, `MUSIC` |
| `fx` (effects)     | `FX`, `SFX`, `EFFECTS` |
| `me` (M&E)         | `ME`, `M&E`, `M_AND_E`, `MANDE` |
| `opt` (optional)   | `OPT`, `OPTIONAL`, `NARR`, `NARRATION` |

Example: `SHOW_S01E03_PM_STEREO.wav` → role `pm`, group `S01E03`, channel hint
`stereo`. `SHOW_S01E04_DX_STEREO.wav` → role `dx`, same group. A filename
that matches two role patterns fails with `AmbiguousClassificationError` —
rename it rather than letting the tool guess.

Group id comes from the first of these that matches the stem: `S##E##`,
`EP##`, `R##`. If none match, the group id is the stem with role + channel
tokens stripped — so `MYSHOW_PM_STEREO.wav` and `MYSHOW_DX_STEREO.wav` group
together under `MYSHOW`.

Override the patterns with `--patterns path/to/patterns.yaml` (same top-level
keys as [_bundled_patterns.yaml](src/finalpass/_bundled_patterns.yaml)).

## Scope — v0.1 release candidate

**Current:** WAV/BWF input, any PCM bit depth, any sample rate (homogeneous across a single asset or checked operation). Mono, stereo, 5.1, 7.1 (SMPTE channel order). Interleaved assets plus canonical split stereo / 5.1 / 7.1 logical assets. Loudness (ITU-R BS.1770-4, 4× oversampled true peak, LRA per EBU Tech 3342), standalone stem-sum null, standalone M&E dialogue-bleed, and `all` with logical-asset selection plus conservative auto-null/auto-M&E. Bundled spec presets plus user-overridable YAML. JSON output + self-contained HTML report + conditional timed-marker AAF export + terminal summary.

**Out (non-goals for v0.1):** Atmos/ADM BWF. MXF audio. DCP audio. Auto time-alignment of misaligned stems. Speech recognition, transcription, diarization, or ML/VAD. Dolby-grade dialog gating. Watch folders. Network/cloud. GUI. Per-platform certification — FinalPass measures, it does not bless.

## Bundled specs (Phase 1)

- `atsc_a85` — stereo, −24 LKFS ±2, TP −2 dBTP, LRA ≤ 18 (ATSC A/85 content-exchange target; current official version is A/85:2013 with Corrigendum No. 1, approved 2021; LRA cap is a FinalPass convention).
- `ebu_r128` — stereo, −23 LUFS ±0.5 (R128 pre-produced), TP −1 dBTP, LRA ≤ 18 (convention).
- `netflix_stereo` — stereo, −27 LKFS ±2, TP −2 dBTP, dialog-anchored, LRA ≤ 18.
- `netflix_51` — 5.1, same targets, dialog-anchored.
- `streaming_-14` — stereo, −14 LUFS ±1, TP −1 dBTP, LRA ≤ 18. This is a streaming *normalization* target (Spotify / YouTube / Tidal), not an official delivery spec.

Values are current best-public-knowledge targets; they are not official platform certifications. Each YAML cites its source URL. Verify against your current delivery paperwork.

## Channel order

SMPTE throughout: `L R C LFE Ls Rs [Lss Rss]`. FinalPass does not silently remap; if your file is in a different order, flag and relabel before measuring.

## Non-goals — ever

No network calls. No telemetry. No cloud. No LLM API calls. No DAW functionality — FinalPass analyzes and reports, it does not edit audio. No reimplementation of proprietary measurement algorithms behind closed specs (e.g. Dolby Dialogue Intelligence); where a spec depends on one, FinalPass measures an honest substitute and says so.

## License

MIT. See [LICENSE](LICENSE).
