# FinalPass

Delivery QC CLI for post-production sound. Runs loudness, stem-sum null, and M&E dialogue-bleed passes against a named spec; writes JSON, an HTML report, and an AAF of Pro Tools markers.

**Status: Phase 1 (loudness pass only).** The `null`, `me`, HTML report, and AAF export passes land in later phases.

## Install (dev)

```
uv sync
uv run finalpass --help
```

Python 3.11+. WAV/BWF input.

## Commands

```
uv run finalpass specs list
uv run finalpass specs show <name>
uv run finalpass loudness <file>... --spec <name> [--dx <file>] [--out <dir>] [--json-only] [--fps <rate>]
uv run finalpass all <folder> --spec <name> [--out <dir>] [--json-only] [--fps <rate>]
                                            [--patterns <config.yaml>] [--include-unclassified]
```

Exit codes: `0` all pass, `1` failures found, `2` tool error.

### `finalpass all` quickstart

Point it at a folder. FinalPass classifies each WAV by role (PM, DX, MX, FX,
M&E, optional), groups files by episode/reel/fallback identifier, and runs
the loudness pass on each group.

```
uv run python examples/_generate.py          # synthesize a two-episode delivery
uv run finalpass all examples/delivery_two_episodes --spec ebu_r128
```

With a dialog-anchored spec (`netflix_stereo`, `netflix_51`), the classified
DX file is automatically measured against `dialog_lufs` — no `--dx` flag.

Schema: `finalpass all` emits `schema_version: 3` with `groups[]`,
`unclassified[]`, `command: "all"`, `folder`, and per-file
`channel_config_hint`/`channel_config_actual`. `finalpass loudness` emits
`schema_version: 1` (unchanged from Phase 1).

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

## Scope — v0.1

**In:** WAV/BWF input, any PCM bit depth, any sample rate (homogeneous across a run). Mono, stereo, 5.1, 7.1 (SMPTE channel order). Three passes: loudness (ITU-R BS.1770-4, 4× oversampled true peak, LRA per EBU Tech 3342), stem sum null, M&E dialogue bleed. Bundled spec presets plus user-overridable YAML. JSON + HTML + AAF markers + summary.

**Out (non-goals for v0.1):** Atmos/ADM BWF. MXF audio. DCP audio. Auto time-alignment of misaligned stems. Dolby-grade dialog gating. Watch folders. Network/cloud. GUI. Per-platform certification — FinalPass measures, it does not bless.

## Bundled specs (Phase 1)

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
