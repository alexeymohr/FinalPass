# FinalPass — Project Overview

## Elevator pitch

One command to QC a sound delivery folder. FinalPass classifies what's there, measures loudness against a named platform spec, nulls the stems against the printmaster to catch gross mix errors, and checks the M&E for dialogue bleed. The current v0.1 release candidate writes a JSON record, a self-contained HTML report, and a conditional AAF of timed markers you can drop onto a Pro Tools marker track and jump directly to flagged null/M&E or true-peak-over regions.

## Why it exists

Delivery QC for post sound is currently a patchwork: a loudness plugin in one DAW, a manual null test nobody bothers with, a last-minute listen-through for M&E bleed. Deliveries still get bounced and it still costs money. There is no single open tool that treats delivery QC as one job.

FinalPass is that tool, CLI-first so it can be automated, scripted, and trusted. The core is MIT-licensed and at release-candidate polish for its v0.1 scope.

## What it does (v0.1)

**Pass 1 — Loudness.** ITU-R BS.1770-4 integrated LUFS, true peak (4× oversampled), LRA, short-term max, momentary max. Checked against a named spec preset (Netflix, ATSC A/85, EBU R128, streaming −14, Apple Podcasts, etc.). Dialog-anchored specs measure the Dx stem separately.

**Pass 2 — Stem sum null.** Sums the supplied stems sample-accurately, subtracts from the printmaster, computes windowed residual RMS, flags windows above threshold with timecode. Catches polarity flips, missing content, wrong-level prints. Does not expect bit-identity (bus limiting on the PM makes that unrealistic) — catches gross errors, not mastering noise.

**Pass 3 — M&E dialogue bleed.** Normalized cross-correlation and band-limited coherence (200 Hz – 4 kHz) between M&E and Dx, windowed, flags regions where dialogue is leaking into the M&E.

## Output

For any normal run, FinalPass writes to `./finalpass-report/`. Artifact names use the historical `report.*` / `markers.aaf` shape when no program name can be inferred; when source stems carry a show/episode pattern, FinalPass prefixes the artifacts, for example `show-s01e03-report.json`:

- `report.json` or `<program>-report.json` — full machine-readable record, the stable contract for downstream tooling.
- `report.html` or `<program>-report.html` — self-contained single file, pass/fail summary, loudness tables, and flagged-region lanes for null, M&E, and timed true-peak overs.
- `markers.aaf` or `<program>-markers.aaf` — written only when the run contains exportable timed `null` / `me` flags or true-peak-over regions.
- `summary.txt` — not a current output artifact.

AAF stays intentionally strict in Phase 6: only persisted timed flags export. That includes `null`, `me`, and timed true-peak-over regions. Integrated/LRA/dialog loudness failures, group-level errors, and skipped checks remain JSON/HTML-only because they have no honest timeline positions.

Exit codes: `0` all pass, `1` failures found, `2` tool error.

## Scope — v0.1

**In:** WAV/BWF input, any PCM bit depth, any sample rate (homogeneous across a run). Mono, stereo, 5.1, 7.1, plus split-mono 5.0 sources padded with silent LFE for 5.1 analysis while preserving five-leg provenance in reports. The three passes. Bundled spec presets plus user-overridable YAML. JSON + HTML output artifacts, plus conditional timed-marker AAF export. Python 3.11+.

**Out (explicit non-goals for v0.1):** Atmos/ADM BWF. MXF audio. DCP audio. Auto time-alignment of misaligned stems. Dolby-grade dialog gating. Watch folders. Network/cloud. GUI. Per-platform certification (we measure, we don't bless).

These non-goals are in the README so users don't file issues for them.

## Non-goals — ever

- Re-implementing proprietary measurement algorithms behind closed specs. Where a spec depends on e.g. Dolby Dialogue Intelligence, FinalPass measures an honest substitute and says so.
- Network/cloud/telemetry of any kind. This is a local tool, run on trusted machines with unreleased content.
- Becoming a DAW. FinalPass analyzes and reports; it does not edit audio.

## Phased roadmap

Each phase ends with a CLI the user can actually run and verify.

- **Phase 1 — Loudness pass.** Project scaffold, spec system, `finalpass loudness` and `finalpass specs` subcommands, JSON output. Fully usable standalone for spot-checking a file against a spec.
- **Phase 2 — File classifier + `all` command (loudness only).** Filename pattern classifier. `finalpass all <folder>` runs the loudness pass across classified files. Summary text output.
- **Phase 3 — Stem sum null pass.** `finalpass null` subcommand and integration into `all`.
- **Phase 4 — M&E dialogue check.** `finalpass me` subcommand and integration into `all`.
- **Phase 5 — HTML report.** Jinja2 template, inline SVG timelines, self-contained single file.
- **Phase 6 — AAF marker export.** Complete. One AAF per CLI run, a single marker track, and persisted timed `null` / `me` / true-peak-over flags exported as timed comment markers.
- **Phase 7 — Polish.** Complete for the release candidate: CI, examples smoke flow, changelog/checklist, terminal-output cleanup, and `finalpass --version`. The local tag step remains manual after human acceptance.

## What lives outside this repo

A future macOS app (not in scope here, not MIT) will build on this core and add: full AAF session round-trip (not just markers), maintained spec library with platform drift tracking, watch folders, Slack/ticket integrations, project history and trend dashboards, ADM/MXF/IMF/DCP format coverage, signed PDF pass certificates. None of that belongs in the OSS core; the OSS core stays honestly complete for its scope.
