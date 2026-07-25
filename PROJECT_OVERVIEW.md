# FinalPass — Project Overview

## Elevator pitch

One command to QC a sound delivery folder. FinalPass classifies what's there, measures loudness against a named platform spec, nulls the stems against the printmaster to catch gross mix errors, checks the M&E for dialogue bleed, audits per-channel integrity, and verifies a delivered 2.0 against a fold-down of its own surround master. It writes a JSON record, a self-contained HTML report, and a conditional AAF of timed markers you can drop onto a DAW marker track to jump straight to each flagged region.

## Why it exists

Delivery QC for post sound is currently a patchwork: a loudness plugin in one DAW, a manual null test nobody bothers with, a last-minute listen-through for M&E bleed. Deliveries still get bounced and it still costs money. There is no single open tool that treats delivery QC as one job.

FinalPass is that tool, CLI-first so it can be automated, scripted, and trusted. The core is MIT-licensed.

## What it does

**Loudness.** ITU-R BS.1770-4 integrated LUFS, true peak (4× oversampled), LRA, short-term max, momentary max. Checked against a named spec preset (Netflix, ATSC A/85, EBU R128, streaming −14, and others). Dialog-anchored specs measure the DX stem separately.

**Stem-sum null.** Sums the supplied stems sample-accurately, subtracts from the printmaster, computes windowed residual RMS, flags windows above threshold with timecode. Catches polarity flips, missing content, wrong-level prints. Does not expect bit-identity — bus limiting on the printmaster makes that unrealistic — so it catches gross errors, not mastering noise.

**M&E dialogue bleed.** Normalized cross-correlation and band-limited coherence (200 Hz – 4 kHz) between M&E and DX, windowed, flagging regions where dialogue is leaking into the M&E.

**Channel integrity.** Per-channel diagnostics on one logical asset: silent legs, duplicated channels, polarity inversion between a pair, broadband content sitting in the LFE slot, and left/right imbalance notes. These findings describe a channel, not a moment, so they are reported but never exported as markers.

**Downmix consistency.** Derives a Lo/Ro-style fold-down from a 5.1/7.1 master and compares it against the delivered 2.0: integrated-loudness delta, windowed similarity between the mono sums, and the delivered stereo's own mono compatibility. It is a plain in-phase fold-down, not an Lt/Rt matrix encode, and no decoder is emulated.

## Working shapes

**Logical assets.** A deliverable is either an interleaved WAV/BWF or a split-mono family (`PM.L.wav`, `PM.R.wav`, `PM.C.wav`, …) assembled and treated as one asset. Split-mono is the dominant real-world delivery shape and is supported natively in every command — no external pre-interleave step.

**Groups.** Assets sharing one episode/reel identifier are checked together and summarized together, so `finalpass all` reports per episode rather than per file.

**Timecode.** The user declares the frame rate; FinalPass never infers it from metadata. All common rates from 23.976 to 120 are supported, with SMPTE ST 12-1 drop-frame counting available at 29.97 and 59.94 via `--drop-frame` (semicolon `HH:MM:SS;FF` notation). When a BWF `bext` time reference is present, sample→timecode is anchored to that embedded start.

**Prep folders.** The wizard can create and scan a `FinalPass Prep/` structure to narrow a messy delivery folder down to the files actually under QC. This scopes input only — it never changes analysis math or output schemas.

## Output

For any normal run, FinalPass writes to `./finalpass-report/` unless `--out` says otherwise. Artifact names use the `report.*` / `markers.aaf` shape when no program name can be inferred; when sources carry a show/episode pattern, the artifacts are prefixed, for example `show-s01e03-report.json`:

- `report.json` or `<program>-report.json` — full machine-readable record, the stable contract for downstream tooling.
- `report.html` or `<program>-report.html` — self-contained single file: pass/fail summary, loudness tables, channel findings, and flagged-region lanes.
- `markers.aaf` or `<program>-markers.aaf` — written only when the run contains exportable timed flags.

AAF export stays deliberately strict: only persisted timed flags are exported. That means `null`, `me`, `downmix`, and timed true-peak-over regions. Integrated/LRA/dialog loudness failures, channel-integrity findings, group-level errors, and skipped checks stay JSON/HTML-only, because none of them have an honest timeline position.

Exit codes: `0` all pass, `1` failures found, `2` tool error.

## Scope

**In:** WAV/BWF input, any PCM bit depth, any sample rate (homogeneous where an operation depends on it). Mono, stereo, 5.1, 7.1, plus split-mono 5.0 sources padded with a silent LFE for 5.1 analysis while preserving five-leg provenance in reports. Interleaved and split-mono logical assets. Bundled spec presets plus user-overridable YAML. JSON + HTML artifacts and conditional timed-marker AAF export. Python 3.11+.

**Out (explicit non-goals):** Atmos/ADM BWF. MXF audio. DCP audio. Auto time-alignment of misaligned stems. Dolby-grade dialog gating. Watch folders. Network/cloud. GUI. Per-platform certification — FinalPass measures, it does not bless.

These non-goals are in the README so users don't file issues for them.

## Non-goals — ever

- Re-implementing proprietary measurement algorithms behind closed specs. Where a spec depends on something like Dolby Dialogue Intelligence, FinalPass measures an honest substitute and says so plainly.
- Network/cloud/telemetry of any kind. This is a local tool, run on trusted machines with unreleased content.
- Becoming a DAW. FinalPass analyzes and reports; it does not edit audio.

## How it got here

Each phase ended with a CLI the user could actually run and verify.

- **Phase 1 — Loudness pass.** Project scaffold, spec system, `finalpass loudness` and `finalpass specs`, JSON output.
- **Phase 2 — Classifier + `all`.** Filename pattern classifier; `finalpass all <folder>` running loudness across classified files.
- **Phase 3 — Stem-sum null.** `finalpass null` and integration into `all`.
- **Phase 4 — M&E dialogue check.** `finalpass me` and integration into `all`.
- **Phase 5 — HTML report.** Self-contained single-file rendering from the persisted report model.
- **Phase 6 — AAF marker export.** One AAF per run, a single marker track, timed flags only.
- **Phase 7 — Polish.** CI, examples smoke flow, changelog and release checklist, terminal-output cleanup, `finalpass --version`.
- **Split-mono series (SM-1 – SM-6).** Native split-mono ingest, standalone and folder-level support, the interactive wizard, and prep-folder curation.
- **Phase 8 — Timecode, channels, downmix.** SMPTE ST 12-1 drop-frame; `finalpass channels`; `finalpass downmix`; both new checks wired into `all` and the wizard.

## What lives outside this repo

A future macOS app (not in scope here, not MIT) would build on this core and add: full AAF session round-trip rather than markers only, a maintained spec library with platform drift tracking, watch folders, ticket integrations, project history and trend dashboards, ADM/MXF/IMF/DCP coverage, and signed PDF pass certificates. None of that belongs in the OSS core; the OSS core stays honestly complete for its scope.
