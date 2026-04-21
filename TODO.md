# TODO — Deferred to later phases

Features outside the current phase's spec go here, not into the code.

## Phase 2 — File classifier + `all`
- Filename pattern classifier (PM, DX, MX, FX, ME stems).
- `finalpass all <folder>` — runs loudness pass across classified files.
- Summary text output.

## Phase 3 — Stem sum null
- `finalpass null <printmaster> <stem>...`
- Windowed residual RMS, flagged regions with TC.
- Integration into `all`.

## Phase 4 — M&E dialogue check
- `finalpass me <me_file> --dx <dx_file>`
- Normalized cross-correlation + 200 Hz – 4 kHz band-limited coherence.

## Phase 5 — HTML report
- `report.html` — jinja2 template, inline SVG timelines, self-contained single file.

## Phase 6 — AAF marker export
- `markers.aaf` — pyaaf2 CompositionMob with DescriptiveMarkers on a marker track.
- Framerate-correct; comment format `[LOUDNESS|NULL|ME] <metric> <value> — <detail>`.

## Phase 7 — Polish
- `examples/` — fixture folders with reference reports.
- CI.
- Tagged v0.1 release.
- `rich` table path wrapping: long paths wrap across lines in the loudness
  table. Polish in Phase 7 (truncate middle, or drop into a separate header
  line).

## Phase 2 — design decisions

- **Open question #1 (Report model)** resolved **separate models**, not unified.
  - Phase 1 (`finalpass loudness`) emits `Report` at `schema_version: 1` with
    `summary: Summary` (no `groups_*` fields).
  - Phase 2 (`finalpass all`) emits `AllReport` at `schema_version: 3` with
    `groups[]`, `unclassified[]`, `command`, `folder`, and `summary: AllSummary`
    (which has `groups_total/passed/failed` on top of the Phase 1 totals).
  - `FileReport`, `Measurements`, `CheckResult` are shared between both reports.
  - **Why not unified**: PHASE_2_SPEC.md says "Phase 1's schema_version: 1
    output from the `loudness` subcommand stays at 1 — don't retrofit it".
    A unified model with optional `groups`/`unclassified`/`groups_*` fields
    would either force those keys into Phase 1 JSON as `null`s (retrofit) or
    force serialization tricks like `exclude_none=True` — which would also
    drop `measured: null` on skipped checks and break the Phase 1 golden.
    Separate classes make the version contract obvious and keep the Phase 1
    golden untouched.

## Known fragilities
- **Private API use: `pyloudnorm.Meter._filters`.** `finalpass.loudness._k_weight`
  reaches into pyloudnorm's private filter dict to reuse its K-weighting
  stages for short-term and momentary max computations. This is stable for
  pyloudnorm 0.2.x but is not part of any public contract.
  **Fallback:** implement BS.1770 K-weighting directly — pre-filter (shelving
  high-boost) + RLB high-pass as two biquads via `scipy.signal.lfilter`, with
  bilinear-transformed coefficients per the sample rate. Triggered by any test
  failure after a pyloudnorm upgrade.

## Dependency pin review
- **`pydantic<2.13`** (currently resolving to 2.12.5): pinned on 2026-04-20
  because 2.13.x was released within the 7-day supply-chain hold window.
  Review after **2026-04-27** — drop the upper bound if 2.13.x has aged in
  cleanly and nothing newer is fresh.
- **`packaging<26.1`** (currently resolving to 26.0): pinned on 2026-04-20
  because 26.1 was <7 days old. Review after **2026-04-21** — drop the upper
  bound if 26.1 is fine.
