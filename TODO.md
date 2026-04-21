# TODO — Deferred to later phases

Features outside the current phase's spec go here, not into the code.

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
  - The integrated folder path now emits `AllReport` at `schema_version: 5`
    with `groups[]`, `groups[].null_test`, `groups[].me_check`, `unclassified[]`, `command`,
    `folder`, and `summary: AllSummary` (which has `groups_total/passed/failed`
    on top of the Phase 1 totals).
  - `FileReport`, `Measurements`, `CheckResult` are shared between both reports.
  - **Why not unified**: PHASE_2_SPEC.md says "Phase 1's schema_version: 1
    output from the `loudness` subcommand stays at 1 — don't retrofit it".
    A unified model with optional `groups`/`unclassified`/`groups_*` fields
    would either force those keys into Phase 1 JSON as `null`s (retrofit) or
    force serialization tricks like `exclude_none=True` — which would also
    drop `measured: null` on skipped checks and break the Phase 1 golden.
    Separate classes make the version contract obvious and keep the Phase 1
    golden untouched.

## Phase 3 — design decisions

- **Standalone null got its own schema.**
  - `finalpass null` emits `NullReport` at `schema_version: 1`.
  - `finalpass all` bumps to `schema_version: 4` and adds `groups[].null_test`
    rather than retrofitting the Phase 1 `loudness` shape.
- **Auto-null stays conservative.**
  - Preferred auto-selection is `dx + mx + fx`.
  - Fallback is `dx + me`.
  - Everything else records `insufficient_stems_for_auto_null` and skips the
    null check rather than pretending a partial reconstruction is meaningful.
- **Detected offsets fail cleanly; they are not auto-corrected.**
  - The null pass raises `AlignmentError` on evidence of a constant global
    offset instead of silently shifting inputs and continuing.
  - This keeps Phase 3 aligned with the explicit "no auto-alignment" rule and
    preserves the loudness/null distinction for later AAF marker work.

## Phase 4 — design decisions

- **Standalone M&E got its own schema.**
  - `finalpass me` emits `MEReport` at `schema_version: 1`.
  - `finalpass all` bumps to `schema_version: 5` and adds `groups[].me_check`
    without retrofitting the Phase 1 or Phase 3 standalone shapes.
- **Auto-M&E is deliberately narrow.**
  - The integrated path runs only when a group has both `dx` and `me`.
  - Missing either role records `missing_dx_or_me` and skips the M&E check
    rather than guessing around partial material.
- **Phase 4 stays DSP-only.**
  - The M&E heuristic is speech-band filtering + zero-lag correlation +
    coherence, with DX gating and an M&E floor.
  - No ML, speech recognition, transcription, diarization, or auto-alignment
    entered the Phase 4 codepath.
- **Shared marker shape stayed shared.**
  - Phase 4 reuses the Phase 3 `FlaggedRegion` contract and adds `code: "ME"`
    / `metric: "dialog_bleed_score"` instead of inventing a second flag schema.

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
