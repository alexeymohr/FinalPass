# TODO — Deferred beyond the v0.1 release candidate

Features outside the current phase's spec go here, not into the code.

## Manual release step after acceptance
- Create the local `v0.1.0` tag only after human sign-off. The exact command
  lives in `RELEASE_CHECKLIST.md`; do not automate it in normal phase work.

## Split-mono follow-up

- **SM-1 + SM-2 landed the ingest foundation plus standalone wiring.**
  - `src/finalpass/assets.py` now discovers logical assets, validates canonical
    stereo / 5.1 / 7.1 split-mono families, assembles valid families in memory,
    and resolves seed-member paths to full logical assets.
  - standalone `loudness` / `null` / `me` now resolve either interleaved files
    or seed-member split-mono paths and persist standalone provenance.
- **SM-3 landed the logical-asset `all` redesign.**
  - `finalpass all` now discovers folder contents as logical assets rather than
    raw files, keeps mixed presentations inside one editorial group, selects
    target-layout assets by actual layout, and persists `groups[].assets[]`
    plus v6 measured-file provenance.
  - explicit split-looking incomplete families now surface as discovery errors
    on the `all` path instead of silently masquerading as mono assets.
- **SM-4 landed the minimal guided wizard / job flow.**
  - `finalpass wizard` now starts folder-first, presents logical assets instead
    of raw mono legs, and can run `all`, standalone `loudness`, standalone
    `null`, and standalone `me` through numbered menus while reusing the same
    internal command runners and artifact-writing path as the direct CLI.
- **SM-5 landed the split-mono UX / execution cleanup polish.**
  - `src/finalpass/jobs.py` is now the authoritative shared execution seam for
    direct commands and the wizard; the old duplicate runner bodies are gone
    from `cli.py`.
  - user-facing split-mono labels are now derived from existing provenance in
    shared presentation helpers and applied across terminal, HTML, and wizard
    surfaces without changing report schemas.
  - maintainer docs now call out the shared-venv/worktree `PYTHONPATH=src`
    verification caveat.
- **SM-6 landed prep-folder guided filtering in the wizard.**
  - `finalpass wizard` can now create and auto-resume a fixed
    `FinalPass Prep/` bucket layout for messy deliveries.
  - prep mode scans only populated non-`Ignore` buckets and ignores files
    outside the prep root without changing any report schema or analysis math.
  - prep bucket hints feed existing logical-asset discovery and job execution
    seams; direct `all` / standalone CLI commands remain unchanged.
- **Still deferred beyond SM-6:**
  - saved presets / persistent wizard state
  - advanced null / M&E tuning inside the wizard
  - Finder tags / color-label integration for prep
  - customizable prep-bucket sets
  - any CLI surface redesign beyond the current commands

## Phase 2 — design decisions

- **Open question #1 (Report model)** resolved **separate models**, not unified.
  - Phase 1 (`finalpass loudness`) originally emitted `Report` at
    `schema_version: 1` with
    `summary: Summary` (no `groups_*` fields).
  - The integrated folder path now emits `AllReport` at `schema_version: 5`
    with `groups[]`, `groups[].null_test`, `groups[].me_check`, `unclassified[]`, `command`,
    `folder`, and `summary: AllSummary` (which has `groups_total/passed/failed`
    on top of the Phase 1 totals).
  - `FileReport`, `Measurements`, `CheckResult` are shared between both reports.
  - **Why not unified**: Phase 2 explicitly forbade retrofitting the original
    Phase 1 `loudness` shape.
    A unified model with optional `groups`/`unclassified`/`groups_*` fields
    would either force those keys into Phase 1 JSON as `null`s (retrofit) or
    force serialization tricks like `exclude_none=True` — which would also
    drop `measured: null` on skipped checks and break the Phase 1 golden.
    Separate classes made the version contract obvious and kept the original
    Phase 1 golden untouched; SM-2 later bumped only the standalone schemas.

## Phase 3 — design decisions

- **Standalone null got its own schema.**
  - `finalpass null` emitted `NullReport` at `schema_version: 1` through Phase 7.
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
  - `finalpass me` emitted `MEReport` at `schema_version: 1` through Phase 7.
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

## Phase 5 — design decisions

- **HTML renders from existing persisted models only.**
  - `report.py` consumes already-built report models and packaged Jinja templates.
  - The renderer never reopens audio files, reruns analysis, or expands JSON schemas.
- **Timelines are flagged-region lanes, not continuous traces.**
  - Phase 5 uses the persisted merged `flags[]` only.
  - Empty-state presentation stays honest: no synthetic per-window curves.
- **Normal runs now write the core report artifacts together.**
  - `report.json` remains the stable machine contract.
  - `report.html` is rendered in-memory first, then the JSON/HTML pair is written.
  - `--json-only` still writes no files.

## Split-mono Phase SM-2 — design decisions

- **Standalone provenance bumped the standalone schemas only.**
  - `finalpass loudness`, `finalpass null`, and `finalpass me` now emit
    `schema_version: 2`.
  - `finalpass all` stays at `schema_version: 5` and does not gain any of the
    standalone split-mono provenance fields in SM-2.
- **One standalone ingest seam builds on SM-1.**
  - `src/finalpass/standalone_ingest.py` resolves interleaved paths and
    split-family seed paths through `assets.py`; the standalone commands do not
    duplicate sibling-family resolution logic.
- **Canonical path stays representative; full source provenance is explicit.**
  - Standalone reports keep `path` as the canonical/representative display path.
  - Split/interleaved truth lives in `source_kind`, `source_paths`,
    `member_legs`, and `presentation_label`.
- **`all` remains intentionally untouched.**
  - Folder-level logical-asset discovery, grouping, and schema redesign are
    still deferred beyond SM-2.

## Split-mono Phase SM-3 — design decisions

- **`all` is now asset-first, not raw-file-first.**
  - Folder discovery runs through `assets.py` plus the new
    `src/finalpass/all_assets.py` seam.
  - Interleaved files and canonical split families become one logical asset
    each; invalid explicit split families surface separately as discovery
    errors.
- **Mixed presentations stay in inventory instead of acting like duplicates.**
  - Duplicate-role failures on `all` now apply only within the chosen target
    layout.
  - Alternate-layout PM/DX/MX/FX/ME assets remain visible in `groups[].assets[]`
    with selection notes and `used_by` provenance.
- **Dialog loudness keeps the earlier fallback, but timed analysis stays strict.**
  - `dialog_lufs` prefers a target-layout DX asset.
  - If none exists, one alternate-layout DX may be used for dialog loudness
    only.
  - Null and M&E still require layout-compatible target assets; alternate DX is
    never reused for timed analysis.

## Phase 6 — design decisions

- **AAF export stays timed-flags-only.**
  - `markers.aaf` exports only persisted timed `null` / `me` `flags[]`.
  - No synthetic markers are created for loudness-only failures, group errors, or skipped checks.
- **One run, one marker track.**
  - Each CLI run writes at most one `markers.aaf`.
  - The exported AAF is a top-level `CompositionMob` with one marker track containing timed comment markers plus minimal timeline guide slots for Pro Tools import.
- **Marker placement uses exact edit-rate math from samples.**
  - Placement comes from persisted sample positions and the shared timecode helper path.
  - The CLI still exposes only `--fps`, so FinalPass formats persisted timecode strings in non-drop notation while placing AAF markers at the exact rational edit rate.

## Future optional artifacts

- `summary.txt` — optional one-page text artifact if it proves useful after HTML and AAF land.

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
- **`packaging` hold review completed on 2026-04-22.**
  - The temporary `packaging<26.1` upper bound was removed after 26.1 aged
    past the 7-day supply-chain window and the release-normalization suite
    stayed green.
