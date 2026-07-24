# Changelog

## Unreleased

### Phase 8T — timecode completion

- SMPTE ST 12-1 drop-frame timecode support for 29.97 and 59.94 via a new
  `--drop-frame` flag on `loudness`, `null`, `me`, and `all`. Drop-frame
  strings use the semicolon convention (`HH:MM:SS;FF`) everywhere a timecode
  appears: terminal tables, JSON, HTML, and AAF marker fields. Default
  remains non-drop; `--drop-frame` with any other rate is a clean validation
  error.
- The whole-hour program-window boundary is found in the selected timecode
  mode: at 29.97 drop-frame, `01:00:00;00` sits at the wall-clock hour
  (107,892 frames), removing the ~3.6 s/hour window offset the non-drop-only
  implementation imposed on drop-frame shows.
- AAF export sets the timecode track's drop-frame flag from the selected
  mode. Marker placement is unchanged (it always used exact rational edit
  rates).
- Frame-rate table gains 119.88 (120000/1001) and 120, non-drop only per
  ST 12-1.
- Schema bumps: `loudness`/`null`/`me` 3 → 4, `all` 8 → 9. Each envelope
  adds `drop_frame: bool` beside `fps`. Non-drop report strings are
  unchanged.
- The 29.97/59.94 stderr note now recommends `--drop-frame` for drop-frame
  shows instead of only describing the drift.

### Internal refactors

- Runner interfaces now take immutable settings values instead of loose
  knob parameters: `NullTunables` / `METunables` (owned by their check
  modules) and `AllSettings` (new `run_settings.py`) carrying folder-run
  scope plus per-check tunables. CLI flags, JSON/HTML output, and defaults
  are unchanged; `fps`/`--drop-frame` remain explicit timecode-mode
  parameters. Added `CONTEXT.md` (domain glossary).
- The two group-processing paths in `jobs.py` (plain spec and bundled-spec
  family) now share one selection/measurement/check/finalize core instead
  of ~160 duplicated lines; the family path keeps its multi-layout
  measurement loop and report ordering. Output is unchanged.
- Flagged-region display (decimal places, unit, terminal column header) and
  loudness target/limit formatting now live once in `presentation.py`; the
  terminal renderer and the HTML templates consume it instead of each
  carrying their own metric conditionals. Terminal and HTML output are
  unchanged.

### Earlier unreleased changes

- Split-mono discovery accepts Pro Tools bounce naming: spelled-out channel
  words (`Left`, `Right`, `Center`/`Centre`, `Left Front`, `Right Front`,
  `Left Surround`, `Right Surround`), commas as token separators, and leading
  track-number prefixes no longer split otherwise-identical families/groups.
- The M&E dialogue-bleed check accepts inputs with different channel layouts
  (for example a mono DX against a stereo M&E); both signals were already
  reduced to the same mono analysis downmix. The null check still requires
  matching channel counts.
- New bundled spec preset `atsc_a85_mono` (joins the ATSC A/85 family).
- Terminal flagged-region tables cap at 20 rows with an explicit
  "showing first N of M" caption; JSON/HTML reports keep the full list.
- Silenced the scipy coherence RuntimeWarning on windows with silent
  segments; undefined coherence bins are now dropped instead of averaged.
- The CLI prints a one-line non-drop timecode note at 29.97/59.94 fps.
- Removed the `pydantic<2.13` supply-chain hold; lock now resolves
  pydantic 2.13.4 / pydantic-core 2.46.4 (both aged past the 7-day window).
- Release checklist artifact names updated to the program-named artifacts the
  smoke flow actually writes.

## v0.1.0

FinalPass v0.1.0 is the first release candidate of the OSS delivery-QC CLI.

### Shipped in v0.1.0

- Phase 1 loudness measurement against named delivery specs
- Phase 2 filename classification and folder-level `all` runs
- Phase 3 standalone stem-sum null plus auto-null in `all`
- Phase 4 standalone M&E dialogue-bleed checks plus auto-M&E in `all`
- Phase 5 self-contained HTML report generation from persisted report models
- Phase 6 conditional AAF marker export for exportable timed `null` / `me` / loudness true-peak flags
- Phase 7 release polish:
  - `finalpass --version`
  - CI workflow
  - deterministic examples generator and smoke flow
  - release checklist and repo/doc cleanup
  - improved terminal readability for long input paths

### Artifact contract

Normal runs write:

- `report.json`
- `report.html`

Conditional artifact:

- `markers.aaf` only when the run contains exportable timed flags

`--json-only` writes no files.

### Important limits still in place

- WAV/BWF only; no Atmos/ADM BWF, MXF, or DCP
- no auto-alignment of misaligned files
- no speech recognition, transcription, diarization, or ML/VAD
- no Dolby Dialogue Intelligence reimplementation
- no GUI, network, telemetry, cloud, or watch-folder behavior
- no synthetic AAF markers for loudness-only failures, group errors, or skipped checks
