# Changelog

## Unreleased

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
