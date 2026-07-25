# FinalPass Design Limitation: No Native Split-Mono Stem Support

> **Status: resolved.** This is a historical design note describing FinalPass
> before native split-mono support existed. The limitation it analyzes was
> addressed by the SM-1 – SM-6 work and shipped in 0.2.0: split-mono families
> are discovered, validated, and analyzed as first-class logical assets in
> every command, with no external pre-interleave step. The document is kept
> because it records why the ingest layer is shaped the way it is.

## Bottom line

FinalPass v0.1 is built around the assumption that each analyzable asset is
already a single finished WAV/BWF file with its full channel layout in one
container.

That works for:
- mono interleaved files
- stereo interleaved files
- 5.1 interleaved files
- 7.1 interleaved files

It does **not** work natively for the most common real-world post delivery
shape: **split-mono stem families**, where one logical asset is delivered as
multiple mono files such as:

- `PM.L.wav`
- `PM.R.wav`
- `PM.C.wav`
- `PM.LFE.wav`
- `PM.Ls.wav`
- `PM.Rs.wav`

In practical terms: FinalPass does not yet have a true MVP for real delivery
folders, because it cannot currently ingest the dominant stem-delivery format
without an external pre-interleave step.

## Precise design limitation

The limitation is not just "classifier patterns need more aliases." The deeper
design issue is:

1. **Audio ingestion is one-file-in, one-audio-object-out.**
   - `read_wav(path)` reads exactly one filesystem path into one `AudioFile`.
   - There is no abstraction for "one logical asset composed of multiple mono
     channel-leg files."
   - Source: [src/finalpass/audio_io.py](src/finalpass/audio_io.py)

2. **All analysis paths assume already-assembled channel layouts.**
   - Loudness validates one file's `channel_count` against the target spec.
   - Null requires one printmaster file plus one or more stem files, all with a
     single shared `channel_count`, `sample_rate`, and `sample_count`.
   - M&E requires one `me_file` and one `dx_file`, also with a single shared
     `channel_count`, `sample_rate`, and `sample_count`.
   - Sources:
     - [src/finalpass/cli.py](src/finalpass/cli.py)
     - [src/finalpass/null_test.py](src/finalpass/null_test.py)
     - [src/finalpass/me_check.py](src/finalpass/me_check.py)

3. **The folder-level `all` path classifies individual files, not channel-leg
   families.**
   - `scan_folder()` and `classify_file()` operate one filename at a time.
   - Grouping is based on the single file's stem after role/channel-token
     stripping.
   - There is no concept of "these six mono files together are one 5.1 PM."
   - Source: [src/finalpass/classify.py](src/finalpass/classify.py)

4. **Reports also assume one path per analyzable input.**
   - `FileReport.path` is one string.
   - `AnalysisInputFile.path` is one string.
   - Standalone `null` and `me` reports likewise describe each input as one
     path, not a family of paths.
   - Source: [src/finalpass/models.py](src/finalpass/models.py)

So the problem is structural: FinalPass lacks a native representation of
"multichannel asset assembled from split-mono source files."

## Concrete evidence from a real short-film delivery

Test folder:

- `/Users/amohr/programming/FinalPass/stems_to_test/INSTBHA_Stems.F/Audio Files`

Observed delivery shape:

- 5.1 printmaster as six mono files:
  - `Comp 5.1.L.wav`
  - `Comp 5.1.R.wav`
  - `Comp 5.1.C.wav`
  - `Comp 5.1.LFE.wav`
  - `Comp 5.1.Ls.wav`
  - `Comp 5.1.Rs.wav`
- 5.1 split stems as six mono files per role:
  - `DX  5.1.*.wav`
  - `MX  5.1.*.wav`
  - `FX  5.1.*.wav`
- stereo LtRt as split pairs:
  - `Comp LtRt.L.wav`
  - `Comp LtRt.R.wav`
  - `MX  LtRt.L.wav`
  - `MX  LtRt.R.wav`
  - `MX-FX LtRt.L.wav`
  - `MX-FX LtRt.R.wav`

Sample header check:

- every checked file was `48000 Hz`, `1 channel`, `PCM_24`

This matters because FinalPass sees these as **many mono files**, not as:

- one 5.1 PM
- one 5.1 DX
- one 5.1 MX
- one 5.1 FX
- one stereo PM
- etc.

## What breaks today

### 1. `loudness`

`finalpass loudness` expects each input file to already have the target channel
count for the chosen spec.

Example failure mode:

- `netflix_51` expects `6` channels
- `Comp 5.1.L.wav` is `1` channel
- so FinalPass treats it as a mono file that does not match a 5.1 spec

This is enforced in `_validate_channels(...)` in
[src/finalpass/cli.py](src/finalpass/cli.py).

### 2. `null`

`finalpass null <pm> <stems>...` accepts one PM path and one path per stem
input. Each path is read independently with `read_wav(...)`.

That means there is no native way to express:

- PM = six mono files that together form one 5.1 printmaster
- DX = six mono files that together form one 5.1 DX stem
- MX = six mono files that together form one 5.1 MX stem
- FX = six mono files that together form one 5.1 FX stem

Null can technically analyze mono files if all inputs are mono and line up, but
that would only be a per-leg workaround, not a native multichannel stem-family
analysis.

### 3. `me`

`finalpass me <me_file> --dx <dx_file>` has the same structural issue:

- one path for M&E
- one path for DX
- both expected to already be complete mono/stereo/5.1/7.1 files

There is no native way to pass:

- six mono files that together make one 5.1 M&E
- six mono files that together make one 5.1 DX

### 4. `all`

This is the most important blocker for real deliveries.

Current `all` behavior on the short-film folder:

```bash
./.venv/bin/python -m finalpass.cli all \
  '/Users/amohr/programming/FinalPass/stems_to_test/INSTBHA_Stems.F/Audio Files' \
  --spec netflix_51 --json-only
```

Observed result:

- `groups_total = 32`
- `files = []`
- every group failed with `MissingPrintmaster`
- many files were recorded under `unclassified[]`

Why:

1. The current bundled classifier does not recognize `Comp` as a printmaster
   alias; PM tokens are currently `PM`, `PRINTMASTER`, `MIX`, `FINAL`.
   - Source: [src/finalpass/_bundled_patterns.yaml](src/finalpass/_bundled_patterns.yaml)
2. More importantly, even if the classifier learned `Comp -> pm`, each mono leg
   would still be treated as a separate classified file.
3. `all` currently expects at most one file per role inside a group for normal
   operation.
4. There is no pre-analysis assembly step that says "these six mono files are
   actually one 5.1 PM asset."

So even with better aliases, `all` still does not have the native asset model
needed for split-mono stem deliveries.

## Why this blocks MVP

Real-world stem deliveries are routinely provided as:

- split-mono 5.1 / 7.1
- split LtRt pairs
- mixed-format folders containing both multichannel split stems and stereo
  split stems

If FinalPass requires an external manual interleave step before meaningful QC,
then:

- the user has to know the correct channel order up front
- the user has to use another tool before FinalPass
- the folder-level `all` workflow is not actually usable on common deliveries
- the current "one command to QC a delivery folder" promise breaks on routine
  source material

That makes native split-mono handling a product-level gap, not a minor edge
case.

## Current workaround

The only practical workaround today is:

1. externally interleave the mono legs into complete stereo / 5.1 / 7.1 WAVs
2. then run FinalPass on those interleaved files

That is acceptable as a temporary manual escape hatch, but not as the intended
core workflow for a delivery QC tool.

## Planning implications

The planner should treat this as a **native-ingest design problem**, not just a
regex/classifier tweak.

Any real solution will need to answer at least these questions:

1. **Asset model**
   - How does FinalPass represent one logical asset assembled from multiple mono
     files?
   - Is that a new internal model separate from `AudioFile`, or a higher-level
     wrapper above it?

2. **Family discovery**
   - How are mono legs recognized as belonging to the same asset?
   - How are channel-leg tokens recognized (`L`, `R`, `C`, `LFE`, `Ls`, `Rs`,
     etc.)?
   - How are stereo split pairs (`.L` / `.R`) handled?

3. **Assembly and validation**
   - Where does interleave/assembly happen in the pipeline?
   - How are missing legs, duplicate legs, mismatched sample counts, and
     mismatched sample rates handled?
   - How is SMPTE channel order enforced?

4. **Command surface**
   - Should standalone commands accept split-mono families directly?
   - If yes, how is that expressed on the CLI without making the commands
     brittle or ambiguous?

5. **`all` behavior**
   - How should the folder scanner classify and group split-mono families before
     analysis?
   - How should mixed folders containing both 5.1 split mono and LtRt split
     pairs be normalized into one coherent grouped run?

6. **Reporting/provenance**
   - Should reports continue to show one path per logical asset, or should they
     expose the source member files that were assembled?
   - If source families matter for debugging, where should that provenance live?

7. **Backwards compatibility**
   - Interleaved inputs already work. Native split-mono support should extend
     FinalPass without degrading the current interleaved path.

## Recommended framing for the planning phase

This is the planning problem to solve:

> FinalPass currently assumes "one analyzable asset = one WAV/BWF file."
> Real deliveries often use "one analyzable asset = a family of mono files."
> We need a native asset-family ingest layer that can discover, validate,
> assemble, and report split-mono stem families without external interleaving.

That is the key design gap between the current release candidate and a true
real-world MVP.
