# FinalPass

Delivery QC for post-production sound: a local CLI that gates a delivery by
measuring loudness and running stem-sum null, M&E dialogue-bleed, and (Phase 8)
channel-integrity and downmix-consistency checks against named specs.

## Language

### Assets

**Delivery**:
The folder of audio deliverables under QC for one program.

**Logical asset**:
One deliverable audio entity — an interleaved WAV/BWF, or a split-mono family
assembled and treated as one.
_Avoid_: file (for the assembled entity), track

**Split-mono family**:
Sibling mono legs (L, R, C, LFE, Ls, Rs, …) that assemble into one logical
asset.
_Avoid_: split files, mono set

**Role**:
The classified function of a logical asset within its group: printmaster, DX,
MX, FX, M&E, or optional stem.

**Printmaster**:
The full program mix deliverable; the reference other checks in its group
measure against.
_Avoid_: final mix, full mix

**Stem**:
A submix deliverable (DX, MX, FX, M&E) that participates in null and bleed
checks.

**Group**:
The logical assets sharing one episode/reel identifier, checked together and
summarized together.

### Checks

**Spec preset**:
A named set of loudness delivery targets (integrated, dialog, true peak, LRA)
for one channel layout.

**Spec family**:
Bundled spec presets spanning multiple layouts under one delivery standard
(e.g. ATSC A/85, Netflix Original).

**Program window**:
The sample span a comparison check actually analyzes: bounded by the
whole-hour timecode boundary when BWF time references allow, with MOS tails
cropped and shorter inputs padded.
_Avoid_: comparison window

**Flagged region**:
A timecoded span where a check exceeded its threshold; the only source of AAF
markers.

### Run configuration

**Timecode mode**:
The frame rate plus drop-frame/non-drop counting choice for a run. Always
declared by the user; never inferred from media.
_Avoid_: fps (when the counting mode is also meant)

**Tunables**:
The per-check analysis knobs — windows, hops, gates, thresholds — a check runs
with. Each check owns its tunables and their defaults.
_Avoid_: options, parameters, config

**Run settings**:
The bundle a folder-level run executes with: analysis scope plus each check's
tunables. Excludes timecode mode, which travels separately.
_Avoid_: run config
