# Examples

FinalPass ships deterministic interleaved and split-mono example deliveries for
local checks, demo runs, and CI smoke coverage.

## What it contains

`examples/_generate.py` synthesizes:

- `examples/delivery_two_episodes/` — interleaved two-episode delivery
- `examples/delivery_split_assets/` — split-mono delivery with clean 5.1
  assets, a mixed-presentation group, and one intentionally incomplete family

The interleaved delivery contains two groups:

- `S01E03` — clean case that passes loudness, null, and M&E
- `S01E04` — intentionally failing case:
  - hot printmaster for loudness failure
  - null defect for timed `NULL` flags
  - dialogue bleed in M&E for timed `ME` flags

The generated delivery is deterministic and disposable. Re-running the script
recreates the same audio.

## Generate the delivery

From the repo root:

```bash
uv run python examples/_generate.py
```

Artifacts land in:

- interleaved delivery: `examples/delivery_two_episodes/`
- split-mono delivery: `examples/delivery_split_assets/`

## One-command smoke flow

The smoke helper regenerates the delivery, then runs a standalone failing
`me` command plus the integrated failing `all` command.

```bash
uv run python examples/_smoke.py
```

Smoke artifacts land in:

- `examples/out/me_failing/`
- `examples/out/all/`

Each of those directories should contain:

- `report.json`
- `report.html`
- `markers.aaf`

`examples/out/` is gitignored.

## Representative manual commands

All commands below assume you already ran `uv sync` and generated the example
deliveries.

### `loudness`

```bash
uv run finalpass loudness \
  examples/delivery_two_episodes/SHOW_S01E03_PM_STEREO.wav \
  --spec ebu_r128 \
  --out examples/out/loudness
```

### `null`

```bash
uv run finalpass null \
  examples/delivery_two_episodes/SHOW_S01E04_PM_STEREO.wav \
  examples/delivery_two_episodes/SHOW_S01E04_DX_STEREO.wav \
  examples/delivery_two_episodes/SHOW_S01E04_MX_STEREO.wav \
  examples/delivery_two_episodes/SHOW_S01E04_FX_STEREO.wav \
  --out examples/out/null
```

### `me`

```bash
uv run finalpass me \
  examples/delivery_two_episodes/SHOW_S01E04_ME_STEREO.wav \
  --dx examples/delivery_two_episodes/SHOW_S01E04_DX_STEREO.wav \
  --out examples/out/me
```

### `all`

```bash
uv run finalpass all \
  examples/delivery_two_episodes \
  --spec ebu_r128 \
  --out examples/out/all
```

## Split-mono seed-path examples

These use a single family member as the seed path. FinalPass resolves the rest
of the canonical split family automatically.

### split `loudness`

```bash
uv run finalpass loudness \
  examples/delivery_split_assets/SHOW_S01E03_Comp_5.1.L.wav \
  --spec netflix_51 \
  --out examples/out/split_loudness
```

### split `null`

```bash
uv run finalpass null \
  examples/delivery_split_assets/SHOW_S01E03_Comp_5.1.L.wav \
  examples/delivery_split_assets/SHOW_S01E03_DX_5.1.L.wav \
  examples/delivery_split_assets/SHOW_S01E03_MX_5.1.L.wav \
  examples/delivery_split_assets/SHOW_S01E03_FX_5.1.L.wav \
  --out examples/out/split_null
```

### split `me`

```bash
uv run finalpass me \
  examples/delivery_split_assets/SHOW_S01E03_ME_5.1.L.wav \
  --dx examples/delivery_split_assets/SHOW_S01E03_DX_5.1.L.wav \
  --out examples/out/split_me
```

### split-folder `all`

```bash
uv run finalpass all \
  examples/delivery_split_assets \
  --spec netflix_51 \
  --out examples/out/split_all
```

### split-folder `wizard`

```bash
uv run finalpass wizard \
  examples/delivery_split_assets \
  --out examples/out/split_wizard
```

## Prep-folder walkthrough

For a messy real-world delivery, the preferred interactive path is:

```bash
uv run finalpass wizard /path/to/delivery
```

Then:

1. Choose `Create FinalPass prep folders`.
2. Move or copy the stems you want analyzed into `FinalPass Prep/`.
3. Leave anything irrelevant in the parent folder, or put explicit non-analysis material in `FinalPass Prep/Ignore/`.
4. Re-run the wizard. If the prep layout already exists, FinalPass auto-resumes prep mode.
5. Run `all` or one of the standalone jobs from the curated prep buckets.

In SM-6 prep mode, FinalPass scans only files placed directly inside populated non-`Ignore` buckets. It does not recurse deeper inside those buckets yet, and it ignores files outside `FinalPass Prep/`.

For `null`, `me`, and the integrated `all` run, `markers.aaf` is written only
when the report contains exportable timed flags. The shipped example delivery
is designed so the failing interleaved E04 case produces those timed flags.
