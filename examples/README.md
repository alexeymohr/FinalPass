# Examples

FinalPass ships a deterministic two-episode example delivery for local checks,
demo runs, and CI smoke coverage.

## What it contains

`examples/_generate.py` synthesizes `examples/delivery_two_episodes/` with two
groups:

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

- delivery: `examples/delivery_two_episodes/`

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

All commands below assume you already ran `uv sync` and generated the delivery.

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

For `null`, `me`, and the integrated `all` run, `markers.aaf` is written only
when the report contains exportable timed flags. The shipped example delivery
is designed so the failing E04 case produces those timed flags.
