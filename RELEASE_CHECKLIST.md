# Release Checklist

Use this after human acceptance of a release candidate. Do not run the tag step
automatically during normal implementation work.

## Local verification

From the repo root:

```bash
uv sync --dev --frozen
uv run pytest
uv run python examples/_smoke.py
uv run finalpass --version
```

## Artifact spot-check

Confirm the smoke flow wrote these disposable outputs (artifact stems are
derived from the example program names):

- `examples/out/me_failing/show-s01e04-report.json`
- `examples/out/me_failing/show-s01e04-report.html`
- `examples/out/me_failing/show-s01e04-markers.aaf`
- `examples/out/channels_failing/show-s01e04-report.json`
- `examples/out/channels_failing/show-s01e04-report.html`
- `examples/out/downmix_failing/show-s01e04-report.json`
- `examples/out/downmix_failing/show-s01e04-report.html`
- `examples/out/downmix_failing/show-s01e04-markers.aaf`
- `examples/out/all/show-s01e03-s01e04-report.json`
- `examples/out/all/show-s01e03-s01e04-report.html`
- `examples/out/all/show-s01e03-s01e04-markers.aaf`

The `channels_failing` run writes no AAF by design: channel-integrity findings
describe a channel, not a moment on the timeline.

## Local DAW marker check

Import:

- `examples/out/all/show-s01e03-s01e04-markers.aaf`

Confirm:

- the DAW imports a marker track successfully
- timed markers land around the known failing E04 regions
- marker names read clearly enough to act on

At least once per release, also check a drop-frame export end to end: a
29.97 or 59.94 run with `--drop-frame` should produce semicolon timecode in the
report and an AAF whose timecode track is flagged drop, so marker positions
agree with a drop-frame session rather than drifting against it.

## Tag and release

Run only after the verification above is clean and human acceptance is explicit.
Replace the version in each command.

```bash
git tag -a v0.2.0 -m "FinalPass v0.2.0"
```

Confirm the tag contents before pushing it:

```bash
git show v0.2.0 --stat
```

Push, then publish the release notes from the matching `CHANGELOG.md` section:

```bash
git push origin main --follow-tags
```

## Explicit non-steps

- do not publish to PyPI — FinalPass is run from source
- do not push a tag before `finalpass --version` agrees with `pyproject.toml`,
  `src/finalpass/__init__.py`, and the `finalpass_version` field in a freshly
  written `report.json`
