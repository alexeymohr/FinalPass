# Release Checklist

Use this after human acceptance of the v0.1 release candidate. Do not run the
tag step automatically during normal implementation phases.

## Local verification

From the repo root:

```bash
uv sync --dev --frozen
uv run pytest
uv run python examples/_smoke.py
uv run finalpass --version
```

## Artifact spot-check

Confirm the smoke flow wrote these disposable outputs:

- `examples/out/me_failing/report.json`
- `examples/out/me_failing/report.html`
- `examples/out/me_failing/markers.aaf`
- `examples/out/all/report.json`
- `examples/out/all/report.html`
- `examples/out/all/markers.aaf`

## Local Pro Tools marker smoke check

Import:

- `examples/out/all/markers.aaf`

Confirm:

- Pro Tools imports a marker track successfully
- timed markers land around the known failing E04 regions
- the marker names are readable enough for current v0.1 use

## Local tag command

Run only after the verification above is clean and human acceptance is explicit:

```bash
git tag -a v0.1.0 -m "FinalPass v0.1.0"
```

Optional local confirmation:

```bash
git show v0.1.0 --stat
```

## Explicit non-steps in this phase

- do not publish to PyPI
- do not create a GitHub Release
- do not push tags automatically
