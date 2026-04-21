"""Spec loading and validation.

Specs are Pydantic v2 models loaded from YAML. Resolution order:

1. Explicit path (anything that exists as a file).
2. User dir: ``~/.finalpass/specs/<name>.yaml``.
3. Bundled: ``finalpass._bundled_specs/<name>.yaml`` via importlib.resources.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import SpecError

SpecSource = Literal["bundled", "user", "explicit"]

CHANNEL_COUNTS: dict[str, int] = {
    "mono": 1,
    "stereo": 2,
    "5.1": 6,
    "7.1": 8,
}


class Tolerance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: float
    tolerance: float = Field(ge=0.0)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str
    integrated_lufs: Tolerance
    dialog_lufs: Tolerance | None = None
    true_peak_max_dbtp: float = Field(le=0.0)
    lra_max: float | None = Field(default=None, gt=0.0)
    channel_config: Literal["mono", "stereo", "5.1", "7.1"]
    notes: str = ""

    @property
    def expected_channel_count(self) -> int:
        return CHANNEL_COUNTS[self.channel_config]

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v


def _user_specs_dir() -> Path:
    return Path.home() / ".finalpass" / "specs"


def _bundled_resource_path(name: str) -> Path | None:
    try:
        spec_file = resources.files("finalpass._bundled_specs").joinpath(f"{name}.yaml")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    if not spec_file.is_file():
        return None
    with resources.as_file(spec_file) as p:
        return Path(p)


def _load_yaml_as_spec(path: Path) -> Spec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"Spec {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"Spec {path} must be a YAML mapping at the top level.")
    try:
        spec = Spec.model_validate(raw)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        raise SpecError(f"Spec {path} is invalid ({fields}): {exc}") from exc
    if spec.name != path.stem:
        raise SpecError(
            f"Spec {path}: field 'name' is '{spec.name}' but filename stem is '{path.stem}'. "
            "They must match."
        )
    return spec


def load_spec(name_or_path: str | Path) -> tuple[Spec, SpecSource, Path]:
    """Resolve a spec by name or path. Returns (spec, source, path)."""
    candidate = Path(name_or_path)
    if candidate.exists() and candidate.is_file():
        return _load_yaml_as_spec(candidate), "explicit", candidate.resolve()

    name = str(name_or_path)
    user_path = _user_specs_dir() / f"{name}.yaml"
    if user_path.is_file():
        return _load_yaml_as_spec(user_path), "user", user_path.resolve()

    bundled = _bundled_resource_path(name)
    if bundled is not None:
        return _load_yaml_as_spec(bundled), "bundled", bundled.resolve()

    known = sorted(list_bundled_names())
    raise SpecError(
        f"Unknown spec '{name}'. Known bundled specs: {', '.join(known) or '(none)'}. "
        f"Or pass a path to a YAML file."
    )


def list_bundled_names() -> list[str]:
    try:
        pkg = resources.files("finalpass._bundled_specs")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name.removesuffix(".yaml")
        for p in pkg.iterdir()
        if p.is_file() and p.name.endswith(".yaml")
    )


def list_bundled() -> list[Spec]:
    out: list[Spec] = []
    for name in list_bundled_names():
        spec, _, _ = load_spec(name)
        out.append(spec)
    return out
