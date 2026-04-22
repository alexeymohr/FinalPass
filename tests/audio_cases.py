from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 48000
SECONDS = 10.0
SEED = 0xC0DE
PM_TARGET_DBFS = -23.0
ROLE_TOKEN = {"pm": "PM", "dx": "DX", "mx": "MX", "fx": "FX", "me": "ME"}
ME_BAND_LOW_HZ = 200.0
ME_BAND_HIGH_HZ = 4000.0
SPLIT_LEG_ORDERS = {
    "stereo": ["L", "R"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Ls", "Rs", "Lss", "Rss"],
}
PRESENTATION_TOKEN = {
    "stereo": "LtRt",
    "5.1": "5.1",
    "7.1": "7.1",
}
ROLE_STEM_TOKEN = {
    "pm": "Comp",
    "dx": "DX",
    "mx": "MX",
    "fx": "FX",
    "me": "ME",
    "opt": "OPT",
    "unknown": "MYSTERY",
}


def pink_noise(n_samples: int, n_channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_samples, n_channels))
    pink = np.zeros_like(white)
    pink[0] = 0.05 * white[0]
    for i in range(1, n_samples):
        pink[i] = 0.99 * pink[i - 1] + 0.05 * white[i]
    return pink.astype(np.float64)


def write_audio(path: Path, data: np.ndarray, *, sr: int = SR, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float64), sr, subtype=subtype)
    return path


def shift_with_zeros(data: np.ndarray, offset_samples: int) -> np.ndarray:
    shifted = np.zeros_like(data)
    if offset_samples > 0:
        shifted[offset_samples:] = data[:-offset_samples]
    elif offset_samples < 0:
        shifted[:offset_samples] = data[-offset_samples:]
    else:
        shifted[:] = data
    return shifted


def scale_to_dbfs(data: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(data ** 2)))
    scale = (10 ** (target_dbfs / 20.0)) / rms if rms > 0 else 1.0
    return data * scale


def speech_band_noise(n_samples: int, seed: int, *, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    sos = butter(4, [ME_BAND_LOW_HZ, ME_BAND_HIGH_HZ], btype="bandpass", fs=sr, output="sos")
    shaped = sosfiltfilt(sos, white.astype(np.float64), padlen=0)
    t = np.arange(n_samples, dtype=np.float64) / float(sr)
    envelope = (0.5 * (1.0 + np.sin(2.0 * np.pi * 1.7 * t + 0.2))) ** 1.5
    return shaped * envelope


def broadcast_mono(mono: np.ndarray, *, n_channels: int) -> np.ndarray:
    if n_channels == 1:
        return mono[:, None]
    if n_channels == 2:
        return np.column_stack([mono, mono * 0.98])
    if n_channels == 6:
        return np.column_stack([
            mono,
            mono * 0.98,
            mono * 0.96,
            np.zeros_like(mono),
            mono * 0.94,
            mono * 0.92,
        ])
    if n_channels == 8:
        return np.column_stack([
            mono,
            mono * 0.98,
            mono * 0.96,
            np.zeros_like(mono),
            mono * 0.94,
            mono * 0.92,
            mono * 0.90,
            mono * 0.88,
        ])
    raise ValueError(f"unsupported test channel count {n_channels}")


def exact_sum_components(
    *,
    seconds: float = SECONDS,
    n_channels: int = 2,
    base_seed: int = SEED,
    pm_target_dbfs: float = PM_TARGET_DBFS,
    sr: int = SR,
) -> dict[str, np.ndarray]:
    n_samples = int(round(seconds * sr))
    dx = pink_noise(n_samples, n_channels, base_seed + 0)
    mx = pink_noise(n_samples, n_channels, base_seed + 1)
    fx = pink_noise(n_samples, n_channels, base_seed + 2)
    pm = dx + mx + fx
    rms = float(np.sqrt(np.mean(pm ** 2)))
    scale = (10 ** (pm_target_dbfs / 20.0)) / rms if rms > 0 else 1.0
    dx *= scale
    mx *= scale
    fx *= scale
    pm *= scale
    return {"pm": pm, "dx": dx, "mx": mx, "fx": fx, "me": mx + fx}


def me_check_components(
    *,
    seconds: float = SECONDS,
    n_channels: int = 2,
    base_seed: int = SEED,
    dx_target_dbfs: float = -24.0,
    me_target_dbfs: float = -48.0,
    bleed_region: tuple[float, float] | None = None,
    bleed_gain: float = 0.08,
    dx_quiet_region: tuple[float, float] | None = None,
    sr: int = SR,
) -> dict[str, np.ndarray]:
    n_samples = int(round(seconds * sr))
    dx_mono = speech_band_noise(n_samples, base_seed + 100, sr=sr)
    me_mono = speech_band_noise(n_samples, base_seed + 101, sr=sr)
    dx_mono = scale_to_dbfs(dx_mono, dx_target_dbfs)
    me_mono = scale_to_dbfs(me_mono, me_target_dbfs)

    if dx_quiet_region is not None:
        start = int(round(dx_quiet_region[0] * sr))
        end = int(round(dx_quiet_region[1] * sr))
        dx_mono[start:end] = 0.0

    if bleed_region is not None:
        start = int(round(bleed_region[0] * sr))
        end = int(round(bleed_region[1] * sr))
        me_mono[start:end] = me_mono[start:end] + (bleed_gain * dx_mono[start:end])

    return {
        "dx": broadcast_mono(dx_mono, n_channels=n_channels),
        "me": broadcast_mono(me_mono, n_channels=n_channels),
    }


def independent_me_from_dx(
    dx: np.ndarray,
    *,
    base_seed: int,
    sr: int = SR,
    target_dbfs: float = -48.0,
    bleed_region: tuple[float, float] | None = None,
    bleed_gain: float = 0.60,
) -> np.ndarray:
    n_samples, n_channels = dx.shape
    me_mono = speech_band_noise(n_samples, base_seed + 200, sr=sr)
    me_mono = scale_to_dbfs(me_mono, target_dbfs)
    me = broadcast_mono(me_mono, n_channels=n_channels)
    if bleed_region is not None:
        start = int(round(bleed_region[0] * sr))
        end = int(round(bleed_region[1] * sr))
        me[start:end] = me[start:end] + (bleed_gain * dx[start:end])
    return me


def build_group(
    root: Path,
    group_id: str,
    *,
    write_roles: tuple[str, ...] = ("pm", "dx", "mx", "fx"),
    pm_gain_db: float = 0.0,
    null_defect: bool = False,
    me_mode: str = "sum",
    me_bleed_defect: bool = False,
    me_bleed_gain: float = 0.60,
    role_labels: dict[str, str] | None = None,
    base_seed: int = SEED,
    seconds: float = SECONDS,
) -> dict[str, Path]:
    role_labels = role_labels or {}
    data = exact_sum_components(seconds=seconds, n_channels=2, base_seed=base_seed)
    gain = 10 ** (pm_gain_db / 20.0)
    for key in data:
        data[key] = data[key] * gain

    if null_defect:
        start = int(round(5.0 * SR))
        end = int(round(7.0 * SR))
        data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]

    if "me" in write_roles and me_mode == "independent":
        data["me"] = independent_me_from_dx(
            data["dx"],
            base_seed=base_seed,
            bleed_region=(5.0, 7.0) if me_bleed_defect else None,
            bleed_gain=me_bleed_gain,
        )

    out: dict[str, Path] = {}
    for role in write_roles:
        label = role_labels.get(role, "STEREO")
        filename = f"SHOW_{group_id}_{ROLE_TOKEN[role]}_{label}.wav"
        out[role] = write_audio(root / filename, data[role])
    return out


def build_two_episodes(
    root: Path,
    *,
    hot_e04_pm: bool = False,
    e04_null_defect: bool = False,
    e04_me_bleed_defect: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    build_group(
        root,
        "S01E03",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        me_mode="independent",
        base_seed=SEED + 0,
    )
    build_group(
        root,
        "S01E04",
        write_roles=("pm", "dx", "mx", "fx", "me"),
        base_seed=SEED + 10,
        pm_gain_db=10.0 if hot_e04_pm else 0.0,
        null_defect=e04_null_defect,
        me_mode="independent",
        me_bleed_defect=e04_me_bleed_defect,
    )
    return root


def write_split_role(
    root: Path,
    group_id: str,
    role: str,
    layout: str,
    data: np.ndarray,
    *,
    presentation_token: str | None = None,
    subtype: str = "PCM_24",
    stem_token: str | None = None,
) -> dict[str, Path]:
    presentation_token = presentation_token or PRESENTATION_TOKEN[layout]
    stem_token = stem_token or ROLE_STEM_TOKEN[role]
    legs = SPLIT_LEG_ORDERS[layout]
    assert data.ndim == 2
    assert data.shape[1] == len(legs)
    out: dict[str, Path] = {}
    for index, leg in enumerate(legs):
        path = root / f"SHOW_{group_id}_{stem_token}_{presentation_token}.{leg}.wav"
        out[leg] = write_audio(path, data[:, [index]], subtype=subtype)
    return out


def build_split_group(
    root: Path,
    group_id: str,
    *,
    layout: str = "5.1",
    write_roles: tuple[str, ...] = ("pm", "dx", "mx", "fx"),
    pm_gain_db: float = 0.0,
    null_defect: bool = False,
    me_mode: str = "sum",
    me_bleed_defect: bool = False,
    me_bleed_gain: float = 0.60,
    base_seed: int = SEED,
    seconds: float = SECONDS,
    presentation_token: str | None = None,
    role_presentation_tokens: dict[str, str] | None = None,
    role_stem_tokens: dict[str, str] | None = None,
) -> dict[str, dict[str, Path]]:
    role_presentation_tokens = role_presentation_tokens or {}
    role_stem_tokens = role_stem_tokens or {}
    n_channels = len(SPLIT_LEG_ORDERS[layout])
    data = exact_sum_components(seconds=seconds, n_channels=n_channels, base_seed=base_seed)
    gain = 10 ** (pm_gain_db / 20.0)
    for key in data:
        data[key] = data[key] * gain

    if null_defect:
        start = int(round(5.0 * SR))
        end = int(round(7.0 * SR))
        data["pm"][start:end] = data["pm"][start:end] - data["fx"][start:end]

    if "me" in write_roles and me_mode == "independent":
        data["me"] = independent_me_from_dx(
            data["dx"],
            base_seed=base_seed,
            bleed_region=(5.0, 7.0) if me_bleed_defect else None,
            bleed_gain=me_bleed_gain,
        )

    out: dict[str, dict[str, Path]] = {}
    for role in write_roles:
        source_key = role if role in data else "mx"
        out[role] = write_split_role(
            root,
            group_id,
            role,
            layout,
            data[source_key],
            presentation_token=role_presentation_tokens.get(role, presentation_token),
            stem_token=role_stem_tokens.get(role),
        )
    return out
