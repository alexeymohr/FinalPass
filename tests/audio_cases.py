from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 48000
SECONDS = 10.0
SEED = 0xC0DE
PM_TARGET_DBFS = -23.0
LOUDNESS_PASSING_SEED = 0xFA57
ROLE_TOKEN = {"pm": "PM", "dx": "DX", "mx": "MX", "fx": "FX", "me": "ME"}
ME_BAND_LOW_HZ = 200.0
ME_BAND_HIGH_HZ = 4000.0
SPLIT_LEG_ORDERS = {
    "stereo": ["L", "R"],
    "5.0": ["L", "R", "C", "Ls", "Rs"],
    "5.1": ["L", "R", "C", "LFE", "Ls", "Rs"],
    "7.1": ["L", "R", "C", "LFE", "Ls", "Rs", "Lss", "Rss"],
}
PRESENTATION_TOKEN = {
    "stereo": "LtRt",
    "5.0": "5.0",
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


def write_audio(
    path: Path,
    data: np.ndarray,
    *,
    sr: int = SR,
    subtype: str = "PCM_24",
    time_reference_samples: int | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float64), sr, subtype=subtype)
    if time_reference_samples is not None:
        write_bext_time_reference(path, time_reference_samples)
    return path


def write_bext_time_reference(path: Path, time_reference_samples: int) -> Path:
    """Insert or replace a minimal bext chunk carrying the sample time reference."""
    path = Path(path)
    original = path.read_bytes()
    if len(original) < 12 or original[:4] != b"RIFF" or original[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file.")

    bext = bytearray(602)
    struct.pack_into("<IIH", bext, 338, time_reference_samples & 0xFFFFFFFF, time_reference_samples >> 32, 1)
    bext_chunk = b"bext" + struct.pack("<I", len(bext)) + bytes(bext)

    body = bytearray()
    offset = 12
    replaced = False
    while offset + 8 <= len(original):
        chunk_id = original[offset : offset + 4]
        size = struct.unpack("<I", original[offset + 4 : offset + 8])[0]
        padded_size = size + (size % 2)
        chunk_end = offset + 8 + padded_size
        if chunk_end > len(original):
            break
        if chunk_id == b"bext":
            body.extend(bext_chunk)
            replaced = True
        else:
            body.extend(original[offset:chunk_end])
        offset = chunk_end

    if offset < len(original):
        body.extend(original[offset:])
    if not replaced:
        body = bytearray(bext_chunk) + body

    riff_size = 4 + len(body)
    updated = bytearray(original[:12])
    updated[4:8] = struct.pack("<I", riff_size)
    updated.extend(body)
    path.write_bytes(updated)
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


def true_peak_over_program(
    *,
    seconds: float = SECONDS,
    n_channels: int = 2,
    base_seed: int = LOUDNESS_PASSING_SEED,
    base_target_dbfs: float = -23.0,
    burst_region: tuple[float, float] = (5.0, 5.002),
    burst_regions: tuple[tuple[float, float], ...] | None = None,
    burst_frequency_hz: float = 997.0,
    sr: int = SR,
) -> np.ndarray:
    data = scale_to_dbfs(
        pink_noise(int(round(seconds * sr)), n_channels, base_seed),
        base_target_dbfs,
    )
    regions = burst_regions or (burst_region,)
    for region in regions:
        start = int(round(region[0] * sr))
        end = int(round(region[1] * sr))
        burst_length = max(1, end - start)
        t = np.arange(burst_length, dtype=np.float64) / float(sr)
        burst = np.clip(np.sin(2.0 * np.pi * burst_frequency_hz * t) * 2.0, -1.0, 1.0)
        for channel in range(n_channels):
            data[start : start + burst_length, channel] = burst
    return data


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


# --- Phase 8A channel-integrity fixtures ------------------------------------
#
# Channel checks need decorrelated content per channel: independent noise is
# the honest stand-in for a real mix's channel independence.


def band_limited_lfe(n_samples: int, seed: int, *, sr: int = SR, cutoff_hz: float = 80.0) -> np.ndarray:
    """A legitimate LFE leg: noise low-passed well under the broadband cutoff."""
    rng = np.random.default_rng(seed)
    sos = butter(4, cutoff_hz / (sr / 2.0), btype="lowpass", output="sos")
    low = sosfiltfilt(sos, rng.standard_normal(n_samples))
    peak = float(np.max(np.abs(low)))
    return (low / peak * 0.3) if peak > 0 else low


def build_clean_51(
    *,
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
) -> np.ndarray:
    """Six decorrelated legs in SMPTE order with a band-limited LFE."""
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, 6)) * 0.1
    data[:, 3] = band_limited_lfe(n, seed + 1, sr=sr)
    return data


def build_fake_51(*, seconds: float = SECONDS, seed: int = SEED, sr: int = SR) -> np.ndarray:
    """A stereo pair smeared across the 5.1 slots — the classic fake surround."""
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(n) * 0.1
    right = rng.standard_normal(n) * 0.1
    data = np.zeros((n, 6))
    data[:, 0] = left
    data[:, 1] = right
    data[:, 2] = left
    data[:, 3] = band_limited_lfe(n, seed + 1, sr=sr)
    data[:, 4] = left
    data[:, 5] = right
    return data


def build_dead_leg_51(
    *,
    leg: str = "Ls",
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
) -> np.ndarray:
    """A normal 5.1 with one leg silenced."""
    data = build_clean_51(seconds=seconds, seed=seed, sr=sr)
    data[:, SPLIT_LEG_ORDERS["5.1"].index(leg)] = 0.0
    return data


def build_polarity_flip(
    *,
    layout: str = "stereo",
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
) -> np.ndarray:
    """A canonical pair whose right member is polarity-inverted."""
    n = int(round(seconds * sr))
    if layout == "stereo":
        rng = np.random.default_rng(seed)
        left = rng.standard_normal(n) * 0.1
        return np.column_stack([left, -left])
    data = build_clean_51(seconds=seconds, seed=seed, sr=sr)
    data[:, 5] = -data[:, 4]
    return data


def build_film_order_51(*, seconds: float = SECONDS, seed: int = SEED, sr: int = SR) -> np.ndarray:
    """Film-order legs (L C R Ls Rs LFE) laid into a SMPTE-ordered container.

    Read as SMPTE, the LFE slot holds a full-range surround leg — the
    fingerprint this check exists to catch.
    """
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    film = [rng.standard_normal(n) * 0.1 for _ in range(5)]
    lfe = band_limited_lfe(n, seed + 1, sr=sr)
    # L C R Ls Rs LFE written into the six SMPTE slots without remapping.
    return np.column_stack([film[0], film[1], film[2], film[3], film[4], lfe])


def build_dual_mono(
    *,
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
    right_gain: float = 1.0,
    dither_lsb: float | None = None,
) -> np.ndarray:
    """One signal on both legs, optionally level-scaled or dither-separated."""
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    left = rng.standard_normal(n) * 0.1
    right = left * right_gain
    if dither_lsb is not None:
        right = right + rng.standard_normal(n) * dither_lsb
    return np.column_stack([left, right])


def build_head_tone_stereo(
    *,
    seconds: float = SECONDS,
    tone_seconds: float = 2.0,
    seed: int = SEED,
    sr: int = SR,
) -> np.ndarray:
    """Identical head tone on both legs, then genuinely decorrelated program.

    A whole-file correlation would call this dual-mono; a windowed median
    must not.
    """
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    data = np.column_stack([rng.standard_normal(n) * 0.1, rng.standard_normal(n) * 0.1])
    tone_samples = int(round(tone_seconds * sr))
    t = np.arange(tone_samples) / sr
    tone = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    data[:tone_samples, 0] = tone
    data[:tone_samples, 1] = tone
    return data


# --- Phase 8B downmix fixtures ----------------------------------------------

DOWNMIX_CENTER_GAIN = 10 ** (-3.0 / 20.0)
DOWNMIX_SURROUND_GAIN = 10 ** (-3.0 / 20.0)


def build_surround_program(
    *,
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
    layout: str = "5.1",
) -> np.ndarray:
    """A 5.1/7.1 program with decorrelated legs and a band-limited LFE."""
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    channels = 6 if layout == "5.1" else 8
    data = rng.standard_normal((n, channels)) * 0.08
    data[:, 3] = band_limited_lfe(n, seed + 1, sr=sr)
    return data


def fold_down_of(surround: np.ndarray) -> np.ndarray:
    """The Lo/Ro-style fold-down FinalPass derives, at default gains."""
    left = surround[:, 0] + DOWNMIX_CENTER_GAIN * surround[:, 2] + DOWNMIX_SURROUND_GAIN * surround[:, 4]
    right = surround[:, 1] + DOWNMIX_CENTER_GAIN * surround[:, 2] + DOWNMIX_SURROUND_GAIN * surround[:, 5]
    if surround.shape[1] == 8:
        left = left + DOWNMIX_SURROUND_GAIN * surround[:, 6]
        right = right + DOWNMIX_SURROUND_GAIN * surround[:, 7]
    return np.column_stack([left, right])


def build_downmix_pair(
    variant: str = "exact",
    *,
    seconds: float = SECONDS,
    seed: int = SEED,
    sr: int = SR,
    level_offset_db: float = 4.0,
    invert_region: tuple[float, float] = (4.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (delivered_stereo, surround) for one downmix scenario.

    Variants: ``exact`` (the mechanical fold), ``discrete`` (a related but
    independently balanced stereo), ``unrelated`` (a different programme),
    ``level_offset`` (the fold, gain-shifted), ``inverted_region`` (the fold
    with its right leg flipped over one span).
    """
    surround = build_surround_program(seconds=seconds, seed=seed, sr=sr)
    fold = fold_down_of(surround)

    if variant == "exact":
        return fold, surround
    if variant == "level_offset":
        return fold * (10 ** (level_offset_db / 20.0)), surround
    if variant == "discrete":
        # Same programme, re-balanced by hand: different centre and surround
        # proportions, which is what a real discrete stereo mix looks like.
        left = surround[:, 0] * 1.1 + 0.5 * surround[:, 2] + 0.3 * surround[:, 4]
        right = surround[:, 1] * 1.1 + 0.5 * surround[:, 2] + 0.3 * surround[:, 5]
        return np.column_stack([left, right]), surround
    if variant == "unrelated":
        rng = np.random.default_rng(seed + 999)
        n = surround.shape[0]
        return rng.standard_normal((n, 2)) * 0.1, surround
    if variant == "inverted_region":
        delivered = fold.copy()
        start = int(round(invert_region[0] * sr))
        end = int(round(invert_region[1] * sr))
        delivered[start:end, 1] = -delivered[start:end, 1]
        return delivered, surround
    raise ValueError(f"unknown downmix variant {variant!r}")
