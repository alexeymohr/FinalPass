"""Phase 8A channel-integrity analysis.

Per-channel diagnostics on one logical asset: dead legs, duplicated channels
(dual-mono "stereo", stereo-duplicated "5.1"), polarity inversion between
canonical pairs, broadband energy in the LFE slot, and channel-imbalance
notes.

All detectors are deterministic windowed statistics — no speech models, no
perceptual weighting, no auto-repair. FinalPass reports a suspected channel
error; it never remaps channels.

Pairwise statistics are windowed and reduced by median rather than measured
across the whole file, because identical head tone, sync pops, and silence
would otherwise dominate a whole-file correlation and manufacture duplicate
findings on legitimately decorrelated content.

Findings are untimed: they describe a channel, not a moment, so nothing here
ever reaches the AAF marker export.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict

from .analysis_common import (
    ANALYSIS_DBFS_FLOOR,
    energy_ratio_above,
    null_depth_db,
    optimal_gain,
    rms,
    signed_correlation,
    to_dbfs,
    window_bounds,
    window_rms_dbfs,
)
from .audio_io import AudioFile

DEFAULT_CHANNELS_WINDOW_MS = 5000.0
DEFAULT_CHANNELS_HOP_MS = 2500.0
DEFAULT_CHANNELS_ACTIVITY_DBFS = -60.0
DEFAULT_CHANNELS_SILENCE_DBFS = -80.0
DEFAULT_CHANNELS_DUPLICATE_NULL_DB = 40.0
DEFAULT_CHANNELS_POLARITY_CORR = -0.95
DEFAULT_CHANNELS_LFE_CUTOFF_HZ = 250.0
DEFAULT_CHANNELS_LFE_ENERGY_RATIO = 0.25
DEFAULT_CHANNELS_IMBALANCE_DB = 6.0

# A pair needs this many gated windows before duplicate/polarity verdicts are
# trustworthy; below it the finding is skipped rather than silently passed.
MIN_QUALIFYING_WINDOWS = 8
# An exact duplicate nulls to a bit-identical zero residual, which would read
# as a ~280 dB null. Cap the reported depth so details stay legible; any value
# at or beyond the threshold means the same thing.
MAX_NULL_DEPTH_DB = 200.0

_LFE_LABEL = "LFE"
# SMPTE order, matching the split-mono assembly order in assets.py.
_CANONICAL_CHANNEL_LABELS: dict[int, list[str]] = {
    1: ["M"],
    2: ["L", "R"],
    6: ["L", "R", "C", "LFE", "Ls", "Rs"],
    8: ["L", "R", "C", "LFE", "Ls", "Rs", "Lss", "Rss"],
}
# Pairs a polarity flip is meaningful for: the two halves of a stereo image.
_CANONICAL_POLARITY_PAIRS = (("L", "R"), ("Ls", "Rs"), ("Lss", "Rss"))

BROADBAND_LFE_DETAIL = (
    "full-range content in the LFE slot — check interleave channel order "
    "(film order L C R Ls Rs LFE vs SMPTE L R C LFE Ls Rs)"
)
NOTE_LFE_SYNTHESIZED = "lfe_synthesized_from_5_0_source"
REASON_INSUFFICIENT_CONTENT = "insufficient_active_content"
REASON_NOT_APPLICABLE_MONO = "not_applicable_mono"


class ChannelsTunables(BaseModel):
    """Analysis knobs for the channel-integrity pass. Runtime-only; never persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_ms: float = DEFAULT_CHANNELS_WINDOW_MS
    hop_ms: float = DEFAULT_CHANNELS_HOP_MS
    activity_dbfs: float = DEFAULT_CHANNELS_ACTIVITY_DBFS
    silence_dbfs: float = DEFAULT_CHANNELS_SILENCE_DBFS
    duplicate_null_db: float = DEFAULT_CHANNELS_DUPLICATE_NULL_DB
    polarity_corr: float = DEFAULT_CHANNELS_POLARITY_CORR
    lfe_cutoff_hz: float = DEFAULT_CHANNELS_LFE_CUTOFF_HZ
    lfe_energy_ratio: float = DEFAULT_CHANNELS_LFE_ENERGY_RATIO
    imbalance_db: float = DEFAULT_CHANNELS_IMBALANCE_DB
    fail_dual_mono: bool = False


@dataclass(frozen=True)
class ChannelFindingData:
    kind: str
    severity: str
    channels: list[str]
    measured: float | None
    threshold: float | None
    detail: str
    skipped: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class ChannelsAnalysis:
    findings: list[ChannelFindingData]
    notes: list[str]


@dataclass(frozen=True)
class _PairStats:
    left_label: str
    right_label: str
    windows: int
    median_null_depth_db: float
    median_gain: float
    median_corr: float


def analyze_channels(
    audio: AudioFile,
    *,
    member_legs: list[str],
    channel_config: str | None,
    tunables: ChannelsTunables = ChannelsTunables(),
) -> ChannelsAnalysis:
    """Return channel-integrity findings for one logical asset.

    ``member_legs`` carries split-mono provenance: it names the real source
    legs, which is how a 5.0 family assembled into a 5.1 container is told
    apart from a delivery that genuinely carries a silent LFE.

    The asset's role is deliberately not an input — a dead leg or a fake
    surround is a defect wherever it appears.
    """
    labels = _channel_labels(audio.channel_count, member_legs)
    lfe_index = labels.index(_LFE_LABEL) if _LFE_LABEL in labels else None
    full_range_indices = [index for index in range(audio.channel_count) if index != lfe_index]

    findings: list[ChannelFindingData] = []
    notes: list[str] = []

    lfe_is_synthesized = _lfe_is_synthesized(
        channel_config=channel_config,
        member_legs=member_legs,
        has_lfe_slot=lfe_index is not None,
    )
    if lfe_is_synthesized:
        notes.append(NOTE_LFE_SYNTHESIZED)

    channel_rms_dbfs = [
        to_dbfs(rms(audio.data[:, index])) for index in range(audio.channel_count)
    ]
    asset_is_active = any(
        channel_rms_dbfs[index] > tunables.activity_dbfs for index in full_range_indices
    )

    findings.extend(_silence_findings(
        labels=labels,
        channel_rms_dbfs=channel_rms_dbfs,
        full_range_indices=full_range_indices,
        lfe_index=lfe_index,
        lfe_is_synthesized=lfe_is_synthesized,
        asset_is_active=asset_is_active,
        tunables=tunables,
    ))

    if audio.channel_count == 1:
        findings.extend(_mono_pairwise_skips())
    else:
        findings.extend(_pairwise_findings(
            audio=audio,
            labels=labels,
            full_range_indices=full_range_indices,
            tunables=tunables,
        ))

    if lfe_index is not None and not lfe_is_synthesized:
        lfe_finding = _broadband_lfe_finding(
            lfe=audio.data[:, lfe_index],
            sample_rate=audio.sample_rate,
            lfe_rms_dbfs=channel_rms_dbfs[lfe_index],
            tunables=tunables,
        )
        if lfe_finding is not None:
            findings.append(lfe_finding)

    imbalance = _imbalance_finding(
        labels=labels,
        channel_rms_dbfs=channel_rms_dbfs,
        tunables=tunables,
    )
    if imbalance is not None:
        findings.append(imbalance)

    return ChannelsAnalysis(findings=findings, notes=notes)


def _channel_labels(channel_count: int, member_legs: list[str]) -> list[str]:
    """Prefer real split-leg names; fall back to the canonical SMPTE order."""
    if len(member_legs) == channel_count:
        return list(member_legs)
    canonical = _CANONICAL_CHANNEL_LABELS.get(channel_count)
    if canonical is not None:
        return list(canonical)
    return [f"ch{index + 1}" for index in range(channel_count)]


def _lfe_is_synthesized(
    *,
    channel_config: str | None,
    member_legs: list[str],
    has_lfe_slot: bool,
) -> bool:
    """True when FinalPass padded a 5.0 split family into a 5.1 container."""
    return (
        has_lfe_slot
        and channel_config == "5.1"
        and bool(member_legs)
        and _LFE_LABEL not in member_legs
    )


def _silence_findings(
    *,
    labels: list[str],
    channel_rms_dbfs: list[float],
    full_range_indices: list[int],
    lfe_index: int | None,
    lfe_is_synthesized: bool,
    asset_is_active: bool,
    tunables: ChannelsTunables,
) -> list[ChannelFindingData]:
    findings: list[ChannelFindingData] = []

    if len(labels) == 1:
        # A mono asset has no sibling to be dead against; the only honest
        # statement is whether the file carries content at all.
        if channel_rms_dbfs[0] < tunables.silence_dbfs:
            findings.append(ChannelFindingData(
                kind="silent_leg",
                severity="info",
                channels=[labels[0]],
                measured=round(channel_rms_dbfs[0], 1),
                threshold=tunables.silence_dbfs,
                detail="file is effectively silent",
            ))
        return findings

    for index in full_range_indices:
        if asset_is_active and channel_rms_dbfs[index] < tunables.silence_dbfs:
            findings.append(ChannelFindingData(
                kind="silent_leg",
                severity="fail",
                channels=[labels[index]],
                measured=round(channel_rms_dbfs[index], 1),
                threshold=tunables.silence_dbfs,
                detail=(
                    f"{labels[index]} carries no content "
                    f"({channel_rms_dbfs[index]:.1f} dBFS full-span RMS)"
                ),
            ))

    if lfe_index is not None and not lfe_is_synthesized:
        if channel_rms_dbfs[lfe_index] < tunables.silence_dbfs:
            # A show that never feeds the LFE is a legitimate delivery.
            findings.append(ChannelFindingData(
                kind="silent_leg",
                severity="info",
                channels=[labels[lfe_index]],
                measured=round(channel_rms_dbfs[lfe_index], 1),
                threshold=tunables.silence_dbfs,
                detail="LFE carries no content; this is legitimate for many programs",
            ))

    return findings


def _mono_pairwise_skips() -> list[ChannelFindingData]:
    return [
        ChannelFindingData(
            kind=kind,
            severity="info",
            channels=[],
            measured=None,
            threshold=None,
            detail="mono asset has no channel pair to compare",
            skipped=True,
            reason=REASON_NOT_APPLICABLE_MONO,
        )
        for kind in ("duplicate_channels", "polarity_inversion")
    ]


def _pairwise_findings(
    *,
    audio: AudioFile,
    labels: list[str],
    full_range_indices: list[int],
    tunables: ChannelsTunables,
) -> list[ChannelFindingData]:
    windows = window_bounds(
        sample_count=audio.data.shape[0],
        sample_rate=audio.sample_rate,
        window_ms=tunables.window_ms,
        hop_ms=tunables.hop_ms,
    )
    window_levels_dbfs = window_rms_dbfs(audio.data, windows=windows)

    is_surround = audio.channel_count >= 6
    duplicate_severity = "fail" if is_surround or tunables.fail_dual_mono else "info"

    findings: list[ChannelFindingData] = []
    evaluated_any = False
    for left, right in combinations(full_range_indices, 2):
        stats = _pair_stats(
            audio.data,
            left=left,
            right=right,
            left_label=labels[left],
            right_label=labels[right],
            windows=windows,
            window_levels_dbfs=window_levels_dbfs,
            activity_dbfs=tunables.activity_dbfs,
        )
        if stats is None:
            continue
        evaluated_any = True
        finding = _pair_finding(
            stats,
            duplicate_severity=duplicate_severity,
            tunables=tunables,
        )
        if finding is not None:
            findings.append(finding)

    if not evaluated_any:
        detail = (
            f"fewer than {MIN_QUALIFYING_WINDOWS} windows had both channels above "
            f"{tunables.activity_dbfs:.1f} dBFS"
        )
        findings.extend(
            ChannelFindingData(
                kind=kind,
                severity="info",
                channels=[],
                measured=None,
                threshold=None,
                detail=detail,
                skipped=True,
                reason=REASON_INSUFFICIENT_CONTENT,
            )
            for kind in ("duplicate_channels", "polarity_inversion")
        )
    return findings


def _pair_finding(
    stats: _PairStats,
    *,
    duplicate_severity: str,
    tunables: ChannelsTunables,
) -> ChannelFindingData | None:
    channels = [stats.left_label, stats.right_label]
    nulls_to_silence = stats.median_null_depth_db >= tunables.duplicate_null_db

    if nulls_to_silence and stats.median_gain < 0:
        # An inverted copy: it nulls deeply, but only against a flipped
        # polarity. That is a polarity defect, not a duplicated channel.
        return ChannelFindingData(
            kind="polarity_inversion",
            severity="fail",
            channels=channels,
            measured=round(stats.median_corr, 3),
            threshold=tunables.polarity_corr,
            detail=(
                f"{stats.left_label} and {stats.right_label} are an inverted copy "
                f"(null depth {stats.median_null_depth_db:.1f} dB against a flipped "
                f"polarity, median correlation {stats.median_corr:.2f} across "
                f"{stats.windows} windows)"
            ),
        )

    if nulls_to_silence:
        return ChannelFindingData(
            kind="duplicate_channels",
            severity=duplicate_severity,
            channels=channels,
            measured=round(stats.median_null_depth_db, 1),
            threshold=tunables.duplicate_null_db,
            detail=(
                f"phase-invert null depth {stats.median_null_depth_db:.1f} dB; "
                f"{stats.left_label} sits {_gain_as_db(stats.median_gain):+.1f} dB "
                f"relative to {stats.right_label} across {stats.windows} windows"
            ),
        )

    if _is_canonical_polarity_pair(stats.left_label, stats.right_label):
        if stats.median_corr <= tunables.polarity_corr:
            return ChannelFindingData(
                kind="polarity_inversion",
                severity="fail",
                channels=channels,
                measured=round(stats.median_corr, 3),
                threshold=tunables.polarity_corr,
                detail=(
                    f"{stats.left_label} and {stats.right_label} are out of polarity "
                    f"(median correlation {stats.median_corr:.2f} across "
                    f"{stats.windows} windows)"
                ),
            )
    return None


def _is_canonical_polarity_pair(left_label: str, right_label: str) -> bool:
    return (left_label, right_label) in _CANONICAL_POLARITY_PAIRS or (
        right_label,
        left_label,
    ) in _CANONICAL_POLARITY_PAIRS


def _pair_stats(
    data: np.ndarray,
    *,
    left: int,
    right: int,
    left_label: str,
    right_label: str,
    windows: list[tuple[int, int]],
    window_levels_dbfs: np.ndarray,
    activity_dbfs: float,
) -> _PairStats | None:
    qualifying = [
        index
        for index in range(len(windows))
        if window_levels_dbfs[index, left] > activity_dbfs
        and window_levels_dbfs[index, right] > activity_dbfs
    ]
    if len(qualifying) < MIN_QUALIFYING_WINDOWS:
        return None

    depths: list[float] = []
    gains: list[float] = []
    correlations: list[float] = []
    for index in qualifying:
        start, end = windows[index]
        a = data[start:end, left]
        b = data[start:end, right]
        gain = optimal_gain(a, b)
        depths.append(null_depth_db(a, b, gain, max_depth_db=MAX_NULL_DEPTH_DB))
        gains.append(gain)
        correlations.append(signed_correlation(a, b))

    return _PairStats(
        left_label=left_label,
        right_label=right_label,
        windows=len(qualifying),
        median_null_depth_db=float(np.median(depths)),
        median_gain=float(np.median(gains)),
        median_corr=float(np.median(correlations)),
    )


def _gain_as_db(gain: float) -> float:
    """The fitted gain as a level offset; sign is reported separately."""
    return to_dbfs(abs(gain))


def _broadband_lfe_finding(
    *,
    lfe: np.ndarray,
    sample_rate: int,
    lfe_rms_dbfs: float,
    tunables: ChannelsTunables,
) -> ChannelFindingData | None:
    if lfe_rms_dbfs <= tunables.silence_dbfs:
        return None
    ratio = energy_ratio_above(lfe, sample_rate=sample_rate, cutoff_hz=tunables.lfe_cutoff_hz)
    if ratio is None or ratio < tunables.lfe_energy_ratio:
        return None
    return ChannelFindingData(
        kind="broadband_lfe",
        severity="fail",
        channels=[_LFE_LABEL],
        measured=round(ratio, 3),
        threshold=tunables.lfe_energy_ratio,
        detail=(
            f"{ratio * 100:.0f}% of LFE energy sits above {tunables.lfe_cutoff_hz:.0f} Hz — "
            f"{BROADBAND_LFE_DETAIL}"
        ),
    )


def _imbalance_finding(
    *,
    labels: list[str],
    channel_rms_dbfs: list[float],
    tunables: ChannelsTunables,
) -> ChannelFindingData | None:
    if "L" not in labels or "R" not in labels:
        return None
    left_dbfs = channel_rms_dbfs[labels.index("L")]
    right_dbfs = channel_rms_dbfs[labels.index("R")]
    delta = abs(left_dbfs - right_dbfs)
    if delta <= tunables.imbalance_db:
        return None
    louder = "L" if left_dbfs > right_dbfs else "R"
    return ChannelFindingData(
        kind="channel_imbalance",
        severity="info",
        channels=["L", "R"],
        measured=round(delta, 1),
        threshold=tunables.imbalance_db,
        detail=f"{louder} is {delta:.1f} dB louder across the full span",
    )


