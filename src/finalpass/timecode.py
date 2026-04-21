"""Sample-index ↔ timecode helper. Used by later phases; Phase 1 writes `fps`
to the JSON report but does not flag any regions yet.
"""

from __future__ import annotations


def samples_to_tc(sample_index: int, sample_rate: int, fps: float) -> str:
    """Convert a sample index to an HH:MM:SS:FF timecode string.

    Uses simple (non-drop-frame) arithmetic. Drop-frame handling lands with
    Phase 6 / AAF export when we actually need framerate-correct markers.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    seconds = sample_index / float(sample_rate)
    total_frames = int(round(seconds * fps))
    frames_per_hour = int(round(fps * 3600))
    frames_per_minute = int(round(fps * 60))
    frames_per_second = int(round(fps))

    hh = total_frames // frames_per_hour
    rem = total_frames - hh * frames_per_hour
    mm = rem // frames_per_minute
    rem -= mm * frames_per_minute
    ss = rem // frames_per_second
    ff = rem - ss * frames_per_second
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
