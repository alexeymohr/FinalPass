"""Emit exact TTS clip-boundary positions per file (FinalPass .venv).

Run separately from the scan: FinalPass depends on pydantic and the ML
evaluation environment deliberately does not. Reuses the shipped `boundaries`
segmentation rather than reimplementing zero-run detection, so the scan's
interior-vs-boundary split uses FinalPass's own segmentation, not a second
opinion about where clips start and end.

Output is numeric only: sample rate, total samples, digital-black gap spans and
clip count per file.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from finalpass.audio_io import read_wav
from finalpass.boundary_check import analyze_boundaries


def main() -> None:
    src_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = {}
    for p in sorted(glob.glob(str(src_dir / "*.wav"))):
        audio = read_wav(Path(p))
        analysis = analyze_boundaries(audio)
        gaps = [(g.start_sample, g.end_sample) for g in analysis.gaps]
        rows[os.path.basename(p)] = {
            "sample_rate": audio.sample_rate,
            "total_samples": int(audio.data.shape[0]),
            "gap_spans": gaps,
            "clip_count": len(analysis.clips),
        }
        print(f"  {os.path.basename(p)}: {len(gaps)} digital-black gaps, {len(analysis.clips)} clips",
              flush=True)
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
