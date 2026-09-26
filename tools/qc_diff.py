"""Sample-exact diff of a submitted audiobook against its post-QC version.

Where the QC operator left audio untouched, the two renders are bit-identical,
so every edit — a pause added or trimmed, a breath or artefact cut out, a block
regenerated — can be located to the sample without transcription or listening.
Each QC comment is then tied to the edit it describes, which turns a drifting
timeline timestamp into an exact span in the SUBMITTED file.

How: index the submitted file in fixed blocks, find every block's exact
occurrence in the post-QC file with a shift-invariant window key (verified
sample-for-sample), keep the longest order-preserving chain of matches, grow
each matched run outward sample by sample, and call everything between runs an
edit.

Output is sample positions, durations, levels and counts — never audio.
"""
from __future__ import annotations

import argparse, bisect, csv, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_compare import category, file_for_chapter, parse_ts  # noqa: E402

BLOCK = 4096
MIX1 = np.uint64(0x9E3779B97F4A7C15)
MIX2 = np.uint64(0xC2B2AE3D27D4EB4F)
MATCH_TOLERANCE_S = 1.0


def _window_stats(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift-invariant key and nonzero count for every length-k window."""
    v = x.astype(np.int64)
    c1 = np.concatenate(([0], np.cumsum(v)))
    c2 = np.concatenate(([0], np.cumsum(v * v)))
    cz = np.concatenate(([0], np.cumsum(v != 0)))
    s1 = (c1[k:] - c1[:-k]).astype(np.uint64)
    s2 = (c2[k:] - c2[:-k]).astype(np.uint64)
    return (s1 * MIX1) ^ (s2 * MIX2), cz[k:] - cz[:-k]


def _anchors(b: np.ndarray, a: np.ndarray) -> list[tuple[int, int]]:
    """Verified (before_pos, after_pos) pairs for distinctive blocks."""
    if len(b) < BLOCK or len(a) < BLOCK:
        return []
    kb, nzb = _window_stats(b, BLOCK)
    starts = np.arange(0, len(b) - BLOCK + 1, BLOCK)
    # Mostly-silent blocks match anywhere; only distinctive audio anchors.
    starts = starts[nzb[starts] >= BLOCK // 2]
    index: dict[int, list[int]] = {}
    for s in starts:
        index.setdefault(int(kb[s]), []).append(int(s))
    ka, nza = _window_stats(a, BLOCK)
    keys = np.fromiter(index.keys(), dtype=np.uint64, count=len(index))
    hits = np.flatnonzero(np.isin(ka, keys) & (nza >= BLOCK // 2))
    out = []
    for j in hits:
        for i in index[int(ka[j])]:
            if np.array_equal(b[i:i + BLOCK], a[j:j + BLOCK]):
                out.append((i, int(j)))
    return out


def _longest_chain(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain increasing in both coordinates (no reordering edits)."""
    pairs = sorted(set(pairs))
    tails: list[int] = []
    tail_idx: list[int] = []
    prev = [-1] * len(pairs)
    for n, (_, j) in enumerate(pairs):
        k = bisect.bisect_left(tails, j)
        if k == len(tails):
            tails.append(j); tail_idx.append(n)
        else:
            tails[k] = j; tail_idx[k] = n
        prev[n] = tail_idx[k - 1] if k else -1
    chain, n = [], tail_idx[-1] if tail_idx else -1
    while n != -1:
        chain.append(pairs[n]); n = prev[n]
    return chain[::-1]


def _first_mismatch(x: np.ndarray, y: np.ndarray) -> int:
    d = np.flatnonzero(x != y)
    return int(d[0]) if d.size else len(x)


@dataclass
class Edit:
    before_start: int
    before_end: int
    after_start: int
    after_end: int
    kind: str
    before_peak_dbfs: float | None
    before_rms_dbfs: float | None
    before_zero_fraction: float | None
    after_peak_dbfs: float | None
    after_zero_fraction: float | None


def _level(x: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if x.size == 0:
        return None, None, None
    f = x.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(f)))
    rms = float(np.sqrt(np.mean(f * f)))
    db = lambda v: round(20 * np.log10(v), 2) if v > 0 else -200.0
    return db(peak), db(rms), round(float(np.mean(x == 0)), 4)


def _classify(bspan: np.ndarray, aspan: np.ndarray) -> str:
    def silent(x):
        return x.size and np.mean(x == 0) >= 0.98
    if bspan.size == 0:
        return "pause_added" if silent(aspan) else "audio_inserted"
    if aspan.size == 0:
        return "pause_shortened" if silent(bspan) else "audio_removed"
    if silent(bspan) and silent(aspan):
        return "pause_changed"
    if silent(aspan):
        return "audio_to_silence"   # cut out, gap kept: breaths, short artefacts
    if silent(bspan):
        return "silence_to_audio"
    return "audio_rerendered"       # a block regenerated


def diff_file(b: np.ndarray, a: np.ndarray) -> tuple[list[Edit], dict]:
    chain = _longest_chain(_anchors(b, a))
    # Group anchors into runs sharing one offset, then grow each run outward.
    runs: list[list[int]] = []
    for i, j in chain:
        if runs and j - i == runs[-1][2] and i == runs[-1][1]:
            runs[-1][1] = i + BLOCK
        else:
            runs.append([i, i + BLOCK, j - i])
    regions: list[tuple[int, int, int]] = []
    for bs, be, off in runs:
        lo_b = regions[-1][1] if regions else 0
        lo_a = (regions[-1][1] + regions[-1][2]) if regions else 0
        # backward
        span = min(bs - lo_b, bs + off - lo_a)
        if span > 0:
            xb, xa = b[bs - span:bs][::-1], a[bs + off - span:bs + off][::-1]
            bs -= _first_mismatch(xb, xa)
        # forward (bounded by the file ends; the next run bounds itself backward)
        span = min(len(b) - be, len(a) - (be + off))
        if span > 0:
            be += _first_mismatch(b[be:be + span], a[be + off:be + off + span])
        # A run whose first block starts inside silence can overlap the previous
        # run on either side; clamp on both so no sample is counted twice.
        if regions:
            prev_b_end = regions[-1][1]
            prev_a_end = regions[-1][1] + regions[-1][2]
            bs = max(bs, prev_b_end, prev_a_end - off)
        if bs < be:
            regions.append((bs, be, off))

    edits: list[Edit] = []
    pb = pa = 0
    for bs, be, off in regions + [(len(b), len(b), len(a) - len(b))]:
        as_ = bs + off
        if bs > pb or as_ > pa:
            bspan, aspan = b[pb:bs], a[pa:max(pa, as_)]
            bp, br, bz = _level(bspan)
            ap, _, az = _level(aspan)
            edits.append(Edit(pb, bs, pa, max(pa, as_), _classify(bspan, aspan),
                              bp, br, bz, ap, az))
        pb, pa = be, be + off
    matched = sum(be - bs for bs, be, _ in regions)
    stats = {"before_samples": len(b), "after_samples": len(a),
             "matched_samples": matched,
             "matched_fraction_of_before": round(matched / len(b), 6) if len(b) else None,
             "anchors": len(chain), "regions": len(regions), "edits": len(edits)}
    return edits, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="submitted chapter folder")
    ap.add_argument("--after", required=True, help="post-QC chapter folder")
    ap.add_argument("--qc", required=True, help="QC comment CSV export")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    before = sorted(Path(a.before).glob("*.wav"))
    after = sorted(Path(a.after).glob("*.wav"))
    if len(before) != len(after):
        raise SystemExit(f"{len(before)} submitted files vs {len(after)} post-QC files")

    inventory, sr_by = {}, {}
    for pb, pa in zip(before, after):
        xb, sr = sf.read(str(pb), dtype="int16", always_2d=False)
        xa, sra = sf.read(str(pa), dtype="int16", always_2d=False)
        if sr != sra or xb.ndim != 1 or xa.ndim != 1:
            raise SystemExit(f"{pb.name}: format mismatch or not mono")
        edits, stats = diff_file(xb, xa)
        inventory[pb.name] = {"after_file": pa.name, "sample_rate": sr, **stats,
                              "edits": [asdict(e) for e in edits]}
        sr_by[pb.name] = sr
        print(f"  {pb.name[:3]}: {stats['edits']:>3} edits, "
              f"{stats['matched_fraction_of_before']:.2%} of submitted audio unchanged", flush=True)
    (out / "edit_inventory.json").write_text(json.dumps(inventory, indent=1))

    names = [p.name for p in before]
    rows = list(csv.DictReader(open(a.qc, newline="", encoding="utf-8-sig")))
    labels = []
    for r in rows:
        f = file_for_chapter(r["chapter_name"], names)
        sr = sr_by[f]
        t = parse_ts(r["timeline_timestamp"])
        best, dist = None, None
        for n, e in enumerate(inventory[f]["edits"]):
            s, en = e["after_start"] / sr, e["after_end"] / sr
            d = 0.0 if s <= t <= en else min(abs(t - s), abs(t - en))
            if dist is None or d < dist:
                best, dist = n, d
        e = inventory[f]["edits"][best] if best is not None else None
        ok = e is not None and dist <= MATCH_TOLERANCE_S
        labels.append({
            "file": f, "qc_time_after_s": t, "qc_category": category(r["comment_content"]),
            "comment": r["comment_content"].strip(),
            "matched": ok, "distance_to_edit_s": round(dist, 3) if dist is not None else None,
            "edit_index": best if ok else None,
            "edit_kind": e["kind"] if ok else None,
            "before_start_sample": e["before_start"] if ok else None,
            "before_end_sample": e["before_end"] if ok else None,
            "before_start_s": round(e["before_start"] / sr, 3) if ok else None,
            "before_duration_s": round((e["before_end"] - e["before_start"]) / sr, 3) if ok else None,
            "after_duration_s": round((e["after_end"] - e["after_start"]) / sr, 3) if ok else None,
            "sample_rate": sr,
        })
    (out / "qc_labels_exact.json").write_text(json.dumps(labels, indent=1))
    print(f"\nQC comments tied to an edit within {MATCH_TOLERANCE_S}s: "
          f"{sum(l['matched'] for l in labels)}/{len(labels)}")


if __name__ == "__main__":
    main()
