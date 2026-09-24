"""Compare FinalPass detector output with a human QC comment export.

Inputs are numbers and the QC operator's comments only — no audio is opened,
no manuscript is read. For every QC comment category and every detector:

* hit rate: fraction of QC comments with a detector event within +-tol seconds;
* chance: the same rate after circularly shifting each file's detector events
  by random offsets (keeps each detector's count and clustering per file, breaks
  any real alignment) — so a detector that fires everywhere cannot look good;
* for the frame-quality model, whether the score is lower at QC-marked times
  than at random times in the same file, independent of any threshold.

The QC operator's own edits (added pauses, removed breaths) shift their timeline
against the submitted audio, so several tolerances are reported rather than one.
"""
from __future__ import annotations

import argparse, csv, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOLERANCES = (1.0, 2.0, 5.0)
SHUFFLES = 500
SEED = 20260924
FRAME_SECONDS = 0.02


def category(text: str) -> str:
    t = text.lower()
    if "room tone" in t:
        return "room_tone"
    if "artifact" in t or "rmoved and fixed" in t:
        return "artifact"
    if "breath" in t:
        return "breath"
    if "typo" in t or "1884" in t or "asterix" in t:
        return "text"
    if ("chapter heading" in t or "chapter break" in t or "0.5-second" in t
            or "0.5 sec" in t or "0.5s" in t or "chapter opening" in t
            or "intro template" in t or "file needs to be removed" in t):
        return "structure"
    if "intonation" in t or "expression" in t:
        return "intonation"
    if "pause" in t or "pace" in t:
        return "pacing"
    return "other"


def parse_ts(s: str) -> float:
    h, m, rest = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def file_for_chapter(name: str, files: list[str]) -> str | None:
    if name.strip().lower().startswith("intro"):
        want = "000_"
    else:
        want = f"{int(name):03d}_"
    hits = [f for f in files if f.startswith(want)]
    return hits[0] if len(hits) == 1 else None


def hit(t: float, events: list[tuple[float, float]], tol: float) -> bool:
    return any(s - tol <= t <= e + tol for s, e in events)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True)
    ap.add_argument("--boundaries", required=True, help="finalpass boundaries source-report.json")
    ap.add_argument("--sqa", required=True, help="sqa_eval scan output dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # --- detectors -------------------------------------------------------
    report = json.loads(Path(a.boundaries).read_text())
    durations: dict[str, float] = {}
    detectors: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for asset in report["assets"]:
        name = Path(asset["path"]).name
        sr = asset["sample_rate"]
        durations[name] = asset["duration_seconds"]
        for f in asset.get("findings") or []:
            s = f["start_sample"] / sr
            e = (f["end_sample"] if f.get("end_sample") is not None else f["start_sample"]) / sr
            detectors[f"boundaries:{f['kind']}"][name].append((s, e))
            detectors["boundaries:any"][name].append((s, e))

    sqa_dir = Path(a.sqa)
    for c in json.loads((sqa_dir / "candidates.json").read_text()):
        detectors["sqa:low_frame_quality"][c["source_id"]].append(
            (c["start_sample"] / c["sample_rate"], c["end_sample"] / c["sample_rate"]))
    files = sorted(durations)

    # --- QC comments -----------------------------------------------------
    rows = list(csv.DictReader(open(a.qc, newline="", encoding="utf-8-sig")))
    qc, unmapped = [], []
    for r in rows:
        f = file_for_chapter(r["chapter_name"], files)
        if f is None:
            unmapped.append(r["chapter_name"])
            continue
        qc.append({"file": f, "t": parse_ts(r["timeline_timestamp"]),
                   "category": category(r["comment_content"]),
                   "text": r["comment_content"].strip()})
    cats = Counter(q["category"] for q in qc)

    # --- hit rates vs chance --------------------------------------------
    rng = random.Random(SEED)
    results = []
    for det, by_file in sorted(detectors.items()):
        n_events = sum(len(v) for v in by_file.values())
        for cat in sorted(cats):
            items = [q for q in qc if q["category"] == cat]
            for tol in TOLERANCES:
                observed = sum(hit(q["t"], by_file.get(q["file"], []), tol) for q in items)
                chance = []
                for _ in range(SHUFFLES):
                    shifted = {}
                    for f, evs in by_file.items():
                        d = durations[f]
                        off = rng.uniform(0, d)
                        shifted[f] = [((s + off) % d, (s + off) % d + (e - s)) for s, e in evs]
                    chance.append(sum(hit(q["t"], shifted.get(q["file"], []), tol) for q in items))
                chance = np.array(chance)
                results.append({
                    "detector": det, "detector_events": n_events, "qc_category": cat,
                    "qc_comments": len(items), "tolerance_s": tol,
                    "hits": int(observed),
                    "hit_rate": round(observed / len(items), 4) if items else None,
                    "chance_hits_mean": round(float(chance.mean()), 2),
                    "chance_hit_rate": round(float(chance.mean()) / len(items), 4) if items else None,
                    "p_chance_at_least_observed": round(float((chance >= observed).mean()), 4),
                })

    # --- frame-quality score at QC times vs random times -----------------
    score_rows = []
    npy = {p.stem: np.load(p) for p in (sqa_dir / "frame_scores").glob("*.npy")}
    srng = np.random.default_rng(SEED)
    for cat in sorted(cats):
        items = [q for q in qc if q["category"] == cat]
        for tol in TOLERANCES:
            half = int(round(tol / FRAME_SECONDS))
            at_qc, at_rand = [], []
            for q in items:
                s = npy.get(Path(q["file"]).stem)
                if s is None:
                    continue
                c = int(q["t"] / FRAME_SECONDS)
                if c >= len(s):
                    continue
                at_qc.append(float(s[max(0, c - half):c + half + 1].min()))
                for _ in range(20):
                    r = int(srng.integers(0, len(s)))
                    at_rand.append(float(s[max(0, r - half):r + half + 1].min()))
            if not at_qc:
                continue
            at_qc_a, at_rand_a = np.array(at_qc), np.array(at_rand)
            # Probability a QC window's minimum is lower than a random window's
            # (0.5 = no relationship; 1.0 = QC spots always score lower).
            auc = float(np.mean(at_qc_a[:, None] < at_rand_a[None, :]))
            score_rows.append({
                "qc_category": cat, "tolerance_s": tol, "n": len(at_qc),
                "median_min_score_at_qc": round(float(np.median(at_qc_a)), 4),
                "median_min_score_random": round(float(np.median(at_rand_a)), 4),
                "p_qc_window_lower_than_random": round(auc, 4),
            })

    out = {
        "qc_comments_total": len(rows), "qc_comments_mapped": len(qc),
        "unmapped_chapters": sorted(set(unmapped)),
        "qc_categories": dict(cats),
        "detector_event_counts": {d: sum(len(v) for v in bf.values()) for d, bf in detectors.items()},
        "hit_rates": results,
        "sqa_score_at_qc_times": score_rows,
        "shuffles": SHUFFLES, "seed": SEED,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("qc_comments_total", "qc_comments_mapped",
                                          "unmapped_chapters", "qc_categories",
                                          "detector_event_counts")}, indent=1))


if __name__ == "__main__":
    main()
