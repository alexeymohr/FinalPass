"""Fit the T-inhale score from labelled onset-feature rows. Numbers only.

Input: a JSON list of rows, one per labelled breath, each carrying every name in
FEATURES plus ``pos`` (operator heard a T-inhale) and ``group`` (the chapter, for
held-out testing). Rows are built locally from the operator's labels; no audio
is read here.

Model: logistic regression on standardised features with a light L2 penalty
(Newton steps, numpy only). Reported quality is leave-one-group-out AUC. The flag
threshold is set so the fitted score flags as many training breaths as the
operator labelled — the same rule the earlier versions used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

FEATURES = ["burst_rise_db", "closure_below_body_db", "context_rel_db", "gap_before_ms",
            "burst_width_ms", "burst_above_4k", "burst_low_rise_db"]
L2 = 1.0


def fit(X: np.ndarray, y: np.ndarray, l2: float = L2) -> np.ndarray:
    X1 = np.c_[np.ones(len(X)), X]
    w = np.zeros(X1.shape[1])
    pen = np.r_[0.0, np.full(X.shape[1], l2)]
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-X1 @ w))
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(H, X1.T @ (p - y) + pen * w)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    p, n = score[y == 1], score[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def held_out_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-12
        w = fit((X[~te] - mu) / sd, y[~te])
        s[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ w
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--trained-on", required=True, help="provenance note (no client identifiers)")
    a = ap.parse_args()

    rows = json.loads(Path(a.rows).read_text())
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([1.0 if r["pos"] else 0.0 for r in rows])
    groups = np.array([str(r["group"]) for r in rows])
    held = held_out_scores(X, y, groups)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    w = fit((X - mu) / sd, y)
    score = np.c_[np.ones(len(y)), (X - mu) / sd] @ w
    threshold = float(np.sort(score)[-int(y.sum())])
    model = {
        "name": "t_inhale_score", "version": a.version, "created": dt.date.today().isoformat(),
        "trained_on": a.trained_on, "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
        "held_out_auc_leave_one_group_out": round(auc(held, y), 3),
        "features": FEATURES, "gap_before_ms_is_log10": True, "l2": L2,
        "mean": mu.tolist(), "std": sd.tolist(), "intercept": float(w[0]),
        "weights": w[1:].tolist(), "flag_threshold": threshold,
    }
    Path(a.out).write_text(json.dumps(model, indent=1) + "\n")
    print(json.dumps({k: model[k] for k in ("positives", "negatives", "held_out_auc_leave_one_group_out",
                                            "flag_threshold")}))


if __name__ == "__main__":
    main()
