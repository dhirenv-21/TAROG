"""Registered predictors and statistics for the appearance audit (prereg §5, §7, §8).

Predictors: logistic regressions on feature sets T (elapsed time), A (appearance), T+A.
Positive class = TAKEOVER, negative = GAP. All development metrics use leave-one-sequence-out
out-of-fold scores; uncertainty by a cluster bootstrap over tracks within (sequence, detector) units.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES: dict[str, list[str]] = {
    "T": ["dt", "log_dt"],
    "A": ["a_glob", "a_loc", "a_loc_missing", "a_glob_missing"],
}
FEATURES["T+A"] = FEATURES["T"] + FEATURES["A"]
GAP_RETENTION_Q = 0.90   # theta = 90th percentile of training GAP scores (<= 10% of gaps rejected)


def feature_matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    """Build the registered features; undefined appearance cues are imputed 1.0 + indicator."""
    cols = []
    for n in names:
        if n == "dt":
            cols.append([r["dt"] for r in rows])
        elif n == "log_dt":
            cols.append([math.log(r["dt"]) for r in rows])
        elif n in ("a_glob", "a_loc"):
            cols.append([1.0 if math.isnan(r[n]) else r[n] for r in rows])
        elif n == "a_loc_missing":
            cols.append([float(math.isnan(r["a_loc"])) for r in rows])
        elif n == "a_glob_missing":
            cols.append([float(math.isnan(r["a_glob"])) for r in rows])
        else:
            raise KeyError(n)
    return np.asarray(cols, dtype=float).T


def make_model():
    """The single registered model family."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))


def fit(rows: Sequence[dict[str, Any]], y: np.ndarray, fs: str):
    """Fit one predictor; returns (model, theta). A constant column is harmless to the scaler."""
    X = feature_matrix(rows, FEATURES[fs])
    m = make_model().fit(X, y)
    s = m.predict_proba(X)[:, 1]
    theta = float(np.quantile(s[y == 0], GAP_RETENTION_Q)) if (y == 0).any() else 0.5
    return m, theta


def score(model, rows: Sequence[dict[str, Any]], fs: str) -> np.ndarray:
    """P(TAKEOVER) from a fitted predictor."""
    if not len(rows):
        return np.zeros(0)
    return model.predict_proba(feature_matrix(rows, FEATURES[fs]))[:, 1]


def loso_scores(rows: list[dict[str, Any]], y: np.ndarray, seqs: Sequence[str], fs: str,
                fit_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-sequence-out out-of-fold scores and per-row thresholds (theta of the training fold).

    Models are fitted on rows of the other sequences with ``fit_mask`` True (default: all rows);
    every row of the held-out sequence is scored. ``y`` is only read where ``fit_mask`` is True.
    """
    s = np.full(len(rows), np.nan)
    th = np.full(len(rows), np.nan)
    seq_of = np.array([r["seq"] for r in rows])
    fm = np.ones(len(rows), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    for q in seqs:
        te = seq_of == q
        tr = (~te) & fm
        if not te.any():
            continue
        if len(set(y[tr])) < 2:
            raise ValueError(f"training fold without both classes when holding out {q}")
        m, theta = fit([rows[i] for i in np.where(tr)[0]], y[tr], fs)
        s[te] = score(m, [rows[i] for i in np.where(te)[0]], fs)
        th[te] = theta
    return s, th


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC via the rank-sum statistic (ties get average ranks); NaN if one class is absent."""
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = np.asarray(s)[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision (step-wise PR-AUC; scikit-learn's definition, tied scores grouped)."""
    from sklearn.metrics import average_precision_score
    if (np.asarray(y) == 1).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def brier(y: np.ndarray, s: np.ndarray) -> float:
    """Mean squared error of P(TAKEOVER)."""
    return float(np.mean((np.asarray(s) - np.asarray(y)) ** 2))


def operating_point(y: np.ndarray, s: np.ndarray, th: np.ndarray) -> dict[str, float]:
    """At per-row thresholds th: gap rejection (GAP scored >= th) and takeover retention/recall."""
    flag = s >= th
    g = y == 0
    t = y == 1
    rej = float(flag[g].mean()) if g.any() else float("nan")
    rec = float(flag[t].mean()) if t.any() else float("nan")
    return {"gap_rejection": rej, "takeover_recall": rec, "takeover_retention": 1 - rec if t.any() else float("nan")}


def cluster_index(units: Sequence[str], tracks: Sequence[int]) -> dict[str, list[np.ndarray]]:
    """unit -> list of row-index arrays, one per track cluster."""
    by: dict[str, dict[int, list[int]]] = {}
    for i, (u, t) in enumerate(zip(units, tracks)):
        by.setdefault(u, {}).setdefault(t, []).append(i)
    return {u: [np.asarray(v) for v in d.values()] for u, d in by.items()}


def bootstrap_delta_auc(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, clusters: dict[str, list[np.ndarray]],
                        rng: np.random.Generator, B: int = 2000) -> np.ndarray:
    """B bootstrap draws of AUC(s_a) - AUC(s_b), resampling track clusters within each unit."""
    out = np.empty(B)
    units = list(clusters)
    for b in range(B):
        idx = []
        for u in units:
            cl = clusters[u]
            pick = rng.integers(0, len(cl), len(cl))
            idx.extend(cl[k] for k in pick)
        ii = np.concatenate(idx)
        out[b] = auc(y[ii], s_a[ii]) - auc(y[ii], s_b[ii])
    return out


def interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile interval of finite draws."""
    d = draws[np.isfinite(draws)]
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def incident_at(incs: list[dict[str, Any]], rows: list[dict[str, Any]], tau: float) -> list[tuple[int, int]]:
    """(incident index, row index) of each GAP/TAKEOVER incident's first row with dt >= tau."""
    out = []
    for k, inc in enumerate(incs):
        if inc["label"] not in ("GAP", "TAKEOVER"):
            continue
        for i in inc["rows"]:
            if rows[i]["dt"] >= tau - 1e-9:
                out.append((k, i))
                break
    return out
