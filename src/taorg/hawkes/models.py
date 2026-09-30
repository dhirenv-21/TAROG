"""Stage 0 models: HK0 (base rates), DUR (duration hazard), HK1 (within-relation), HK2 (multivariate).

Hawkes rows are fitted independently: each row k maximises the concave
    sum_{e: type k} log(mu_k + alpha_k . x_e) - mu_k T_k - alpha_k . I_k - pen ||alpha_k||^2
subject to mu_k >= mu_min, alpha_k >= 0 on the model's support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize as so

from taorg.hawkes.dataset import HawkesData, type_totals
from taorg.hawkes.vidor import PairStream


def support(model: str, K: int) -> np.ndarray:
    """Boolean mask (K, K+1) of allowed alpha entries."""
    M = np.zeros((K, K + 1), dtype=bool)
    if model == "HK0":
        return M
    if model == "HK1":
        for k in range(K):
            r = k // 2
            M[k, [2 * r, 2 * r + 1, K]] = True
        return M
    if model == "HK2":
        M[:] = True
        return M
    raise ValueError(model)


@dataclass
class HawkesParams:
    """mu (K,), A (K, K+1), beta."""

    mu: np.ndarray
    A: np.ndarray
    beta: float


def fit(d: HawkesData, videos: np.ndarray, model: str, penalty: float, mu_min: float) -> HawkesParams:
    """Penalised MLE on a subset of videos."""
    K = d.K
    T, I = type_totals(d, videos)
    sel = np.isin(d.ev_video, videos)
    mask = support(model, K)
    mu = np.zeros(K)
    A = np.zeros((K, K + 1))
    for k in range(K):
        rows = np.where(sel & (d.ev_type == k))[0]
        n = len(rows)
        cols = np.where(mask[k])[0]
        if n == 0 or T[k] <= 0:
            mu[k] = mu_min
            continue
        if len(cols) == 0:
            mu[k] = max(n / T[k], mu_min)
            continue
        X = d.X[rows][:, cols].toarray()
        Ik, Tk = I[k, cols], T[k]

        def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
            m, a = theta[0], theta[1:]
            lam = m + X @ a
            inv = 1.0 / lam
            val = -np.log(lam).sum() + m * Tk + a @ Ik + penalty * a @ a
            g = np.concatenate([[-inv.sum() + Tk], -(X.T @ inv) + Ik + 2 * penalty * a])
            return val, g
        x0 = np.concatenate([[max(n / Tk, mu_min)], np.zeros(len(cols))])
        res = so.minimize(f, x0, jac=True, method="L-BFGS-B",
                          bounds=[(mu_min, None)] + [(0.0, None)] * len(cols),
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        mu[k] = res.x[0]
        A[k, cols] = res.x[1:]
    return HawkesParams(mu, A, d.beta)


def ll_per_video(d: HawkesData, p: HawkesParams) -> np.ndarray:
    """Held-out log-likelihood contribution of every video (events + compensator)."""
    assert abs(d.beta - p.beta) < 1e-12, "features and parameters use different beta"
    K, R = d.K, d.R
    out = np.zeros(d.n_videos)
    for k in range(K):
        rows = np.where(d.ev_type == k)[0]
        if len(rows) == 0:
            continue
        lam = p.mu[k] + d.X[rows] @ p.A[k]
        np.add.at(out, d.ev_video[rows], np.log(lam))
    on = np.arange(0, K, 2)
    off = on + 1
    out -= d.TW * p.mu[on].sum() + d.J @ p.A[on].sum(axis=0)
    for r in range(R):
        out -= d.dT[:, r] * p.mu[2 * r] + d.dJ[r] @ p.A[2 * r]
        out -= d.Toff[:, r] * p.mu[2 * r + 1] + d.Ioff[r] @ p.A[2 * r + 1]
    return out


# ---------------------------------------------------------------------------
# DUR: onset Poisson (as HK0), offset piecewise-constant hazard in time since onset
# ---------------------------------------------------------------------------

@dataclass
class DurStats:
    """Per-video exposure / event counts for the duration model."""

    on_n: np.ndarray        # (V, R) onset events
    on_T: np.ndarray        # (V, R) onset risk time
    age_n: np.ndarray       # (V, R, B+1) offsets per age bin (last index = unknown age)
    age_T: np.ndarray       # (V, R, B+1) exposure per age bin


def dur_stats(streams: list[list[PairStream]], R: int, bins: list[float], d: HawkesData) -> DurStats:
    """Exposure in age bins; unknown-age intervals (left-censored) use the extra bin."""
    V = len(streams)
    B = len(bins)
    edges = list(bins) + [np.inf]
    on_n = np.zeros((V, R))
    age_n = np.zeros((V, R, B + 1))
    age_T = np.zeros((V, R, B + 1))
    for vi, ps_list in enumerate(streams):
        for ps in ps_list:
            for t, typ in ps.events:
                if typ < 2 * R and typ % 2 == 0:
                    on_n[vi, typ // 2] += 1
            for r, dur, known, observed in ps.ages:
                if not known:
                    age_T[vi, r, B] += dur
                    age_n[vi, r, B] += observed
                    continue
                for bi in range(B):
                    lo, hi = edges[bi], edges[bi + 1]
                    age_T[vi, r, bi] += max(0.0, min(dur, hi) - lo)
                    if observed and lo <= dur < hi:
                        age_n[vi, r, bi] += 1
    on_T = d.TW[:, None] + d.dT
    return DurStats(on_n, on_T, age_n, age_T)


def dur_fit(s: DurStats, videos: np.ndarray, rate_min: float) -> tuple[np.ndarray, np.ndarray]:
    """MLE rates: onset (R,), offset hazard (R, B+1)."""
    on = np.maximum(s.on_n[videos].sum(0) / np.maximum(s.on_T[videos].sum(0), 1e-300), rate_min)
    hz = np.maximum(s.age_n[videos].sum(0) / np.maximum(s.age_T[videos].sum(0), 1e-300), rate_min)
    return on, hz


def dur_ll_per_video(s: DurStats, on: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """Per-video log-likelihood of the duration model."""
    return ((s.on_n * np.log(on) - s.on_T * on).sum(1)
            + (s.age_n * np.log(hz) - s.age_T * hz).sum((1, 2)))
