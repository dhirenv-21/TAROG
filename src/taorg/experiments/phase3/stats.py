"""Statistical utilities for Phase 3 (descriptive, paired, frontier).

Conventions (pre-registered, research/phase3_preregistration.md §7):
    - The statistical unit is the evaluation SEED (one synthetic video).
      Per-seed values are the frozen per-sequence metric values.
    - Descriptive: mean, sample SD, n, and a two-sided 95% t-interval.
    - Paired differences (A - B on identical instances): mean, SD, percentile
      bootstrap 95% CI (seeded), Cohen's d_z = mean/SD, and a two-sided
      Wilcoxon signed-rank p-value (zeros dropped; all-zero -> p = 1).
    - Multiplicity: Holm step-down over the pre-registered family.
    - Seeds with NaN in either arm are dropped pairwise (count reported).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats as sps


def descriptive(values: Sequence[float], ci: float = 0.95) -> dict[str, float]:
    """Mean, SD, n and t-interval of the non-NaN values."""
    x = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    n = int(x.size)
    if n == 0:
        return {"mean": math.nan, "sd": math.nan, "n": 0, "ci_lo": math.nan, "ci_hi": math.nan}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "sd": math.nan, "n": 1, "ci_lo": math.nan, "ci_hi": math.nan}
    sd = float(x.std(ddof=1))
    half = float(sps.t.ppf(0.5 + ci / 2, n - 1)) * sd / math.sqrt(n)
    return {"mean": mean, "sd": sd, "n": n, "ci_lo": mean - half, "ci_hi": mean + half}


def bootstrap_mean_ci(
    x: np.ndarray, n_resamples: int, ci: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of x."""
    n = x.size
    if n == 0:
        return math.nan, math.nan
    if np.all(x == x[0]):
        return float(x[0]), float(x[0])
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.5 - ci / 2, 0.5 + ci / 2])
    return float(lo), float(hi)


def wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if all differences are zero."""
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0
    return float(sps.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired comparison of arm A vs arm B (difference = A - B).

    Args:
        a: Per-seed values of arm A.
        b: Per-seed values of arm B, aligned with ``a``.
        n_resamples: Bootstrap resamples.
        ci: Confidence level.
        rng: Seeded generator.

    Returns:
        Dict with n, n_dropped, mean_a, mean_b, mean_diff, sd_diff, ci_lo,
        ci_hi, dz, p_wilcoxon.
    """
    if len(a) != len(b):
        raise ValueError("paired_comparison: arms must have equal length")
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    keep = ~(np.isnan(A) | np.isnan(B))
    A, B = A[keep], B[keep]
    d = A - B
    n = int(d.size)
    out = {"n": n, "n_dropped": int((~keep).sum())}
    if n == 0:
        out.update(mean_a=math.nan, mean_b=math.nan, mean_diff=math.nan, sd_diff=math.nan,
                   ci_lo=math.nan, ci_hi=math.nan, dz=math.nan, p_wilcoxon=math.nan)
        return out
    sd = float(d.std(ddof=1)) if n > 1 else math.nan
    mean = float(d.mean())
    if n > 1 and sd > 0:
        dz = mean / sd
    elif mean == 0:
        dz = 0.0
    else:
        dz = math.copysign(math.inf, mean)
    lo, hi = bootstrap_mean_ci(d, n_resamples, ci, rng)
    out.update(mean_a=float(A.mean()), mean_b=float(B.mean()), mean_diff=mean, sd_diff=sd,
               ci_lo=lo, ci_hi=hi, dz=dz, p_wilcoxon=wilcoxon_p(d))
    return out


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN entries are passed through)."""
    p = np.asarray(pvals, dtype=float)
    valid = np.where(~np.isnan(p))[0]
    adj = np.full(p.shape, np.nan)
    m = valid.size
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def hypervolume_2d(
    points: Sequence[tuple[float, float]], cost_min: float, cost_ref: float
) -> float:
    """Normalized dominated hypervolume for (quality up, cost down) points.

    The region dominated by the point set inside the box
    ``quality in [0, 1]`` x ``cost in [cost_min, cost_ref]`` is measured and
    divided by the box area ``(cost_ref - cost_min)``.  Points with cost >=
    cost_ref or NaN coordinates contribute nothing; costs below cost_min are
    clipped to cost_min; quality is clipped to [0, 1].

    Args:
        points: (quality, cost) pairs, e.g. (Persistence F1, termination delay).
        cost_min: Best achievable cost (lower box edge).
        cost_ref: Reference (worst) cost.

    Returns:
        Hypervolume in [0, 1].
    """
    if cost_ref <= cost_min:
        raise ValueError("cost_ref must exceed cost_min")
    pts = sorted(
        (max(cost_min, c), min(1.0, max(0.0, q)))
        for q, c in points
        if not (math.isnan(q) or math.isnan(c)) and c < cost_ref
    )
    area = 0.0
    best_q = 0.0
    for i, (c, q) in enumerate(pts):
        best_q = max(best_q, q)
        c_next = pts[i + 1][0] if i + 1 < len(pts) else cost_ref
        area += (c_next - c) * best_q
    return area / (cost_ref - cost_min)
