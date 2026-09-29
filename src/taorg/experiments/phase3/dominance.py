"""Formal Pareto-dominance classification of frontiers (EXPLORATORY, not H1).

Defined and committed AFTER registration but BEFORE any evaluation data
existed (see research/phase3_research_log.md).  It is NOT part of the
pre-registered H1 rule and can never change the H1 outcome; it exists so that
"Pareto-dominates" is used in the paper only in this exact sense.

Definitions (quality q = mean Persistence F1 up, cost c = mean termination
delay down, both over evaluation seeds; points are each method's CALIBRATION
Pareto set evaluated on evaluation seeds):

    Box B = {c < hv_delay_ref} (the region measured by the registered
    hypervolume; points outside B contribute nothing to either method).

    Point dominance:  a >= b  iff  q_a >= q_b and c_a <= c_b.
    Frontier set dominance G >> M (within B):
        every point of M in B is weakly dominated by some point of G in B,
        and at least one point of M in B is strictly dominated
        (q_g > q_m or c_g < c_m) by some point of G.
        If M has no point in B while G has one, G >> M.

    Uncertainty: the same test on per-seed bootstrap means (paired resampling
    of evaluation seeds, shared across methods); ``p_dom`` = fraction of
    resamples in which G >> M holds.

Classification of the treatment against the simpler methods U over all
(condition, gap) pairs of the H1 conditions:
    A  "Pareto-dominates"            G >> m for every m in U at every
                                      (condition, gap), point estimate, AND
                                      p_dom >= 0.95 everywhere.
    B  "condition-specific"          G >> m for every m in U in at least one
                                      (condition, gap) (point estimate) but not A.
    C  "no meaningful advantage"     otherwise.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = tuple[float, float]  # (quality, cost)


def _in_box(points: Sequence[Point], cost_ref: float) -> list[Point]:
    return [(q, c) for q, c in points if not (math.isnan(q) or math.isnan(c)) and c < cost_ref]


def set_dominates(g: Sequence[Point], m: Sequence[Point], cost_ref: float) -> bool:
    """True iff frontier ``g`` Pareto-dominates frontier ``m`` inside the box."""
    G, M = _in_box(g, cost_ref), _in_box(m, cost_ref)
    if not G:
        return False
    if not M:
        return True
    strict = False
    for qm, cm in M:
        covering = [(qg, cg) for qg, cg in G if qg >= qm and cg <= cm]
        if not covering:
            return False
        if any(qg > qm or cg < cm for qg, cg in covering):
            strict = True
    return strict


def bootstrap_dominance(
    Fg: np.ndarray, Dg: np.ndarray, Fm: np.ndarray, Dm: np.ndarray,
    idx: np.ndarray, cost_ref: float,
) -> float:
    """Fraction of bootstrap resamples in which g dominates m.

    Args:
        Fg, Dg: (n_configs_g, n_seeds) per-seed F1 / delay of g's frontier configs.
        Fm, Dm: same for m.
        idx: (R, n_seeds) resample indices shared across methods.
        cost_ref: Box reference cost.
    """
    hits = 0
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for cols in idx:
                g = list(zip(np.nanmean(Fg[:, cols], axis=1), np.nanmean(Dg[:, cols], axis=1)))
                m = list(zip(np.nanmean(Fm[:, cols], axis=1), np.nanmean(Dm[:, cols], axis=1)))
                hits += set_dominates(g, m, cost_ref)
    return hits / len(idx)


def classify(results: dict[tuple[str, int], dict[str, dict[str, float]]], comparators: Sequence[str]) -> str:
    """Return "A", "B" or "C" (see module docstring).

    Args:
        results: (condition, gap) -> comparator -> {"dominates": bool, "p_dom": float}.
        comparators: Simpler methods U.
    """
    all_dom = all(results[k][m]["dominates"] and results[k][m]["p_dom"] >= 0.95
                  for k in results for m in comparators)
    if results and all_dom:
        return "A"
    if any(all(results[k][m]["dominates"] for m in comparators) for k in results):
        return "B"
    return "C"
