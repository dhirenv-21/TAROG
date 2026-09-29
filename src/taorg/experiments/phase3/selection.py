"""Budget-constrained parameter selection and Pareto-set extraction.

Selection protocol (identical for every method; see
research/phase3_preregistration.md §5):

    Using CALIBRATION instances only, for termination-delay budget B:
        feasible = {configs with mean termination delay <= B}
        choose argmax mean Persistence F1 over feasible configs,
        ties (|dF1| <= TIE_TOL) broken by: lower mean churn, lower mean delay,
        lower mean false-persistence rate, lower grid index.
    If no configuration is feasible, the minimum-delay configuration is
    chosen and flagged ``feasible=False``.

Pareto set: configurations not dominated in (mean F1 up, mean delay down) on
calibration.  Exact (F1, delay) duplicates are collapsed to the member that
wins the same tie-break order.
"""

from __future__ import annotations

import math
from typing import Any

TIE_TOL = 1e-9


def _v(x: Any) -> float:
    """Convert NaN/None to +inf-safe floats for ordering."""
    if x is None:
        return math.nan
    return float(x)


def _tiebreak_key(s: dict[str, Any]) -> tuple:
    def nz(x: float, default: float) -> float:
        return default if math.isnan(x) else x
    return (
        nz(_v(s["churn_mean"]), math.inf),
        nz(_v(s["term_delay_mean"]), math.inf),
        nz(_v(s["false_pers_mean"]), math.inf),
        int(s["grid_index"]),
    )


def select_under_budget(
    summaries: list[dict[str, Any]],
    budget: float,
    top_k: int = 5,
) -> dict[str, Any]:
    """Select one configuration of ONE method under a delay budget.

    Args:
        summaries: Calibration summaries for configurations of a single method.
            Each needs keys config_id, grid_index, persistence_f1,
            term_delay_mean, churn_mean, false_pers_mean.
        budget: Maximum allowed mean termination delay (frames).
        top_k: Number of runner-up configurations to report.

    Returns:
        Dict with ``config_id``, ``feasible``, ``n_feasible`` and ``ranking``
        (top_k config summaries in selection order).
    """
    if not summaries:
        raise ValueError("select_under_budget: no summaries given")
    methods = {s["method"] for s in summaries}
    if len(methods) != 1:
        raise ValueError(f"select_under_budget expects one method, got {sorted(methods)}")

    feasible = [
        s for s in summaries
        if not math.isnan(_v(s["term_delay_mean"])) and _v(s["term_delay_mean"]) <= budget
    ]
    if feasible:
        def f1(s: dict) -> float:
            v = _v(s["persistence_f1"])
            return -math.inf if math.isnan(v) else v
        best_f1 = max(f1(s) for s in feasible)
        # Order: F1 bucketed by tolerance first, then tie-break key.
        ranked = sorted(
            feasible,
            key=lambda s: ((0, 0.0) if f1(s) >= best_f1 - TIE_TOL else (1, -f1(s)))
            + _tiebreak_key(s),
        )
        chosen, ok = ranked[0], True
    else:
        ranked = sorted(summaries, key=lambda s: (
            math.inf if math.isnan(_v(s["term_delay_mean"])) else _v(s["term_delay_mean"]),
            _tiebreak_key(s)))
        chosen, ok = ranked[0], False
    return {
        "config_id": chosen["config_id"],
        "feasible": ok,
        "n_feasible": len(feasible),
        "ranking": [
            {k: r[k] for k in ("config_id", "persistence_f1", "term_delay_mean",
                               "churn_mean", "false_pers_mean")}
            for r in ranked[:top_k]
        ],
    }


def pareto_front(summaries: list[dict[str, Any]]) -> list[str]:
    """Config ids on the calibration Pareto front (max F1, min delay).

    Args:
        summaries: Calibration summaries (any number of methods — callers
            normally pass one method at a time).

    Returns:
        Config ids sorted by increasing delay.
    """
    pts = [
        s for s in summaries
        if not math.isnan(_v(s["persistence_f1"])) and not math.isnan(_v(s["term_delay_mean"]))
    ]
    # Sort by delay asc, then F1 desc, then tie-break: a sweep keeps a point
    # iff its F1 strictly exceeds every F1 seen at lower-or-equal delay.
    pts.sort(key=lambda s: (_v(s["term_delay_mean"]), -_v(s["persistence_f1"]), _tiebreak_key(s)))
    front: list[str] = []
    best = -math.inf
    for s in pts:
        f1 = _v(s["persistence_f1"])
        if f1 > best + TIE_TOL:
            front.append(s["config_id"])
            best = f1
    return front
