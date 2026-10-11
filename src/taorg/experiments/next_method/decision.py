"""Next-method study: pre-registered decision rules (research/taorg_next_method_preregistration.md §8).

Registered with the pre-registration, BEFORE any registered comparison.  Pure functions of
paired statistics in the format of ``taorg.experiments.phase3.stats.paired_comparison``
(difference = X - Y).  Per-condition stats dicts use the metric names
``J, churn, fpt_s, frr, std_mean_s, ghost_time_s, rdd_mean_s``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

Stats = Mapping[str, float]
HARMS = ("fpt_s", "frr", "std_mean_s", "ghost_time_s", "rdd_mean_s")


def _ok(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _def(s: Optional[Stats], n: int) -> bool:
    return s is not None and _ok(s.get("mean_diff")) and _ok(s.get("ci_lo")) and int(s.get("n", 0)) >= n


def pairwise(st: Mapping[str, Stats], crit: Mapping[str, Any]) -> dict[str, Any]:
    """Win / loss of X vs Y in one condition.

    win:  dJ >= g_J with CI_lo > 0, churn CI_hi <= max(abs, rel * Y churn), and no harm loss.
    loss: dJ <= -g_J with CI_hi < 0; or a harm metric increases by >= max(floor, harm_rel * Y mean)
          with CI_lo > 0; or churn increases by >= max(abs, churn_loss_rel * Y churn) with CI_lo > 0.
    Loss takes precedence.
    """
    n = int(crit["min_paired"])
    J = st["J"]
    reasons = []
    if _ok(J["mean_diff"]) and J["mean_diff"] <= -crit["g_J"] and J["ci_hi"] < 0:
        reasons.append("J")
    for h in HARMS:
        s = st.get(h)
        if not _def(s, n):
            continue
        ref = abs(s["mean_b"]) if _ok(s.get("mean_b")) else 0.0
        margin = max(crit["harm_floor"][h], crit["harm_rel"] * ref)
        if s["mean_diff"] >= margin and s["ci_lo"] > 0:
            reasons.append(h)
    C = st.get("churn")
    churn_ok = True
    if _def(C, 2):
        cb = C["mean_b"] if _ok(C.get("mean_b")) else 0.0
        if C["mean_diff"] >= max(crit["churn_margin_abs"], crit["churn_loss_rel"] * cb) and C["ci_lo"] > 0:
            reasons.append("churn")
        churn_ok = C["ci_hi"] <= max(crit["churn_margin_abs"], crit["churn_margin_rel"] * cb)
    loss = bool(reasons)
    win = (_ok(J["mean_diff"]) and J["mean_diff"] >= crit["g_J"] and J["ci_lo"] > 0 and churn_ok and not loss)
    return {"win": bool(win), "loss": loss, "loss_reasons": reasons, "churn_guard": churn_ok}


def count(results: Mapping[str, Mapping[str, Any]], conds) -> tuple[list[str], list[str]]:
    """(wins, losses) over the given conditions."""
    w = [c for c in conds if results[c]["win"]]
    l_ = [c for c in conds if results[c]["loss"]]
    return w, l_


def gate(vs_b2: Mapping[str, Mapping[str, Any]], vs_buf: Mapping[str, Mapping[str, Any]],
         vs_static: Mapping[str, Mapping[str, Any]], vs_geo: Mapping[str, Mapping[str, Any]],
         half_vs_b2: Optional[Mapping[str, Mapping[str, Any]]], half_vs_buf: Optional[Mapping[str, Mapping[str, Any]]],
         latency_ratio: float, crit: Mapping[str, Any]) -> dict[str, Any]:
    """Algorithm gate G1-G7 and the GO / NARROW / STOP decision (prereg §8).

    Each ``vs_*`` maps condition -> pairwise() result of K1 against that comparator.
    """
    allc = list(vs_b2)
    prim, adapt = crit["primary_conditions"], crit["adapt_conditions"]
    w_b2, l_b2 = count(vs_b2, allc)
    w_buf, l_buf = count(vs_buf, allc)
    w_st, l_st = count(vs_static, allc)
    w_geo, l_geo = count(vs_geo, allc)
    G = {
        "G1_primary_wins_vs_B2": all(vs_b2[c]["win"] for c in prim),
        "G2_no_loss_vs_B2": not l_b2,
        "G3_beyond_fixed_buffer": len([c for c in adapt if vs_buf[c]["win"]]) >= crit["g3_min_wins"] and not l_buf,
        "G4_online_adaptation": len([c for c in adapt if vs_static[c]["win"]]) >= crit["g4_min_wins"] and not l_st,
        "G5_beyond_geometric_learner": len([c for c in adapt if vs_geo[c]["win"]]) >= crit["g5_min_wins"] and not l_geo,
        "G7_cost": _ok(latency_ratio) and latency_ratio <= crit["max_latency_ratio"],
    }
    if half_vs_b2 is not None and half_vs_buf is not None:
        hw_b2, hl_b2 = count(half_vs_b2, allc)
        G["G6_tuning_control"] = (all(half_vs_b2[c]["win"] for c in prim)
                                  and len([c for c in adapt if half_vs_buf[c]["win"]]) >= crit["g3_min_wins"])
    else:
        G["G6_tuning_control"] = False
    prim_wins = [c for c in prim if vs_b2[c]["win"]]
    if not prim_wins or len(l_b2) >= crit["stop_min_losses"]:
        decision = "STOP"
    elif all(G.values()):
        decision = "GO"
    else:
        decision = "NARROW"
    distinct = G["G3_beyond_fixed_buffer"] or G["G5_beyond_geometric_learner"]
    contribution = ("SUPPORTED" if decision == "GO" else
                    "PARTIAL" if (decision == "NARROW" and distinct) else "UNSUPPORTED")
    return {"decision": decision, "algorithm_contribution": contribution, "gates": G,
            "wins_vs_B2": w_b2, "losses_vs_B2": l_b2, "wins_vs_BUF": w_buf, "losses_vs_BUF": l_buf,
            "wins_vs_static": w_st, "losses_vs_static": l_st, "wins_vs_GEO": w_geo, "losses_vs_GEO": l_geo,
            "primary_wins_vs_B2": prim_wins}
