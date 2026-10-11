"""Episode study: pre-registered decision rules (research/taorg_episode_preregistration.md §8).

Registered together with the pre-registration BEFORE any episode-study method,
calibration or comparison existed.  Pure functions over already-computed paired
statistics.  A *stats* mapping has the keys produced by
``taorg.experiments.phase3.stats.paired_comparison``: ``n``, ``mean_a``, ``mean_b``,
``mean_diff``, ``ci_lo``, ``ci_hi`` (difference = treatment - comparator).

Metric names used as keys of a per-condition stats dict:
    ``J``, ``gap_recall``, ``churn``, ``rdd_mean_s`` and the target harm metric of the
    condition (``std_mean_s``, ``fpt_s``, ``frr``, ``ghost_time_s``).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

Stats = Mapping[str, float]


def _ok(x: Optional[float]) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _defined(s: Optional[Stats], min_n: int = 1) -> bool:
    return s is not None and _ok(s.get("mean_diff")) and int(s.get("n", 0)) >= min_n


def pairwise(stats: Mapping[str, Stats], condition: str, crit: Mapping[str, Any],
             min_paired_rdd: int = 10) -> dict[str, Any]:
    """Win / loss of treatment X against ONE comparator Y in one condition (§8.1).

    Args:
        stats: metric -> paired stats (X - Y).  ``J`` is required; others optional.
        condition: Condition name (selects the target harm endpoint).
        crit: The ``criteria`` block of the config.
        min_paired_rdd: Minimum paired seeds for the recurrence-delay guard to apply.

    Returns:
        ``{"win", "loss", "route", "failed_guards", "loss_reasons"}``.
    """
    J = stats["J"]
    target = crit["target_harm"].get(condition)
    H = stats.get(target) if target else None
    R = stats.get("rdd_mean_s")
    G = stats.get("gap_recall")
    C = stats.get("churn")

    route_a = _ok(J["mean_diff"]) and J["mean_diff"] >= crit["g_J"] and J["ci_lo"] > 0
    route_b = False
    if target and _defined(H):
        ref = abs(H["mean_b"]) if _ok(H.get("mean_b")) else 0.0
        route_b = (ref > 0 and H["mean_diff"] <= -crit["r_h"] * ref and H["ci_hi"] < 0
                   and _ok(J["ci_lo"]) and J["ci_lo"] >= crit["ni_J"])
    guards = {"J_noninferior": _ok(J["ci_lo"]) and J["ci_lo"] >= crit["ni_J"]}
    if _defined(R, min_paired_rdd):
        guards["rdd_noninferior"] = R["ci_hi"] <= crit["rdd_margin_s"]
    if _defined(G, 2):
        guards["gap_recall_noninferior"] = G["ci_lo"] >= crit["gap_recall_ni"]
    if _defined(C, 2):
        margin = max(crit["churn_margin_abs"],
                     crit["churn_margin_rel"] * (C["mean_b"] if _ok(C.get("mean_b")) else 0.0))
        guards["churn_noninferior"] = C["ci_hi"] <= margin

    loss_reasons = []
    if _ok(J["mean_diff"]) and J["mean_diff"] <= -crit["g_J"] and J["ci_hi"] < 0:
        loss_reasons.append("J")
    if target and _defined(H):
        ref = abs(H["mean_b"]) if _ok(H.get("mean_b")) else 0.0
        if ref > 0 and H["mean_diff"] >= crit["r_h"] * ref and H["ci_lo"] > 0:
            loss_reasons.append(target)
        elif ref == 0 and H["mean_diff"] > 0 and H["ci_lo"] > 0:
            loss_reasons.append(target)          # comparator has zero harm, treatment adds harm
    if _defined(R, min_paired_rdd) and R["mean_diff"] >= crit["rdd_margin_s"] and R["ci_lo"] > 0:
        loss_reasons.append("rdd_mean_s")
    loss = bool(loss_reasons)
    win = (route_a or route_b) and all(guards.values()) and not loss
    return {"win": win, "loss": loss, "route": "J" if route_a else ("harm" if route_b else ""),
            "failed_guards": [k for k, v in guards.items() if not v], "loss_reasons": loss_reasons}


def cell_outcome(per_comparator: Mapping[str, Mapping[str, Any]]) -> str:
    """'win' if a win vs every comparator, 'loss' if a loss vs any, else 'neutral'."""
    if any(r["loss"] for r in per_comparator.values()):
        return "loss"
    if per_comparator and all(r["win"] for r in per_comparator.values()):
        return "win"
    return "neutral"


def comparators_for(condition: str, crit: Mapping[str, Any]) -> list[str]:
    """Registered comparator set of a condition (§8.2)."""
    out = list(crit["comparators"])
    if condition in crit["identity_comparator_conditions"]:
        out.append(crit["identity_comparator"])
    return out


# ---------------------------------------------------------------------------
# Ladder retention (§8.3)
# ---------------------------------------------------------------------------

def _j_loss(s: Stats, crit: Mapping[str, Any]) -> bool:
    return _ok(s["mean_diff"]) and s["mean_diff"] <= -crit["g_J"] and s["ci_hi"] < 0


def _j_win(s: Stats, crit: Mapping[str, Any]) -> bool:
    return _ok(s["mean_diff"]) and s["mean_diff"] >= crit["g_J"] and s["ci_lo"] > 0


def retain_identity(per_cond: Mapping[str, Mapping[str, Stats]], crit: Mapping[str, Any],
                    calibrated: Sequence[str]) -> dict[str, Any]:
    """P2 vs P1.  ``per_cond[c]`` holds stats for ``J`` and the harm metric."""
    lad = crit["ladder"]["identity"]
    t = per_cond[lad["target"]]
    H, J = t[lad["harm"]], t["J"]
    ref = abs(H["mean_b"]) if _ok(H.get("mean_b")) else 0.0
    target_ok = (ref > 0 and H["mean_diff"] <= -lad["rel"] * ref and H["ci_hi"] < 0
                 and J["ci_lo"] >= crit["ni_J"])
    losses = [c for c in calibrated if _j_loss(per_cond[c]["J"], crit)]
    return {"retained": bool(target_ok and not losses), "target_ok": bool(target_ok), "J_losses": losses}


def retain_context(per_cond: Mapping[str, Mapping[str, Stats]], crit: Mapping[str, Any],
                   calibrated: Sequence[str]) -> dict[str, Any]:
    """P3 vs P2."""
    lad = crit["ladder"]["context"]
    hits = []
    for c in lad["targets"]:
        J, S = per_cond[c]["J"], per_cond[c].get("std_mean_s")
        std_ok = (_defined(S) and S["mean_diff"] <= -lad["std_gain_s"] and S["ci_hi"] < 0
                  and J["ci_lo"] >= crit["ni_J"])
        if _j_win(J, crit) or std_ok:
            hits.append(c)
    losses = [c for c in calibrated if _j_loss(per_cond[c]["J"], crit)]
    return {"retained": bool(hits and not losses), "target_hits": hits, "J_losses": losses}


def retain_history(per_cond: Mapping[str, Mapping[str, Stats]], pooled_J: Stats,
                   crit: Mapping[str, Any], calibrated: Sequence[str]) -> dict[str, Any]:
    """P4 vs P3: must beat simple context on downstream quality."""
    lad = crit["ladder"]["history"]
    pooled_ok = _ok(pooled_J["mean_diff"]) and pooled_J["mean_diff"] >= lad["pooled_min"] and pooled_J["ci_lo"] > 0
    hits = [c for c in lad["targets"] if _j_win(per_cond[c]["J"], crit)]
    losses = [c for c in calibrated if _j_loss(per_cond[c]["J"], crit)]
    return {"retained": bool(pooled_ok and hits and not losses), "pooled_ok": bool(pooled_ok),
            "target_hits": hits, "J_losses": losses}


def select_treatment(identity: bool, context: bool, history: bool, rungs: Sequence[str]) -> str:
    """Highest rung reachable through consecutive retentions (P1 always available)."""
    t = rungs[0]
    for retained, rung in zip((identity, context, history), rungs[1:]):
        if not retained:
            break
        t = rung
    return t


# ---------------------------------------------------------------------------
# Study decision (§8.4) and registered answers (§8.5)
# ---------------------------------------------------------------------------

def study_decision(cells: Mapping[str, str], vs_simple: Mapping[str, Mapping[str, Any]],
                   latency_ms: float, latency_ratio: float, half_control: Optional[str],
                   crit: Mapping[str, Any]) -> dict[str, Any]:
    """GO / NARROW / STOP.

    Args:
        cells: condition -> cell outcome of T vs its comparator set (all 11 conditions).
        vs_simple: relapse condition -> pairwise result of T vs B2 alone.
        latency_ms: T mean ms/frame; latency_ratio: T / GatedEMA.
        half_control: decision obtained under the half-sample selection (None = not computed).
    """
    ben, rel = crit["benefit_conditions"], crit["relapse_conditions"]
    cal = ben + crit["ni_conditions"]
    tr = crit["transfer_conditions"]
    ben_wins = [c for c in ben if cells.get(c) == "win"]
    rel_wins = [c for c in rel if cells.get(c) == "win"]
    cal_losses = [c for c in cal if cells.get(c) == "loss"]
    tr_losses = [c for c in tr if cells.get(c) == "loss"]
    wins_all = sum(v == "win" for v in cells.values())
    losses_all = sum(v == "loss" for v in cells.values())
    g = crit["go"]
    go_checks = {
        "benefit_wins": len(ben_wins) >= g["min_benefit_wins"],
        "relapse_wins": len(rel_wins) >= g["min_relapse_wins"],
        "no_calibrated_losses": len(cal_losses) <= g["max_calibrated_losses"],
        "transfer_losses": len(tr_losses) <= g["max_transfer_losses"],
        "latency": _ok(latency_ms) and latency_ms <= g["max_latency_ms"]
        and _ok(latency_ratio) and latency_ratio <= g["max_latency_ratio"],
        "half_control_not_stop": half_control != "STOP",
    }
    simple_beaten = [c for c in rel if vs_simple.get(c, {}).get("win")]
    stop_checks = {
        "benefit_wins_le_max": len(ben_wins) <= crit["stop"]["max_benefit_wins"],
        "losses_ge_wins": losses_all >= wins_all,
        "simple_baseline_not_beaten": len(simple_beaten) == 0,
    }
    if any(stop_checks.values()):
        decision = "STOP"
    elif all(go_checks.values()):
        decision = "GO"
    else:
        decision = "NARROW"
    return {"decision": decision, "benefit_wins": ben_wins, "relapse_wins": rel_wins,
            "calibrated_losses": cal_losses, "transfer_losses": tr_losses, "wins_all": wins_all,
            "losses_all": losses_all, "simple_baseline_beaten_in": simple_beaten,
            "go_checks": go_checks, "stop_checks": stop_checks}


def answer_q1(p1_vs: Mapping[str, Mapping[str, Mapping[str, Any]]], crit: Mapping[str, Any]) -> dict[str, Any]:
    """Q1: P1 wins vs each of B1-B4 in C3 or C5.  ``p1_vs[cond][comparator]`` = pairwise()."""
    conds = ["C3_misleading_post", "C5_boundary_flicker"]
    hits = [c for c in conds if all(p1_vs[c][b]["win"] for b in crit["comparators"])]
    return {"answer": "YES" if hits else "NO", "conditions": hits}


def answer_q2(p1_rdd: Mapping[str, Mapping[str, Stats]], crit: Mapping[str, Any],
              min_paired_rdd: int = 10) -> dict[str, Any]:
    """Q2: P1 delays recurrence iff a ΔRDD loss vs any B1-B4 in C4 or C5."""
    hits = []
    for c in ("C4_recurrence", "C5_boundary_flicker"):
        for b in crit["comparators"]:
            s = p1_rdd[c][b]
            if _defined(s, min_paired_rdd) and s["mean_diff"] >= crit["rdd_margin_s"] and s["ci_lo"] > 0:
                hits.append(f"{c}:{b}")
    return {"answer": "YES (delays)" if hits else "NO", "cells": hits}


def answer_q3(p2_vs_b5: Mapping[str, Any], interaction_ghost: Stats) -> dict[str, Any]:
    """Q3: P2 wins vs B5 in C6 AND (P2-P1)-(B5-B1) ghost-time interaction CI_hi < 0."""
    inter = _ok(interaction_ghost["ci_hi"]) and interaction_ghost["ci_hi"] < 0
    return {"answer": "YES" if (p2_vs_b5["win"] and inter) else "NO",
            "p2_wins_vs_b5": bool(p2_vs_b5["win"]), "interaction_negative": bool(inter)}


def answer_q5(cells: Mapping[str, str], crit: Mapping[str, Any]) -> dict[str, Any]:
    """Q5: <= 1 transfer loss and >= 2 transfer wins (needs some BEN win to be meaningful)."""
    tr = crit["transfer_conditions"]
    wins = [c for c in tr if cells.get(c) == "win"]
    losses = [c for c in tr if cells.get(c) == "loss"]
    if not any(cells.get(c) == "win" for c in crit["benefit_conditions"]):
        ans = "NO IN-DISTRIBUTION ADVANTAGE TO TRANSFER"
    else:
        ans = "YES" if (len(losses) <= crit["go"]["max_transfer_losses"]
                        and len(wins) >= crit["q5_min_transfer_wins"]) else "NO"
    return {"answer": ans, "transfer_wins": wins, "transfer_losses": losses}
