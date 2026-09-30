"""Pre-registered H1 decision rule (research/phase3_preregistration.md §8).

This module encodes the PASS / MIXED / FAIL criteria exactly as registered
BEFORE any calibration or evaluation run.  Thresholds come from the
experiment config (``h1_criteria``) and must not be changed after results
are viewed without a documented amendment.

Notation: d = GatedEMA - comparator, computed per evaluation seed.
Cell = (condition, gap, budget).  Unaware comparators U = criteria["comparators"].

Win vs comparator m (all must hold):
    mean dF1 >= min_f1_gain
    bootstrap CI(dF1).lo > 0   and   Holm-adjusted Wilcoxon p(dF1) < alpha
    CI(dDelay).hi <= delay_margin
    CI(dFPR).hi  <= fpr_margin
    CI(dChurn).hi <= churn_margin(m)
Persistence loss vs m (all must hold):
    mean dF1 <= -min_f1_gain, CI(dF1).hi < 0, Holm p < alpha,
    CI(dDelay).lo >= -delay_margin, CI(dFPR).lo >= -fpr_margin,
    CI(dChurn).lo >= -churn_margin(gated)
Termination/churn loss vs m:
    mean dF1 < min_f1_gain AND (CI(dDelay).lo > term_loss_delay
                                OR CI(dFPR).lo > term_loss_fpr
                                OR CI(dChurn).lo > churn_margin(m))
churn_margin(x) = max(churn_margin_abs, churn_margin_rel * mean churn of x).

Cell WIN  iff win vs every m in U.
Cell LOSS iff any loss vs some m in U.

Frontier criterion for (condition, gap): for every m in U,
    HV(gated) - HV(m) >= min_hv_gain and paired-bootstrap CI lower bound > 0.

Decision:
    PASS iff  W(primary) >= pass_primary_min_wins and L(primary) <= pass_primary_max_losses
          and for each robustness condition: W >= pass_robust_min_wins, L <= pass_robust_max_losses
          and frontier gaps(primary) >= pass_primary_min_hv_gaps
          and frontier gaps(each robustness) >= pass_robust_min_hv_gaps
    FAIL iff  W(primary) < fail_primary_max_wins_exclusive
          or  sum(L) >= sum(W) over primary+robustness conditions
          or  frontier gaps(primary) < fail_primary_min_hv_gaps_exclusive
    MIXED otherwise.
"""

from __future__ import annotations

import math
from typing import Any


def _churn_margin(mean_churn: float, crit: dict[str, Any]) -> float:
    base = crit["churn_margin_abs"]
    if math.isnan(mean_churn):
        return base
    return max(base, crit["churn_margin_rel"] * mean_churn)


def compare_vs(stats: dict[str, dict[str, float]], crit: dict[str, Any]) -> dict[str, Any]:
    """Classify GatedEMA vs one comparator in one cell.

    Args:
        stats: Mapping metric -> paired_comparison dict (gated - comparator).
            Must contain persistence_f1 (with ``p_holm``), term_delay_mean,
            false_pers_mean, churn_mean.
        crit: ``h1_criteria`` config mapping.

    Returns:
        Dict with ``win`` (bool), ``loss`` (bool), ``loss_type`` and ``reasons``.
    """
    f1, dl, fp, ch = (stats["persistence_f1"], stats["term_delay_mean"],
                      stats["false_pers_mean"], stats["churn_mean"])
    m_churn_comp = _churn_margin(ch["mean_b"], crit)
    m_churn_gated = _churn_margin(ch["mean_a"], crit)
    alpha = crit["alpha"]
    p = f1.get("p_holm", math.nan)

    def ok(x: float) -> bool:
        return not math.isnan(x)

    reasons: list[str] = []
    win_checks = {
        "f1_gain": ok(f1["mean_diff"]) and f1["mean_diff"] >= crit["min_f1_gain"],
        "f1_ci": ok(f1["ci_lo"]) and f1["ci_lo"] > 0,
        "f1_holm": ok(p) and p < alpha,
        "delay_ni": ok(dl["ci_hi"]) and dl["ci_hi"] <= crit["delay_margin"],
        "fpr_ni": ok(fp["ci_hi"]) and fp["ci_hi"] <= crit["fpr_margin"],
        "churn_ni": ok(ch["ci_hi"]) and ch["ci_hi"] <= m_churn_comp,
    }
    win = all(win_checks.values())
    if not win:
        reasons.append("win_failed:" + ",".join(k for k, v in win_checks.items() if not v))

    pers_loss = (
        ok(f1["mean_diff"]) and f1["mean_diff"] <= -crit["min_f1_gain"]
        and ok(f1["ci_hi"]) and f1["ci_hi"] < 0 and ok(p) and p < alpha
        and ok(dl["ci_lo"]) and dl["ci_lo"] >= -crit["delay_margin"]
        and ok(fp["ci_lo"]) and fp["ci_lo"] >= -crit["fpr_margin"]
        and ok(ch["ci_lo"]) and ch["ci_lo"] >= -m_churn_gated
    )
    f1_not_better = not (ok(f1["mean_diff"]) and f1["mean_diff"] >= crit["min_f1_gain"])
    term_loss = f1_not_better and (
        (ok(dl["ci_lo"]) and dl["ci_lo"] > crit["term_loss_delay"])
        or (ok(fp["ci_lo"]) and fp["ci_lo"] > crit["term_loss_fpr"])
        or (ok(ch["ci_lo"]) and ch["ci_lo"] > m_churn_comp)
    )
    loss_type = "persistence" if pers_loss else ("termination_or_churn" if term_loss else "")
    if loss_type:
        reasons.append(f"loss:{loss_type}")
    return {"win": win, "loss": bool(loss_type), "loss_type": loss_type,
            "checks": win_checks, "reasons": reasons}


def classify_cell(
    per_comparator: dict[str, dict[str, dict[str, float]]], crit: dict[str, Any]
) -> dict[str, Any]:
    """Classify one cell as win / loss / neutral against all comparators."""
    detail = {m: compare_vs(per_comparator[m], crit) for m in crit["comparators"]}
    win = all(d["win"] for d in detail.values())
    loss = any(d["loss"] for d in detail.values())
    outcome = "loss" if loss else ("win" if win else "neutral")
    return {"outcome": outcome, "detail": detail}


def frontier_advantage(hv: dict[str, Any], crit: dict[str, Any]) -> bool:
    """True if GatedEMA's frontier beats every comparator's (see module doc).

    Args:
        hv: {"diff_vs": {m: {"mean_diff": .., "ci_lo": ..}}} for one
            (condition, gap).
    """
    for m in crit["comparators"]:
        d = hv["diff_vs"][m]
        if math.isnan(d["mean_diff"]) or d["mean_diff"] < crit["min_hv_gain"]:
            return False
        if math.isnan(d["ci_lo"]) or d["ci_lo"] <= 0:
            return False
    return True


def decide_h1(
    cell_outcomes: dict[str, list[str]],
    hv_gaps_won: dict[str, int],
    primary: str,
    robustness: list[str],
    crit: dict[str, Any],
) -> dict[str, Any]:
    """Apply the registered PASS / MIXED / FAIL rule.

    Args:
        cell_outcomes: condition -> list of cell outcomes ("win"/"loss"/"neutral").
        hv_gaps_won: condition -> number of gaps meeting the frontier criterion.
        primary: Primary condition name.
        robustness: Robustness condition names.
        crit: ``h1_criteria`` mapping.

    Returns:
        Dict with ``decision`` and a transparent breakdown of each criterion.
    """
    W = {c: sum(o == "win" for o in v) for c, v in cell_outcomes.items()}
    L = {c: sum(o == "loss" for o in v) for c, v in cell_outcomes.items()}
    n_cells = {c: len(v) for c, v in cell_outcomes.items()}
    conds = [primary, *robustness]

    pass_checks = {
        "primary_wins": W[primary] >= crit["pass_primary_min_wins"],
        "primary_losses": L[primary] <= crit["pass_primary_max_losses"],
        "primary_frontier": hv_gaps_won[primary] >= crit["pass_primary_min_hv_gaps"],
    }
    for r in robustness:
        pass_checks[f"{r}_wins"] = W[r] >= crit["pass_robust_min_wins"]
        pass_checks[f"{r}_losses"] = L[r] <= crit["pass_robust_max_losses"]
        pass_checks[f"{r}_frontier"] = hv_gaps_won[r] >= crit["pass_robust_min_hv_gaps"]
    fail_checks = {
        "primary_wins_below_min": W[primary] < crit["fail_primary_max_wins_exclusive"],
        "losses_ge_wins": sum(L[c] for c in conds) >= sum(W[c] for c in conds),
        "primary_frontier_below_min": hv_gaps_won[primary] < crit["fail_primary_min_hv_gaps_exclusive"],
    }
    if all(pass_checks.values()):
        decision = "PASS"
    elif any(fail_checks.values()):
        decision = "FAIL"
    else:
        decision = "MIXED"
    return {
        "decision": decision, "wins": W, "losses": L, "n_cells": n_cells,
        "hv_gaps_won": hv_gaps_won, "pass_checks": pass_checks, "fail_checks": fail_checks,
    }
