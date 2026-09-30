"""Phase 4 pre-registered decision rules (H2, H3, H4, ablation retention).

Registered together with ``research/phase4_preregistration.md`` §8 BEFORE any
V-HPED implementation or Phase 4 run.  The functions are pure: they consume
already-computed statistics (per cell and comparator) and thresholds from the
``criteria`` block of ``configs/experiments/phase4.yaml``.

Statistic dicts use the keys produced by ``taorg.experiments.phase3.stats``:
``mean_diff``, ``ci_lo``, ``ci_hi`` (and ``p_holm`` where a Holm family exists).
Differences are always TREATMENT minus COMPARATOR.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

Stats = Mapping[str, float]


def _ok(x: float) -> bool:
    return x is not None and not math.isnan(x)


def churn_margin(comparator_churn: float, abs_margin: float, rel_margin: float) -> float:
    """Churn non-inferiority margin max(abs, rel * comparator mean churn)."""
    c = comparator_churn if _ok(comparator_churn) else 0.0
    return max(abs_margin, rel_margin * c)


# ---------------------------------------------------------------------------
# Frontier (H2 / H3) comparisons
# ---------------------------------------------------------------------------

def frontier_vs(
    d_hv: Stats,
    d_churn: Stats | None,
    comparator_churn: float,
    crit: Mapping[str, Any],
    d_hv_default: Stats | None = None,
    default_ni: float | None = None,
) -> dict[str, Any]:
    """Win/loss of the treatment against ONE comparator on one cell.

    Args:
        d_hv: Paired-bootstrap stats of HV(treatment) - HV(comparator) on the
            cell's registered frontier.
        d_churn: Paired stats of churn difference at the primary-budget
            operating point (None -> churn check not applicable, fails the win).
        comparator_churn: Comparator mean churn at that operating point.
        crit: Criteria with ``min_hv_gain``, ``churn_margin_abs``, ``churn_margin_rel``.
        d_hv_default: For identity conditions, stats of the default-frontier
            HV difference (must satisfy CI lower > ``default_ni``).
        default_ni: Non-inferiority bound for ``d_hv_default``.

    Returns:
        Dict with ``win``, ``loss`` and the failed checks.
    """
    g = crit["min_hv_gain"]
    checks = {
        "hv_gain": _ok(d_hv["mean_diff"]) and d_hv["mean_diff"] >= g,
        "hv_ci": _ok(d_hv["ci_lo"]) and d_hv["ci_lo"] > 0,
        "churn_ni": d_churn is not None and _ok(d_churn["ci_hi"]) and d_churn["ci_hi"] <= churn_margin(
            comparator_churn, crit["churn_margin_abs"], crit["churn_margin_rel"]),
    }
    if d_hv_default is not None:
        checks["default_frontier_ni"] = _ok(d_hv_default["ci_lo"]) and d_hv_default["ci_lo"] > default_ni
    win = all(checks.values())
    loss = _ok(d_hv["mean_diff"]) and d_hv["mean_diff"] <= -g and _ok(d_hv["ci_hi"]) and d_hv["ci_hi"] < 0
    return {"win": win, "loss": loss, "failed_checks": [k for k, v in checks.items() if not v]}


def cell_outcome(per_comparator: Mapping[str, Mapping[str, Any]]) -> str:
    """'win' if win vs every comparator, 'loss' if loss vs any, else 'neutral'."""
    if any(r["loss"] for r in per_comparator.values()):
        return "loss"
    if per_comparator and all(r["win"] for r in per_comparator.values()):
        return "win"
    return "neutral"


def decide_h2(
    failure_cells: Mapping[tuple[str, int], str],
    ni_cells: Mapping[tuple[str, int], str],
    crit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered H2 rule.

    Args:
        failure_cells: (condition, gap) -> 'win' | 'loss' | 'neutral' for the
            7 x 3 failure cells.
        ni_cells: same for the 3 x 3 non-inferiority cells.
        crit: ``criteria.h2`` block.
    """
    wins = [k for k, v in failure_cells.items() if v == "win"]
    f_losses = [k for k, v in failure_cells.items() if v == "loss"]
    ni_losses = [k for k, v in ni_cells.items() if v == "loss"]
    conds_with_win = sorted({c for c, _ in wins})
    pass_checks = {
        "min_wins": len(wins) >= crit["pass_min_wins"],
        "min_conditions_with_win": len(conds_with_win) >= crit["pass_min_conditions_with_win"],
        "max_failure_losses": len(f_losses) <= crit["pass_max_failure_losses"],
        "max_ni_losses": len(ni_losses) <= crit["pass_max_ni_losses"],
    }
    fail_checks = {
        "wins_le_max": len(wins) <= crit["fail_max_wins"],
        "conditions_le_max": len(conds_with_win) <= crit["fail_max_conditions_with_win"],
        "losses_ge_wins": (len(f_losses) + len(ni_losses)) >= len(wins),
    }
    if all(pass_checks.values()):
        decision = "PASS"
    elif any(fail_checks.values()):
        decision = "FAIL"
    else:
        decision = "MIXED"
    return {"decision": decision, "n_wins": len(wins), "n_failure_losses": len(f_losses),
            "n_ni_losses": len(ni_losses), "conditions_with_win": conds_with_win,
            "n_failure_cells": len(failure_cells), "n_ni_cells": len(ni_cells),
            "pass_checks": pass_checks, "fail_checks": fail_checks}


def decide_h3(
    benefit_cells: Mapping[tuple[str, int], str],
    ni_cells: Mapping[tuple[str, int], str],
    crit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered H3 rule (M6 vs M5)."""
    wins = sum(v == "win" for v in benefit_cells.values())
    ni_losses = sum(v == "loss" for v in ni_cells.values())
    if wins >= crit["pass_min_wins"] and ni_losses <= crit["pass_max_ni_losses"]:
        decision = "PASS"
    elif wins <= crit["fail_max_wins"] or ni_losses >= crit["fail_min_ni_losses"]:
        decision = "FAIL"
    else:
        decision = "MIXED"
    return {"decision": decision, "n_wins": wins, "n_ni_losses": ni_losses,
            "n_benefit_cells": len(benefit_cells), "n_ni_cells": len(ni_cells)}


# ---------------------------------------------------------------------------
# H4 (operating points under shift)
# ---------------------------------------------------------------------------

def shift_vs(
    f1: Stats, delay_s: Stats, fpr: Stats, churn: Stats,
    comparator_churn: float, treatment_churn: float, crit: Mapping[str, Any],
    abs_churn_margin: float, rel_churn_margin: float,
) -> dict[str, Any]:
    """H4 win/loss of the treatment against one comparator on one cell.

    ``f1`` must carry ``p_holm`` (Holm-adjusted over the registered family).
    All differences are treatment - comparator; the loss rules are the win
    rules with roles swapped (negated differences / swapped CI bounds).
    """
    def pers_win(d_f1, d_del, d_fpr, d_ch, other_churn) -> bool:
        return (_ok(d_f1["mean"]) and d_f1["mean"] >= crit["min_f1_gain"] and d_f1["lo"] > 0
                and _ok(d_f1["p"]) and d_f1["p"] < crit["alpha"]
                and d_del["hi"] <= crit["delay_margin_s"] and d_fpr["hi"] <= crit["fpr_margin"]
                and d_ch["hi"] <= churn_margin(other_churn, abs_churn_margin, rel_churn_margin))

    def term_win(d_f1, d_del, d_ch, other_churn) -> bool:
        if not (_ok(d_f1["lo"]) and d_f1["lo"] >= crit["f1_noninferiority"]):
            return False
        delay_win = _ok(d_del["mean"]) and d_del["mean"] <= -crit["term_win_delay_s"] and d_del["hi"] < 0
        need = max(abs_churn_margin, crit["churn_win_rel"] * (other_churn if _ok(other_churn) else 0.0))
        churn_win = _ok(d_ch["mean"]) and d_ch["mean"] <= -need and d_ch["hi"] < 0
        return delay_win or churn_win

    def fwd(s: Stats) -> dict[str, float]:
        return {"mean": s["mean_diff"], "lo": s["ci_lo"], "hi": s["ci_hi"], "p": s.get("p_holm", math.nan)}

    def rev(s: Stats) -> dict[str, float]:
        return {"mean": -s["mean_diff"], "lo": -s["ci_hi"], "hi": -s["ci_lo"], "p": s.get("p_holm", math.nan)}

    F, D, P, C = fwd(f1), fwd(delay_s), fwd(fpr), fwd(churn)
    Fr, Dr, Pr, Cr = rev(f1), rev(delay_s), rev(fpr), rev(churn)
    win_p = pers_win(F, D, P, C, comparator_churn)
    win_t = term_win(F, D, C, comparator_churn)
    loss_p = pers_win(Fr, Dr, Pr, Cr, treatment_churn)
    loss_t = term_win(Fr, Dr, Cr, treatment_churn)
    return {"win": win_p or win_t, "loss": loss_p or loss_t,
            "win_kind": "persistence" if win_p else ("termination_or_churn" if win_t else ""),
            "loss_kind": "persistence" if loss_p else ("termination_or_churn" if loss_t else "")}


def decide_h4(cells: Mapping[tuple[str, int], str], crit: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the registered H4 rule."""
    wins = sum(v == "win" for v in cells.values())
    losses = sum(v == "loss" for v in cells.values())
    if wins >= crit["pass_min_wins"] and losses <= crit["pass_max_losses"]:
        decision = "PASS"
    elif wins <= crit["fail_max_wins"] or losses >= wins:
        decision = "FAIL"
    else:
        decision = "MIXED"
    return {"decision": decision, "n_wins": wins, "n_losses": losses, "n_cells": len(cells)}


# ---------------------------------------------------------------------------
# Ablation retention
# ---------------------------------------------------------------------------

def rung_retained(
    target_hv: Sequence[Stats], ni_hv: Sequence[Stats], crit: Mapping[str, Any],
) -> dict[str, Any]:
    """Mechanism retention: ΔHV(Mk - M(k-1)) gains in >= 1/3 of target cells, no NI loss."""
    g = crit["min_hv_gain"]
    gains = sum(1 for s in target_hv if _ok(s["mean_diff"]) and s["mean_diff"] >= g and s["ci_lo"] > 0)
    ni_losses = sum(1 for s in ni_hv if _ok(s["mean_diff"]) and s["mean_diff"] <= -g and s["ci_hi"] < 0)
    need = crit["min_target_cell_fraction"] * len(target_hv)
    return {"retained": len(target_hv) > 0 and gains >= need - 1e-12 and ni_losses == 0,
            "n_target_gains": gains, "n_target_cells": len(target_hv), "n_ni_losses": ni_losses}
