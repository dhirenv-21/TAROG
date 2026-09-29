"""Tests for Phase 3 evaluation, selection, statistics and the H1 decision rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from taorg.experiments.phase3.conditions import ConditionSpec, build_instance
from taorg.experiments.phase3.evaluation import METRIC_FIELDS, evaluate_config
from taorg.experiments.phase3.grids import expand_all, expand_grid
from taorg.experiments.phase3.h1 import classify_cell, compare_vs, decide_h1, frontier_advantage
from taorg.experiments.phase3.selection import pareto_front, select_under_budget
from taorg.experiments.phase3.stats import (
    descriptive, holm_adjust, hypervolume_2d, paired_comparison, wilcoxon_p,
)

OCC = {"num_gaps_per_edge": 1, "min_pre_gap_frames": 2, "min_post_gap_frames": 2, "max_edges": None}
C0 = ConditionSpec(name="C0", role="h1_primary", calibrate=True)


# ---------------------------------------------------------------------------
# Evaluation on known behaviour
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_metric_record_complete(self):
        inst = build_instance(C0, 5, 42, {}, OCC)
        m, _ = evaluate_config(inst, "no_memory", {"score_threshold": 0.0})
        assert set(METRIC_FIELDS) <= set(m)

    def test_no_memory_zero_persistence_immediate_drop(self):
        inst = build_instance(C0, 5, 42, {}, OCC)
        m, _ = evaluate_config(inst, "no_memory", {"score_threshold": 0.0})
        assert m["persistence_recall"] == 0.0
        assert m["persistence_f1"] == 0.0
        assert m["term_delay_mean"] == 1.0          # drops at t_end + 1
        assert m["false_pers_mean"] == 0.0
        assert m["diag_clean_alive_recall"] == 1.0
        assert m["diag_spurious_rate"] == 0.0

    def test_gated_ema_oracle_bridges_and_drops(self):
        # With oracle visibility: slow occluded decay bridges, zero visible decay drops.
        inst = build_instance(C0, 10, 42, {}, OCC)
        params = {"alpha": 1.0, "decay_visible": 0.0, "decay_occluded": 0.99,
                  "score_threshold": 0.05}
        m, _ = evaluate_config(inst, "gated_ema", params)
        assert m["persistence_f1"] == 1.0
        assert m["term_delay_mean"] == 1.0
        assert m["false_pers_mean"] == 0.0

    def test_ema_bridging_costs_termination_delay(self):
        inst = build_instance(C0, 10, 42, {}, OCC)
        m, _ = evaluate_config(inst, "ema", {"alpha": 1.0, "decay": 0.99, "score_threshold": 0.05})
        assert m["persistence_f1"] == 1.0
        assert m["term_delay_mean"] > 10.0

    def test_undefined_metrics_are_nan(self):
        spec = ConditionSpec(name="S2", role="stress", tuned_on="C0",
                             id_switch={"swap_duration": 5, "num_swaps": 1, "require_same_class": True})
        inst = build_instance(spec, 0, 42, {}, OCC)
        m, _ = evaluate_config(inst, "ema", {})
        assert math.isnan(m["persistence_f1"])
        assert not math.isnan(m["diag_swap_recall"])

    def test_recovery_lag_zero_when_bridged(self):
        inst = build_instance(C0, 5, 42, {}, OCC)
        m, _ = evaluate_config(inst, "fixed_window", {"window_size": 40, "score_threshold": 0.0})
        assert m["diag_recovery_lag"] == 0.0


# ---------------------------------------------------------------------------
# Grids and selection
# ---------------------------------------------------------------------------

class TestGridsSelection:
    def test_expand_grid_order_and_ids(self):
        cfgs = expand_grid("ema", {"alpha": [0.2, 0.5], "decay": [0.9, 0.95, 0.99]})
        assert len(cfgs) == 6
        assert cfgs[0].config_id == "ema#0000"
        assert cfgs[1].params == {"alpha": 0.2, "decay": 0.95}
        assert expand_grid("no_memory", {})[0].params == {}

    def test_expand_all(self):
        cfgs = expand_all({"no_memory": {"score_threshold": [0.0]}, "ema": {"alpha": [0.3, 0.5]}})
        assert [c.config_id for c in cfgs] == ["no_memory#0000", "ema#0000", "ema#0001"]

    def _s(self, i, f1, d, churn=0.0, fpr=0.0, method="ema"):
        return {"config_id": f"{method}#{i:04d}", "method": method, "grid_index": i,
                "persistence_f1": f1, "term_delay_mean": d, "churn_mean": churn,
                "false_pers_mean": fpr}

    def test_budget_feasibility(self):
        s = [self._s(0, 0.9, 12.0), self._s(1, 0.6, 4.0), self._s(2, 0.5, 2.0)]
        assert select_under_budget(s, 5.0)["config_id"] == "ema#0001"
        assert select_under_budget(s, 2.0)["config_id"] == "ema#0002"
        assert select_under_budget(s, 20.0)["config_id"] == "ema#0000"

    def test_tie_breaks(self):
        s = [self._s(0, 0.8, 3.0, churn=0.02), self._s(1, 0.8, 3.0, churn=0.01),
             self._s(2, 0.8, 2.0, churn=0.01)]
        assert select_under_budget(s, 5.0)["config_id"] == "ema#0002"

    def test_infeasible_falls_back_to_min_delay(self):
        s = [self._s(0, 0.9, 12.0), self._s(1, 0.6, 8.0)]
        sel = select_under_budget(s, 5.0)
        assert sel["config_id"] == "ema#0001" and sel["feasible"] is False

    def test_single_method_enforced(self):
        with pytest.raises(ValueError):
            select_under_budget([self._s(0, 0.9, 1.0), self._s(1, 0.9, 1.0, method="hmm")], 5.0)

    def test_pareto_front(self):
        s = [self._s(0, 0.5, 1.0), self._s(1, 0.4, 2.0), self._s(2, 0.8, 5.0),
             self._s(3, 0.8, 6.0), self._s(4, 1.0, 10.0)]
        assert pareto_front(s) == ["ema#0000", "ema#0002", "ema#0004"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_descriptive(self):
        d = descriptive([1.0, 2.0, 3.0, float("nan")])
        assert d["n"] == 3 and d["mean"] == 2.0 and d["sd"] == pytest.approx(1.0)
        assert d["ci_lo"] < 2.0 < d["ci_hi"]

    def test_holm(self):
        adj = holm_adjust([0.01, 0.04, 0.03, float("nan")])
        assert adj[0] == pytest.approx(0.03)
        assert adj[2] == pytest.approx(0.06)
        assert adj[1] == pytest.approx(0.06)
        assert math.isnan(adj[3])

    def test_wilcoxon_all_zero(self):
        assert wilcoxon_p(np.zeros(10)) == 1.0

    def test_paired_comparison(self):
        rng = np.random.default_rng(0)
        a = [1.0, 2.0, 3.0, 4.0, float("nan")]
        b = [0.5, 1.5, 2.5, 3.5, 1.0]
        pc = paired_comparison(a, b, 1000, 0.95, rng)
        assert pc["n"] == 4 and pc["n_dropped"] == 1
        assert pc["mean_diff"] == pytest.approx(0.5)
        assert pc["ci_lo"] == pc["ci_hi"] == pytest.approx(0.5)  # constant differences
        assert pc["dz"] == math.inf

    def test_hypervolume(self):
        assert hypervolume_2d([(1.0, 1.0)], 1.0, 11.0) == pytest.approx(1.0)
        assert hypervolume_2d([(0.5, 1.0)], 1.0, 11.0) == pytest.approx(0.5)
        # Two-step staircase: 0.5 on [1,6) and 1.0 on [6,11) -> (2.5 + 5) / 10
        assert hypervolume_2d([(0.5, 1.0), (1.0, 6.0)], 1.0, 11.0) == pytest.approx(0.75)
        assert hypervolume_2d([(1.0, 20.0)], 1.0, 11.0) == 0.0
        assert hypervolume_2d([], 1.0, 11.0) == 0.0


# ---------------------------------------------------------------------------
# H1 decision rule
# ---------------------------------------------------------------------------

CRIT = {
    "comparators": ["ema"], "alpha": 0.05, "min_f1_gain": 0.05, "delay_margin": 1.0,
    "fpr_margin": 0.02, "churn_margin_abs": 0.005, "churn_margin_rel": 0.1,
    "term_loss_delay": 2.0, "term_loss_fpr": 0.05, "min_hv_gain": 0.02,
    "pass_primary_min_wins": 9, "pass_primary_max_losses": 0, "pass_robust_min_wins": 6,
    "pass_robust_max_losses": 2, "pass_primary_min_hv_gaps": 4, "pass_robust_min_hv_gaps": 3,
    "fail_primary_max_wins_exclusive": 3, "fail_primary_min_hv_gaps_exclusive": 2,
}


def _pc(mean, lo, hi, a=0.0, b=0.0, p=math.nan):
    return {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "mean_a": a, "mean_b": b, "p_holm": p}


class TestH1Rule:
    def test_clear_win(self):
        st = {"persistence_f1": _pc(0.3, 0.2, 0.4, p=0.001), "term_delay_mean": _pc(0.0, -0.5, 0.5),
              "false_pers_mean": _pc(0.0, -0.01, 0.01), "churn_mean": _pc(0.0, -0.001, 0.001, 0.02, 0.02)}
        r = compare_vs(st, CRIT)
        assert r["win"] and not r["loss"]

    def test_win_blocked_by_termination_cost(self):
        st = {"persistence_f1": _pc(0.3, 0.2, 0.4, p=0.001), "term_delay_mean": _pc(3.0, 2.5, 3.5),
              "false_pers_mean": _pc(0.0, -0.01, 0.01), "churn_mean": _pc(0.0, 0.0, 0.0)}
        r = compare_vs(st, CRIT)
        assert not r["win"] and not r["loss"]  # F1 gain present -> not a termination loss

    def test_termination_loss(self):
        st = {"persistence_f1": _pc(0.0, -0.01, 0.01, p=0.9), "term_delay_mean": _pc(3.0, 2.5, 3.5),
              "false_pers_mean": _pc(0.0, 0.0, 0.0), "churn_mean": _pc(0.0, 0.0, 0.0)}
        assert compare_vs(st, CRIT)["loss_type"] == "termination_or_churn"

    def test_persistence_loss(self):
        st = {"persistence_f1": _pc(-0.2, -0.3, -0.1, p=0.001), "term_delay_mean": _pc(0.0, -0.2, 0.2),
              "false_pers_mean": _pc(0.0, 0.0, 0.0), "churn_mean": _pc(0.0, 0.0, 0.0)}
        assert compare_vs(st, CRIT)["loss_type"] == "persistence"

    def test_not_significant_is_neutral(self):
        st = {"persistence_f1": _pc(0.1, 0.02, 0.2, p=0.2), "term_delay_mean": _pc(0.0, 0.0, 0.0),
              "false_pers_mean": _pc(0.0, 0.0, 0.0), "churn_mean": _pc(0.0, 0.0, 0.0)}
        assert classify_cell({"ema": st}, CRIT)["outcome"] == "neutral"

    def test_frontier_advantage(self):
        assert frontier_advantage({"diff_vs": {"ema": {"mean_diff": 0.1, "ci_lo": 0.05}}}, CRIT)
        assert not frontier_advantage({"diff_vs": {"ema": {"mean_diff": 0.01, "ci_lo": 0.005}}}, CRIT)
        assert not frontier_advantage({"diff_vs": {"ema": {"mean_diff": 0.1, "ci_lo": -0.01}}}, CRIT)

    def _outcomes(self, w, l, n=18):
        return ["win"] * w + ["loss"] * l + ["neutral"] * (n - w - l)

    def test_decide_pass(self):
        d = decide_h1({"C0": self._outcomes(10, 0), "R1": self._outcomes(7, 1), "R2": self._outcomes(6, 2)},
                      {"C0": 5, "R1": 3, "R2": 4}, "C0", ["R1", "R2"], CRIT)
        assert d["decision"] == "PASS"

    def test_decide_mixed_when_robustness_fails(self):
        d = decide_h1({"C0": self._outcomes(12, 0), "R1": self._outcomes(1, 0), "R2": self._outcomes(6, 0)},
                      {"C0": 6, "R1": 0, "R2": 3}, "C0", ["R1", "R2"], CRIT)
        assert d["decision"] == "MIXED"

    def test_decide_fail(self):
        d = decide_h1({"C0": self._outcomes(2, 0), "R1": self._outcomes(0, 0), "R2": self._outcomes(0, 0)},
                      {"C0": 1, "R1": 0, "R2": 0}, "C0", ["R1", "R2"], CRIT)
        assert d["decision"] == "FAIL"

    def test_decide_fail_when_losses_dominate(self):
        d = decide_h1({"C0": self._outcomes(5, 1), "R1": self._outcomes(0, 5), "R2": self._outcomes(0, 5)},
                      {"C0": 4, "R1": 0, "R2": 0}, "C0", ["R1", "R2"], CRIT)
        assert d["decision"] == "FAIL"
