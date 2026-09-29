"""Tests for Phase 4 selection helpers (budget selection, Pareto sets, tuning control)."""

from __future__ import annotations

from pathlib import Path

from taorg.experiments.phase4 import pipeline as P

CFG = P.load_config(Path("configs/experiments/phase4.yaml"))


def _row(cond, method, i, f1, delay, sr=0.5, ghost=10.0):
    return {"condition": cond, "config_id": f"{method}#{i:04d}", "method": method, "grid_index": i,
            "params_json": f'{{"i": {i}}}', "persistence_f1": f1, "term_delay_mean": delay,
            "churn_mean": 0.01, "false_pers_mean": 0.02, "diag_swap_recall": sr,
            "diag_ghost_edge_frames": ghost}


def test_pareto_generic_identity_axes():
    rows = [_row("F5_id_switch", "m", 0, 0.9, 5, sr=0.2, ghost=5),
            _row("F5_id_switch", "m", 1, 0.9, 5, sr=0.8, ghost=50),
            _row("F5_id_switch", "m", 2, 0.9, 5, sr=0.5, ghost=100)]    # dominated
    assert P.pareto_generic(rows, "diag_swap_recall", "diag_ghost_edge_frames") == ["m#0000", "m#0001"]


def test_select_group_budgets_frontiers_and_half_control():
    rows = [_row("F5_id_switch", "vhped_m5", i, 0.5 + i / 400, 3 + i / 20, sr=i / 200, ghost=i)
            for i in range(200)]
    sel, par = P.select_group(CFG, rows, half_for=("vhped_m5",))
    s = sel["F5_id_switch"]
    assert set(s) == {"vhped_m5", "vhped_m5__half"}
    for name in s:
        for b, d in s[name].items():
            r = next(x for x in rows if x["config_id"] == d["config_id"])
            assert r["term_delay_mean"] <= float(b)
    assert set(par["F5_id_switch"]) == {"default", "identity"}
    half_ids = {r for r in par["F5_id_switch"]["default"]["vhped_m5__half"]}
    assert 0 < len(half_ids) <= 100
    again, _ = P.select_group(CFG, rows, half_for=("vhped_m5",))
    assert again == sel


def test_methods_for_m6_restriction():
    assert "vhped_m6" in P.methods_for(CFG, "proposed", "T1_succession_unknown")
    assert "vhped_m6" not in P.methods_for(CFG, "proposed", "F3_visnoise")
    assert P.methods_for(CFG, "baseline", "F3_visnoise")[-1] == "va_hmm"
