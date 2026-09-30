"""Tests for the pre-registered Phase 4 decision rules."""

from __future__ import annotations

from pathlib import Path

import yaml

from taorg.experiments.phase4.decision import (
    cell_outcome, churn_margin, decide_h2, decide_h3, decide_h4, frontier_vs, rung_retained,
    shift_vs,
)

CFG = yaml.safe_load(Path("configs/experiments/phase4.yaml").read_text(encoding="utf-8"))
CRIT = CFG["criteria"]
FR = {"min_hv_gain": CRIT["min_hv_gain"], "churn_margin_abs": CRIT["churn_margin_abs"],
      "churn_margin_rel": CRIT["churn_margin_rel"]}


def s(mean, lo, hi, p=float("nan")):
    return {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi, "p_holm": p}


def test_churn_margin():
    assert churn_margin(0.01, 0.005, 0.1) == 0.005
    assert abs(churn_margin(0.2, 0.005, 0.1) - 0.02) < 1e-12


def test_frontier_win_requires_all_checks():
    ok = frontier_vs(s(0.05, 0.03, 0.07), s(0.0, -0.001, 0.001), 0.01, FR)
    assert ok["win"] and not ok["loss"]
    small = frontier_vs(s(0.015, 0.01, 0.02), s(0, 0, 0), 0.01, FR)
    assert not small["win"] and "hv_gain" in small["failed_checks"]
    ci = frontier_vs(s(0.05, -0.01, 0.1), s(0, 0, 0), 0.01, FR)
    assert not ci["win"] and "hv_ci" in ci["failed_checks"]
    churn = frontier_vs(s(0.05, 0.03, 0.07), s(0.02, 0.01, 0.03), 0.01, FR)
    assert not churn["win"] and "churn_ni" in churn["failed_checks"]
    no_churn = frontier_vs(s(0.05, 0.03, 0.07), None, 0.01, FR)
    assert not no_churn["win"]


def test_frontier_identity_default_ni():
    bad = frontier_vs(s(0.05, 0.03, 0.07), s(0, 0, 0), 0.01, FR, s(-0.05, -0.08, -0.03), -0.02)
    assert not bad["win"] and "default_frontier_ni" in bad["failed_checks"]
    good = frontier_vs(s(0.05, 0.03, 0.07), s(0, 0, 0), 0.01, FR, s(0.0, -0.01, 0.01), -0.02)
    assert good["win"]


def test_frontier_loss():
    assert frontier_vs(s(-0.05, -0.07, -0.03), s(0, 0, 0), 0.01, FR)["loss"]
    assert not frontier_vs(s(-0.05, -0.07, 0.01), s(0, 0, 0), 0.01, FR)["loss"]
    assert not frontier_vs(s(-0.01, -0.02, -0.005), s(0, 0, 0), 0.01, FR)["loss"]


def test_cell_outcome():
    w = {"win": True, "loss": False}
    n = {"win": False, "loss": False}
    lo = {"win": False, "loss": True}
    assert cell_outcome({"a": w, "b": w}) == "win"
    assert cell_outcome({"a": w, "b": n}) == "neutral"
    assert cell_outcome({"a": w, "b": lo}) == "loss"


def _cells(conds, gaps, outcomes):
    it = iter(outcomes)
    return {(c, g): next(it) for c in conds for g in gaps}


F = ["F1", "F2", "F3", "F4", "F5", "F7", "F8"]
G = [3, 10, 20]
NI = _cells(["N0", "N1", "N2"], G, ["neutral"] * 9)


def test_h2_pass_mixed_fail():
    c = CRIT["h2"]
    passing = _cells(F, G, ["win"] * 12 + ["neutral"] * 9)
    assert decide_h2(passing, NI, c)["decision"] == "PASS"
    # 12 wins but concentrated in 4 conditions -> still PASS (>= 4 conditions)
    ni_loss = dict(NI)
    ni_loss[("N0", 3)] = "loss"
    assert decide_h2(passing, ni_loss, c)["decision"] == "MIXED"
    few = _cells(F, G, ["win"] * 4 + ["neutral"] * 17)
    assert decide_h2(few, NI, c)["decision"] == "FAIL"
    one_cond = _cells(F, G, ["win"] * 3 + ["neutral"] * 18)
    assert decide_h2(one_cond, NI, c)["decision"] == "FAIL"
    mid = _cells(F, G, ["win", "neutral", "neutral"] * 3 + ["neutral"] * 12)  # 3 wins in 3 conds
    assert decide_h2(mid, NI, c)["decision"] == "FAIL"
    mixed = _cells(F, G, ["win", "win", "neutral"] * 3 + ["neutral"] * 12)  # 6 wins, 3 conds
    assert decide_h2(mixed, NI, c)["decision"] == "MIXED"
    lossy = _cells(F, G, ["win", "win", "loss"] * 3 + ["loss"] * 3 + ["neutral"] * 9)
    assert decide_h2(lossy, NI, c)["decision"] == "FAIL"  # 6 losses >= 6 wins


def test_h3_rule():
    c = CRIT["h3"]
    ben = {("T1", g): "win" for g in G} | {("T2", 3): "win", ("T2", 10): "neutral", ("T2", 20): "neutral"}
    ni = {("T3", g): "neutral" for g in G} | {("F2", g): "neutral" for g in G}
    assert decide_h3(ben, ni, c)["decision"] == "PASS"
    ni1 = dict(ni)
    ni1[("T3", 3)] = "loss"
    assert decide_h3(ben, ni1, c)["decision"] == "MIXED"
    ni1[("T3", 10)] = "loss"
    assert decide_h3(ben, ni1, c)["decision"] == "FAIL"
    ben1 = {k: "neutral" for k in ben} | {("T1", 3): "win"}
    assert decide_h3(ben1, ni, c)["decision"] == "FAIL"


def _z():
    return s(0.0, -0.001, 0.001, 1.0)


def test_shift_persistence_win_and_mirror_loss():
    c = CRIT["h4"]
    a, r = CRIT["churn_margin_abs"], CRIT["churn_margin_rel"]
    r1 = shift_vs(s(0.1, 0.05, 0.15, 0.001), _z(), _z(), _z(), 0.01, 0.01, c, a, r)
    assert r1["win"] and r1["win_kind"] == "persistence" and not r1["loss"]
    r2 = shift_vs(s(-0.1, -0.15, -0.05, 0.001), _z(), _z(), _z(), 0.01, 0.01, c, a, r)
    assert r2["loss"] and r2["loss_kind"] == "persistence" and not r2["win"]
    # Significant but delay NI violated -> no persistence win
    r3 = shift_vs(s(0.1, 0.05, 0.15, 0.001), s(0.1, 0.05, 0.2), _z(), _z(), 0.01, 0.01, c, a, r)
    assert not r3["win"]


def test_shift_termination_and_churn_wins():
    c = CRIT["h4"]
    a, r = CRIT["churn_margin_abs"], CRIT["churn_margin_rel"]
    d = shift_vs(s(0.0, -0.01, 0.01, 1.0), s(-0.1, -0.15, -0.08), _z(), _z(), 0.01, 0.01, c, a, r)
    assert d["win"] and d["win_kind"] == "termination_or_churn"
    ch = shift_vs(s(0.0, -0.01, 0.01, 1.0), _z(), _z(), s(-0.1, -0.12, -0.08), 0.15, 0.05, c, a, r)
    assert ch["win"]
    # F1 non-inferiority violated -> no termination win
    bad = shift_vs(s(-0.05, -0.08, -0.02, 1.0), s(-0.1, -0.15, -0.08), _z(), _z(), 0.01, 0.01, c, a, r)
    assert not bad["win"]
    # mirror: comparator much faster -> treatment loss
    lo = shift_vs(s(0.0, -0.01, 0.01, 1.0), s(0.1, 0.08, 0.15), _z(), _z(), 0.01, 0.01, c, a, r)
    assert lo["loss"]


def test_h4_rule():
    c = CRIT["h4"]
    cells = {(f"SH{i}", g): "neutral" for i in range(1, 9) for g in G}
    keys = list(cells)
    for k in keys[:12]:
        cells[k] = "win"
    assert decide_h4(cells, c)["decision"] == "PASS"
    for k in keys[12:15]:
        cells[k] = "loss"
    assert decide_h4(cells, c)["decision"] == "MIXED"
    few = {k: ("win" if i < 4 else "neutral") for i, k in enumerate(keys)}
    assert decide_h4(few, c)["decision"] == "FAIL"


def test_rung_retained():
    c = CRIT["ablation"]
    tgt = [s(0.03, 0.01, 0.05)] + [s(0.0, -0.01, 0.01)] * 2
    assert rung_retained(tgt, [s(0, -0.01, 0.01)], c)["retained"]
    assert not rung_retained(tgt, [s(-0.05, -0.07, -0.03)], c)["retained"]
    assert not rung_retained([s(0.0, -0.01, 0.01)] * 3, [], c)["retained"]
