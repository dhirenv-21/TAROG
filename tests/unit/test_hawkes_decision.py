"""Tests for the Phase 5H Stage 0 decision rule."""

from taorg.hawkes.decision import decide_stage0


def st(m, lo, hi):
    return {"mean": m, "ci_lo": lo, "ci_hi": hi}


BASE = dict(g_dur=st(0.2, 0.1, 0.3), g_base=st(0.5, 0.4, 0.6), g_self=0.3, s_real=40,
            s_null=[3, 5, 2, 4, 6], rho_real=0.6, rho_null=[0.1, 0.05, 0.12, 0.08, 0.02], c3_ratio=0.10)
NULL = [0.001 * i for i in range(-9, 10)]      # max 0.009, median 0.0


def test_strong():
    r = decide_stage0(st(0.05, 0.03, 0.07), NULL, **BASE)
    assert r["label"] == "STRONG" and r["C1"] and r["C3"] and abs(r["perm_p"] - 0.05) < 1e-12


def test_absent_when_ci_includes_zero_or_below_null():
    assert decide_stage0(st(0.05, -0.01, 0.1), NULL, **BASE)["label"] == "ABSENT"
    assert decide_stage0(st(0.005, 0.001, 0.009), NULL, **BASE)["label"] == "ABSENT"


def test_weak_mixed_paths():
    small = decide_stage0(st(0.02, 0.01, 0.03), NULL, **BASE)       # excess 0.02 < 0.03
    assert small["label"] == "WEAK-MIXED" and not small["C3"]
    b = dict(BASE)
    b["g_dur"] = st(-0.1, -0.2, 0.0)
    assert decide_stage0(st(0.05, 0.03, 0.07), NULL, **b)["label"] == "WEAK-MIXED"
    b = dict(BASE)
    b["s_real"] = 5
    r = decide_stage0(st(0.05, 0.03, 0.07), NULL, **b)
    assert r["label"] == "WEAK-MIXED" and not r["C4"]


def test_nonpositive_self_gain():
    b = dict(BASE)
    b["g_self"] = -0.01
    assert decide_stage0(st(0.05, 0.03, 0.07), NULL, **b)["C3"]
