"""Episode study decision rules (registered with the pre-registration, before any comparison)."""

from __future__ import annotations

from pathlib import Path

import yaml

from taorg.experiments.episodes import decision as D

CRIT = yaml.safe_load(Path("configs/experiments/episodes.yaml").read_text(encoding="utf-8"))["criteria"]
CAL = CRIT["benefit_conditions"] + CRIT["ni_conditions"]


def st(diff, lo, hi, mean_b=1.0, n=50):
    return {"mean_diff": diff, "ci_lo": lo, "ci_hi": hi, "mean_b": mean_b, "mean_a": mean_b + diff, "n": n}


def test_win_by_quality_requires_guards():
    s = {"J": st(0.03, 0.01, 0.05)}
    assert D.pairwise(s, "C1_gaps", CRIT)["win"]
    s["churn"] = st(0.02, 0.01, 0.03, mean_b=0.01)          # churn guard violated
    r = D.pairwise(s, "C1_gaps", CRIT)
    assert not r["win"] and "churn_noninferior" in r["failed_guards"]


def test_win_by_target_harm_route():
    s = {"J": st(0.005, -0.005, 0.015), "frr": st(-0.2, -0.3, -0.1, mean_b=0.5)}
    r = D.pairwise(s, "C5_boundary_flicker", CRIT)
    assert r["win"] and r["route"] == "harm"
    # harm reduction too small (< 25 % of comparator)
    s["frr"] = st(-0.05, -0.08, -0.02, mean_b=0.5)
    assert not D.pairwise(s, "C5_boundary_flicker", CRIT)["win"]
    # no target harm in C1 -> harm route impossible
    assert not D.pairwise({"J": st(0.005, -0.005, 0.015)}, "C1_gaps", CRIT)["win"]


def test_loss_takes_precedence_and_recurrence_delay_loss():
    s = {"J": st(0.05, 0.03, 0.07), "rdd_mean_s": st(0.2, 0.15, 0.25)}
    r = D.pairwise(s, "C4_recurrence", CRIT)
    assert r["loss"] and not r["win"] and r["loss_reasons"] == ["rdd_mean_s"]
    s = {"J": st(-0.03, -0.05, -0.01)}
    assert D.pairwise(s, "C1_gaps", CRIT)["loss"]
    # harm increase with zero comparator harm is a loss
    s = {"J": st(0.0, -0.005, 0.005), "ghost_time_s": st(0.5, 0.2, 0.8, mean_b=0.0)}
    assert D.pairwise(s, "C6_id_switch", CRIT)["loss"]


def test_rdd_guard_needs_enough_pairs():
    s = {"J": st(0.03, 0.01, 0.05), "rdd_mean_s": st(0.05, -0.1, 0.2, n=5)}
    assert D.pairwise(s, "C4_recurrence", CRIT)["win"]           # guard not applicable (n < 10)
    s["rdd_mean_s"] = st(0.05, -0.1, 0.2, n=40)
    assert not D.pairwise(s, "C4_recurrence", CRIT)["win"]


def test_cell_outcome_and_comparator_sets():
    w, n, l = {"win": True, "loss": False}, {"win": False, "loss": False}, {"win": False, "loss": True}
    assert D.cell_outcome({"a": w, "b": w}) == "win"
    assert D.cell_outcome({"a": w, "b": n}) == "neutral"
    assert D.cell_outcome({"a": w, "b": l}) == "loss"
    assert "gated_ema_id" in D.comparators_for("C6_id_switch", CRIT)
    assert "gated_ema_id" not in D.comparators_for("C2_hidden_termination", CRIT)


def _ladder(j_diff=0.0, j_lo=-0.005, j_hi=0.005):
    return {c: {"J": st(j_diff, j_lo, j_hi), "std_mean_s": st(0.0, -0.01, 0.01),
                "ghost_time_s": st(0.0, -0.01, 0.01)} for c in CAL}


def test_ladder_rules_and_treatment_selection():
    pc = _ladder()
    pc["C6_id_switch"]["ghost_time_s"] = st(-0.5, -0.7, -0.3, mean_b=1.0)
    assert D.retain_identity(pc, CRIT, CAL)["retained"]
    pc["C7_motion"]["J"] = st(-0.05, -0.07, -0.03)               # identity false alarms cost quality
    assert not D.retain_identity(pc, CRIT, CAL)["retained"]

    pc = _ladder()
    pc["C2_hidden_termination"]["std_mean_s"] = st(-0.2, -0.3, -0.1)
    assert D.retain_context(pc, CRIT, CAL)["retained"]

    pc = _ladder()
    pc["C2_hidden_termination"]["J"] = st(0.03, 0.01, 0.05)
    assert D.retain_history(pc, st(0.012, 0.002, 0.02), CRIT, CAL)["retained"]
    assert not D.retain_history(pc, st(0.005, 0.001, 0.01), CRIT, CAL)["retained"]   # pooled gain too small

    rungs = ["episode", "episode_id", "episode_ctx", "episode_hist"]
    assert D.select_treatment(False, True, True, rungs) == "episode"
    assert D.select_treatment(True, False, True, rungs) == "episode_id"
    assert D.select_treatment(True, True, True, rungs) == "episode_hist"


def _cells(ben="win", ni="neutral", tr="neutral"):
    out = {c: ben for c in CRIT["benefit_conditions"]}
    out.update({c: ni for c in CRIT["ni_conditions"]})
    out.update({c: tr for c in CRIT["transfer_conditions"]})
    return out


def test_study_decision_go_narrow_stop():
    vs_simple = {c: {"win": True} for c in CRIT["relapse_conditions"]}
    assert D.study_decision(_cells(), vs_simple, 0.05, 3.0, "GO", CRIT)["decision"] == "GO"
    # latency too high -> not GO
    assert D.study_decision(_cells(), vs_simple, 2.0, 3.0, "GO", CRIT)["decision"] == "NARROW"
    # a calibrated loss -> not GO
    c = _cells()
    c["C4_recurrence"] = "loss"
    assert D.study_decision(c, vs_simple, 0.05, 3.0, None, CRIT)["decision"] == "NARROW"
    # simple baseline never beaten in relapse conditions -> STOP (simplification)
    none = {c: {"win": False} for c in CRIT["relapse_conditions"]}
    r = D.study_decision(_cells(), none, 0.05, 3.0, None, CRIT)
    assert r["decision"] == "STOP" and r["stop_checks"]["simple_baseline_not_beaten"]
    # <= 1 benefit win -> STOP
    c = _cells(ben="neutral")
    c["C5_boundary_flicker"] = "win"
    assert D.study_decision(c, vs_simple, 0.05, 3.0, None, CRIT)["decision"] == "STOP"
    # half-sample control STOP blocks GO
    assert D.study_decision(_cells(), vs_simple, 0.05, 3.0, "STOP", CRIT)["decision"] == "NARROW"


def test_registered_answers():
    w = {"win": True, "loss": False}
    p1 = {c: {b: w for b in CRIT["comparators"]} for c in ("C3_misleading_post", "C5_boundary_flicker")}
    assert D.answer_q1(p1, CRIT)["answer"] == "YES"
    rdd = {c: {b: st(0.0, -0.05, 0.05) for b in CRIT["comparators"]} for c in ("C4_recurrence", "C5_boundary_flicker")}
    assert D.answer_q2(rdd, CRIT)["answer"] == "NO"
    rdd["C4_recurrence"]["hsmm"] = st(0.2, 0.1, 0.3)
    assert D.answer_q2(rdd, CRIT)["answer"].startswith("YES")
    assert D.answer_q3({"win": True}, st(-0.3, -0.5, -0.1))["answer"] == "YES"
    assert D.answer_q3({"win": True}, st(-0.1, -0.3, 0.1))["answer"] == "NO"
    assert D.answer_q5(_cells(tr="win"), CRIT)["answer"] == "YES"
    assert D.answer_q5(_cells(ben="neutral", tr="win"), CRIT)["answer"].startswith("NO IN-DISTRIBUTION")
