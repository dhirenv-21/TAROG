"""Next-method decision rules (registered with the pre-registration)."""

from __future__ import annotations

from pathlib import Path

import yaml

from taorg.experiments.next_method import decision as D

CFG = yaml.safe_load(Path("configs/experiments/next_method.yaml").read_text(encoding="utf-8"))
CRIT = CFG["criteria"]
CONDS = list(CFG["conditions"])


def st(diff, lo, hi, mean_b=1.0, n=50):
    return {"mean_diff": diff, "ci_lo": lo, "ci_hi": hi, "mean_b": mean_b, "n": n}


def test_pairwise_win_loss_and_floors():
    assert D.pairwise({"J": st(0.03, 0.01, 0.05)}, CRIT)["win"]
    assert not D.pairwise({"J": st(0.015, 0.01, 0.02)}, CRIT)["win"]            # below practical margin
    r = D.pairwise({"J": st(0.05, 0.03, 0.07), "fpt_s": st(1.0, 0.5, 1.5, mean_b=2.0)}, CRIT)
    assert r["loss"] and not r["win"] and r["loss_reasons"] == ["fpt_s"]          # +50% false persistence
    # harm floor: a +0.1 s FPT increase on a 0.1 s base is not a loss (floor 0.25 s)
    assert D.pairwise({"J": st(0.03, 0.01, 0.05), "fpt_s": st(0.1, 0.05, 0.15, mean_b=0.1)}, CRIT)["win"]
    r = D.pairwise({"J": st(0.03, 0.01, 0.05), "churn": st(0.004, 0.002, 0.006, mean_b=0.01)}, CRIT)
    assert not r["win"] and not r["loss"]                                          # churn guard fails, not a loss
    assert D.pairwise({"J": st(-0.03, -0.05, -0.01)}, CRIT)["loss"]
    # too few paired seeds: harm metric ignored
    assert D.pairwise({"J": st(0.03, 0.01, 0.05), "rdd_mean_s": st(1.0, 0.5, 1.5, n=5)}, CRIT)["win"]


W = {"win": True, "loss": False}
N = {"win": False, "loss": False}
L = {"win": False, "loss": True}


def _all(default, **over):
    d = {c: dict(default) for c in CONDS}
    d.update(over)
    return d


def test_gate_go_narrow_stop():
    prim = {c: W for c in CRIT["primary_conditions"]}
    adapt = {c: W for c in CRIT["adapt_conditions"]}
    go = D.gate(_all(N, **prim), _all(N, **adapt), _all(N, **adapt), _all(N, **adapt),
                _all(N, **prim), _all(N, **adapt), 1.5, CRIT)
    assert go["decision"] == "GO" and go["algorithm_contribution"] == "SUPPORTED"
    # gain over B2 but the fixed buffer and the geometric learner match it -> NARROW, UNSUPPORTED
    r = D.gate(_all(N, **prim), _all(N), _all(N, **adapt), _all(N), _all(N, **prim), _all(N), 1.5, CRIT)
    assert r["decision"] == "NARROW" and r["algorithm_contribution"] == "UNSUPPORTED"
    # beats the geometric learner only -> NARROW, PARTIAL
    r = D.gate(_all(N, **prim), _all(N), _all(N, **adapt), _all(N, **adapt), _all(N, **prim), _all(N), 1.5, CRIT)
    assert r["decision"] == "NARROW" and r["algorithm_contribution"] == "PARTIAL"
    # no primary win -> STOP
    assert D.gate(_all(N), _all(N, **adapt), _all(N, **adapt), _all(N, **adapt), None, None, 1.0, CRIT)["decision"] == "STOP"
    # two losses vs B2 -> STOP even with primary wins
    r = D.gate(_all(N, **prim, C2_hidden_termination=L, C6_id_switch=L), _all(N, **adapt), _all(N, **adapt),
               _all(N, **adapt), _all(N, **prim), _all(N, **adapt), 1.0, CRIT)
    assert r["decision"] == "STOP"
    # cost gate
    r = D.gate(_all(N, **prim), _all(N, **adapt), _all(N, **adapt), _all(N, **adapt),
               _all(N, **prim), _all(N, **adapt), 5.0, CRIT)
    assert r["decision"] == "NARROW" and not r["gates"]["G7_cost"]
