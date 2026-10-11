"""B2 + survival-calibrated non-visible hold (formulation §4-§9): numerics, properties P1-P3,
B2 equivalence, recurrence, irregular timestamps, determinism, causality, leakage."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal import METHODS
from taorg.temporal.cure import DT_REF, B2SurvivalHold, RunCounts, fit_prior, km_survival
from taorg.temporal.gated_variants import GatedEMAHysteresis

K = EdgeKey(1, 3, 0)
B2P = dict(alpha=0.4, decay_visible=0.6, decay_occluded=0.9, decay_unknown=0.5, score_threshold=0.4, off_ratio=0.5)
H = 90


def flat_prior(h: float = 0.05, hu: float | None = None) -> dict:
    hu = h if hu is None else hu
    return {"horizon_s": 3.0, "h0": {"OCCLUDED_OR_UNRELIABLE": [h] * H, "UNKNOWN": [hu] * H, "ALL": [h] * H},
            "geom": {s: {"p": 0.05, "pi": 0.7} for s in ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN", "ALL")}}


def rec(f, t, ch, key=K):
    if ch == "D":
        return EdgeObservation(key, "v", f, t, 0.8, True)
    vis = {"V": VisibilityState.VISIBLE, "O": VisibilityState.OCCLUDED_OR_UNRELIABLE, "U": VisibilityState.UNKNOWN}[ch]
    return EdgeObservation(key, "v", f, t, 0.0, False, object_visibility=vis)


def stream(pat, times=None, key=K):
    return [FrameGraph("v", f, times[f] if times else f / 30, [], [] if ch == "-" else [rec(f, times[f] if times else f / 30, ch, key)])
            for f, ch in enumerate(pat)]


def outs(m, fs):
    m.reset_video("v")
    return [K in {o.edge_key for o in m.step(fr).edge_observations} for fr in fs]


def test_identical_to_b2_without_non_visible_absences():
    rng = random.Random(1)
    pat = "".join(rng.choice("DDDV") for _ in range(400))
    b2 = outs(GatedEMAHysteresis(**B2P), stream(pat))
    for mode in ("km", "static", "pooled", "geometric", "buffer"):
        assert outs(B2SurvivalHold(**B2P, mode=mode, prior=flat_prior()), stream(pat)) == b2, mode


def test_holds_through_occlusion_and_uses_b2_for_visible():
    pat = "D" * 30 + "O" * 40 + "D" * 10 + "V" * 20
    b2 = outs(GatedEMAHysteresis(**B2P), stream(pat))
    hold = outs(B2SurvivalHold(**B2P, mode="buffer", n_occ=60, n_unk=0), stream(pat))
    assert not all(b2[30:70]), "B2 (decay 0.9) drops during a 40-frame occlusion"
    assert all(hold[30:80]), "buffer of 60 bins bridges it"
    assert hold[80:] == b2[80:], "visible absences follow B2 exactly"


def test_hold_expires_and_visible_after_occlusion_terminates():
    m = B2SurvivalHold(**B2P, mode="buffer", n_occ=20, n_unk=20)
    o = outs(m, stream("D" * 30 + "O" * 60))
    assert all(o[30:50]) and not any(o[51:]), "hold of 20 bins from the last detection"
    o = outs(B2SurvivalHold(**B2P, mode="buffer", n_occ=80, n_unk=80), stream("D" * 30 + "O" * 10 + "V" * 30))
    assert not o[-1], "a visible absence after the hidden period lets B2's rule terminate"


def test_property_p1_monotone_and_p2_static_equals_buffer():
    rng = random.Random(2)
    h0 = [min(0.9, max(0.001, rng.random() * 0.2)) for _ in range(H)]
    prior = {"horizon_s": 3.0, "h0": {"OCCLUDED_OR_UNRELIABLE": h0, "UNKNOWN": h0, "ALL": h0},
             "geom": {s: {"p": 0.05, "pi": 0.7} for s in ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN", "ALL")}}
    m = B2SurvivalHold(**B2P, mode="static", tau=0.4, prior=prior)
    m.reset_video("v")
    q = [m.gap_posterior("OCCLUDED_OR_UNRELIABLE", l) for l in range(H + 3)]
    assert all(b <= a + 1e-12 for a, b in zip(q, q[1:])), "P1: q non-increasing"
    Lstar = max(l for l in range(H) if q[l] >= 0.4)
    pat = "D" * 30 + "O" * 120
    a = outs(B2SurvivalHold(**B2P, mode="static", tau=0.4, prior=prior), stream(pat))
    b = outs(B2SurvivalHold(**B2P, mode="buffer", n_occ=Lstar, n_unk=Lstar), stream(pat))
    assert a == b, "P2: static rule == per-flag buffer with L* = max{l: q(l) >= tau}"


def test_property_p3_threshold_rule_minimises_expected_cost():
    """Brute force over hold lengths L for random cure models and cost ratios."""
    rng = random.Random(3)
    for _ in range(200):
        Hs = 30
        pi = rng.uniform(0.05, 0.95)
        hz = [0.0] + [rng.uniform(0.0, 0.4) for _ in range(Hs)]
        hz[Hs] = 1.0                                      # every gap returns by H (S_G(H) = 0)
        SG = km_survival(hz)
        cfp, cfn = rng.uniform(0.1, 5), rng.uniform(0.1, 5)
        tau = cfp / (cfp + cfn)
        cost = [cfp * (1 - pi) * L + cfn * pi * sum(SG[l] for l in range(L + 1, Hs + 1)) for L in range(Hs + 1)]
        best = min(cost)
        q = [pi * SG[l] / (pi * SG[l] + 1 - pi) for l in range(Hs + 1)]
        Lq = max([l for l in range(1, Hs + 1) if q[l] >= tau] or [0])
        assert cost[Lq] <= best + 1e-9
        # the mixture identity q(l) = 1 - S(H)/S(l) with S = 1 - pi + pi S_G
        S = [1 - pi + pi * x for x in SG]
        for l in range(1, Hs):
            assert q[l] == pytest.approx(1 - S[Hs] / S[l], abs=1e-12)


def test_online_counts_match_brute_force_km():
    """Risk sets from online bookkeeping equal those computed from the run list."""
    rng = random.Random(4)
    pat = ""
    lengths = []
    for _ in range(25):
        L = rng.randint(1, 120)
        pat += "D" * 5 + ("O" * L)
        lengths.append(L)
    pat += "D" * 5
    m = B2SurvivalHold(**{**B2P, "score_threshold": 0.3}, mode="km", tau=0.01, kappa=1.0, prior=flat_prior(0.02))
    m.reset_video("v")
    for fr in stream(pat):
        m.step(fr)
    Y = [0.0] * (H + 1)
    d = [0.0] * (H + 1)
    for L in lengths:                     # re-detected at bin L+1 if L+1 <= H, else censored after bin H
        for b in range(1, min(L, H) + 1):
            Y[b] += 1
        if L + 1 <= H:
            Y[L + 1] += 1
            d[L + 1] += 1
    assert m._counts.Y["OCCLUDED_OR_UNRELIABLE"] == Y
    assert m._counts.d["OCCLUDED_OR_UNRELIABLE"] == d


def test_online_learning_moves_the_horizon_and_static_does_not():
    gappy = ("D" * 10 + "O" * 40) * 6           # every OCC run returns after 40 bins
    h0 = [0.3] * 5 + [1e-4] * (H - 5)           # prior says: return within 5 bins or never
    prior = {**flat_prior(), "h0": {"OCCLUDED_OR_UNRELIABLE": h0, "UNKNOWN": h0, "ALL": h0}}
    km = B2SurvivalHold(**B2P, mode="km", tau=0.5, kappa=5.0, prior=prior)
    st = B2SurvivalHold(**B2P, mode="static", tau=0.5, kappa=5.0, prior=prior)
    o_km, o_st = outs(km, stream(gappy)), outs(st, stream(gappy))
    late_km = sum(o_km[-50:-10])
    late_st = sum(o_st[-50:-10])
    assert late_km > late_st, "after observing long gaps, the online rule holds longer than the static prior"


def test_recurrence_after_visible_termination_is_a_new_birth_not_a_hold():
    m = B2SurvivalHold(**B2P, mode="buffer", n_occ=90, n_unk=90)
    o = outs(m, stream("D" * 30 + "V" * 20 + "D" * 20))
    assert not o[45] and o[-1], "visible termination ends the episode; recurrence re-enters via theta_on"


def test_irregular_timestamps_count_elapsed_bins():
    times = [i / 30 for i in range(30)] + [1.0 + 0.4, 1.0 + 1.2]       # big jumps during occlusion
    m = B2SurvivalHold(**B2P, mode="buffer", n_occ=20, n_unk=20)
    o = outs(m, stream("D" * 30 + "OO", times=times))
    l1 = round((times[30] - times[29]) / DT_REF)
    l2 = round((times[31] - times[29]) / DT_REF)
    assert o[30] == (l1 <= 20) and o[31] == (l2 <= 20)


def test_numerics_determinism_causality_and_validation():
    rng = random.Random(5)
    pat = "".join(rng.choice("DDDOOUV-") for _ in range(600))
    for mode in ("km", "pooled", "geometric"):
        a = outs(B2SurvivalHold(**B2P, mode=mode, tau=0.3, kappa=2.0, prior=flat_prior(0.999, 1e-4)), stream(pat))
        b = outs(B2SurvivalHold(**B2P, mode=mode, tau=0.3, kappa=2.0, prior=flat_prior(0.999, 1e-4)), stream(pat))
        assert a == b
        alt = pat[:300] + "".join(rng.choice("DOV") for _ in range(300))
        c = outs(B2SurvivalHold(**B2P, mode=mode, tau=0.3, kappa=2.0, prior=flat_prior(0.999, 1e-4)), stream(alt))
        assert a[:300] == c[:300], "output at n must not depend on frames after n"
        m = B2SurvivalHold(**B2P, mode=mode, tau=0.3, kappa=2.0, prior=flat_prior(0.999, 1e-4))
        m.reset_video("v")
        for fr in stream(pat):
            m.step(fr)
            for s in m._strata:
                for l in (1, 10, 89, 90, 200):
                    q = m.gap_posterior(s, l)
                    assert 0.0 <= q <= 1.0 and math.isfinite(q)
    with pytest.raises(ValueError):
        B2SurvivalHold(**B2P, mode="km")                                  # prior required
    with pytest.raises(ValueError):
        B2SurvivalHold(**B2P, mode="km", prior={**flat_prior(), "horizon_s": 2.0})
    with pytest.raises(ValueError):
        B2SurvivalHold(**B2P, mode="nope", prior=flat_prior())


def test_fit_prior_uses_only_input_records_and_matches_counts():
    fs = stream("D" * 5 + "O" * 10 + "D" * 5 + "U" * 100 + "D")
    pr = fit_prior([fs], 3.0)
    h = pr["h0"]["OCCLUDED_OR_UNRELIABLE"]
    assert h[10] == pytest.approx(1.0 - 1e-3, abs=2e-3) or h[10] > 0.9, "the OCC run returned at bin 11"
    assert pr["n_runs"]["OCCLUDED_OR_UNRELIABLE"] == 1 and pr["n_runs"]["UNKNOWN"] == 1
    assert METHODS["b2_hold_km"](**B2P, prior=pr).mode == "km"
    # the function signature only accepts frame streams: no ground truth can be passed
    import inspect
    assert list(inspect.signature(fit_prior).parameters) == ["streams", "horizon_s"]
