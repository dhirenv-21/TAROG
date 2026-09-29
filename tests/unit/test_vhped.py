"""Tests: V-HPED eager reference matches research/vhped_formulation.md."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.temporal.vhped import (
    EPS_OCC, VHPED, emission, generator, propagate_constant, stationary, vhped_m2, vhped_m5, vhped_m6,
)

V = VisibilityState
K = EdgeKey(1, 2, 0)


def frames_for(pattern, fps=30.0, key=K, tracks=None):
    out = []
    for f, rec in enumerate(pattern):
        edges = []
        if rec is not None:
            obs, vis = rec
            edges = [EdgeObservation(key, "v", f, f / fps, 0.8 if obs else 0.0, observed=obs,
                                     object_visibility=vis)]
        out.append(FrameGraph("v", f, f / fps, tracks[f] if tracks else [], edges))
    return out


def run(model, frames, key=K):
    model.reset_video("v")
    bel, emitted = [], []
    for fr in frames:
        o = model.step(fr)
        bel.append(model.belief(key))
        emitted.append(any(e.edge_key == key for e in o.edge_observations))
    return bel, emitted


def test_generator_and_stationary():
    for eta in (0.0, 0.4):
        Q = generator(0.5, 0.05, 1.0, 4.0, eta)
        assert np.allclose(Q.sum(axis=1), 0.0) and (Q - np.diag(np.diag(Q)) >= 0).all()
        pi = stationary(Q)
        assert np.allclose(pi @ Q, 0.0, atol=1e-12) and abs(pi.sum() - 1) < 1e-12
    pi = stationary(generator(0.5, 0.05, 1.0, 4.0))
    assert abs((pi[2] + pi[3]) - 0.05 / 0.55) < 1e-12        # Z marginal = nu/(mu+nu)


def test_prior_relaxation_closed_form():
    mu, nu = 0.8, 0.1
    P = expm(generator(mu, nu, 1.0, 3.0) * (1 / 30))
    x = np.array([0.0, 0.0, 0.1, 0.9])
    pi = nu / (mu + nu)
    for n in (1, 10, 45):
        y = propagate_constant(x, P, (1, 1, 1, 1), n)
        expect = pi + (1.0 - pi) * math.exp(-(mu + nu) * n / 30)
        assert abs((y[2] + y[3]) - expect) < 1e-10


def test_occluded_absence_is_no_evidence_and_hold_horizon():
    # false_pos == EPS_OCC makes P(D=0|Z=1,V=0) == P(D=0|Z=0): occlusion carries no Z evidence.
    m = vhped_m2(mu=0.6, nu=0.02, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=EPS_OCC, threshold=0.5)
    on = [(True, V.VISIBLE)] * 30
    bel, emitted = run(m, frames_for(on + [(False, V.OCCLUDED_OR_UNRELIABLE)] * 90))
    pi = 0.02 / 0.62
    p0 = bel[29]
    for n in (1, 15, 60):
        expect = pi + (p0 - pi) * math.exp(-0.62 * n / 30)
        # M2 trusts flags with confusion 1e-6 (not exactly 0), so a residual V=1 mass leaks
        # O(1e-5) evidence; "no evidence" holds to that order.
        assert abs(bel[29 + n] - expect) < 1e-4
    t_hold = math.log((p0 - pi) / (0.5 - pi)) / 0.62
    drop = next(i for i in range(30, 120) if not emitted[i]) - 29
    assert abs(drop / 30 - t_hold) <= 1 / 30 + 1e-9


def test_visible_absence_is_finite_negative_evidence():
    strict = vhped_m2(mu=0.3, nu=0.01, gamma_o=0.5, gamma_v=3.0, miss=0.01, false_pos=0.01, threshold=0.5)
    lenient = vhped_m2(mu=0.3, nu=0.01, gamma_o=0.5, gamma_v=3.0, miss=0.4, false_pos=0.01, threshold=0.5)
    pat = [(True, V.VISIBLE)] * 30 + [(False, V.VISIBLE)] + [(True, V.VISIBLE)] * 5
    _, e_s = run(strict, frames_for(pat))
    _, e_l = run(lenient, frames_for(pat))
    assert not e_s[30] and e_l[30]           # one visible miss: dropped only if miss is implausible
    _, e_end = run(lenient, frames_for([(True, V.VISIBLE)] * 30 + [(False, V.VISIBLE)] * 30))
    assert not e_end[-1]                     # sustained visible absence terminates


def test_uncertain_flags_discount_isolated_visible_flag():
    base = dict(mu=0.3, nu=0.01, gamma_o=0.5, gamma_v=1.0, miss=0.01, false_pos=0.01, threshold=0.5)
    exact = vhped_m2(**base)
    noisy = VHPED(**base, r_ov=0.3, r_vo=0.3, rung="m3")
    occ = (False, V.OCCLUDED_OR_UNRELIABLE)
    pat = [(True, V.VISIBLE)] * 30 + [occ] * 5 + [(False, V.VISIBLE)] + [occ] * 5
    b_exact, _ = run(exact, frames_for(pat))
    b_noisy, _ = run(noisy, frames_for(pat))
    assert b_noisy[35] > b_exact[35] + 0.2   # the lone VIS flag is not taken as certain


def test_frame_rate_invariance():
    kw = dict(mu=0.8, nu=0.02, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=EPS_OCC, threshold=0.5)
    drops = {}
    for fps in (15.0, 30.0, 60.0):
        n_on, n_off = int(fps), int(3 * fps)
        pat = [(True, V.VISIBLE)] * n_on + [(False, V.OCCLUDED_OR_UNRELIABLE)] * n_off
        _, em = run(vhped_m2(**kw, dt_ref=1 / 30), frames_for(pat, fps=fps))
        drops[fps] = (next(i for i in range(n_on, n_on + n_off) if not em[i]) - n_on + 1) / fps
    assert abs(drops[15.0] - drops[30.0]) <= 1 / 15 + 1e-9
    assert abs(drops[60.0] - drops[30.0]) <= 1 / 30 + 1e-9


def test_lazy_equals_eager_within_tolerance():
    rng = np.random.default_rng(0)
    for _ in range(20):
        Q = generator(*rng.uniform(0.05, 3.0, 4), eta=float(rng.uniform(0, 0.8)))
        P = expm(Q / 30)
        L = rng.uniform(0.01, 1.0, 4)
        x = rng.dirichlet(np.ones(4))
        n = int(rng.integers(1, 400))
        eager = x.copy()
        for _ in range(n):
            eager = (eager @ P) * L
            eager /= eager.sum()
        lazy = propagate_constant(x, P, L, n)
        assert np.allclose(eager, lazy, rtol=1e-9, atol=1e-12)


def _tracks(n, jump_at=None):
    out = []
    for f in range(n):
        x2 = 100.0 if (jump_at is None or f < jump_at) else 250.0
        out.append([TrackObservation("v", f, f / 30, 1, 0, (0, 0, 60, 90)),
                    TrackObservation("v", f, f / 30, 2, 1, (x2, 0, x2 + 60, 90))])
    return out


def test_identity_reliability_holds_key_through_suspected_switch():
    kw = dict(mu=2.0, nu=0.02, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=0.01, threshold=0.5,
              r_ov=0.01, r_vo=0.01, eta=0.0, sigma_id=0.05, r_id=1.0)
    pat = [(True, V.VISIBLE)] * 30 + [None] * 15          # key vanishes (no record) at a box jump
    tr = _tracks(45, jump_at=30)
    b_id, _ = run(vhped_m5(**kw), frames_for(pat, tracks=tr))
    b_no, _ = run(VHPED(**kw, rung="m4"), frames_for(pat, tracks=tr))
    assert b_id[40] > b_no[40] + 0.1
    # without a jump, identity reliability changes (almost) nothing: the posterior switch
    # probability given d = 0 is small but not zero (formulation §7): u settles near 0.9994 and
    # beliefs differ by <= 0.004 here, so M5 ≈ M4.
    b_id2, _ = run(vhped_m5(**kw), frames_for(pat, tracks=_tracks(45)))
    b_no2, _ = run(VHPED(**kw, rung="m4"), frames_for(pat, tracks=_tracks(45)))
    assert np.allclose(b_id2, b_no2, atol=1e-2)


def test_cross_predicate_exclusivity():
    E = [[0.0] * 6 for _ in range(6)]
    E[0][1] = 0.9
    kw = dict(mu=0.2, nu=0.01, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=0.01, threshold=0.5,
              r_ov=0.01, r_vo=0.01, eta=0.0, sigma_id=0.3, r_id=3.0)
    succ = EdgeKey(1, 2, 1)

    def fr(f):
        edges = []
        if f < 30:
            edges.append(EdgeObservation(K, "v", f, f / 30, 0.8, observed=True))
        else:
            edges.append(EdgeObservation(K, "v", f, f / 30, 0.0, observed=False,
                                         object_visibility=V.UNKNOWN))
            edges.append(EdgeObservation(succ, "v", f, f / 30, 0.8, observed=True))
        return FrameGraph("v", f, f / 30, [], edges)
    frames = [fr(f) for f in range(45)]
    b_x, _ = run(vhped_m6(**kw, w_x=1.0, exclusivity=E), frames)
    b_0, _ = run(vhped_m6(**kw, w_x=0.0, exclusivity=E), frames)
    assert b_x[33] < 0.5 < b_0[33]
    zero = [[0.0] * 6 for _ in range(6)]
    b_z, _ = run(vhped_m6(**kw, w_x=1.0, exclusivity=zero), frames)
    assert np.allclose(b_z, b_0)


def test_validation_and_emission_rows():
    with pytest.raises(ValueError):
        VHPED(mu=0.0)
    with pytest.raises(ValueError):
        VHPED(threshold=1.0)
    with pytest.raises(ValueError):
        VHPED(cross=True)
    unk = emission("unk", 0.1, 0.01, 0.2, 0.2)
    vis = emission("vis", 0.1, 0.01, 0.2, 0.2)
    occ = emission("occ", 0.1, 0.01, 0.2, 0.2)
    assert np.allclose(np.add(vis, occ), unk)       # flag likelihoods partition the absence event
