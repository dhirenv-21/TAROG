"""Episode-study baselines B2 (hysteresis), B4 (HSMM), B5 (GatedEMA + identity) and the identity module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.temporal import METHODS
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.gated_variants import GatedEMAHysteresis, GatedEMAIdentity
from taorg.temporal.hsmm import DT_REF, DurationHSMM, weibull_bin_hazard
from taorg.temporal.identity import IdentityReliability

K = EdgeKey(1, 3, 0)


def frames(pattern: str, times=None, tracks=None):
    out = []
    for f, ch in enumerate(pattern):
        t = times[f] if times is not None else f / 30
        e = []
        if ch == "D":
            e = [EdgeObservation(K, "v", f, t, 0.8, True)]
        elif ch in "VOU":
            vis = {"V": VisibilityState.VISIBLE, "O": VisibilityState.OCCLUDED_OR_UNRELIABLE,
                   "U": VisibilityState.UNKNOWN}[ch]
            e = [EdgeObservation(K, "v", f, t, 0.0, False, object_visibility=vis)]
        out.append(FrameGraph("v", f, t, tracks(f, t) if tracks else [], e))
    return out


def outputs(model, fs):
    model.reset_video("v")
    return [K in {o.edge_key for o in model.step(fr).edge_observations} for fr in fs]


def test_hysteresis_needs_on_threshold_and_holds_until_off_threshold():
    g = dict(alpha=0.3, decay_visible=0.8, decay_occluded=0.95, decay_unknown=0.95)
    plain = outputs(GatedEMA(**g, score_threshold=0.2), frames("D" * 30 + "V" * 20))
    hyst = outputs(GatedEMAHysteresis(**g, score_threshold=0.5, off_ratio=0.4), frames("D" * 30 + "V" * 20))
    assert plain[0] and not hyst[0], "on-threshold 0.5 needs several detections"
    on_at = hyst.index(True)
    assert on_at >= 2
    # stays on below theta_on while above theta_off = 0.2, i.e. as long as the plain 0.2-threshold output
    assert hyst[30:] == plain[30:]
    with pytest.raises(ValueError):
        GatedEMAHysteresis(**g, score_threshold=0.5, off_ratio=1.0)


def test_gated_ema_identity_equals_gated_ema_without_tracks():
    g = dict(alpha=0.4, decay_visible=0.5, decay_occluded=0.97, decay_unknown=0.9, score_threshold=0.1)
    pat = "DDDDVVODDDOOOOUUU-DDVVVVVVVVVVVV"
    a = GatedEMA(**g)
    b = GatedEMAIdentity(**g, sigma_id=0.1, r_id=3.0)
    a.reset_video("v")
    b.reset_video("v")
    for fr in frames(pat):
        oa = {o.edge_key: o.predicate_score for o in a.step(fr).edge_observations}
        ob = {o.edge_key: o.predicate_score for o in b.step(fr).edge_observations}
        assert oa.keys() == ob.keys()
        for k in oa:
            assert oa[k] == pytest.approx(ob[k], rel=1e-12)


def _tracks(jump_at=None, speed=60.0, curve=False):
    def f(i, t):
        x = 100 + speed * t
        y = 100 + (40 * math.sin(2 * t) if curve else 0)
        if jump_at is not None and jump_at <= i < jump_at + 20:
            x += 300
        return [TrackObservation("v", i, t, 1, 0, (x, y, x + 60, y + 90)),
                TrackObservation("v", i, t, 3, 1, (400, 100, 460, 190))]
    return f


def test_identity_reliability_swap_vs_ordinary_motion():
    def min_u(trk, n=120, dt=1 / 30):
        idm = IdentityReliability(sigma_id=0.05, r_id=2.0)
        us = []
        for i in range(n):
            idm.update(trk(i, i * dt), dt)
            us.append(idm.track(1))
        return min(us), us
    u_swap, us = min_u(_tracks(jump_at=60))
    u_move, _ = min_u(_tracks(jump_at=None, speed=150.0, curve=True))
    assert u_swap < 0.05, "a 300-px identity jump is detected"
    assert u_move > 0.9, "smooth (even curving) motion is predicted by the velocity model"
    assert us[-1] > 0.5, "reliability recovers after the swap ends"
    idm = IdentityReliability(sigma_id=0.05, r_id=2.0)
    assert idm.switch_posterior(10.0, 1 / 30) == 1.0
    assert 0.0 <= idm.switch_posterior(0.0, 1 / 30) < 1e-3


def test_identity_irregular_dt_widens_the_null():
    idm = IdentityReliability(sigma_id=0.05, r_id=2.0)
    # a residual of 0.3 diagonals is a switch at dt = 1/30 but plausible after a 1-s gap
    assert idm.switch_posterior(0.3, 1 / 30) > 0.5 > idm.switch_posterior(0.3, 1.0)


def test_weibull_bin_hazard_shape_one_is_constant():
    h = weibull_bin_hazard(1.0, 2.0, 50, DT_REF)
    assert np.allclose(h, 1 - math.exp(-DT_REF / 2.0))
    inc = weibull_bin_hazard(3.0, 1.0, 50, DT_REF)
    assert np.all(np.diff(inc[:-1]) > 0)


def test_hsmm_normalised_substeps_and_relapse_capability():
    p = dict(theta_vis=0.05, theta_occ=0.95, theta_unk=0.95, false_pos=0.05, threshold=0.5, prior=0.5,
             on_shape=1.5, on_scale=3.0, off_shape=1.0, off_scale=5.0)
    m = DurationHSMM(**p)
    m.reset_video("v")
    for fr in frames("D" * 30 + "O" * 30):
        m.step(fr)
        on, off = m._on[K], m._off[K]
        assert on.sum() + off.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(on >= 0) and np.all(off >= 0)
    # irregular dt: one 2-bin step == two 1-bin steps under uninformative evidence (theta_unk = 1 - f)
    q = {**p, "theta_unk": 0.95}
    a, b = DurationHSMM(**q), DurationHSMM(**q)
    fa = frames("DDD" + "-" * 4, times=[0, 1 / 30, 2 / 30, 3 / 30, 4 / 30, 5 / 30, 6 / 30])
    fb = frames("DDD" + "-" * 2, times=[0, 1 / 30, 2 / 30, 4 / 30, 6 / 30])
    for model, fs in ((a, fa), (b, fb)):
        model.reset_video("v")
        for fr in fs:
            model.step(fr)
    assert a.p_on(K) == pytest.approx(b.p_on(K), rel=1e-9)


def test_registry_and_constructor_validation():
    for name in ("gated_ema_hyst", "hsmm", "gated_ema_id", "episode", "episode_id"):
        assert name in METHODS
    with pytest.raises(ValueError):
        DurationHSMM(on_shape=0.0)
    with pytest.raises(ValueError):
        IdentityReliability(sigma_id=0.0)
