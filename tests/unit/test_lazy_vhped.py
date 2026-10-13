"""Phase 5: lazy maintenance equals the eager V-HPED reference (research/derivations/lazy_maintenance.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from taorg.experiments.phase5.traces import TraceConfig, make_trace
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.temporal.lazy_vhped import (LazyVHPED, hilbert_distance, logit, make_certificate,
                                       projective_diameter)
from taorg.temporal.vhped import VHPED, emission, generator
from taorg.utils.seed import set_seed

BASE = dict(mu=0.5, nu=0.01, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=0.01, threshold=0.5)
RUNGS = {
    "m2": {},
    "m3": {"r_ov": 0.05, "r_vo": 0.05},
    "m4": {"r_ov": 0.05, "r_vo": 0.05, "eta": 0.3},
    "m5": {"r_ov": 0.05, "r_vo": 0.05, "identity": True},
}
TOL = 1e-9


def exclusivity(P: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 1, (P, P)) * (rng.random((P, P)) < 0.4)
    np.fill_diagonal(E, 0)
    return E.tolist()


def compare(params: dict, frames: list[FrameGraph]) -> tuple[LazyVHPED, int]:
    """Run eager and lazy side by side; assert emitted sets/scores and all beliefs agree."""
    eager, lazy = VHPED(**params), LazyVHPED(**params)
    eager.reset_video("v")
    lazy.reset_video("v")
    near = 0
    for fr in frames:
        e = {o.edge_key: o.predicate_score for o in eager.step(fr).edge_observations}
        z = {o.edge_key: o.predicate_score for o in lazy.step(fr).edge_observations}
        for k in set(e) | set(z):
            if (k in e) != (k in z):
                b = eager.belief(k)
                assert abs(b - params["threshold"]) < TOL, (fr.frame_idx, k, b)
                near += 1
            else:
                assert abs(e[k] - z[k]) < TOL
    for k in eager._x:
        assert abs(eager.belief(k) - lazy.belief(k)) < TOL
    assert set(eager._x) == set(lazy._x) | set(lazy._frozen)
    return lazy, near


@pytest.mark.parametrize("rung", list(RUNGS))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_randomized_equivalence(rung, seed):
    set_seed(52000 + seed)
    cfg = TraceConfig(n_frames=500, n_keys=150, p_jitter=0.01, p_jump=0.01 if rung == "m5" else 0.0)
    frames = make_trace(cfg, np.random.default_rng(52000 + seed))
    lazy, near = compare({**BASE, **RUNGS[rung]}, frames)
    c = lazy.counters()
    if rung == "m5":
        assert c["freezes"] == 0 and c["identity_fallback_steps"] == len(frames)   # eager fallback
    else:
        assert c["freezes"] > 0 and c["frozen"] > 0                                 # laziness used


@pytest.mark.parametrize("seed", [4, 5])
def test_cross_predicate_equivalence_and_pair_materialisation(seed):
    set_seed(52100 + seed)
    cfg = TraceConfig(n_frames=500, n_keys=150, n_tracks=8, n_predicates=6)
    frames = make_trace(cfg, np.random.default_rng(52100 + seed))
    p = {**BASE, "r_ov": 0.05, "r_vo": 0.05, "cross": True, "w_x": 0.7, "exclusivity": exclusivity(6, seed)}
    lazy, _ = compare(p, frames)
    assert lazy.counters()["mat_pair"] > 0


def test_frame_interval_changes_materialise_and_stay_equal():
    cfg = TraceConfig(n_frames=400, n_keys=100, p_jitter=0.2)
    frames = make_trace(cfg, np.random.default_rng(52201))
    lazy, _ = compare({**BASE, **RUNGS["m3"]}, frames)
    assert lazy.counters()["mat_regime"] > 0


def test_recurrence_after_freeze_is_exact():
    k = EdgeKey(1, 2, 0)
    frames = []
    for f in range(900):
        det = f < 30 or 700 <= f < 720
        edges = [EdgeObservation(k, "v", f, f / 30, 0.9, True)] if det else []
        frames.append(FrameGraph("v", f, f / 30, [], edges))
    lazy, _ = compare(BASE, frames)
    c = lazy.counters()
    assert c["freezes"] >= 2 and c["mat_record"] >= 1                 # frozen, re-materialised, re-frozen
    fresh = LazyVHPED(**BASE)                                         # usable without reset_video, like VHPED
    ref = VHPED(**BASE)
    assert ([o.predicate_score for o in fresh.step(frames[0]).edge_observations]
            == [o.predicate_score for o in ref.step(frames[0]).edge_observations])


def test_deadlines_cancel_and_stale_entries_under_interrupted_evidence():
    cfg = TraceConfig(n_frames=600, n_keys=120, p_vis_record=0.6, p_occ=0.05)
    frames = make_trace(cfg, np.random.default_rng(52301))
    lazy, _ = compare({**BASE, **RUNGS["m3"]}, frames)
    c = lazy.counters()
    assert c["deadlines"] > 0 and c["stale_pops"] + c["rebuilds"] > 0
    assert c["heap_size"] <= 3 * c["scheduled_keys"] + 64


def test_long_trace_numerical_stability():
    cfg = TraceConfig(n_frames=20000, n_keys=40, n_tracks=10, p_birth=0.0005, p_retire=0.0005)
    frames = make_trace(cfg, np.random.default_rng(52401))
    lazy, _ = compare({**BASE, **RUNGS["m3"]}, frames)
    for x, _, _ in lazy._frozen.values():
        assert all(math.isfinite(v) and v >= 0 for v in x) and abs(sum(x) - 1) < 1e-9


def test_belief_read_does_not_change_state():
    cfg = TraceConfig(n_frames=300, n_keys=60)
    frames = make_trace(cfg, np.random.default_rng(52501))
    lazy = LazyVHPED(**BASE)
    lazy.reset_video("v")
    for fr in frames:
        lazy.step(fr)
    snap = {k: (list(x), s0) for k, (x, s0, _) in lazy._frozen.items()}
    for k in snap:
        lazy.belief(k)
    assert {k: (list(x), s0) for k, (x, s0, _) in lazy._frozen.items()} == snap


def test_certificate_is_sound_on_random_positive_maps():
    """d_H to the fixed point never increases, and odds(p_n) <= odds(p*) e^{d_n} along trajectories."""
    rng = np.random.default_rng(52601)
    for _ in range(50):
        Q = generator(*rng.uniform(0.05, 3.0, 4), eta=float(rng.uniform(0, 0.8)))
        P = expm(Q / 30)
        L = emission("unk", *rng.uniform(0.01, 0.3, 2), *rng.uniform(1e-6, 0.3, 2))
        c = make_certificate(P, L, 0.5)
        x = rng.dirichlet(np.ones(4))
        d_prev = math.inf
        for _ in range(300):
            x = x @ c.M
            x /= x.sum()
            d = hilbert_distance(x, c.x_star)
            assert d <= d_prev + 1e-9
            p = x[2] + x[3]
            assert logit(p) <= logit(c.p_star) + d + 1e-9
            d_prev = d


def test_projective_diameter_and_contraction():
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert projective_diameter(M) == pytest.approx(math.log(4.0))       # max cross-ratio 2*2/(1*1)
    assert projective_diameter(np.array([[1.0, 0.0], [0.0, 1.0]])) == math.inf
    assert hilbert_distance(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(0.0)
