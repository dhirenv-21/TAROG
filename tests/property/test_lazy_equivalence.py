"""Randomized eager-vs-lazy equivalence harness (Rules §9, §10; Phase 5 exit gate).

Declared tolerance: alive scores agree to rel 1e-9 / abs 1e-12; keep/drop
decisions agree exactly except where the eager belief lies within 1e-9 of the
threshold (floating-point near-ties from product-of-exponentials vs closed
form).  Event ordering: events are applied at the first frame whose
timestamp is strictly greater than the scheduled crossing time.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.vhped import HAND_SET_CONFIG, VHPED, LazyVHPED, VHPEDParams
from taorg.temporal.vhped.expiry_heap import VersionedHeap, crossing_delay, ctmc_belief

VIS, OCC = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE
TIE = 1e-9


def _stream(seed: int, n_frames: int = 80, n_pairs: int = 8, n_pred: int = 3,
            p_present: float = 0.5) -> list[FrameGraph]:
    """Pairs flicker in and out of the frame entirely; irregular timestamps."""
    rng = random.Random(seed)
    frames, ts = [], 0.0
    alive = {(i, p): rng.random() < 0.7 for i in range(n_pairs) for p in range(n_pred)}
    for t in range(n_frames):
        ts += rng.choice([1 / 30, 1 / 30, 2 / 30, 0.25])
        edges = []
        for i in range(n_pairs):
            if rng.random() > p_present:
                continue  # pair has no record -> untouched
            vis = rng.choice([VIS, VIS, OCC])
            for p in range(n_pred):
                if rng.random() < 0.03:
                    alive[(i, p)] = not alive[(i, p)]
                seen = vis == VIS and alive[(i, p)] and rng.random() < 0.8
                if seen or rng.random() < 0.7:
                    edges.append(EdgeObservation(
                        edge_key=EdgeKey(i, 100 + i, p), video_id="v", frame_idx=t, timestamp=ts,
                        predicate_score=rng.uniform(0.3, 1.0) if seen else 0.0, observed=seen,
                        subject_visibility=VIS, object_visibility=vis))
        frames.append(FrameGraph("v", t, ts, edge_observations=edges))
    return frames


def _params(n_pred=3, **kw) -> VHPEDParams:
    cfg = dict(HAND_SET_CONFIG, num_predicates=n_pred, rho_unk=0.0)
    cfg.update(kw)
    return VHPEDParams.from_config(cfg)


def _compare(params, frames, threshold, prune):
    eager = VHPED(params=params, score_threshold=threshold, prune_below=prune)
    lazy = LazyVHPED(params=params, score_threshold=threshold, prune_below=prune)
    for m in (eager, lazy):
        m.reset_video("v")
    for f in frames:
        e_out = {o.edge_key: o.predicate_score for o in eager.step(f).edge_observations}
        l_out = {o.edge_key: o.predicate_score for o in lazy.step(f).edge_observations}
        for key in set(e_out) ^ set(l_out):
            b = eager.alive_probability(key)
            assert abs(b - threshold) < TIE, f"decision mismatch {key} at frame {f.frame_idx}: {b}"
        for key in set(e_out) & set(l_out):
            assert l_out[key] == pytest.approx(e_out[key], rel=1e-9, abs=1e-12)
    assert lazy.log_likelihood == pytest.approx(eager.log_likelihood, rel=1e-9, abs=1e-9)
    return eager, lazy


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("rebirth", [0.0, 0.4])
def test_lazy_matches_eager(seed, rebirth):
    rng = random.Random(1000 + seed)
    params = _params(rebirth_rate=rebirth, alpha_cross=rng.uniform(-1, 1),
                     eta=rng.uniform(-2, 1))
    threshold = rng.choice([0.3, 0.5, 0.7])
    prune = rng.choice([0.0, 1e-6, 1e-2])
    _compare(params, _stream(seed, p_present=rng.choice([0.2, 0.5, 0.9])), threshold, prune)


def test_lazy_saves_work_on_sparse_streams():
    _, lazy = _compare(_params(), _stream(3, n_frames=150, n_pairs=20, p_present=0.15), 0.5, 1e-6)
    c = lazy.counters
    assert lazy.lazy_exact
    assert c.pair_steps < 0.5 * c.eager_equivalent_pair_steps


def test_fallback_when_conditions_violated_is_exact():
    gamma = np.zeros((3, 3))
    gamma[1, 0] = 2.0
    params = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=3, rho_unk=0.5,
                                          gamma=gamma.tolist()))
    frames = _stream(7)
    eager = VHPED(params=params, score_threshold=0.5)
    lazy = LazyVHPED(params=params, score_threshold=0.5)
    assert not lazy.lazy_exact
    for m in (eager, lazy):
        m.reset_video("v")
    for f in frames:
        a = {o.edge_key: o.predicate_score for o in eager.step(f).edge_observations}
        b = {o.edge_key: o.predicate_score for o in lazy.step(f).edge_observations}
        assert a == b


# ---------------------------------------------------------------
# Closed-form pieces
# ---------------------------------------------------------------

@pytest.mark.parametrize("a0,h,r", [(0.9, 0.5, 0.0), (0.2, 0.3, 0.6), (0.99, 1.2, 0.1)])
def test_ctmc_semigroup_matches_per_frame_product(a0, h, r):
    a = a0
    for _ in range(40):
        pi = r / (h + r)
        a = pi + (a - pi) * math.exp(-(h + r) / 30)  # one eager frame step
    assert ctmc_belief(a0, h, r, 40 / 30) == pytest.approx(a, rel=1e-12)


@pytest.mark.parametrize("a0,h,r,level", [
    (0.9, 0.5, 0.0, 0.5), (0.2, 0.3, 0.6, 0.5), (0.6, 0.3, 0.6, 0.5), (0.99, 1.2, 0.1, 1e-6),
])
def test_crossing_delay_brute_force(a0, h, r, level):
    d = crossing_delay(a0, h, r, level)
    grid = np.linspace(0, 60, 600_001)
    side0 = a0 >= level
    vals = np.array([ctmc_belief(a0, h, r, x) for x in grid[::100]])
    crossed = np.flatnonzero((vals >= level) != side0)
    if math.isinf(d):
        assert crossed.size == 0
    else:
        assert grid[::100][crossed[0]] == pytest.approx(d, abs=0.011)


def test_versioned_heap_skips_stale_and_rebuilds():
    heap = VersionedHeap()
    versions = {EdgeKey(0, 1, 0): 2}
    for v in range(200):
        heap.push(float(v), EdgeKey(0, 1, 0), v % 10)
    due = heap.pop_due(50.0, lambda k: versions.get(k))
    assert len(due) == 5 and all(e.version == 2 for e in due)
    assert heap.counters.stale_pops == 45
    heap.maybe_rebuild(lambda k: versions.get(k))
    assert heap.counters.rebuilds == 1 and len(heap) == 15



def test_heap_pushes_bounded_by_touches_plus_wakeups():
    _, lazy = _compare(_params(rebirth_rate=0.3), _stream(11, n_frames=200, n_pairs=15,
                                                          p_present=0.3), 0.5, 1e-6)
    c, h = lazy.counters, lazy._heap.counters
    assert h.pushes <= c.pair_steps + c.wakeups
    assert h.stale_pops <= h.pushes


# ---------------------------------------------------------------
# Necessity of L1 / L2 (research/derivations/lazy_expiry.md §9)
# ---------------------------------------------------------------

def _untouched_belief(params, partition, a0=0.9, e0=3.0):
    """Eager evolution of one untouched tracked predicate over the given frame intervals."""
    from taorg.temporal.vhped import dynamics
    from taorg.temporal.vhped.alive import filter_step

    traces = np.array([e0, 0.0, 0.0])
    a = np.array([a0])
    beta_unk = float(params.beta[2])
    for dt in partition:
        traces = dynamics.decay_traces(traces, beta_unk, dt)
        lam = dynamics.intensity(traces, params)[:1]
        haz = dynamics.hazard(traces, params)[:1]
        a, _ = filter_step(a, np.array([False]), lam, haz, rho=float(params.rho[2]),
                           eps_rate=params.eps_rate, rebirth_rate=params.rebirth_rate, dt=dt)
    return float(a[0])


PARTITIONS = ([0.3], [0.1, 0.1, 0.1], [0.05, 0.25], [0.01] * 30)


def test_conditions_hold_belief_depends_only_on_elapsed_time():
    p = _params(rho_unk=0.0, gamma_self=0.0, rebirth_rate=0.2)
    vals = [_untouched_belief(p, part) for part in PARTITIONS]
    assert max(vals) - min(vals) < 1e-12


@pytest.mark.parametrize("violation", ["rho_unk", "gamma"])
def test_violating_a_condition_makes_belief_partition_dependent(violation):
    """Counterexample: same elapsed time, different frame partitions, different beliefs.

    Hence no function of elapsed time can reproduce the eager filter, and the
    closed-form wake-up schedule cannot be exact without L1 and L2.
    """
    kw = {"rho_unk": 0.5} if violation == "rho_unk" else {"rho_unk": 0.0, "gamma_self": 2.0}
    p = _params(**kw)
    vals = [_untouched_belief(p, part) for part in PARTITIONS]
    assert max(vals) - min(vals) > 1e-4
