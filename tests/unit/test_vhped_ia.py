"""Tests for IA-VHPED (research/derivations/identity_aware.md)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from taorg.noise.identity_confidence import (
    IdentityScoreConfig,
    annotate_identity_confidence,
    jump_frames,
)
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph, PairKey
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.temporal.vhped import HAND_SET_CONFIG, VHPED, VHPEDParams
from taorg.temporal.vhped.likelihood import params_to_tensors
from taorg.temporal.vhped.trainer import TrainerConfig
from taorg.temporal.vhped_ia import IAVHPED, IAParams, LazyIAVHPED, fit_ia_vhped
from taorg.temporal.vhped_ia.identity import TrackIdentityFilter
from taorg.temporal.vhped_ia.likelihood import (
    encode_ia_video,
    ia_log_likelihood,
    stack_ia_videos,
)

VIS, OCC = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE
FPS = 30.0


def _stream(seed: int, n_frames: int = 50, n_pairs: int = 4, n_pred: int = 3,
            kappa: str = "random", p_present: float = 0.8) -> list[FrameGraph]:
    """Pairs (i, 100+i); tracks carry identity_confidence; irregular timestamps."""
    rng = random.Random(seed)
    frames, ts = [], 0.0
    alive = {(i, p): rng.random() < 0.7 for i in range(n_pairs) for p in range(n_pred)}
    for t in range(n_frames):
        ts += rng.choice([1 / 30, 1 / 30, 2 / 30, 0.2])
        tracks, edges = [], []
        for i in range(n_pairs):
            if rng.random() > p_present:
                continue
            for tid in (i, 100 + i):
                k = None if kappa == "none" else (1.0 if kappa == "one" else
                                                  rng.choice([1.0, 0.97, 0.9, 0.3, 0.05]))
                tracks.append(TrackObservation("v", t, ts, tid, 0, (0, 0, 1, 1),
                                               identity_confidence=k))
            vis = rng.choice([VIS, VIS, OCC])
            for p in range(n_pred):
                if rng.random() < 0.04:
                    alive[(i, p)] = not alive[(i, p)]
                seen = vis == VIS and alive[(i, p)] and rng.random() < 0.8
                if seen or rng.random() < 0.7:
                    edges.append(EdgeObservation(
                        edge_key=EdgeKey(i, 100 + i, p), video_id="v", frame_idx=t, timestamp=ts,
                        predicate_score=rng.uniform(0.3, 1.0) if seen else 0.0, observed=seen,
                        subject_visibility=VIS, object_visibility=vis))
        frames.append(FrameGraph("v", t, ts, track_observations=tracks, edge_observations=edges))
    return frames


def _base(**kw) -> VHPEDParams:
    return VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=3, **kw))


def _beliefs(model, frames):
    model.reset_video("v")
    out = []
    for f in frames:
        out.append({o.edge_key: o.predicate_score for o in model.step(f).edge_observations})
    return out


# ---------------------------------------------------------------- identity filter

def test_identity_toggle_and_recovery():
    filt = TrackIdentityFilter(recovery_rate=0.0)
    mk = lambda t, k: FrameGraph("v", t, t / 10, track_observations=[  # noqa: E731
        TrackObservation("v", t, t / 10, 7, 0, (0, 0, 1, 1), identity_confidence=k)])
    filt.update(mk(0, 0.0))
    assert filt.track(7, 0.0) == 1.0          # first appearance
    filt.update(mk(1, 0.0))
    assert filt.track(7, 0.1) == 0.0          # certain jump -> foreign
    filt.update(mk(2, 0.0))
    assert filt.track(7, 0.2) == 1.0          # second jump -> home again
    filt.update(mk(3, 0.7))
    assert filt.track(7, 0.3) == pytest.approx(0.7)
    rec = TrackIdentityFilter(recovery_rate=2.0)
    rec.update(mk(0, 1.0))
    rec.update(mk(1, 0.0))
    assert rec.track(7, 0.1 + 1.0) == pytest.approx(1 - np.exp(-2.0))
    assert filt.pair(PairKey(7, 999), 0.3) == pytest.approx(0.7)  # unseen track -> 1


def test_identity_rejects_invalid_score():
    filt = TrackIdentityFilter()
    with pytest.raises(ValueError):
        filt.update(FrameGraph("v", 0, 0.0, track_observations=[
            TrackObservation("v", 0, 0.0, 1, 0, (0, 0, 1, 1), identity_confidence=1.5)]))


# ---------------------------------------------------------------- reduction / validity

@pytest.mark.parametrize("kappa", ["none", "one"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reduces_exactly_to_vhped_when_identity_certain(kappa, seed):
    base = _base(alpha_cross=0.4, gamma_cross=0.3, rebirth_rate=0.2)
    frames = _stream(seed, kappa=kappa)
    a = VHPED(params=base, score_threshold=0.0, prune_below=0.0)
    b = IAVHPED(ia=IAParams(base=base, foreign_rate=2.0, foreign_birth=0.2),
                score_threshold=0.0, prune_below=0.0)
    for x, y in zip(_beliefs(a, frames), _beliefs(b, frames)):
        assert x.keys() == y.keys()
        for k in x:
            assert y[k] == pytest.approx(x[k], rel=1e-12, abs=1e-15)
    assert b.log_likelihood == pytest.approx(a.log_likelihood, rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_beliefs_are_probabilities(seed):
    m = IAVHPED(ia=IAParams(base=_base(alpha_self=-50.0), foreign_rate=50.0),
                score_threshold=0.0, prune_below=0.0)
    for out in _beliefs(m, _stream(seed)):
        assert all(0.0 <= s <= 1.0 for s in out.values())
    assert np.isfinite(m.log_likelihood)


def test_phantom_birth_uses_foreign_belief():
    """A predicate first seen on a pair whose identity just jumped gets ~b, not the V-HPED birth."""
    base = _base()
    ia = IAParams(base=base, foreign_rate=1.0, foreign_birth=0.05)
    t0 = [TrackObservation("v", 0, 0.0, tid, 0, (0, 0, 1, 1), identity_confidence=1.0)
          for tid in (1, 2)]
    f0 = FrameGraph("v", 0, 0.0, track_observations=t0)
    t1 = [TrackObservation("v", 1, 1 / 30, 1, 0, (0, 0, 1, 1), identity_confidence=1.0),
          TrackObservation("v", 1, 1 / 30, 2, 0, (0, 0, 1, 1), identity_confidence=0.0)]
    e1 = [EdgeObservation(EdgeKey(1, 2, 0), "v", 1, 1 / 30, 0.9, True)]
    f1 = FrameGraph("v", 1, 1 / 30, track_observations=t1, edge_observations=e1)
    ia_m = IAVHPED(ia=ia, score_threshold=0.0)
    plain = VHPED(params=base, score_threshold=0.0)
    for m in (ia_m, plain):
        m.reset_video("v")
        m.step(f0)
        m.step(f1)
    assert ia_m.alive_probability(EdgeKey(1, 2, 0)) == pytest.approx(0.05)
    assert plain.alive_probability(EdgeKey(1, 2, 0)) > 0.5


def test_ablation_switches_change_behaviour():
    frames = _stream(3)
    base = _base()
    full = _beliefs(IAVHPED(ia=IAParams(base=base), score_threshold=0.0, prune_below=0.0), frames)
    nopost = _beliefs(IAVHPED(ia=IAParams(base=base, posterior_weight=False), score_threshold=0.0,
                              prune_below=0.0), frames)
    nobirth = _beliefs(IAVHPED(ia=IAParams(base=base, identity_births=False),
                               score_threshold=0.0, prune_below=0.0), frames)
    assert full != nopost and full != nobirth


# ---------------------------------------------------------------- likelihood / training

@pytest.mark.parametrize("calib", [None, (-3.0, 1.2)])
@pytest.mark.parametrize("posterior_weight", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_numpy_and_torch_ia_likelihood_agree(seed, posterior_weight, calib):
    base = _base(alpha_cross=-0.5, gamma_cross=0.4, rebirth_rate=0.3)
    ia = IAParams(base=base, foreign_rate=1.7, foreign_birth=0.15,
                  posterior_weight=posterior_weight, jump_calibration=calib)
    frames = _stream(seed)
    m = IAVHPED(ia=ia, score_threshold=0.0, prune_below=0.0)
    m.reset_video("v")
    for f in frames:
        m.step(f)
    theta = params_to_tensors(base)
    theta["foreign_rate"] = torch.tensor(1.7, dtype=torch.float64)
    theta["foreign_birth"] = torch.tensor(0.15, dtype=torch.float64)
    if calib is not None:
        theta["jump_a"] = torch.tensor(calib[0], dtype=torch.float64)
        theta["jump_b"] = torch.tensor(calib[1], dtype=torch.float64)
    enc = encode_ia_video(frames, 3, FPS, True, 0.0)
    ll = ia_log_likelihood(enc, theta, posterior_weight=posterior_weight)
    assert float(ll) == pytest.approx(m.log_likelihood, rel=1e-9, abs=1e-9)
    # stacking with another (shorter) video preserves the sum
    enc2 = encode_ia_video(_stream(seed + 10, n_frames=20), 3, FPS, True, 0.0)
    (st,) = stack_ia_videos([enc, enc2])
    assert float(ia_log_likelihood(st, theta, posterior_weight)) == pytest.approx(
        float(ll) + float(ia_log_likelihood(enc2, theta, posterior_weight)), rel=1e-12)


def test_gradients_finite_including_certain_identity():
    base = _base()
    theta = {k: v.clone().requires_grad_(True) for k, v in params_to_tensors(base).items()}
    theta["foreign_rate"] = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    theta["foreign_birth"] = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    for kappa in ("one", "random"):
        enc = encode_ia_video(_stream(1, kappa=kappa), 3, FPS, True, 0.0)
        ll = ia_log_likelihood(enc, theta)
        ll.backward()
        for name, t in theta.items():
            assert torch.all(torch.isfinite(t.grad)), (kappa, name)
            t.grad = None


def test_trainer_runs_and_decreases_nll():
    videos = [_stream(s) for s in range(3)]
    res = fit_ia_vhped(videos, IAParams.default(3), TrainerConfig(num_iters=25), fps=FPS)
    assert res.nll_history[-1] < res.nll_history[0]
    assert res.params.foreign_rate > 0 and 0 < res.params.foreign_birth < 1


# ---------------------------------------------------------------- lazy equivalence

@pytest.mark.parametrize("seed", range(6))
def test_lazy_ia_matches_eager_ia(seed):
    base = _base(rho_unk=0.0, rebirth_rate=random.Random(seed).choice([0.0, 0.3]))
    ia = IAParams(base=base, foreign_rate=1.3, foreign_birth=0.1)
    frames = _stream(seed, p_present=0.4)
    eager = IAVHPED(ia=ia, score_threshold=0.5)
    lazy = LazyIAVHPED(ia=ia, score_threshold=0.5)
    assert lazy.lazy_exact
    for m in (eager, lazy):
        m.reset_video("v")
    for f in frames:
        a = {o.edge_key: o.predicate_score for o in eager.step(f).edge_observations}
        b = {o.edge_key: o.predicate_score for o in lazy.step(f).edge_observations}
        for k in set(a) ^ set(b):
            assert abs(eager.alive_probability(k) - 0.5) < 1e-9
        for k in set(a) & set(b):
            assert b[k] == pytest.approx(a[k], rel=1e-9, abs=1e-12)
    assert lazy.log_likelihood == pytest.approx(eager.log_likelihood, rel=1e-9, abs=1e-9)
    assert lazy.counters.pair_steps < lazy.counters.eager_equivalent_pair_steps


# ---------------------------------------------------------------- simulated tracker scores

def _manifest():
    m = CorruptionManifest(seed=0)
    m.entries.append(CorruptionEntry("id_switch", "v", track_ids=(1, 2), start_frame=3,
                                     end_frame=5, duration=3))
    return m


def test_jump_frames_are_swap_start_and_return():
    assert jump_frames(_manifest()) == {3: {1, 2}, 6: {1, 2}}


def test_identity_scores_deterministic_and_informative():
    frames = [FrameGraph("v", t, t / 30, track_observations=[
        TrackObservation("v", t, t / 30, tid, 0, (0, 0, 1, 1)) for tid in (1, 2, 3)])
        for t in range(10)]
    cfg = IdentityScoreConfig(informativeness=1.0, false_alarm_prob=0.0)
    a = annotate_identity_confidence(frames, _manifest(), 5, cfg)
    b = annotate_identity_confidence(frames, _manifest(), 5, cfg)
    ka = [[t.identity_confidence for t in f.track_observations] for f in a]
    assert ka == [[t.identity_confidence for t in f.track_observations] for f in b]
    assert max(ka[3][:2] + ka[6][:2]) < 0.2 and min(ka[0] + ka[3][2:]) >= 0.9
    flat = annotate_identity_confidence(frames, _manifest(), 5,
                                        IdentityScoreConfig(informativeness=0.0,
                                                            false_alarm_prob=0.0))
    assert all(t.identity_confidence >= 0.9 for f in flat for t in f.track_observations)


def test_scores_survive_real_id_switch_injection():
    frames = _stream(0, n_frames=30, kappa="none", p_present=1.0)
    for f in frames:
        f.track_observations = [TrackObservation("v", f.frame_idx, f.timestamp, t.track_id, 1,
                                                 (0, 0, 1, 1)) for t in f.track_observations]
    sw, man = inject_id_switch(frames, seed=1, config=IDSwitchConfig(swap_duration=4,
                                                                     require_same_class=True))
    out = annotate_identity_confidence(sw, man, 0)
    assert all(t.identity_confidence is not None for f in out for t in f.track_observations)


def test_calibrated_jump_probability_reduction_and_monotonicity():
    from taorg.temporal.vhped_ia.identity import jump_probability
    assert jump_probability(None, (-3.0, 1.0)) == 0.0
    assert jump_probability(1.0, (-3.0, 1.0)) == 0.0
    assert jump_probability(0.3, None) == pytest.approx(0.7)
    vals = [jump_probability(k, (-3.0, 1.5)) for k in (0.99, 0.9, 0.5, 0.1)]
    assert vals == sorted(vals) and vals[0] < 0.01


def test_trainer_learns_calibration():
    res = fit_ia_vhped([_stream(s) for s in range(2)],
                       IAParams.default(3, jump_calibration=(-3.0, 1.0)),
                       TrainerConfig(num_iters=10), fps=FPS)
    a, b = res.params.jump_calibration
    assert np.isfinite(a) and b > 0
