"""Phase 4 tests for the V-HPED reference model.

Covers the Phases.md Phase 4 test list plus the propositions in
research/derivations/vhped_model.md:
    - zero-alpha / disabled cross terms -> independent predicate dynamics
    - inhibition cannot create invalid intensity
    - gradients finite; finite-difference check of the likelihood
    - NumPy inference and PyTorch likelihood agree exactly
    - Proposition 1 (censoring), 2 (self-calibrating evidence), 3 (nesting = HMM)
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph, PairKey
from taorg.schemas.tracks import VisibilityState
from taorg.temporal import METHODS
from taorg.temporal.base import TemporalEdgeModel
from taorg.temporal.hmm import HMM
from taorg.temporal.vhped import HAND_SET_CONFIG, VHPED, TrainerConfig, VHPEDParams, fit_vhped
from taorg.temporal.vhped import dynamics
from taorg.temporal.vhped.alive import filter_step
from taorg.temporal.vhped.likelihood import (
    encode_video,
    params_to_tensors,
    sequence_log_likelihood,
)
from tests.fixtures.synthetic_fixtures import (
    make_simple_video_frames,
    make_termination_video_frames,
)

VIS = VisibilityState.VISIBLE
OCC = VisibilityState.OCCLUDED_OR_UNRELIABLE
UNK = VisibilityState.UNKNOWN
FPS = 30.0
DT = 1.0 / FPS
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _params(num_predicates: int = 3, **overrides) -> VHPEDParams:
    cfg = dict(HAND_SET_CONFIG, num_predicates=num_predicates)
    cfg.update(overrides)
    return VHPEDParams.from_config(cfg)


def _frame(t: int, edges: list[tuple[int, int, int, bool, VisibilityState]],
           score: float = 0.9) -> FrameGraph:
    """edges: (subject, object, predicate, observed, object_visibility)."""
    obs = [
        EdgeObservation(
            edge_key=EdgeKey(s, o, p), video_id="v", frame_idx=t, timestamp=t / FPS,
            predicate_score=score if seen else 0.0, observed=seen,
            subject_visibility=VIS, object_visibility=vis,
        )
        for s, o, p, seen, vis in edges
    ]
    return FrameGraph(video_id="v", frame_idx=t, timestamp=t / FPS, edge_observations=obs)


def _random_video(seed: int, num_frames: int = 40, num_predicates: int = 3) -> list[FrameGraph]:
    """Multi-pair, multi-predicate stream with mixed visibility and gaps."""
    rng = random.Random(seed)
    pairs = [(1, 2), (1, 3), (4, 2)]
    frames = []
    for t in range(num_frames):
        edges = []
        for s, o in pairs:
            if rng.random() < 0.15:
                continue  # pair absent from frame -> UNKNOWN
            vis = rng.choice([VIS, VIS, VIS, OCC])
            for p in range(num_predicates):
                if rng.random() < 0.5:
                    seen = vis == VIS and rng.random() < 0.7
                    edges.append((s, o, p, seen, vis))
        frames.append(_frame(t, edges, score=rng.uniform(0.3, 1.0)))
    return frames


def _run(model: VHPED, frames: list[FrameGraph], key: EdgeKey) -> list[float]:
    model.reset_video(frames[0].video_id)
    out = []
    for f in frames:
        model.step(f)
        out.append(model.alive_probability(key))
    model.finalize_video()
    return out


KEY = EdgeKey(1, 2, 0)


# ---------------------------------------------------------------
# Interface / config
# ---------------------------------------------------------------

class TestInterface:
    def test_protocol_and_registry(self):
        assert isinstance(VHPED(), TemporalEdgeModel)
        assert METHODS["vhped"] is VHPED

    def test_accepts_score_threshold_kwarg(self):
        # synthetic_eval constructs methods with cls(score_threshold=...)
        assert VHPED(score_threshold=0.3).score_threshold == 0.3

    def test_hand_set_matches_yaml(self):
        cfg = yaml.safe_load((REPO_ROOT / "configs/temporal/vhped.yaml").read_text())["vhped"]
        for key, val in HAND_SET_CONFIG.items():
            assert cfg[key] == val, f"{key}: yaml {cfg[key]} != code {val}"

    def test_from_config_builds_model(self):
        cfg = yaml.safe_load((REPO_ROOT / "configs/temporal/vhped.yaml").read_text())["vhped"]
        model = VHPED.from_config(cfg)
        assert model.params.num_predicates == 26

    def test_params_roundtrip(self):
        p = _params()
        q = VHPEDParams.from_dict(p.to_dict())
        np.testing.assert_array_equal(p.alpha, q.alpha)

    @pytest.mark.parametrize("override", [
        {"beta_vis": 0.0}, {"rho_occ": 1.5}, {"eps_rate": 0.0}, {"prior_alive": 1.0},
    ])
    def test_invalid_params_raise(self, override):
        with pytest.raises(ValueError):
            _params(**override)

    def test_predicate_outside_vocabulary_raises(self):
        model = VHPED(params=_params(num_predicates=2))
        model.reset_video("v")
        with pytest.raises(ValueError, match="vocabulary"):
            model.step(_frame(0, [(1, 2, 5, True, VIS)]))

    def test_non_increasing_timestamp_raises(self):
        model = VHPED(params=_params())
        model.reset_video("v")
        model.step(_frame(3, [(1, 2, 0, True, VIS)]))
        with pytest.raises(ValueError, match="Non-increasing"):
            model.step(_frame(3, [(1, 2, 0, True, VIS)]))

    def test_deterministic_and_reset(self):
        frames = _random_video(0)
        m = VHPED(params=_params())
        a = _run(m, frames, KEY)
        b = _run(m, frames, KEY)
        assert a == b


# ---------------------------------------------------------------
# Dynamics validity
# ---------------------------------------------------------------

class TestDynamics:
    def test_extreme_inhibition_keeps_intensity_valid(self):
        p = _params(alpha_self=-1e4, alpha_cross=-1e4)
        traces = np.array([50.0, 50.0, 50.0])
        lam = dynamics.intensity(traces, p)
        assert np.all(np.isfinite(lam)) and np.all(lam >= 0.0)

    def test_extreme_inhibition_model_runs(self):
        m = VHPED(params=_params(alpha_self=-1e4, gamma_self=1e4), prune_below=0.0)
        beliefs = _run(m, _random_video(1), KEY)
        assert all(math.isfinite(b) and 0.0 <= b <= 1.0 for b in beliefs)

    def test_trace_decay_is_exponential(self):
        e = dynamics.decay_traces(np.array([2.0]), beta=3.0, dt=0.5)
        assert e[0] == pytest.approx(2.0 * math.exp(-1.5))


# ---------------------------------------------------------------
# Propositions
# ---------------------------------------------------------------

class TestPropositions:
    def test_prop1_full_occlusion_is_censored(self):
        """rho_occ = 0: occluded absence leaves only the prior transition."""
        belief, lam, haz = np.array([0.8]), np.array([60.0]), np.array([0.5])
        post, log_ev = filter_step(belief, np.array([False]), lam, haz,
                                   rho=0.0, eps_rate=1.0, rebirth_rate=0.0, dt=DT)
        assert post[0] == pytest.approx(0.8 * math.exp(-0.5 * DT))  # r = 0
        assert log_ev[0] == pytest.approx(0.0)

    def test_prop2_visible_absence_log_bayes_factor(self):
        """log L1/L0 for a visible absence equals -(lambda - eps) dt."""
        lam, eps = 42.0, 1.5
        # Recover the likelihood ratio from the posterior odds with no transition.
        prior = np.array([0.5])
        post, _ = filter_step(prior, np.array([False]), np.array([lam]), np.array([0.0]),
                              rho=1.0, eps_rate=eps, rebirth_rate=0.0, dt=DT)
        log_bf = math.log(post[0] / (1 - post[0]))
        assert log_bf == pytest.approx(-(lam - eps) * DT, rel=1e-9)

    def test_prop2_dense_edges_drop_faster_than_sparse(self):
        """Evidence is self-calibrating: same visible misses, different histories."""
        m = VHPED(params=_params(num_predicates=1, eta=-30.0), prune_below=0.0)
        dense = [_frame(t, [(1, 2, 0, True, VIS)]) for t in range(10)]
        sparse = [_frame(t, [(1, 2, 0, t % 3 == 0, VIS)]) for t in range(10)]
        misses = [_frame(t, [(1, 2, 0, False, VIS)]) for t in range(10, 13)]

        def drop_in_log_odds(frames):
            b = _run(m, frames, KEY)
            lo = lambda x: math.log(x / (1 - x))  # noqa: E731
            return lo(b[9]) - lo(b[-1])

        assert drop_in_log_odds(dense + misses) > drop_in_log_odds(sparse + misses)

    def test_visible_absence_beats_occluded_absence(self):
        m = VHPED(params=_params(), prune_below=0.0)
        head = [_frame(t, [(1, 2, 0, True, VIS)]) for t in range(8)]
        vis_tail = [_frame(t, [(1, 2, 0, False, VIS)]) for t in range(8, 12)]
        occ_tail = [_frame(t, [(1, 2, 0, False, OCC)]) for t in range(8, 12)]
        assert _run(m, head + vis_tail, KEY)[-1] < 0.5 < _run(m, head + occ_tail, KEY)[-1]

    def test_fixtures_bridge_occlusion_and_detect_termination(self):
        m = VHPED()
        occl = _run(m, make_simple_video_frames(num_frames=12), KEY)
        term = _run(m, make_termination_video_frames(num_frames=20), KEY)
        assert min(occl) > 0.5  # bridged frames 4-6
        assert term[-1] < 0.5   # dropped after visible termination

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_prop3_nesting_equals_hmm(self, seed):
        """alpha = gamma = 0, rho = 1 reduces exactly to the HMM baseline."""
        mu, eta, eps, r, prior = 2.0, -1.0, 1.2, 0.3, 0.4
        params = VHPEDParams.from_config(dict(
            HAND_SET_CONFIG, num_predicates=1, mu=mu, eta=eta, eps_rate=eps,
            rebirth_rate=r, prior_alive=prior,
        )).apply_ablation(use_visibility=False, use_excitation=False)
        sp = lambda x: math.log1p(math.exp(x))  # noqa: E731
        k = sp(eta) + r
        pi = r / k
        hmm = HMM(
            stay_alive_prob=pi + (1 - pi) * math.exp(-k * DT),
            resurrect_prob=pi * (1 - math.exp(-k * DT)),
            emit_obs_alive=1 - math.exp(-sp(mu) * DT),
            emit_obs_dead=1 - math.exp(-eps * DT),
            prior_alive=prior, score_threshold=0.5,
        )
        rng = random.Random(seed)
        frames = [_frame(t, [(1, 2, 0, t == 0 or rng.random() < 0.6, rng.choice([VIS, OCC]))])
                  for t in range(30)]
        vh = _run(VHPED(params=params, prune_below=0.0), frames, KEY)
        hmm.reset_video("v")
        hb = []
        for f in frames:
            hmm.step(f)
            hb.append(hmm._belief[KEY])
        np.testing.assert_allclose(vh, hb, rtol=1e-10, atol=1e-12)

    def test_prop4_no_cross_effect_when_disabled(self):
        params = _params(alpha_cross=3.0, gamma_cross=2.0).apply_ablation(use_cross_predicate=False)
        base = [_frame(t, [(1, 2, 0, t < 5, VIS)]) for t in range(12)]
        with_other = [_frame(t, [(1, 2, 0, t < 5, VIS), (1, 2, 1, True, VIS)]) for t in range(12)]
        m = VHPED(params=params, prune_below=0.0)
        np.testing.assert_allclose(_run(m, base, KEY), _run(m, with_other, KEY), rtol=0, atol=1e-15)

    def test_cross_hazard_has_effect_when_enabled(self):
        """gamma[1, 0] > 0: observing predicate 1 accelerates predicate 0's termination."""
        gamma = np.zeros((3, 3))
        gamma[1, 0] = 3.0
        params = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=3, gamma=gamma.tolist()))
        base = [_frame(t, [(1, 2, 0, True, VIS)]) for t in range(12)]
        with_other = [_frame(t, [(1, 2, 0, True, VIS), (1, 2, 1, t >= 6, VIS)]) for t in range(12)]
        m = VHPED(params=params, prune_below=0.0)
        assert _run(m, with_other, KEY)[-1] < _run(m, base, KEY)[-1]


# ---------------------------------------------------------------
# Likelihood / training
# ---------------------------------------------------------------

class TestLikelihood:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_numpy_and_torch_likelihood_agree(self, seed):
        params = _params(alpha_cross=-0.7, gamma_cross=0.4, gamma_self=-0.2, rebirth_rate=0.2)
        frames = _random_video(seed)
        model = VHPED(params=params, prune_below=0.0)
        model.reset_video("v")
        for f in frames:
            model.step(f)
        enc = encode_video(frames, params.num_predicates, FPS, params.use_marks)
        ll = sequence_log_likelihood(enc, params_to_tensors(params))
        assert float(ll) == pytest.approx(model.log_likelihood, rel=1e-9, abs=1e-9)

    def test_gradients_finite_and_match_finite_difference(self):
        params = _params(alpha_cross=0.5)
        enc = encode_video(_random_video(5, num_frames=25), 3, FPS, True)
        theta = {k: v.clone().requires_grad_(v.is_floating_point())
                 for k, v in params_to_tensors(params).items()}
        ll = sequence_log_likelihood(enc, theta)
        ll.backward()
        for name, t in theta.items():
            assert torch.all(torch.isfinite(t.grad)), name
        # central finite difference on alpha[1, 0]
        h = 1e-6
        def ll_at(delta):
            th = {k: v.detach().clone() for k, v in theta.items()}
            th["alpha"][1, 0] += delta
            return float(sequence_log_likelihood(enc, th))
        fd = (ll_at(h) - ll_at(-h)) / (2 * h)
        assert float(theta["alpha"].grad[1, 0]) == pytest.approx(fd, rel=1e-5, abs=1e-7)

    def test_trainer_decreases_nll_and_respects_ablation(self):
        videos = [_random_video(s) for s in range(3)]
        init = _params(alpha_cross=0.2)
        cfg = TrainerConfig(num_iters=40, lr=0.05, use_cross_predicate=False)
        res = fit_vhped(videos, init, cfg, fps=FPS)
        assert res.nll_history[-1] < res.nll_history[0]
        off = res.params.alpha - np.diag(np.diag(res.params.alpha))
        assert np.all(off == 0.0)
        assert res.params.rho[0] == 1.0

    def test_trainer_no_visibility_pins_rho(self):
        res = fit_vhped([_random_video(0)], _params(), TrainerConfig(num_iters=5, use_visibility=False))
        np.testing.assert_array_equal(res.params.rho, [1.0, 1.0, 1.0])
        assert len(set(res.params.beta.tolist())) == 1

    def test_trainer_rejects_empty(self):
        with pytest.raises(ValueError):
            fit_vhped([], _params())

    def test_current_intensity_rises_after_observation(self):
        m = VHPED(params=_params(), prune_below=0.0)
        m.reset_video("v")
        m.step(_frame(0, [(1, 2, 0, True, VIS)]))
        lam = m.current_intensity(PairKey(1, 2))
        assert lam[0] > dynamics.softplus(np.array([HAND_SET_CONFIG["mu"]]))[0]


def test_stacked_videos_preserve_likelihood():
    from taorg.temporal.vhped.likelihood import stack_videos
    params = _params(alpha_cross=0.3)
    encs = [encode_video(_random_video(s), 3, FPS, True) for s in range(3)]
    theta = params_to_tensors(params)
    separate = sum(float(sequence_log_likelihood(e, theta)) for e in encs)
    stacked = stack_videos(encs)
    assert len(stacked) == 1
    assert float(sequence_log_likelihood(stacked[0], theta)) == pytest.approx(separate, rel=1e-12)


def test_padded_stacking_of_unequal_videos_preserves_likelihood():
    """Videos of different lengths and frame rates: dt=0 padding is an exact no-op."""
    from taorg.temporal.vhped.likelihood import stack_videos
    params = _params(alpha_cross=0.3, rebirth_rate=0.5)
    encs = [encode_video(_random_video(s, num_frames=n), 3, fps, True)
            for s, n, fps in [(0, 40, FPS), (1, 17, 24.0), (2, 29, FPS)]]
    encs[1].dt[5] = 0.4  # irregular interval
    theta = params_to_tensors(params)
    separate = sum(float(sequence_log_likelihood(e, theta)) for e in encs)
    stacked = stack_videos(encs)
    assert len(stacked) == 1 and stacked[0].dt.shape == (40, sum(len(e.pairs) for e in encs))
    assert float(sequence_log_likelihood(stacked[0], theta)) == pytest.approx(separate, rel=1e-12)


def test_trainer_lazy_exact_variant():
    from taorg.temporal.vhped import LazyVHPED
    res = fit_vhped([_random_video(0)], _params(gamma_cross=0.5),
                    TrainerConfig(num_iters=5, fit_gamma=False, rho_unk_fixed=0.0))
    assert not res.params.gamma.any() and res.params.rho[2] == 0.0
    assert LazyVHPED(params=res.params).lazy_exact
