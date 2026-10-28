"""The vectorised threshold sweep must equal the frozen metric implementations."""

from __future__ import annotations

import random

import numpy as np
import pytest

from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.termination import compute_false_persistence, compute_termination_delay
from taorg.metrics.threshold_sweep import SweepAccumulator, hypervolume, video_scores
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.experiments.pilot import SWEEPABLE_METHODS, build_method
from taorg.temporal import METHODS

VIS = VisibilityState.VISIBLE


def _gt_video(seed: int, n_frames: int = 30) -> list[FrameGraph]:
    rng = random.Random(seed)
    spans = {}
    for o in (1, 2, 3):
        for p in (0, 1):
            start = rng.randint(0, 10)
            end = rng.randint(start, n_frames + 5)
            spans[EdgeKey(0, o, p)] = (start, end)
    frames = []
    for t in range(n_frames):
        edges = [EdgeObservation(edge_key=k, video_id="v", frame_idx=t, timestamp=t / 30,
                                 predicate_score=rng.uniform(0.4, 1.0) if s <= t <= e else 0.0,
                                 observed=s <= t <= e and rng.random() > 0.1)
                 for k, (s, e) in spans.items() if t >= s]
        frames.append(FrameGraph("v", t, t / 30, edge_observations=edges))
    return frames


def _filter(frames: list[FrameGraph], thr: float) -> list[FrameGraph]:
    return [FrameGraph(f.video_id, f.frame_idx, f.timestamp,
                       edge_observations=[o for o in f.edge_observations
                                          if o.predicate_score >= thr]) for f in frames]


def _run(cls, frames, thr):
    m = cls(score_threshold=thr)
    m.reset_video("v")
    return [m.step(f) for f in frames]


@pytest.mark.parametrize("name", sorted(SWEEPABLE_METHODS))
@pytest.mark.parametrize("thr", [0.05, 0.3, 0.62])
def test_post_hoc_threshold_equals_direct_run(name, thr):
    gt = _gt_video(3)
    noisy, _ = inject_occlusion(gt, seed=5, config=OcclusionConfig(gap_lengths=[3]))
    direct = _run(METHODS[name], noisy, thr)
    filtered = _filter(_run(METHODS[name], noisy, 0.0), thr)
    for a, b in zip(direct, filtered):
        assert {(o.edge_key, round(o.predicate_score, 12)) for o in a.edge_observations} == \
               {(o.edge_key, round(o.predicate_score, 12)) for o in b.edge_observations}


@pytest.mark.parametrize("seed", range(6))
def test_sweep_matches_frozen_metrics(seed):
    gt = _gt_video(seed)
    noisy, manifest = inject_occlusion(gt, seed=seed, config=OcclusionConfig(gap_lengths=[2, 4]))
    pred0 = _run(METHODS["gated_ema"], noisy, 0.0)
    thresholds = [0.05, 0.2, 0.45, 0.8]
    acc = SweepAccumulator(np.array(thresholds))
    acc.add(video_scores(gt, pred0, manifest))
    for thr, res in zip(thresholds, acc.results()):
        pred = _filter(pred0, thr)
        pers = compute_persistence_f1(gt, pred, manifest)
        term = compute_termination_delay(gt, pred)
        fpr = compute_false_persistence(gt, pred)
        churn = compute_churn_rate(pred, gt)
        assert res["persistence_f1"] == pytest.approx(pers.f1)
        assert res["persistence_recall"] == pytest.approx(pers.recall)
        assert res["termination_delay_mean"] == pytest.approx(term.mean_delay)
        assert res["termination_delay_median"] == pytest.approx(term.median_delay)
        assert res["num_terminating_edges"] == term.num_evaluated
        assert res["termination_censored_frac"] == pytest.approx(
            term.num_censored / term.num_evaluated if term.num_evaluated else 0.0)
        assert res["false_persistence_rate"] == pytest.approx(fpr.mean_rate)
        assert res["churn"] == pytest.approx(churn.mean_churn)


def test_hypervolume():
    assert hypervolume([(0.0, 1.0)]) == pytest.approx(1.0)
    assert hypervolume([(0.5, 1.0)]) == pytest.approx(0.5)
    assert hypervolume([(0.0, 0.5), (0.5, 1.0)]) == pytest.approx(0.75)
    assert hypervolume([]) == 0.0


def test_fixed_window_is_not_sweepable_and_pilot_rejects_it():
    gt = _gt_video(3)
    noisy, _ = inject_occlusion(gt, seed=5, config=OcclusionConfig(gap_lengths=[3]))
    direct = _run(METHODS["fixed_window"], noisy, 0.62)
    filtered = _filter(_run(METHODS["fixed_window"], noisy, 0.0), 0.62)
    assert any({o.edge_key for o in a.edge_observations} != {o.edge_key for o in b.edge_observations}
               for a, b in zip(direct, filtered))
    with pytest.raises(ValueError, match="post-hoc"):
        build_method("fw", {"method": "fixed_window"}, {}, 3, None)
