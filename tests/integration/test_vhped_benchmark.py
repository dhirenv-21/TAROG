"""Integration test: synthetic V-HPED benchmark pipeline runs end to end."""

from __future__ import annotations

from taorg.experiments.vhped_benchmark import (
    ScenarioConfig,
    evaluate,
    generate_video,
    run_benchmark,
    run_method,
)
from taorg.temporal.vhped import VHPED, TrainerConfig

SMALL = ScenarioConfig(num_frames=60, num_pairs=2, terminate_range=(20, 40))


def test_generator_is_deterministic():
    a = generate_video("v", 7, SMALL)
    b = generate_video("v", 7, SMALL)
    assert [[o.observed for o in f.edge_observations] for f in a.noisy] == \
           [[o.observed for o in f.edge_observations] for f in b.noisy]
    assert a.manifest.to_dict() == b.manifest.to_dict()


def test_vhped_evaluates_on_generated_video():
    videos = [generate_video("v", s, SMALL) for s in range(2)]
    metrics = evaluate(videos, run_method(lambda t: VHPED(score_threshold=t), videos, 0.5))
    assert 0.0 <= metrics["frame_f1"] <= 1.0


def test_run_benchmark_tiny():
    res = run_benchmark(0, SMALL, n_train=2, n_val=1, n_test=1,
                        trainer=TrainerConfig(num_iters=3))
    assert "vhped_full_fit" in res.per_method and "gated_ema" in res.per_method
