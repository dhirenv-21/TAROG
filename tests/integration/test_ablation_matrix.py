"""Tiny end-to-end run of the Phase 8 ablation matrix."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from taorg.experiments.ablation_matrix import paired_bootstrap, run_matrix

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def test_matrix_tiny():
    cfg = yaml.safe_load((REPO / "configs/experiments/ablation_matrix_synthetic.yaml").read_text(
        encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = [1, 2]
    cfg["splits"] = {"train": 2, "val": 1, "test": 2}
    cfg["scenario"] = {"num_frames": 60, "num_pairs": 3, "terminate_range": [20, 40]}
    cfg["thresholds"] = [0.2, 0.6]
    cfg["trainer"]["num_iters"] = 2
    cfg["bootstrap"]["n"] = 200
    res = run_matrix(cfg)
    assert len(res["cells"]) == 2 * 2
    assert {r["method"] for r in res["table"]} == set(cfg["methods"])
    lazy = [c["fitted"]["vhped_lazyexact_fit"]["params"] for c in res["cells"]]
    assert all(not np.any(p["gamma"]) and p["rho"][2] == 0.0 for p in lazy)
    assert len(res["paired_vs_reference"]) == 2 * (len(cfg["methods"]) - 1)


def test_paired_bootstrap_ci_contains_mean():
    d = np.random.default_rng(0).normal(1.0, 0.5, 200)
    lo, hi = paired_bootstrap(d, 2000, 0)
    assert lo < d.mean() < hi and lo > 0.8
