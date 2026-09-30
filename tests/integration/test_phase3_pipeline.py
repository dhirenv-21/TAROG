"""Miniature end-to-end run of the Phase 3 pipeline (all stages).

Uses a tiny world, tiny grids, two seeds and two workers.  Checks artifact
completeness and determinism — NOT results.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import yaml

from taorg.experiments.phase3.analysis import run_analysis
from taorg.experiments.phase3.failures import run_failures
from taorg.experiments.phase3.pipeline import (
    load_phase3_config, read_csv_gz, run_calibration, run_evaluation, run_latency,
    run_selection,
)
from taorg.experiments.phase3.plots import run_plots

MINI = {
    "experiment": "phase3_mini_test",
    "world": {"num_frames": 150, "fps": 30.0, "num_subject_tracks": 2, "num_object_tracks": 2,
              "num_predicates": 3, "num_edges": 4, "start_frame_max": 10, "terminating_prob": 0.7,
              "min_alive_frames": 40, "max_alive_frames": 60, "min_post_termination_frames": 30},
    "seeds": {"calibration": [1, 2], "evaluation": [3, 4], "smoke": [5], "bootstrap": 7},
    "occlusion": {"num_gaps_per_edge": 1, "max_edges": None, "min_pre_gap_frames": 2,
                  "min_post_gap_frames": 2},
    "gap_lengths": [2, 10],
    "budgets_frames": [2, 5],
    "primary_budget_frames": 5,
    "representative_gap_frames": 10,
    "conditions": {
        "C0_oracle": {"role": "h1_primary", "gaps": [2, 10], "calibrate": True},
        "R1_visnoise_0.2": {"role": "h1_robustness", "gaps": [2, 10], "calibrate": True,
                            "visibility_noise": {"occluded_to_visible_prob": 0.2,
                                                 "visible_to_occluded_prob": 0.2}},
        "R2_flicker_0.1": {"role": "h1_robustness", "gaps": [2, 10], "calibrate": True,
                           "flicker_miss_prob": 0.1},
        "S1_termination_under_occlusion": {"role": "stress", "gaps": [5], "tuned_on": "C0_oracle",
                                           "world_overrides": {"termination_occlusion_prob": 1.0}},
        "S2_id_switch_d20": {"role": "stress", "gaps": [0], "tuned_on": "C0_oracle",
                             "id_switch": {"swap_duration": 10, "num_swaps": 1,
                                           "require_same_class": True}},
    },
    "grids": {
        "no_memory": {"score_threshold": [0.0]},
        "fixed_window": {"window_size": [1, 3, 12], "score_threshold": [0.0]},
        "ema": {"alpha": [0.5], "decay": [0.0, 0.8, 0.97], "score_threshold": [0.1]},
        "gated_ema": {"alpha": [0.5], "decay_visible": [0.0, 0.8], "decay_occluded": [0.8, 0.97],
                      "score_threshold": [0.1]},
        "hmm": {"stay_alive_prob": [0.9], "emit_obs_alive": [0.9], "emit_obs_dead": [0.05],
                "score_threshold": [0.3, 0.7]},
    },
    "method_labels": {"no_memory": "No Memory", "fixed_window": "Fixed Window", "ema": "EMA",
                      "gated_ema": "Vis-Gated EMA", "hmm": "HMM"},
    "statistics": {"ci_level": 0.95, "bootstrap_resamples": 200, "hv_bootstrap_resamples": 50,
                   "hv_delay_min": 1.0, "hv_delay_ref": 30.0, "hv_fpr_ref": 1.0},
    "h1_criteria": {
        "treatment": "gated_ema", "comparators": ["ema", "fixed_window", "hmm"], "alpha": 0.05,
        "min_f1_gain": 0.05, "delay_margin": 1.0, "fpr_margin": 0.02, "churn_margin_abs": 0.005,
        "churn_margin_rel": 0.1, "term_loss_delay": 2.0, "term_loss_fpr": 0.05, "min_hv_gain": 0.02,
        "pass_primary_min_wins": 2, "pass_primary_max_losses": 0, "pass_robust_min_wins": 1,
        "pass_robust_max_losses": 1, "pass_primary_min_hv_gaps": 1, "pass_robust_min_hv_gaps": 1,
        "fail_primary_max_wins_exclusive": 1, "fail_primary_min_hv_gaps_exclusive": 1,
    },
}


def _run_all(tmp: Path) -> tuple[dict, Path]:
    cfg_path = tmp / "mini.yaml"
    cfg_path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    cfg = load_phase3_config(cfg_path)
    out = tmp / "out"
    run_calibration(cfg, out, workers=2)
    run_selection(cfg, out)
    run_evaluation(cfg, out, workers=2)
    run_latency(cfg, out)
    decision = run_analysis(cfg, out)
    run_plots(cfg, out)
    run_failures(cfg, out)
    return decision, out


def test_phase3_pipeline_end_to_end(tmp_path: Path):
    decision, out = _run_all(tmp_path)
    assert decision["decision"] in {"PASS", "MIXED", "FAIL"}
    for rel in ["raw/calibration_summary.csv.gz", "raw/evaluation_runs.csv.gz",
                "raw/manifests_evaluation.jsonl.gz", "raw/latency.csv.gz",
                "raw/stage_calibrate.json", "raw/stage_evaluate.json",
                "configs/selected_params.json", "configs/pareto_sets.json",
                "aggregated/descriptive_selected.csv", "aggregated/paired_tests.csv",
                "aggregated/h1_cells.csv", "aggregated/h1_decision.json",
                "aggregated/hypervolume.csv", "tables/h1_decision.md",
                "plots/figA_persistence_f1_vs_gap.pdf", "plots/figE_pareto_gap10.pdf",
                "failures/failure_index.json"]:
        assert (out / rel).exists(), rel

    calib = read_csv_gz(out / "raw" / "calibration_summary.csv.gz")
    n_cfg = 1 + 3 + 3 + 4 + 2
    assert len(calib) == 3 * n_cfg  # three calibrated conditions
    assert all(int(r["n_instances"]) == 4 for r in calib)  # 2 seeds x 2 gaps

    rows = read_csv_gz(out / "raw" / "evaluation_runs.csv.gz")
    assert {r["seed"] for r in rows} == {"3", "4"}  # evaluation seeds only
    meta = json.loads((out / "raw" / "stage_evaluate.json").read_text())
    assert {"git_commit", "git_dirty", "git_branch", "config_sha256", "environment"} <= set(meta)
    with gzip.open(out / "raw" / "manifests_evaluation.jsonl.gz", "rt") as f:
        first = json.loads(f.readline())
    assert "manifests" in first and "lifetimes" in first


def test_phase3_calibration_is_deterministic(tmp_path: Path):
    cfg_path = tmp_path / "mini.yaml"
    cfg_path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    cfg = load_phase3_config(cfg_path)
    run_calibration(cfg, tmp_path / "a", workers=2)
    run_calibration(cfg, tmp_path / "b", workers=1)
    strip = lambda rows: [{k: v for k, v in r.items()} for r in rows]  # noqa: E731
    a = strip(read_csv_gz(tmp_path / "a" / "raw" / "calibration_summary.csv.gz"))
    b = strip(read_csv_gz(tmp_path / "b" / "raw" / "calibration_summary.csv.gz"))
    # Runtime diagnostics are not part of the summary; everything else must match exactly.
    assert a == b
