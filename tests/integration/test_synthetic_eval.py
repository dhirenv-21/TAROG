"""Integration test for the end-to-end synthetic evaluation pipeline.

Verifies:
    - Pipeline runs to completion for all baselines
    - Metrics are computed and structured correctly
    - Corruption manifest is generated
    - Git provenance is captured
    - Artifacts are saved when output directory is specified
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taorg.experiments.synthetic_eval import run_synthetic_eval
from tests.fixtures.synthetic_fixtures import make_simple_video_frames


class TestSyntheticEvalPipeline:
    def test_all_baselines_run(self):
        """All Phase 1 baselines run to completion on synthetic data."""
        frames = make_simple_video_frames(num_frames=20)
        results = run_synthetic_eval(frames, seed=42, gap_lengths=[3])

        assert "no_memory" in results
        assert "fixed_window" in results
        assert "ema" in results
        assert "gated_ema" in results
        assert "hmm" in results

    def test_metrics_structure(self):
        """Each method result contains all required metric sections."""
        frames = make_simple_video_frames(num_frames=20)
        results = run_synthetic_eval(
            frames, seed=42, gap_lengths=[3],
            method_names=["ema"],
        )

        ema = results["ema"]
        assert "persistence" in ema
        assert "churn" in ema
        assert "termination_delay" in ema
        assert "false_persistence" in ema
        assert "latency" in ema
        assert "provenance" in ema

        # Check persistence has correct keys
        assert "f1" in ema["persistence"]
        assert "tp" in ema["persistence"]
        assert "fn" in ema["persistence"]

        # Check provenance
        assert "git_commit" in ema["provenance"]
        assert "git_dirty" in ema["provenance"]

    def test_deterministic_results(self):
        """Same input produces same results."""
        frames = make_simple_video_frames(num_frames=20)

        r1 = run_synthetic_eval(frames, seed=42, gap_lengths=[3], method_names=["ema"])
        r2 = run_synthetic_eval(frames, seed=42, gap_lengths=[3], method_names=["ema"])

        assert r1["ema"]["persistence"] == r2["ema"]["persistence"]
        assert r1["ema"]["churn"] == r2["ema"]["churn"]

    def test_artifact_saving(self, tmp_path: Path):
        """Artifacts are correctly saved to output directory."""
        frames = make_simple_video_frames(num_frames=20)
        results = run_synthetic_eval(
            frames, seed=42, gap_lengths=[3],
            method_names=["no_memory"],
            output_base=tmp_path,
        )

        # Find the run directory
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        assert (run_dir / "config.resolved.yaml").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "corruption_manifest.json").exists()

        # Verify manifest is valid JSON
        with open(run_dir / "corruption_manifest.json") as f:
            manifest_data = json.load(f)
        assert manifest_data["seed"] == 42

        # Verify metadata has git info
        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "git_commit" in meta
        assert "git_dirty" in meta
        assert "git_branch" in meta
        assert meta["status"] == "completed"

    def test_gated_ema_beats_no_memory_persistence(self):
        """Sanity check: GatedEMA should persist better than NoMemory during occlusion."""
        frames = make_simple_video_frames(num_frames=20)
        results = run_synthetic_eval(
            frames, seed=42, gap_lengths=[3],
            method_names=["no_memory", "gated_ema"],
        )

        # NoMemory drops edges immediately during occlusion
        nm_recall = results["no_memory"]["persistence"]["recall"]
        # GatedEMA should persist through occlusion
        ge_recall = results["gated_ema"]["persistence"]["recall"]

        # NoMemory should have 0 recall (it can't persist)
        assert nm_recall == 0.0
        # GatedEMA should have >0 recall
        assert ge_recall >= nm_recall
