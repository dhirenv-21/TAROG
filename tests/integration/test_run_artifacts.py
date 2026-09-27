"""Integration test: dummy experiment creates a complete run artifact folder.

This is the key Phase 0 exit gate test: verify that a dummy experiment
run creates all required artifact files without any external dataset.
"""

import json
from pathlib import Path

import yaml
import pytest

from taorg.config import load_config, validate_experiment_config
from taorg.experiments.artifacts import RunArtifactWriter
from taorg.schemas.runs import RunMetadata
from taorg.utils.seed import set_seed
from tests.fixtures.synthetic_fixtures import make_simple_video_frames


class TestDummyExperiment:
    """End-to-end test: create a run artifact directory from synthetic data."""

    def test_complete_run_artifact(self, tmp_path: Path):
        """A dummy experiment creates all required artifact files."""

        # 1. Load and validate config
        config = {
            "dataset": "synthetic_fixture",
            "dataset_split": "test",
            "temporal_method": "no_memory",
            "seed": 42,
            "temporal_params": {},
            "noise": {},
            "metrics": ["edge_persistence_f1"],
            "runtime": {"device": "cpu", "log_level": "INFO"},
        }
        errors = validate_experiment_config(config)
        assert errors == [], f"Config validation errors: {errors}"

        # 2. Set seed
        set_seed(config["seed"])

        # 3. Create run metadata
        metadata = RunMetadata(
            config=config,
            dataset_id="synthetic_fixture/test",
            temporal_method="no_memory",
            seeds=[42],
        )
        metadata.mark_started()

        # 4. Create artifact writer and run directory
        writer = RunArtifactWriter(metadata, output_base=tmp_path)
        run_dir = writer.create_run_directory()
        assert run_dir.exists()

        # 5. Generate synthetic data (no external dataset needed)
        frames = make_simple_video_frames()
        assert len(frames) == 10

        # 6. Simulate a trivial experiment (no temporal processing yet)
        dummy_metrics = {
            "edge_persistence_f1": 0.0,
            "churn_rate": 0.0,
            "num_frames_processed": len(frames),
        }
        dummy_timing = {
            "total_seconds": 0.001,
            "frames_per_second": 10000.0,
        }

        # 7. Save all artifacts
        metadata.mark_completed()
        writer.save_config(config)
        writer.save_metrics(dummy_metrics)
        writer.save_timing(dummy_timing)
        writer.save_metadata()
        writer.save_warnings()

        # 8. Verify all required files exist
        assert (run_dir / "config.resolved.yaml").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "timing.json").exists()
        assert (run_dir / "warnings.log").exists()

        # 9. Verify file contents are parseable
        with open(run_dir / "config.resolved.yaml") as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config["seed"] == 42

        with open(run_dir / "metadata.json") as f:
            loaded_meta = json.load(f)
        assert loaded_meta["status"] == "completed"
        assert loaded_meta["run_id"] == metadata.run_id

        with open(run_dir / "metrics.json") as f:
            loaded_metrics = json.load(f)
        assert loaded_metrics["num_frames_processed"] == 10

    def test_duplicate_run_id_fails(self, tmp_path: Path):
        """Cannot create two run directories with the same ID."""
        metadata = RunMetadata(run_id="test_dup_001")
        writer = RunArtifactWriter(metadata, output_base=tmp_path)
        writer.create_run_directory()

        # Second attempt should fail
        writer2 = RunArtifactWriter(metadata, output_base=tmp_path)
        with pytest.raises(FileExistsError):
            writer2.create_run_directory()
