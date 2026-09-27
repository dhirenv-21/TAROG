"""Unit tests for configuration loading and validation."""

from pathlib import Path
import tempfile

import pytest
import yaml

from taorg.config import load_config, merge_configs, validate_experiment_config


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_yaml(self, tmp_path: Path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("dataset: test\nseed: 42\ntemporal_method: ema\n")
        config = load_config(config_file)
        assert config["dataset"] == "test"
        assert config["seed"] == 42

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_file(self, tmp_path: Path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config == {}

    def test_load_non_dict(self, tmp_path: Path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        with pytest.raises(TypeError):
            load_config(config_file)


class TestMergeConfigs:
    """Tests for merge_configs."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 3, "z": 4}}
        result = merge_configs(base, override)
        assert result == {"nested": {"x": 1, "y": 3, "z": 4}}

    def test_no_mutation(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        merge_configs(base, override)
        assert base == {"a": {"b": 1}}  # original unchanged


class TestValidateExperimentConfig:
    """Tests for validate_experiment_config."""

    def test_valid_config(self):
        config = {"dataset": "ag", "temporal_method": "ema", "seed": 42}
        errors = validate_experiment_config(config)
        assert errors == []

    def test_missing_fields(self):
        config = {}
        errors = validate_experiment_config(config)
        assert len(errors) == 3  # dataset, temporal_method, seed

    def test_invalid_seed_type(self):
        config = {"dataset": "ag", "temporal_method": "ema", "seed": "bad"}
        errors = validate_experiment_config(config)
        assert any("seed" in e and "integer" in e for e in errors)
