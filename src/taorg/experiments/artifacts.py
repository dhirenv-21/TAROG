"""Run artifact writer for experiment provenance.

Creates a structured output directory for each experiment run,
saves resolved configs, metadata, and metrics in machine-readable format.

Corresponds to Architecture.md §9.

Directory structure per run:
    outputs/runs/<run_id>/
    ├── config.resolved.yaml
    ├── metadata.json
    ├── metrics.json
    ├── timing.json
    ├── warnings.log
    └── corruption_manifest.json  (if applicable)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from taorg.schemas.runs import RunMetadata

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_DIR = Path("outputs/runs")


class RunArtifactWriter:
    """Manages the output directory and artifact files for one experiment run.

    Usage:
        writer = RunArtifactWriter(metadata, output_base=Path("outputs/runs"))
        writer.create_run_directory()
        writer.save_config(resolved_config)
        # ... run experiment ...
        writer.save_metrics(metrics_dict)
        writer.save_timing(timing_dict)
        writer.save_metadata()
    """

    def __init__(
        self,
        metadata: RunMetadata,
        output_base: Path = DEFAULT_OUTPUTS_DIR,
    ) -> None:
        """Initialize the artifact writer.

        Args:
            metadata: Run metadata object (will be mutated on save).
            output_base: Base directory for all run outputs.
        """
        self.metadata = metadata
        self.run_dir = output_base / metadata.run_id

    def create_run_directory(self) -> Path:
        """Create the run output directory.

        Returns:
            Path to the created run directory.

        Raises:
            FileExistsError: If the run directory already exists.
        """
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")

        self.run_dir.mkdir(parents=True, exist_ok=False)
        logger.info("Created run directory: %s", self.run_dir)
        return self.run_dir

    def save_config(self, config: dict[str, Any]) -> Path:
        """Save the fully resolved configuration.

        Args:
            config: Resolved configuration dictionary.

        Returns:
            Path to the saved config file.
        """
        config_path = self.run_dir / "config.resolved.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=True)
        logger.info("Saved resolved config to %s", config_path)
        return config_path

    def save_metadata(self) -> Path:
        """Save the run metadata as JSON.

        Returns:
            Path to the saved metadata file.
        """
        meta_path = self.run_dir / "metadata.json"
        meta_dict = asdict(self.metadata)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, default=str)
        logger.info("Saved metadata to %s", meta_path)
        return meta_path

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        """Save experiment metrics.

        Args:
            metrics: Metrics dictionary.

        Returns:
            Path to the saved metrics file.
        """
        self.metadata.metrics = metrics
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Saved metrics to %s", metrics_path)
        return metrics_path

    def save_timing(self, timing: dict[str, Any]) -> Path:
        """Save timing statistics.

        Args:
            timing: Timing dictionary.

        Returns:
            Path to the saved timing file.
        """
        self.metadata.timing = timing
        timing_path = self.run_dir / "timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2, default=str)
        logger.info("Saved timing to %s", timing_path)
        return timing_path

    def save_warnings(self) -> Path:
        """Save accumulated warnings.

        Returns:
            Path to the warnings log file.
        """
        warnings_path = self.run_dir / "warnings.log"
        with open(warnings_path, "w", encoding="utf-8") as f:
            for w in self.metadata.warnings:
                f.write(w + "\n")
        return warnings_path

    def save_all(self, config: dict[str, Any]) -> None:
        """Convenience: save config, metadata, and warnings.

        Args:
            config: Resolved configuration dictionary.
        """
        self.save_config(config)
        self.save_metadata()
        self.save_warnings()
