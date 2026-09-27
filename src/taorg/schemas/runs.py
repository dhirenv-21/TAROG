"""Run metadata schema for experiment tracking.

Every experiment run must save a complete provenance record.
Corresponds to Architecture.md §9 and PRD FR-11.
"""

from __future__ import annotations

import datetime
import platform
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _generate_run_id() -> str:
    """Generate a unique run identifier.

    Format: YYYYMMDD_HHMMSS_<short-uuid> for human readability
    while remaining unique.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    short_uuid = uuid.uuid4().hex[:8]
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{short_uuid}"


@dataclass(slots=True)
class RunMetadata:
    """Complete provenance record for one experiment run.

    Attributes:
        run_id: Unique identifier for this run.
        git_commit: Git commit hash at run time, or "unknown".
        command: Command line or config path that launched this run.
        config: Resolved configuration dictionary.
        dataset_id: Dataset name and split identifier.
        base_model_id: Base VidSGG model name and checkpoint.
        temporal_method: Temporal method name.
        seeds: List of random seeds used.
        corruption_manifest: Path to corruption manifest, if applicable.
        environment: Platform and package information.
        start_time: ISO 8601 start time.
        end_time: ISO 8601 end time, set on completion.
        status: "running", "completed", "failed".
        warnings: Collected warnings during the run.
        metrics: Final metrics dictionary, populated on completion.
        timing: Timing statistics, populated on completion.
    """

    run_id: str = field(default_factory=_generate_run_id)
    git_commit: str = "unknown"
    git_dirty: bool = True  # conservative default
    git_branch: str = "unknown"
    command: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    dataset_id: str = ""
    base_model_id: str = ""
    temporal_method: str = ""
    seeds: list[int] = field(default_factory=list)
    corruption_manifest: Optional[str] = None
    environment: dict[str, str] = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)

    def populate_environment(self) -> None:
        """Fill environment info from current runtime."""
        self.environment = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        }
        # Optionally add torch version if available
        try:
            import torch
            self.environment["torch_version"] = torch.__version__
            self.environment["cuda_available"] = str(torch.cuda.is_available())
        except ImportError:
            pass

    def mark_started(self) -> None:
        """Record start time."""
        self.start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status = "running"
        self.populate_environment()

    def mark_completed(self) -> None:
        """Record completion time."""
        self.end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status = "completed"

    def mark_failed(self, error: str) -> None:
        """Record failure."""
        self.end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status = "failed"
        self.warnings.append(f"FATAL: {error}")
