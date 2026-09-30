"""Small end-to-end synthetic evaluation pipeline.

Generates canonical FrameGraph sequences, applies deterministic corruption,
runs Phase 1 baselines, computes Phase 2 metrics, and saves structured
outputs to run artifact directories.

These are SYNTHETIC TEST OUTPUTS — not paper evidence.  Do not add
results to claims.md.

Corresponds to Phase 2 exit gate requirements.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from taorg.experiments.artifacts import RunArtifactWriter
from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.latency import measure_latency
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.termination import (
    compute_false_persistence,
    compute_termination_delay,
)
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.schemas.edges import FrameGraph
from taorg.schemas.runs import RunMetadata
from taorg.temporal import METHODS
from taorg.utils.git import get_git_branch, get_git_commit, get_git_dirty

logger = logging.getLogger(__name__)


def run_synthetic_eval(
    gt_frames: list[FrameGraph],
    seed: int = 42,
    gap_lengths: list[int] | None = None,
    method_names: list[str] | None = None,
    output_base: Path | None = None,
    score_threshold: float = 0.1,
) -> dict[str, Any]:
    """Run a complete synthetic evaluation pipeline.

    Steps:
        1. Apply deterministic occlusion corruption
        2. Run each baseline on corrupted input
        3. Compute all Phase 2 metrics
        4. Save results to run artifact directory

    Args:
        gt_frames: Ground-truth (uncorrupted) frame sequence.
        seed: Random seed for corruption.
        gap_lengths: Occlusion gap lengths. Defaults to [3].
        method_names: Methods to evaluate. Defaults to the five Phase 1 baselines
            (Phase 4 methods need explicit parameters and are evaluated by
            taorg.experiments.phase4).
        output_base: Output directory for run artifacts. If None, no artifacts saved.
        score_threshold: Score threshold for method construction.

    Returns:
        Dict mapping method_name -> metrics dict.
    """
    if gap_lengths is None:
        gap_lengths = [3]
    if method_names is None:
        method_names = ["no_memory", "fixed_window", "ema", "gated_ema", "hmm"]

    video_id = gt_frames[0].video_id if gt_frames else "unknown"

    # Step 1: Apply corruption
    config = OcclusionConfig(gap_lengths=gap_lengths)
    corrupted_frames, manifest = inject_occlusion(gt_frames, seed=seed, config=config)

    # Collect git provenance
    git_commit = get_git_commit()
    git_dirty = get_git_dirty()
    git_branch = get_git_branch()

    all_results: dict[str, Any] = {}

    for method_name in method_names:
        if method_name not in METHODS:
            logger.warning("Unknown method: %s, skipping", method_name)
            continue

        cls = METHODS[method_name]

        # Construct method with score_threshold if supported
        try:
            model = cls(score_threshold=score_threshold)
        except TypeError:
            model = cls()

        # Step 2: Run method on corrupted input
        model.reset_video(video_id)
        pred_frames: list[FrameGraph] = []
        for frame in corrupted_frames:
            out = model.step(frame)
            pred_frames.append(out)
        model.finalize_video()

        # Step 3: Compute metrics
        persistence = compute_persistence_f1(gt_frames, pred_frames, manifest)
        churn = compute_churn_rate(pred_frames, gt_frames)
        termination = compute_termination_delay(gt_frames, pred_frames)
        false_pers = compute_false_persistence(gt_frames, pred_frames)
        latency = measure_latency(model, corrupted_frames, video_id)

        method_metrics = {
            "method": method_name,
            "seed": seed,
            "gap_lengths": gap_lengths,
            "video_id": video_id,
            "persistence": {
                "tp": persistence.tp,
                "fp": persistence.fp,
                "fn": persistence.fn,
                "precision": persistence.precision,
                "recall": persistence.recall,
                "f1": persistence.f1,
            },
            "churn": {
                "mean_churn": churn.mean_churn,
                "total_flips": churn.total_flips,
            },
            "termination_delay": {
                "mean_delay": termination.mean_delay,
                "median_delay": termination.median_delay,
                "num_censored": termination.num_censored,
                "num_evaluated": termination.num_evaluated,
            },
            "false_persistence": {
                "mean_rate": false_pers.mean_rate,
                "num_evaluated": false_pers.num_evaluated,
            },
            "latency": {
                "mean_ms": latency.mean_ms,
                "median_ms": latency.median_ms,
                "p95_ms": latency.p95_ms,
                "p99_ms": latency.p99_ms,
                "total_frames": latency.total_frames,
            },
            "provenance": {
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "git_branch": git_branch,
            },
        }

        all_results[method_name] = method_metrics

        # Step 4: Save artifacts if output_base specified
        if output_base is not None:
            metadata = RunMetadata(
                git_commit=git_commit,
                git_dirty=git_dirty,
                git_branch=git_branch,
                dataset_id=f"synthetic/{video_id}",
                temporal_method=method_name,
                seeds=[seed],
            )
            metadata.mark_started()

            writer = RunArtifactWriter(metadata, output_base=output_base)
            writer.create_run_directory()
            writer.save_config({
                "method": method_name,
                "seed": seed,
                "gap_lengths": gap_lengths,
                "score_threshold": score_threshold,
                "corruption_type": "occlusion",
            })
            writer.save_metrics(method_metrics)
            writer.save_timing({
                "latency_mean_ms": latency.mean_ms,
                "latency_median_ms": latency.median_ms,
            })

            # Save corruption manifest
            manifest.save(writer.run_dir / "corruption_manifest.json")

            metadata.mark_completed()
            writer.save_metadata()
            logger.info("Saved artifacts for %s to %s", method_name, writer.run_dir)

    return all_results
