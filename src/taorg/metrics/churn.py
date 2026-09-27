"""Relationship Churn Rate metric.

Measures temporal instability by counting predicted-state flips.

See research/metric_definitions.md §3 for the frozen definition.
"""

from __future__ import annotations

from dataclasses import dataclass

from taorg.schemas.edges import EdgeKey, FrameGraph


@dataclass(frozen=True)
class ChurnResult:
    """Result of churn rate computation.

    Attributes:
        per_edge_churn: Dict mapping EdgeKey to churn rate for that edge.
        mean_churn: Mean churn rate across edges.
        total_flips: Total number of state flips across all edges.
    """

    per_edge_churn: dict[EdgeKey, float]
    mean_churn: float
    total_flips: int


def compute_churn_rate(
    pred_frames: list[FrameGraph],
    gt_frames: list[FrameGraph] | None = None,
) -> ChurnResult:
    """Compute Relationship Churn Rate from predicted frames.

    A flip occurs when the predicted state of an edge changes between
    consecutive frames (alive→dead or dead→alive).

    Initial appearance and final disappearance do NOT count as flips.

    Args:
        pred_frames: Predicted frame sequence from a temporal method.
        gt_frames: Optional ground-truth frames to define evaluation window.
            If provided, evaluation is restricted to frames between the first
            and last GT observation of each edge.  If None, uses the full
            prediction sequence.

    Returns:
        ChurnResult with per-edge and aggregate churn rates.
    """
    if not pred_frames:
        return ChurnResult(per_edge_churn={}, mean_churn=0.0, total_flips=0)

    # Determine evaluation window per edge from GT if provided
    edge_eval_range: dict[EdgeKey, tuple[int, int]] = {}
    if gt_frames is not None:
        for frame in gt_frames:
            for obs in frame.edge_observations:
                if obs.observed:
                    ek = obs.edge_key
                    if ek not in edge_eval_range:
                        edge_eval_range[ek] = (frame.frame_idx, frame.frame_idx)
                    else:
                        lo, hi = edge_eval_range[ek]
                        edge_eval_range[ek] = (min(lo, frame.frame_idx), max(hi, frame.frame_idx))

    # Build per-frame predicted edge sets
    frame_indices = sorted(set(f.frame_idx for f in pred_frames))
    pred_per_frame: dict[int, set[EdgeKey]] = {}
    all_pred_edges: set[EdgeKey] = set()
    for frame in pred_frames:
        edges = set()
        for obs in frame.edge_observations:
            edges.add(obs.edge_key)
            all_pred_edges.add(obs.edge_key)
        pred_per_frame[frame.frame_idx] = edges

    # If no GT, use full prediction range for each edge
    if gt_frames is None:
        for ek in all_pred_edges:
            first = None
            last = None
            for fi in frame_indices:
                if ek in pred_per_frame.get(fi, set()):
                    if first is None:
                        first = fi
                    last = fi
            if first is not None and last is not None:
                edge_eval_range[ek] = (first, last)

    # Also include edges from GT that were never predicted
    all_edges = set(edge_eval_range.keys()) | all_pred_edges

    # Compute churn per edge
    per_edge_churn: dict[EdgeKey, float] = {}
    total_flips = 0

    for ek in all_edges:
        if ek not in edge_eval_range:
            # Edge never in GT — churn = 0
            per_edge_churn[ek] = 0.0
            continue

        lo, hi = edge_eval_range[ek]
        eval_frames = [fi for fi in frame_indices if lo <= fi <= hi]

        if len(eval_frames) < 2:
            per_edge_churn[ek] = 0.0
            continue

        flips = 0
        for i in range(1, len(eval_frames)):
            prev_alive = ek in pred_per_frame.get(eval_frames[i - 1], set())
            curr_alive = ek in pred_per_frame.get(eval_frames[i], set())
            if prev_alive != curr_alive:
                flips += 1

        churn = flips / (len(eval_frames) - 1)
        per_edge_churn[ek] = churn
        total_flips += flips

    mean_churn = (
        sum(per_edge_churn.values()) / len(per_edge_churn)
        if per_edge_churn
        else 0.0
    )

    return ChurnResult(
        per_edge_churn=per_edge_churn,
        mean_churn=mean_churn,
        total_flips=total_flips,
    )
