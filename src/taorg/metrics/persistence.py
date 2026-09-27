"""Edge Persistence Precision / Recall / F1.

Evaluates a temporal method's ability to persist edges through corruption
gaps (synthetic occlusion).

See research/metric_definitions.md §2 for the frozen definition.

Scope: evaluated over corrupted intervals only — frames where the edge's
ground-truth state is ALIVE but observations were synthetically suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from taorg.schemas.edges import EdgeKey, FrameGraph
from taorg.noise.manifest import CorruptionManifest


@dataclass(frozen=True)
class PersistenceResult:
    """Result of edge persistence evaluation.

    Attributes:
        tp: True positives (correctly persisted during corruption).
        fp: False positives (falsely output during corruption when GT is dead).
        fn: False negatives (failed to persist during corruption when GT is alive).
        precision: TP / (TP + FP), or 0.0 if undefined.
        recall: TP / (TP + FN), or 0.0 if undefined.
        f1: Harmonic mean of precision and recall, or 0.0 if undefined.
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def compute_persistence_f1(
    gt_frames: list[FrameGraph],
    pred_frames: list[FrameGraph],
    manifest: CorruptionManifest,
) -> PersistenceResult:
    """Compute Edge Persistence F1 over corrupted intervals.

    Args:
        gt_frames: Ground-truth (uncorrupted) frame sequence.
        pred_frames: Predicted (temporal method output) frame sequence.
        manifest: Corruption manifest identifying which frames were corrupted.

    Returns:
        PersistenceResult with TP/FP/FN/precision/recall/F1.
    """
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"Frame count mismatch: GT has {len(gt_frames)}, "
            f"pred has {len(pred_frames)}"
        )

    # Build lookup: which (edge_key, frame_idx) pairs are in corruption intervals
    corrupted_frames: dict[EdgeKey, set[int]] = {}
    for entry in manifest.entries:
        if entry.corruption_type != "occlusion":
            continue
        ek = entry.edge_key
        if ek is None:
            continue
        corrupted_frames.setdefault(ek, set()).update(
            range(entry.start_frame, entry.end_frame + 1)
        )

    if not corrupted_frames:
        return PersistenceResult(tp=0, fp=0, fn=0, precision=0.0, recall=0.0, f1=0.0)

    # Build GT alive set: edges that are observed in uncorrupted GT
    gt_alive: dict[int, set[EdgeKey]] = {}
    for frame in gt_frames:
        alive_keys: set[EdgeKey] = set()
        for obs in frame.edge_observations:
            if obs.observed:
                alive_keys.add(obs.edge_key)
        gt_alive[frame.frame_idx] = alive_keys

    # Build prediction set
    pred_alive: dict[int, set[EdgeKey]] = {}
    for frame in pred_frames:
        alive_keys = set()
        for obs in frame.edge_observations:
            alive_keys.add(obs.edge_key)
        pred_alive[frame.frame_idx] = alive_keys

    # Count TP/FP/FN over corrupted intervals only
    tp = 0
    fp = 0
    fn = 0

    for edge_key, frame_set in corrupted_frames.items():
        for f_idx in sorted(frame_set):
            gt_is_alive = edge_key in gt_alive.get(f_idx, set())
            pred_is_alive = edge_key in pred_alive.get(f_idx, set())

            if gt_is_alive and pred_is_alive:
                tp += 1
            elif not gt_is_alive and pred_is_alive:
                fp += 1
            elif gt_is_alive and not pred_is_alive:
                fn += 1
            # gt_is_dead and pred_is_dead: true negative, not counted

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return PersistenceResult(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
    )
