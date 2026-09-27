"""Termination Delay and False Persistence Rate metrics.

Measures how quickly a temporal method correctly drops an edge after
the ground-truth relationship genuinely terminates.

See research/metric_definitions.md §4 and §5 for frozen definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, FrameGraph


@dataclass(frozen=True)
class TerminationDelayResult:
    """Result of termination delay computation.

    Attributes:
        per_edge_delay: Dict mapping EdgeKey to delay in frames.
            Negative = early termination.
        per_edge_censored: Dict mapping EdgeKey to whether the delay is censored.
        mean_delay: Mean delay across all edges (including censored).
        median_delay: Median delay.
        num_censored: Number of edges where the method never dropped the edge.
        num_evaluated: Total number of edges with known GT termination.
    """

    per_edge_delay: dict[EdgeKey, int]
    per_edge_censored: dict[EdgeKey, bool]
    mean_delay: float
    median_delay: float
    num_censored: int
    num_evaluated: int


@dataclass(frozen=True)
class FalsePersistenceResult:
    """Result of false persistence rate computation.

    Attributes:
        per_edge_rate: Dict mapping EdgeKey to false persistence rate [0, 1].
        mean_rate: Mean false persistence rate across edges.
        num_evaluated: Number of edges evaluated.
    """

    per_edge_rate: dict[EdgeKey, float]
    mean_rate: float
    num_evaluated: int


def compute_termination_delay(
    gt_frames: list[FrameGraph],
    pred_frames: list[FrameGraph],
) -> TerminationDelayResult:
    """Compute termination delay for each edge with known GT termination.

    For each edge where the ground truth terminates (stops being observed
    while remaining visible), compute how many frames elapse before the
    temporal method drops the edge.

    Args:
        gt_frames: Ground-truth (uncorrupted) frame sequence.
        pred_frames: Predicted frame sequence from a temporal method.

    Returns:
        TerminationDelayResult.
    """
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"Frame count mismatch: GT={len(gt_frames)}, pred={len(pred_frames)}"
        )

    seq_len = len(gt_frames)
    frame_indices = [f.frame_idx for f in gt_frames]

    # Find GT termination frame for each edge:
    # last frame where edge is observed in GT
    gt_last_obs: dict[EdgeKey, int] = {}
    gt_first_obs: dict[EdgeKey, int] = {}
    for frame in gt_frames:
        for obs in frame.edge_observations:
            if obs.observed:
                ek = obs.edge_key
                if ek not in gt_first_obs:
                    gt_first_obs[ek] = frame.frame_idx
                gt_last_obs[ek] = frame.frame_idx

    # Only consider edges that terminate BEFORE the sequence ends
    # (i.e., gt_last_obs < last frame index)
    last_frame_idx = frame_indices[-1] if frame_indices else 0
    terminating_edges = {
        ek: t_end for ek, t_end in gt_last_obs.items()
        if t_end < last_frame_idx
    }

    if not terminating_edges:
        return TerminationDelayResult(
            per_edge_delay={}, per_edge_censored={},
            mean_delay=0.0, median_delay=0.0,
            num_censored=0, num_evaluated=0,
        )

    # Build prediction set per frame
    pred_per_frame: dict[int, set[EdgeKey]] = {}
    for frame in pred_frames:
        edges = set()
        for obs in frame.edge_observations:
            edges.add(obs.edge_key)
        pred_per_frame[frame.frame_idx] = edges

    # Compute delay per edge
    per_edge_delay: dict[EdgeKey, int] = {}
    per_edge_censored: dict[EdgeKey, bool] = {}

    for ek, t_end in terminating_edges.items():
        # Find first frame after t_end where method does NOT output edge
        t_drop = None
        for f_idx in frame_indices:
            if f_idx <= t_end:
                continue
            if ek not in pred_per_frame.get(f_idx, set()):
                t_drop = f_idx
                break

        if t_drop is None:
            # Never dropped — censored
            delay = seq_len - t_end
            per_edge_delay[ek] = delay
            per_edge_censored[ek] = True
        else:
            delay = t_drop - t_end
            per_edge_delay[ek] = delay
            per_edge_censored[ek] = False

    # Aggregate
    delays = sorted(per_edge_delay.values())
    mean_delay = sum(delays) / len(delays) if delays else 0.0
    median_delay = delays[len(delays) // 2] if delays else 0.0
    num_censored = sum(1 for c in per_edge_censored.values() if c)

    return TerminationDelayResult(
        per_edge_delay=per_edge_delay,
        per_edge_censored=per_edge_censored,
        mean_delay=mean_delay,
        median_delay=median_delay,
        num_censored=num_censored,
        num_evaluated=len(terminating_edges),
    )


def compute_false_persistence(
    gt_frames: list[FrameGraph],
    pred_frames: list[FrameGraph],
) -> FalsePersistenceResult:
    """Compute false persistence rate for each edge with known GT termination.

    Fraction of post-termination frames where the method still outputs the edge.

    Args:
        gt_frames: Ground-truth (uncorrupted) frame sequence.
        pred_frames: Predicted frame sequence from a temporal method.

    Returns:
        FalsePersistenceResult.
    """
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"Frame count mismatch: GT={len(gt_frames)}, pred={len(pred_frames)}"
        )

    frame_indices = [f.frame_idx for f in gt_frames]
    last_frame_idx = frame_indices[-1] if frame_indices else 0

    # Find GT termination
    gt_last_obs: dict[EdgeKey, int] = {}
    for frame in gt_frames:
        for obs in frame.edge_observations:
            if obs.observed:
                gt_last_obs[obs.edge_key] = frame.frame_idx

    terminating_edges = {
        ek: t_end for ek, t_end in gt_last_obs.items()
        if t_end < last_frame_idx
    }

    if not terminating_edges:
        return FalsePersistenceResult(
            per_edge_rate={}, mean_rate=0.0, num_evaluated=0,
        )

    # Build prediction set
    pred_per_frame: dict[int, set[EdgeKey]] = {}
    for frame in pred_frames:
        edges = set()
        for obs in frame.edge_observations:
            edges.add(obs.edge_key)
        pred_per_frame[frame.frame_idx] = edges

    per_edge_rate: dict[EdgeKey, float] = {}
    for ek, t_end in terminating_edges.items():
        post_term_frames = [fi for fi in frame_indices if fi > t_end]
        if not post_term_frames:
            continue  # no post-termination frames to evaluate

        falsely_persisted = sum(
            1 for fi in post_term_frames
            if ek in pred_per_frame.get(fi, set())
        )
        rate = falsely_persisted / len(post_term_frames)
        per_edge_rate[ek] = rate

    mean_rate = (
        sum(per_edge_rate.values()) / len(per_edge_rate)
        if per_edge_rate
        else 0.0
    )

    return FalsePersistenceResult(
        per_edge_rate=per_edge_rate,
        mean_rate=mean_rate,
        num_evaluated=len(per_edge_rate),
    )
