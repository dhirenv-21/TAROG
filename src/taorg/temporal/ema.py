"""Exponential Moving Average (EMA) temporal baseline.

Maintains an exponentially decaying score for each edge.  When the edge is
observed, the score is updated toward the observation; when absent, the score
decays toward zero at a fixed rate regardless of visibility.

This is the standard visibility-UNAWARE EMA baseline.  Compare with
`gated_ema.py` which uses different decay rates based on visibility.

Method name in configs/plots: "EMA"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass
class EMA:
    """Exponential Moving Average temporal baseline.

    For each tracked edge, maintains a smoothed alive score:

        On observation:   score = alpha * predicate_score + (1 - alpha) * score
        On absence:       score = decay * score

    The edge is output if score >= score_threshold.

    Configuration:
        alpha: Update weight for new observations in (0, 1].
            Higher alpha means the current observation dominates.
        decay: Multiplicative decay per absent frame in [0, 1).
            Lower decay means faster forgetting.
        score_threshold: Minimum score to output an edge.
    """

    alpha: float = 0.3
    decay: float = 0.9
    score_threshold: float = 0.1

    # Internal state
    _video_id: str = field(default="", init=False, repr=False)
    _edge_scores: dict[EdgeKey, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (0.0 <= self.decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {self.decay}")

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._edge_scores.clear()

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame using EMA temporal smoothing.

        Args:
            frame: Input frame with observations.

        Returns:
            Frame with temporally-smoothed edges.
        """
        # Collect which edges are observed this frame
        observed_keys: set[EdgeKey] = set()
        obs_map: dict[EdgeKey, EdgeObservation] = {}
        for obs in frame.edge_observations:
            if obs.observed:
                observed_keys.add(obs.edge_key)
                obs_map[obs.edge_key] = obs

        # Update observed edges
        for edge_key, obs in obs_map.items():
            old_score = self._edge_scores.get(edge_key, 0.0)
            new_score = self.alpha * obs.predicate_score + (1.0 - self.alpha) * old_score
            self._edge_scores[edge_key] = new_score

        # Decay unobserved edges
        dead_keys: list[EdgeKey] = []
        for edge_key in list(self._edge_scores.keys()):
            if edge_key not in observed_keys:
                self._edge_scores[edge_key] *= self.decay
                # Prune edges that have decayed below a tiny threshold
                if self._edge_scores[edge_key] < 1e-8:
                    dead_keys.append(edge_key)

        for k in dead_keys:
            del self._edge_scores[k]

        # Build output: all edges above threshold
        output_edges: list[EdgeObservation] = []
        for edge_key, score in self._edge_scores.items():
            if score < self.score_threshold:
                continue
            if edge_key in obs_map:
                # Use the original observation but note the smoothed score
                orig = obs_map[edge_key]
                output_edges.append(EdgeObservation(
                    edge_key=edge_key,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    predicate_score=score,
                    observed=True,
                    subject_visibility=orig.subject_visibility,
                    object_visibility=orig.object_visibility,
                    source_model=orig.source_model,
                    source_checkpoint=orig.source_checkpoint,
                ))
            else:
                output_edges.append(EdgeObservation(
                    edge_key=edge_key,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    predicate_score=score,
                    observed=False,
                    source_model="ema",
                ))

        return FrameGraph(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=frame.track_observations,
            edge_observations=output_edges,
        )

    def finalize_video(self) -> None:
        """Finalize current video."""
        pass

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {
            "method": "ema",
            "alpha": self.alpha,
            "decay": self.decay,
            "score_threshold": self.score_threshold,
            "active_edges": len(self._edge_scores),
        }
