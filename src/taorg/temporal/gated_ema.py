"""Visibility-Gated Exponential Moving Average temporal baseline.

This is the CRITICAL baseline for the TAORG research question.  It uses
different decay rates depending on the visibility state of the edge:

    VISIBLE absence:               decay_visible   (fast — treat as termination evidence)
    OCCLUDED_OR_UNRELIABLE absence: decay_occluded  (slow — treat as censored/weak evidence)
    UNKNOWN absence:               decay_unknown   (configurable, defaults to occluded)

The central research hypothesis H1 (PRD §3) is that this visibility-aware
behavior alone provides substantial benefit.  If Vis-Gated EMA already
performs well, V-HPED must earn its additional complexity through ablation.

DO NOT artificially weaken this baseline.

Method name in configs/plots: "Vis-Gated EMA"

Mathematical note:
    On observation:  score = alpha * predicate_score + (1 - alpha) * score
    On VISIBLE absence:              score *= decay_visible
    On OCCLUDED/UNRELIABLE absence:  score *= decay_occluded
    On UNKNOWN absence:              score *= decay_unknown

    The key difference from plain EMA: during occlusion, the score decays
    slowly (high decay_occluded, close to 1.0), effectively "remembering"
    the edge.  When both objects are visible but the relation is absent,
    the score decays quickly (low decay_visible), treating the absence as
    evidence of genuine termination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass
class GatedEMA:
    """Visibility-Gated EMA temporal baseline.

    Uses different decay rates depending on the visibility state of the
    edge when the edge is absent.

    Configuration:
        alpha: Update weight for new observations in (0, 1].
        decay_visible: Decay per frame when BOTH objects are VISIBLE
            but relation is absent.  Should be LOW (fast decay) to
            reflect that visible absence is evidence of termination.
        decay_occluded: Decay per frame when at least one object is
            OCCLUDED_OR_UNRELIABLE.  Should be HIGH (slow decay) to
            preserve the edge through occlusion gaps.
        decay_unknown: Decay per frame when visibility is UNKNOWN.
            Defaults to decay_occluded (conservative).
        score_threshold: Minimum alive score to output an edge.
    """

    alpha: float = 0.3
    decay_visible: float = 0.7
    decay_occluded: float = 0.95
    decay_unknown: float = -1.0  # sentinel; will default to decay_occluded
    score_threshold: float = 0.1

    # Internal state
    _video_id: str = field(default="", init=False, repr=False)
    _edge_scores: dict[EdgeKey, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (0.0 <= self.decay_visible < 1.0):
            raise ValueError(
                f"decay_visible must be in [0, 1), got {self.decay_visible}"
            )
        if not (0.0 <= self.decay_occluded < 1.0):
            raise ValueError(
                f"decay_occluded must be in [0, 1), got {self.decay_occluded}"
            )
        # Default decay_unknown to decay_occluded if sentinel
        if self.decay_unknown < 0:
            self.decay_unknown = self.decay_occluded
        if not (0.0 <= self.decay_unknown < 1.0):
            raise ValueError(
                f"decay_unknown must be in [0, 1), got {self.decay_unknown}"
            )
        # Research-relevant validation: warn if not gated
        if self.decay_visible >= self.decay_occluded:
            logger.warning(
                "decay_visible (%.3f) >= decay_occluded (%.3f): "
                "gating is effectively disabled.  This reduces Vis-Gated EMA "
                "to standard EMA behavior.",
                self.decay_visible,
                self.decay_occluded,
            )

    def _get_decay_for_visibility(
        self, visibility: VisibilityState
    ) -> float:
        """Select the decay rate based on edge visibility.

        Args:
            visibility: Combined visibility state of the edge.

        Returns:
            Appropriate decay rate.
        """
        if visibility == VisibilityState.VISIBLE:
            return self.decay_visible
        elif visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE:
            return self.decay_occluded
        else:
            return self.decay_unknown

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._edge_scores.clear()

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame using visibility-gated EMA.

        For each edge:
          - If observed: EMA update toward predicate_score.
          - If absent: decay using the rate selected by visibility state.

        The visibility of an absent edge is determined from the
        EdgeObservation.edge_visibility property (conservative: VISIBLE
        only if both subject and object are VISIBLE).

        Args:
            frame: Input frame with observations.

        Returns:
            Frame with temporally-smoothed edges.
        """
        # Separate observed from absent edges
        observed_map: dict[EdgeKey, EdgeObservation] = {}
        absent_map: dict[EdgeKey, EdgeObservation] = {}
        for obs in frame.edge_observations:
            if obs.observed:
                observed_map[obs.edge_key] = obs
            else:
                absent_map[obs.edge_key] = obs

        # Update observed edges via EMA
        for edge_key, obs in observed_map.items():
            old_score = self._edge_scores.get(edge_key, 0.0)
            new_score = self.alpha * obs.predicate_score + (1.0 - self.alpha) * old_score
            self._edge_scores[edge_key] = new_score

        # Decay absent edges using visibility-gated rates
        dead_keys: list[EdgeKey] = []
        for edge_key in list(self._edge_scores.keys()):
            if edge_key in observed_map:
                continue  # already updated

            # Determine visibility for this absent edge
            if edge_key in absent_map:
                visibility = absent_map[edge_key].edge_visibility
            else:
                # Edge not in current frame at all — use UNKNOWN
                visibility = VisibilityState.UNKNOWN

            decay = self._get_decay_for_visibility(visibility)
            self._edge_scores[edge_key] *= decay

            if self._edge_scores[edge_key] < 1e-8:
                dead_keys.append(edge_key)

        for k in dead_keys:
            del self._edge_scores[k]

        # Build output
        output_edges: list[EdgeObservation] = []
        for edge_key, score in self._edge_scores.items():
            if score < self.score_threshold:
                continue
            if edge_key in observed_map:
                orig = observed_map[edge_key]
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
                # Construct persisted-edge observation
                if edge_key in absent_map:
                    abs_obs = absent_map[edge_key]
                    output_edges.append(EdgeObservation(
                        edge_key=edge_key,
                        video_id=frame.video_id,
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                        predicate_score=score,
                        observed=False,
                        subject_visibility=abs_obs.subject_visibility,
                        object_visibility=abs_obs.object_visibility,
                        source_model="gated_ema",
                    ))
                else:
                    output_edges.append(EdgeObservation(
                        edge_key=edge_key,
                        video_id=frame.video_id,
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                        predicate_score=score,
                        observed=False,
                        source_model="gated_ema",
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
            "method": "gated_ema",
            "alpha": self.alpha,
            "decay_visible": self.decay_visible,
            "decay_occluded": self.decay_occluded,
            "decay_unknown": self.decay_unknown,
            "score_threshold": self.score_threshold,
            "active_edges": len(self._edge_scores),
        }
