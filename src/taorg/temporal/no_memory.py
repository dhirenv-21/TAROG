"""No-memory temporal baseline.

The simplest possible baseline: passes through only the edges that are
observed in the current frame.  No temporal state is maintained.

This is the lower bound for temporal methods — it has zero persistence
but also zero false persistence after genuine termination.

Method name in configs/plots: "No Memory"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeObservation, FrameGraph

logger = logging.getLogger(__name__)


@dataclass
class NoMemory:
    """No temporal memory — pass through current-frame observations only.

    Configuration:
        score_threshold: Minimum predicate_score to include an edge.
            Edges below this threshold are dropped even if observed.
    """

    score_threshold: float = 0.0
    _video_id: str = field(default="", init=False, repr=False)

    def reset_video(self, video_id: str) -> None:
        """Reset for a new video (no-op — no state to reset)."""
        self._video_id = video_id

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Return only currently observed edges above threshold.

        Args:
            frame: Input frame with observations.

        Returns:
            Frame with only currently observed, above-threshold edges.
        """
        kept: list[EdgeObservation] = [
            obs for obs in frame.edge_observations
            if obs.observed and obs.predicate_score >= self.score_threshold
        ]
        return FrameGraph(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=frame.track_observations,
            edge_observations=kept,
        )

    def finalize_video(self) -> None:
        """Finalize current video (no-op)."""
        pass

    def state_dict(self) -> dict:
        """Return empty state dict."""
        return {"method": "no_memory", "score_threshold": self.score_threshold}
