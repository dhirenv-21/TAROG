"""Fixed-window temporal baseline.

Maintains a sliding window of recent observations.  An edge is considered
alive if it was observed at least once within the last `window_size` frames.

The alive score is the maximum predicate_score seen within the window.

Method name in configs/plots: "Fixed Window"
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass
class FixedWindow:
    """Fixed-window temporal memory.

    Keeps a sliding window of the last `window_size` frames of observations
    for each edge.  An edge is output if it has been observed at least once
    within the window and its best score is above `score_threshold`.

    Configuration:
        window_size: Number of past frames to remember (inclusive of current).
        score_threshold: Minimum score to output an edge.
    """

    window_size: int = 5
    score_threshold: float = 0.0

    # Internal state
    _video_id: str = field(default="", init=False, repr=False)
    # Map: EdgeKey -> deque of (frame_idx, predicate_score, observed)
    _edge_history: dict[EdgeKey, deque[tuple[int, float, bool]]] = field(
        default_factory=lambda: defaultdict(deque), init=False, repr=False
    )

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._edge_history.clear()

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame using fixed-window memory.

        For each observed edge, add it to the window.
        For all tracked edges, evict entries older than window_size.
        Output edges that have at least one observation in the window.

        Args:
            frame: Input frame with observations.

        Returns:
            Frame with temporally-filtered edges.
        """
        current_frame = frame.frame_idx
        cutoff_frame = current_frame - self.window_size + 1

        # Record current observations
        current_obs: dict[EdgeKey, EdgeObservation] = {}
        for obs in frame.edge_observations:
            if obs.observed:
                self._edge_history[obs.edge_key].append(
                    (obs.frame_idx, obs.predicate_score, True)
                )
                current_obs[obs.edge_key] = obs

        # Evict old entries and build output
        output_edges: list[EdgeObservation] = []
        dead_keys: list[EdgeKey] = []

        for edge_key, history in self._edge_history.items():
            # Evict entries outside window
            while history and history[0][0] < cutoff_frame:
                history.popleft()

            if not history:
                dead_keys.append(edge_key)
                continue

            # Best score in window
            best_score = max(score for _, score, _ in history)
            if best_score < self.score_threshold:
                continue

            # If we have a current observation for this edge, use it
            if edge_key in current_obs:
                output_edges.append(current_obs[edge_key])
            else:
                # Persist edge from window — construct a persistence observation
                # Use the most recent entry's score
                most_recent_score = history[-1][1]
                output_edges.append(EdgeObservation(
                    edge_key=edge_key,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    predicate_score=most_recent_score,
                    observed=False,  # this is a persisted, not directly observed, edge
                    source_model="fixed_window",
                ))

        # Cleanup dead keys
        for k in dead_keys:
            del self._edge_history[k]

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
            "method": "fixed_window",
            "window_size": self.window_size,
            "score_threshold": self.score_threshold,
            "active_edges": len(self._edge_history),
        }
