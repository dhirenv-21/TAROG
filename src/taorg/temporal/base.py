"""Base protocol for temporal edge methods.

All temporal methods — baselines and V-HPED — must implement this
interface so the evaluation pipeline has no method-specific branches.

Corresponds to Architecture.md §5.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from taorg.schemas.edges import FrameGraph


@runtime_checkable
class TemporalEdgeModel(Protocol):
    """Protocol for temporal edge methods.

    Every method consumes a stream of FrameGraphs and produces
    temporally-processed FrameGraphs with the same schema.

    Lifecycle:
        model.reset_video(video_id)
        for frame in video_frames:
            output_frame = model.step(frame)
        model.finalize_video()
    """

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video.

        Args:
            video_id: Identifier of the new video.
        """
        ...

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame and return temporally-filtered output.

        Args:
            frame: Input frame graph with observations.

        Returns:
            Output frame graph with temporally-processed edges.
        """
        ...

    def finalize_video(self) -> None:
        """Finalize processing for the current video.

        Called after the last frame.  Methods may use this for cleanup,
        final statistics, etc.
        """
        ...

    def state_dict(self) -> dict:
        """Return serializable internal state for checkpointing.

        Returns:
            Dictionary of method state.
        """
        ...
