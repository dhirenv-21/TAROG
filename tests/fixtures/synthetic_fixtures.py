"""Synthetic fixtures for testing without external datasets.

Generates tiny deterministic edge observation sequences that exercise
the canonical schema and temporal method interface.  No downloads required.

This is required by Phase 0 exit gate.
"""

from __future__ import annotations

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import (
    TrackObservation,
    TrackerState,
    VisibilityState,
)


def make_simple_video_frames(
    video_id: str = "synthetic_001",
    num_frames: int = 10,
    fps: float = 30.0,
) -> list[FrameGraph]:
    """Generate a simple synthetic video with one relationship edge.

    Scenario:
        - Track 1 (person, class 0) and Track 2 (bag, class 1) are present
          for all frames.
        - Predicate 0 ("holding") is observed for frames 0-3, absent frames
          4-6 (simulating occlusion), then returns frames 7-9.
        - Subject is always VISIBLE.
        - Object goes OCCLUDED_OR_UNRELIABLE during frames 4-6.

    This is the simplest possible test case for visibility-aware temporal
    methods: can the method bridge the gap during occlusion?

    Args:
        video_id: Video identifier.
        num_frames: Number of frames to generate.
        fps: Frames per second for timestamp calculation.

    Returns:
        List of FrameGraph objects, one per frame.
    """
    frames: list[FrameGraph] = []

    for f in range(num_frames):
        timestamp = f / fps
        in_occlusion = 4 <= f <= 6

        # Subject track: always visible
        subject = TrackObservation(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_id=1,
            class_id=0,
            bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
            detector_confidence=0.95,
            tracker_state=TrackerState.TRACKED,
            visibility_state=VisibilityState.VISIBLE,
        )

        # Object track: occluded during frames 4-6
        obj_vis = (
            VisibilityState.OCCLUDED_OR_UNRELIABLE if in_occlusion
            else VisibilityState.VISIBLE
        )
        obj_tracker = TrackerState.LOST if in_occlusion else TrackerState.TRACKED
        obj = TrackObservation(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_id=2,
            class_id=1,
            bbox_xyxy=(150.0, 150.0, 220.0, 280.0),
            detector_confidence=0.0 if in_occlusion else 0.90,
            tracker_state=obj_tracker,
            visibility_state=obj_vis,
        )

        # Edge: observed when not occluded
        edge_key = EdgeKey(subject_track_id=1, object_track_id=2, predicate_id=0)
        edge_observed = not in_occlusion
        edge = EdgeObservation(
            edge_key=edge_key,
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            predicate_score=0.85 if edge_observed else 0.0,
            observed=edge_observed,
            subject_visibility=VisibilityState.VISIBLE,
            object_visibility=obj_vis,
            source_model="synthetic",
        )

        frames.append(FrameGraph(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_observations=[subject, obj],
            edge_observations=[edge],
        ))

    return frames


def make_termination_video_frames(
    video_id: str = "synthetic_term_001",
    num_frames: int = 10,
    fps: float = 30.0,
) -> list[FrameGraph]:
    """Generate a synthetic video where the relationship genuinely ends.

    Scenario:
        - Track 1 (person) and Track 2 (cup) are always VISIBLE.
        - Predicate 0 ("holding") is observed frames 0-4.
        - Predicate 0 is NOT observed frames 5-9, but both objects are VISIBLE.
        - This represents genuine termination: both visible, relation absent.

    This tests whether temporal methods can detect true termination vs
    incorrectly persisting the edge.

    Args:
        video_id: Video identifier.
        num_frames: Number of frames.
        fps: Frames per second.

    Returns:
        List of FrameGraph objects.
    """
    frames: list[FrameGraph] = []

    for f in range(num_frames):
        timestamp = f / fps
        relation_active = f < 5

        subject = TrackObservation(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_id=1,
            class_id=0,
            bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
            detector_confidence=0.95,
            tracker_state=TrackerState.TRACKED,
            visibility_state=VisibilityState.VISIBLE,
        )

        obj = TrackObservation(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_id=2,
            class_id=2,
            bbox_xyxy=(250.0, 150.0, 350.0, 280.0),
            detector_confidence=0.92,
            tracker_state=TrackerState.TRACKED,
            visibility_state=VisibilityState.VISIBLE,
        )

        edge_key = EdgeKey(subject_track_id=1, object_track_id=2, predicate_id=0)
        edge = EdgeObservation(
            edge_key=edge_key,
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            predicate_score=0.88 if relation_active else 0.05,
            observed=relation_active,
            subject_visibility=VisibilityState.VISIBLE,
            object_visibility=VisibilityState.VISIBLE,
            source_model="synthetic",
        )

        frames.append(FrameGraph(
            video_id=video_id,
            frame_idx=f,
            timestamp=timestamp,
            track_observations=[subject, obj],
            edge_observations=[edge],
        ))

    return frames
