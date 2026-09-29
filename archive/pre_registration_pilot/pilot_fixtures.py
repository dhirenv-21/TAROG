"""Extended synthetic fixtures for Phase 3 go/no-go pilot.

Provides richer scenarios beyond the simple occlusion gap:
    - Multi-edge sequences with staggered lifetimes
    - Long sequences for realistic corruption injection
    - All tracks VISIBLE — corruption injector adds occlusion

These fixtures generate ground-truth sequences that the corruption
injector modifies.  Do NOT hard-code occlusion here.
"""

from __future__ import annotations

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, TrackerState, VisibilityState


def make_multi_edge_video(
    video_id: str = "pilot_multi_001",
    num_frames: int = 100,
    fps: float = 30.0,
) -> list[FrameGraph]:
    """Generate a multi-edge synthetic video with staggered lifetimes.

    Scenario:
        - 4 tracks (person=1, bag=2, cup=3, chair=4)
        - Edge (1,2,pred=0) "holding": alive frames 0-79
        - Edge (1,3,pred=1) "drinking": alive frames 20-59
        - Edge (1,4,pred=2) "sitting": alive entire sequence
        - Edge (3,4,pred=3) "on_top_of": alive frames 30-69

    All tracks VISIBLE for all frames.  No built-in occlusion.
    """
    edge_lifetimes = {
        EdgeKey(1, 2, 0): (0, min(79, num_frames - 1)),
        EdgeKey(1, 3, 1): (20, min(59, num_frames - 1)),
        EdgeKey(1, 4, 2): (0, num_frames - 1),
        EdgeKey(3, 4, 3): (30, min(69, num_frames - 1)),
    }

    track_classes = {1: 0, 2: 1, 3: 2, 4: 3}
    track_bboxes = {
        1: (100.0, 100.0, 200.0, 300.0),
        2: (150.0, 150.0, 220.0, 280.0),
        3: (250.0, 200.0, 300.0, 260.0),
        4: (80.0, 280.0, 300.0, 400.0),
    }

    frames: list[FrameGraph] = []
    for f in range(num_frames):
        timestamp = f / fps
        tracks = []
        for tid in sorted(track_classes.keys()):
            tracks.append(TrackObservation(
                video_id=video_id, frame_idx=f, timestamp=timestamp,
                track_id=tid, class_id=track_classes[tid],
                bbox_xyxy=track_bboxes[tid], detector_confidence=0.92,
                tracker_state=TrackerState.TRACKED,
                visibility_state=VisibilityState.VISIBLE,
            ))

        edges = []
        for ek, (start, end) in edge_lifetimes.items():
            alive = start <= f <= end
            edges.append(EdgeObservation(
                edge_key=ek, video_id=video_id, frame_idx=f,
                timestamp=timestamp,
                predicate_score=0.85 if alive else 0.05,
                observed=alive,
                subject_visibility=VisibilityState.VISIBLE,
                object_visibility=VisibilityState.VISIBLE,
                source_model="synthetic",
            ))

        frames.append(FrameGraph(
            video_id=video_id, frame_idx=f, timestamp=timestamp,
            track_observations=tracks, edge_observations=edges,
        ))

    return frames


def make_long_single_edge_video(
    video_id: str = "pilot_long_001",
    num_frames: int = 200,
    edge_end_frame: int | None = None,
    fps: float = 30.0,
) -> list[FrameGraph]:
    """Long single-edge video for corruption sweep testing.

    Edge alive from frame 0 to edge_end_frame (default=entire sequence).
    Both tracks always VISIBLE.  No built-in occlusion.
    """
    if edge_end_frame is None:
        edge_end_frame = num_frames - 1

    ek = EdgeKey(1, 2, 0)
    frames: list[FrameGraph] = []

    for f in range(num_frames):
        timestamp = f / fps
        alive = f <= edge_end_frame

        subject = TrackObservation(
            video_id=video_id, frame_idx=f, timestamp=timestamp,
            track_id=1, class_id=0,
            bbox_xyxy=(100.0, 100.0, 200.0, 300.0),
            detector_confidence=0.95,
            tracker_state=TrackerState.TRACKED,
            visibility_state=VisibilityState.VISIBLE,
        )
        obj = TrackObservation(
            video_id=video_id, frame_idx=f, timestamp=timestamp,
            track_id=2, class_id=1,
            bbox_xyxy=(150.0, 150.0, 220.0, 280.0),
            detector_confidence=0.90,
            tracker_state=TrackerState.TRACKED,
            visibility_state=VisibilityState.VISIBLE,
        )
        edge = EdgeObservation(
            edge_key=ek, video_id=video_id, frame_idx=f,
            timestamp=timestamp,
            predicate_score=0.85 if alive else 0.05,
            observed=alive,
            subject_visibility=VisibilityState.VISIBLE,
            object_visibility=VisibilityState.VISIBLE,
            source_model="synthetic",
        )

        frames.append(FrameGraph(
            video_id=video_id, frame_idx=f, timestamp=timestamp,
            track_observations=[subject, obj],
            edge_observations=[edge],
        ))

    return frames
