"""Track-level observation schema.

Represents a single object detection/track observation at one frame.
Corresponds to Architecture.md §4 TrackObservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrackerState(Enum):
    """Tracker-reported state for a tracked object.

    These map to visibility categories used downstream by temporal methods.
    The exact mapping from tracker internals to these states is configurable
    (see visibility policy).
    """

    TRACKED = "tracked"           # actively matched detection
    LOST = "lost"                 # tracker has no detection, coasting
    TENTATIVE = "tentative"       # new track, not yet confirmed
    REMOVED = "removed"           # track terminated by tracker


class VisibilityState(Enum):
    """Visibility state used by temporal methods (FR-2 in PRD.md).

    The mapping from TrackerState + detector confidence to VisibilityState
    is a configurable policy decision.  UNKNOWN must not silently become
    VISIBLE.
    """

    VISIBLE = "visible"
    OCCLUDED_OR_UNRELIABLE = "occluded_or_unreliable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """A single object observation at one frame.

    Attributes:
        video_id: Identifier for the source video.
        frame_idx: Zero-based frame index within the video.
        timestamp: Frame timestamp in seconds (frame_idx / fps if unavailable).
        track_id: Unique track identifier assigned by the tracker or GT.
        class_id: Object class index.
        bbox_xyxy: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        detector_confidence: Detection confidence score in [0, 1].
        tracker_state: Raw tracker state.
        visibility_state: Derived visibility state (may be set by visibility policy).
        occlusion_ratio: Optional estimated occlusion ratio in [0, 1].
        identity_confidence: Optional tracker score in [0, 1] that this track label
            still refers to the same physical object as in the previous frame
            (association confidence).  None = not supplied; consumers that need it
            treat None as 1.0.  Used only by identity-aware methods (vhped_ia).
    """

    video_id: str
    frame_idx: int
    timestamp: float
    track_id: int
    class_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float = 1.0
    tracker_state: TrackerState = TrackerState.TRACKED
    visibility_state: VisibilityState = VisibilityState.VISIBLE
    occlusion_ratio: Optional[float] = None
    identity_confidence: Optional[float] = None
