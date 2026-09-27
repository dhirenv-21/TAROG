"""Edge-level schemas for the canonical edge stream.

Defines EdgeKey, EdgeObservation, TemporalEdgeState, ExpiryEntry, and
FrameGraph — the core data structures that flow between adapters, temporal
methods, noise injectors, and metrics.

Corresponds to Architecture.md §4 and PRD FR-1, FR-4, FR-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from taorg.schemas.tracks import TrackObservation, VisibilityState


# ---------------------------------------------------------------------------
# EdgeKey — immutable identifier for a directed (subject, object, predicate)
# ---------------------------------------------------------------------------

class EdgeKey(NamedTuple):
    """Immutable key for a directed relationship edge.

    Using NamedTuple for hashability, immutability, and zero-overhead
    comparison — important for dict keys and heap entries.
    """

    subject_track_id: int
    object_track_id: int
    predicate_id: int


class PairKey(NamedTuple):
    """Immutable key for a subject–object pair (predicate-independent).

    Used when maintaining cross-predicate state for the same object pair.
    """

    subject_track_id: int
    object_track_id: int


# ---------------------------------------------------------------------------
# EdgeObservation — one relationship prediction at one frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EdgeObservation:
    """A single relationship observation from a base VidSGG model.

    This is the canonical unit of the edge stream consumed by every
    temporal method.  Adapters must produce these.

    Attributes:
        edge_key: The (subject, object, predicate) identifier.
        video_id: Source video identifier.
        frame_idx: Zero-based frame index.
        timestamp: Frame timestamp in seconds.
        predicate_score: Model-predicted score for this predicate in [0, 1].
        observed: Whether this edge was explicitly predicted (True) or is
            an absence record constructed for temporal processing (False).
        subject_visibility: Visibility state of the subject at this frame.
        object_visibility: Visibility state of the object at this frame.
        source_model: Name of the base model that produced this observation.
        source_checkpoint: Optional checkpoint/config identifier for provenance.
    """

    edge_key: EdgeKey
    video_id: str
    frame_idx: int
    timestamp: float
    predicate_score: float
    observed: bool = True
    subject_visibility: VisibilityState = VisibilityState.VISIBLE
    object_visibility: VisibilityState = VisibilityState.VISIBLE
    source_model: str = ""
    source_checkpoint: str = ""

    @property
    def pair_key(self) -> PairKey:
        """Return the predicate-independent pair key."""
        return PairKey(self.edge_key.subject_track_id, self.edge_key.object_track_id)

    @property
    def edge_visibility(self) -> VisibilityState:
        """Derive combined visibility for the edge.

        Conservative rule: the edge is VISIBLE only if both subject and
        object are VISIBLE.  If either is OCCLUDED_OR_UNRELIABLE, the edge
        is OCCLUDED_OR_UNRELIABLE.  If either is UNKNOWN, the edge is UNKNOWN
        (unless the other is OCCLUDED_OR_UNRELIABLE, which dominates).

        This is a default policy; temporal methods may override.
        """
        if (
            self.subject_visibility == VisibilityState.VISIBLE
            and self.object_visibility == VisibilityState.VISIBLE
        ):
            return VisibilityState.VISIBLE
        if (
            self.subject_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE
            or self.object_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE
        ):
            return VisibilityState.OCCLUDED_OR_UNRELIABLE
        return VisibilityState.UNKNOWN


# ---------------------------------------------------------------------------
# TemporalEdgeState — mutable internal state for a temporal method
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TemporalEdgeState:
    """Internal temporal state for one edge maintained by a temporal method.

    This is the mutable state object that temporal methods update over time.
    It is NOT frozen because temporal methods mutate it in-place for
    efficiency.

    Attributes:
        edge_key: The (subject, object, predicate) identifier.
        z: Pre-activation / latent state scalar (or vector for cross-predicate).
        last_update_time: Timestamp of last state materialization.
        last_update_frame: Frame index of last state materialization.
        visibility_regime: Current visibility regime for decay selection.
        alive_score: Current edge-alive score/probability in [0, 1].
        scheduled_expiry: Predicted expiry timestamp, or None if unknown.
        version: Generation counter for heap invalidation.
    """

    edge_key: EdgeKey
    z: float = 0.0
    last_update_time: float = 0.0
    last_update_frame: int = 0
    visibility_regime: VisibilityState = VisibilityState.VISIBLE
    alive_score: float = 0.0
    scheduled_expiry: Optional[float] = None
    version: int = 0


# ---------------------------------------------------------------------------
# ExpiryEntry — min-heap element for lazy expiry scheduling
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, order=True)
class ExpiryEntry:
    """Min-heap entry for scheduled edge expiry.

    Ordered by expiry_time so heapq operations produce earliest-first.
    When popped, the version must be compared against the current
    TemporalEdgeState.version — stale entries are discarded.

    Attributes:
        expiry_time: Scheduled expiry timestamp.
        edge_key: The edge this entry refers to.
        version: Must match TemporalEdgeState.version to be valid.
    """

    expiry_time: float
    edge_key: EdgeKey = field(compare=False)
    version: int = field(compare=False)


# ---------------------------------------------------------------------------
# FrameGraph — all observations and tracks for one frame
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FrameGraph:
    """All observations for a single video frame.

    This is the per-frame unit consumed and produced by temporal methods.

    Attributes:
        video_id: Source video identifier.
        frame_idx: Zero-based frame index.
        timestamp: Frame timestamp in seconds.
        track_observations: All object tracks observed at this frame.
        edge_observations: All relationship edges observed/absent at this frame.
    """

    video_id: str
    frame_idx: int
    timestamp: float
    track_observations: list[TrackObservation] = field(default_factory=list)
    edge_observations: list[EdgeObservation] = field(default_factory=list)
