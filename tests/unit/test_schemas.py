"""Unit tests for canonical schemas.

Verifies that all domain objects can be instantiated, are hashable
where required, and have correct properties.
"""

from taorg.schemas.edges import (
    EdgeKey,
    EdgeObservation,
    ExpiryEntry,
    FrameGraph,
    PairKey,
    TemporalEdgeState,
)
from taorg.schemas.tracks import (
    TrackObservation,
    TrackerState,
    VisibilityState,
)


class TestTrackObservation:
    """Tests for TrackObservation schema."""

    def test_create_basic(self):
        obs = TrackObservation(
            video_id="v1",
            frame_idx=0,
            timestamp=0.0,
            track_id=1,
            class_id=0,
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        )
        assert obs.video_id == "v1"
        assert obs.frame_idx == 0
        assert obs.track_id == 1
        assert obs.visibility_state == VisibilityState.VISIBLE  # default

    def test_frozen(self):
        obs = TrackObservation(
            video_id="v1", frame_idx=0, timestamp=0.0,
            track_id=1, class_id=0, bbox_xyxy=(0, 0, 1, 1),
        )
        try:
            obs.track_id = 999  # type: ignore
            assert False, "Should not allow mutation"
        except AttributeError:
            pass  # expected for frozen dataclass

    def test_visibility_states(self):
        assert VisibilityState.VISIBLE.value == "visible"
        assert VisibilityState.OCCLUDED_OR_UNRELIABLE.value == "occluded_or_unreliable"
        assert VisibilityState.UNKNOWN.value == "unknown"

    def test_tracker_states(self):
        assert TrackerState.TRACKED.value == "tracked"
        assert TrackerState.LOST.value == "lost"


class TestEdgeKey:
    """Tests for EdgeKey and PairKey."""

    def test_hashable(self):
        k1 = EdgeKey(subject_track_id=1, object_track_id=2, predicate_id=0)
        k2 = EdgeKey(subject_track_id=1, object_track_id=2, predicate_id=0)
        assert k1 == k2
        assert hash(k1) == hash(k2)
        # Usable as dict key
        d = {k1: "test"}
        assert d[k2] == "test"

    def test_different_predicates(self):
        k1 = EdgeKey(1, 2, 0)
        k2 = EdgeKey(1, 2, 1)
        assert k1 != k2

    def test_pair_key(self):
        pk = PairKey(subject_track_id=1, object_track_id=2)
        assert pk.subject_track_id == 1
        assert pk.object_track_id == 2


class TestEdgeObservation:
    """Tests for EdgeObservation."""

    def test_create(self):
        ek = EdgeKey(1, 2, 0)
        obs = EdgeObservation(
            edge_key=ek,
            video_id="v1",
            frame_idx=5,
            timestamp=5.0,
            predicate_score=0.9,
        )
        assert obs.observed is True  # default
        assert obs.predicate_score == 0.9

    def test_pair_key_property(self):
        ek = EdgeKey(1, 2, 0)
        obs = EdgeObservation(
            edge_key=ek, video_id="v1", frame_idx=0,
            timestamp=0.0, predicate_score=0.5,
        )
        assert obs.pair_key == PairKey(1, 2)

    def test_edge_visibility_both_visible(self):
        obs = EdgeObservation(
            edge_key=EdgeKey(1, 2, 0), video_id="v1",
            frame_idx=0, timestamp=0.0, predicate_score=0.5,
            subject_visibility=VisibilityState.VISIBLE,
            object_visibility=VisibilityState.VISIBLE,
        )
        assert obs.edge_visibility == VisibilityState.VISIBLE

    def test_edge_visibility_one_occluded(self):
        obs = EdgeObservation(
            edge_key=EdgeKey(1, 2, 0), video_id="v1",
            frame_idx=0, timestamp=0.0, predicate_score=0.5,
            subject_visibility=VisibilityState.VISIBLE,
            object_visibility=VisibilityState.OCCLUDED_OR_UNRELIABLE,
        )
        assert obs.edge_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE

    def test_edge_visibility_unknown(self):
        obs = EdgeObservation(
            edge_key=EdgeKey(1, 2, 0), video_id="v1",
            frame_idx=0, timestamp=0.0, predicate_score=0.5,
            subject_visibility=VisibilityState.UNKNOWN,
            object_visibility=VisibilityState.VISIBLE,
        )
        assert obs.edge_visibility == VisibilityState.UNKNOWN

    def test_frozen(self):
        obs = EdgeObservation(
            edge_key=EdgeKey(1, 2, 0), video_id="v1",
            frame_idx=0, timestamp=0.0, predicate_score=0.5,
        )
        try:
            obs.predicate_score = 0.1  # type: ignore
            assert False, "Should not allow mutation"
        except AttributeError:
            pass


class TestTemporalEdgeState:
    """Tests for TemporalEdgeState."""

    def test_mutable(self):
        state = TemporalEdgeState(edge_key=EdgeKey(1, 2, 0))
        state.z = 1.5
        state.version = 3
        assert state.z == 1.5
        assert state.version == 3

    def test_defaults(self):
        state = TemporalEdgeState(edge_key=EdgeKey(1, 2, 0))
        assert state.z == 0.0
        assert state.alive_score == 0.0
        assert state.version == 0
        assert state.scheduled_expiry is None


class TestExpiryEntry:
    """Tests for ExpiryEntry."""

    def test_ordering(self):
        e1 = ExpiryEntry(expiry_time=1.0, edge_key=EdgeKey(1, 2, 0), version=0)
        e2 = ExpiryEntry(expiry_time=2.0, edge_key=EdgeKey(3, 4, 1), version=0)
        assert e1 < e2

    def test_heap_compatible(self):
        import heapq
        entries = [
            ExpiryEntry(3.0, EdgeKey(1, 2, 0), 0),
            ExpiryEntry(1.0, EdgeKey(3, 4, 0), 0),
            ExpiryEntry(2.0, EdgeKey(5, 6, 0), 0),
        ]
        heapq.heapify(entries)
        assert heapq.heappop(entries).expiry_time == 1.0
        assert heapq.heappop(entries).expiry_time == 2.0
        assert heapq.heappop(entries).expiry_time == 3.0


class TestFrameGraph:
    """Tests for FrameGraph."""

    def test_empty_frame(self):
        fg = FrameGraph(video_id="v1", frame_idx=0, timestamp=0.0)
        assert fg.track_observations == []
        assert fg.edge_observations == []

    def test_with_observations(self):
        track = TrackObservation(
            video_id="v1", frame_idx=0, timestamp=0.0,
            track_id=1, class_id=0, bbox_xyxy=(0, 0, 1, 1),
        )
        edge = EdgeObservation(
            edge_key=EdgeKey(1, 2, 0), video_id="v1",
            frame_idx=0, timestamp=0.0, predicate_score=0.9,
        )
        fg = FrameGraph(
            video_id="v1", frame_idx=0, timestamp=0.0,
            track_observations=[track],
            edge_observations=[edge],
        )
        assert len(fg.track_observations) == 1
        assert len(fg.edge_observations) == 1
