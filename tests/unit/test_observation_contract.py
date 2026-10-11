"""Observation contract: legacy path unchanged, defaults explicit, invalid states rejected."""

from __future__ import annotations

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation
from taorg.schemas.observation import (
    EvidenceSource, MissingReason, RelationRecord, TrackRecord, to_legacy_edge, track_presence_legacy,
)
from taorg.schemas.tracks import TrackerState, VisibilityState


def test_unknown_meaning_and_legacy_schema_unchanged():
    assert VisibilityState.UNKNOWN.value == "unknown"
    o = EdgeObservation(EdgeKey(1, 2, 0), "v", 0, 0.0, 0.0, False)
    assert o.edge_visibility is VisibilityState.VISIBLE          # legacy absence default unchanged
    assert [f for f in EdgeObservation.__dataclass_fields__] == [
        "edge_key", "video_id", "frame_idx", "timestamp", "predicate_score", "observed",
        "subject_visibility", "object_visibility", "source_model", "source_checkpoint"]


def test_defaults_are_unspecified_and_no_invented_reliability():
    r = TrackRecord(track_id=3, frame_idx=5, timestamp=0.2, detected=False, tracker_state=TrackerState.LOST)
    assert r.missing_reason is MissingReason.UNSPECIFIED and r.missing_reason_source is EvidenceSource.NONE
    assert r.association_confidence is None and r.visibility is VisibilityState.UNKNOWN


def test_invalid_combinations_rejected():
    with pytest.raises(ValueError):
        TrackRecord(1, 0, 0.0, True, TrackerState.TRACKED)                       # detected without confidence
    with pytest.raises(ValueError):
        TrackRecord(1, 0, 0.0, False, TrackerState.LOST, detector_confidence=0.5)
    with pytest.raises(ValueError):
        TrackRecord(1, 0, 0.0, True, TrackerState.TRACKED, detector_confidence=0.9,
                    missing_reason=MissingReason.TRACK_LOST, missing_reason_source=EvidenceSource.TRACKER)
    with pytest.raises(ValueError):
        TrackRecord(1, 0, 0.0, False, TrackerState.LOST, missing_reason=MissingReason.IDENTITY_SUSPECT)
    with pytest.raises(ValueError):
        RelationRecord(EdgeKey(1, 2, 0), 0, 0.0, evaluated=False, score=0.0)     # no fabricated 0.0
    with pytest.raises(ValueError):
        RelationRecord(EdgeKey(1, 2, 0), 0, 0.0, evaluated=True)


def test_legacy_conversion_reproduces_existing_convention():
    pos = RelationRecord(EdgeKey(1, 2, 0), 3, 0.1, evaluated=True, score=0.8, positive_evidence=True)
    neg = RelationRecord(EdgeKey(1, 2, 0), 3, 0.1, evaluated=True, score=0.1, positive_evidence=False,
                         missing_reason=MissingReason.EVALUATED_NEGATIVE, missing_reason_source=EvidenceSource.TRACKER)
    unk = RelationRecord(EdgeKey(1, 2, 0), 3, 0.1)
    e = to_legacy_edge(pos, "v")
    assert e.observed and e.predicate_score == 0.8 and e.edge_visibility is VisibilityState.VISIBLE
    assert to_legacy_edge(neg, "v") is None and to_legacy_edge(unk, "v") is None
    d = TrackRecord(4, 1, 0.03, True, TrackerState.TRACKED, detector_confidence=0.7)
    u = TrackRecord(4, 2, 0.07, False, TrackerState.LOST, time_since_detection_s=0.03,
                    missing_reason=MissingReason.OCCLUSION_EVIDENCE, missing_reason_source=EvidenceSource.HEURISTIC)
    assert track_presence_legacy(d, "v").predicate_score == 0.7
    assert track_presence_legacy(u, "v") is None                     # legacy: reason discarded, no record
