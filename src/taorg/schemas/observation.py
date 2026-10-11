"""Minimal, backward-compatible observation contract (research/taorg_observation_contract.md).

ADDITIONAL record types; nothing here changes ``EdgeObservation``, ``TrackObservation``,
any temporal method, or the meaning of ``VisibilityState.UNKNOWN``.  Legacy streams are
obtained with :func:`to_legacy_edge` / :func:`track_presence_legacy`, which reproduce the
current convention exactly (undetected => no record => UNKNOWN in every method).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation
from taorg.schemas.tracks import TrackerState, VisibilityState

Box = tuple[float, float, float, float]


class MissingReason(Enum):
    """Why an expected observation is absent (``UNSPECIFIED`` = unknown / legacy)."""

    UNSPECIFIED = "unspecified"
    NOT_EVALUATED = "not_evaluated"
    EVALUATED_NEGATIVE = "evaluated_negative"
    TRACK_TENTATIVE = "track_tentative"
    TRACK_LOST = "track_lost"
    TRACK_TERMINATED = "track_terminated"
    OCCLUSION_EVIDENCE = "occlusion_evidence"
    OUT_OF_VIEW_EVIDENCE = "out_of_view_evidence"
    IDENTITY_SUSPECT = "identity_suspect"


class EvidenceSource(Enum):
    """Provenance of an inferred field."""

    NONE = "none"
    TRACKER = "tracker"          # directly reported by the tracker / detector
    HEURISTIC = "heuristic"      # derived at inference time from available outputs
    ORACLE = "oracle"            # ground truth; evaluation / labelled diagnostics only


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """One track at one frame under the clarified contract (see the contract document)."""

    track_id: int
    frame_idx: int
    timestamp: float
    detected: bool
    tracker_state: TrackerState
    detector_confidence: Optional[float] = None
    bbox: Optional[Box] = None
    predicted_bbox: Optional[Box] = None
    time_since_detection_s: Optional[float] = None
    association_confidence: Optional[float] = None
    visibility: VisibilityState = VisibilityState.UNKNOWN
    visibility_source: EvidenceSource = EvidenceSource.NONE
    missing_reason: MissingReason = MissingReason.UNSPECIFIED
    missing_reason_source: EvidenceSource = EvidenceSource.NONE

    def __post_init__(self) -> None:
        if self.detected and self.missing_reason is not MissingReason.UNSPECIFIED:
            raise ValueError("a detected track cannot carry a missing reason")
        if self.detected and self.detector_confidence is None:
            raise ValueError("a detected track needs its detector confidence")
        if not self.detected and self.detector_confidence is not None:
            raise ValueError("an undetected track has no detector confidence")
        if self.association_confidence is not None and not 0.0 <= self.association_confidence <= 1.0:
            raise ValueError("association_confidence must be in [0, 1] or None")
        if self.missing_reason is not MissingReason.UNSPECIFIED and self.missing_reason_source is EvidenceSource.NONE:
            raise ValueError("a missing reason needs its evidence source")


@dataclass(frozen=True, slots=True)
class RelationRecord:
    """One relationship triple at one frame under the clarified contract (future VidSGG adapters)."""

    edge_key: EdgeKey
    frame_idx: int
    timestamp: float
    evaluated: Optional[bool] = None
    score: Optional[float] = None
    positive_evidence: Optional[bool] = None
    missing_reason: MissingReason = MissingReason.UNSPECIFIED
    missing_reason_source: EvidenceSource = EvidenceSource.NONE

    def __post_init__(self) -> None:
        if self.evaluated is not True and self.score is not None:
            raise ValueError("a score exists only for evaluated triples (never a fabricated 0.0)")
        if self.evaluated is True and self.score is None:
            raise ValueError("an evaluated triple needs its score")


def to_legacy_edge(r: RelationRecord, video_id: str, source_model: str = "") -> Optional[EdgeObservation]:
    """Legacy conversion: a positive, evaluated triple -> observed record; anything else -> no record.

    This reproduces the existing convention for thresholded predictors (absent => no record =>
    every temporal method treats it as UNKNOWN).
    """
    if r.evaluated and r.positive_evidence is not False and r.score is not None:
        return EdgeObservation(edge_key=r.edge_key, video_id=video_id, frame_idx=r.frame_idx,
                               timestamp=r.timestamp, predicate_score=r.score, observed=True,
                               source_model=source_model)
    return None


def track_presence_legacy(r: TrackRecord, video_id: str) -> Optional[EdgeObservation]:
    """Track presence as a unary 'edge' (key (track, track, -1)) under the LEGACY contract.

    detected -> observed record with the detector confidence; undetected -> no record.
    """
    if not r.detected:
        return None
    return EdgeObservation(edge_key=EdgeKey(r.track_id, r.track_id, -1), video_id=video_id,
                           frame_idx=r.frame_idx, timestamp=r.timestamp,
                           predicate_score=float(r.detector_confidence), observed=True,
                           source_model="tracker")
