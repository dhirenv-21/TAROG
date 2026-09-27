"""Comprehensive tests for Phase 2 metrics.

All expected values are hand-computable.  Tests cover:
    - Perfect persistence through occlusion
    - Premature deletion during occlusion
    - False persistence after true termination
    - Repeated oscillation causing churn
    - Relationship never predicted
    - Relationship always predicted
    - Immediate correct termination
    - Never-dropped edge
    - Sequence boundary cases
    - Aggregation over multiple edges
"""

from __future__ import annotations

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.termination import (
    compute_termination_delay,
    compute_false_persistence,
)
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

EK = EdgeKey(1, 2, 0)  # default test edge
EK2 = EdgeKey(3, 4, 0)  # second test edge


def _frame(
    frame_idx: int,
    edges: dict[EdgeKey, bool] | None = None,
    video_id: str = "test",
) -> FrameGraph:
    """Build a FrameGraph with specified edge presence.

    Args:
        frame_idx: Frame index.
        edges: Dict mapping EdgeKey -> observed (True/False).
            If None, frame has no edges.
    """
    fps = 30.0
    timestamp = frame_idx / fps
    edge_obs = []
    if edges:
        for ek, observed in edges.items():
            edge_obs.append(EdgeObservation(
                edge_key=ek,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                predicate_score=0.9 if observed else 0.0,
                observed=observed,
                source_model="test",
            ))
    return FrameGraph(
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        track_observations=[],
        edge_observations=edge_obs,
    )


def _pred_frame(frame_idx: int, edges_present: set[EdgeKey] | None = None, video_id: str = "test") -> FrameGraph:
    """Build a predicted FrameGraph — edges in edges_present are output."""
    fps = 30.0
    timestamp = frame_idx / fps
    edge_obs = []
    if edges_present:
        for ek in edges_present:
            edge_obs.append(EdgeObservation(
                edge_key=ek,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                predicate_score=0.8,
                observed=True,
                source_model="test",
            ))
    return FrameGraph(
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp=timestamp,
        track_observations=[],
        edge_observations=edge_obs,
    )


def _manifest(entries: list[CorruptionEntry]) -> CorruptionManifest:
    return CorruptionManifest(seed=42, entries=entries)


# ---------------------------------------------------------------
# Edge Persistence F1
# ---------------------------------------------------------------

class TestPersistenceF1:
    """Tests for Edge Persistence precision/recall/F1."""

    def test_perfect_persistence(self):
        """Method persists edge through entire corruption gap → F1=1.0."""
        # GT: edge alive frames 0-9
        gt = [_frame(i, {EK: True}) for i in range(10)]
        # Corruption: frames 4-6
        manifest = _manifest([CorruptionEntry(
            corruption_type="occlusion", video_id="test",
            edge_key_tuple=(1, 2, 0),
            start_frame=4, end_frame=6, duration=3,
        )])
        # Prediction: edge present all frames (persisted through gap)
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_persistence_f1(gt, pred, manifest)
        assert result.tp == 3  # frames 4,5,6
        assert result.fn == 0
        assert result.f1 == pytest.approx(1.0)

    def test_premature_deletion(self):
        """Method drops edge during corruption → low recall."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        manifest = _manifest([CorruptionEntry(
            corruption_type="occlusion", video_id="test",
            edge_key_tuple=(1, 2, 0),
            start_frame=4, end_frame=6, duration=3,
        )])
        # Prediction: edge dropped during corruption (frames 4-6 missing)
        pred = [
            _pred_frame(i, {EK} if i < 4 or i > 6 else set())
            for i in range(10)
        ]

        result = compute_persistence_f1(gt, pred, manifest)
        assert result.tp == 0
        assert result.fn == 3
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_partial_persistence(self):
        """Method persists for part of the gap."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        manifest = _manifest([CorruptionEntry(
            corruption_type="occlusion", video_id="test",
            edge_key_tuple=(1, 2, 0),
            start_frame=3, end_frame=6, duration=4,
        )])
        # Prediction: persists frames 3,4 but drops 5,6
        pred = [
            _pred_frame(i, {EK} if i <= 4 else set())
            for i in range(10)
        ]

        result = compute_persistence_f1(gt, pred, manifest)
        assert result.tp == 2  # frames 3,4
        assert result.fn == 2  # frames 5,6
        assert result.precision == pytest.approx(1.0)  # no FP
        assert result.recall == pytest.approx(0.5)

    def test_no_corruption_intervals(self):
        """No corruption → F1=0.0 (undefined)."""
        gt = [_frame(i, {EK: True}) for i in range(5)]
        pred = [_pred_frame(i, {EK}) for i in range(5)]
        manifest = _manifest([])

        result = compute_persistence_f1(gt, pred, manifest)
        assert result.f1 == 0.0

    def test_frame_count_mismatch(self):
        """Mismatched frame counts raise ValueError."""
        gt = [_frame(i) for i in range(5)]
        pred = [_pred_frame(i) for i in range(3)]
        manifest = _manifest([])

        with pytest.raises(ValueError, match="Frame count mismatch"):
            compute_persistence_f1(gt, pred, manifest)


# ---------------------------------------------------------------
# Relationship Churn Rate
# ---------------------------------------------------------------

class TestChurnRate:
    """Tests for Relationship Churn Rate metric."""

    def test_constant_prediction_zero_churn(self):
        """Edge always predicted → churn = 0."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_churn_rate(pred, gt)
        assert result.per_edge_churn[EK] == 0.0
        assert result.total_flips == 0

    def test_alternating_max_churn(self):
        """Edge alternates every frame → churn = 1.0."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        pred = [
            _pred_frame(i, {EK} if i % 2 == 0 else set())
            for i in range(10)
        ]

        result = compute_churn_rate(pred, gt)
        assert result.per_edge_churn[EK] == pytest.approx(1.0)

    def test_single_drop_and_return(self):
        """Edge present, drops for 1 frame, returns → 2 flips."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        # Present except frame 5
        pred = [
            _pred_frame(i, {EK} if i != 5 else set())
            for i in range(10)
        ]

        result = compute_churn_rate(pred, gt)
        # 2 flips: alive→dead at frame 5, dead→alive at frame 6
        assert result.total_flips == 2
        # churn = 2/9 (9 consecutive pairs in [0..9])
        assert result.per_edge_churn[EK] == pytest.approx(2 / 9)

    def test_never_predicted_zero_churn(self):
        """Edge never predicted → churn = 0."""
        gt = [_frame(i, {EK: True}) for i in range(5)]
        pred = [_pred_frame(i, set()) for i in range(5)]

        result = compute_churn_rate(pred, gt)
        assert result.per_edge_churn[EK] == 0.0

    def test_single_frame_edge(self):
        """Single-frame edge → churn = 0 (no consecutive pairs)."""
        gt = [_frame(0, {EK: True})]
        pred = [_pred_frame(0, {EK})]

        result = compute_churn_rate(pred, gt)
        assert result.per_edge_churn[EK] == 0.0


# ---------------------------------------------------------------
# Termination Delay
# ---------------------------------------------------------------

class TestTerminationDelay:
    """Tests for Termination Delay metric."""

    def test_immediate_correct_drop(self):
        """Method drops edge at t_end + 1 → delay = 1."""
        # GT: edge alive frames 0-4, dead frames 5-9
        gt = [
            _frame(i, {EK: True} if i <= 4 else {EK: False})
            for i in range(10)
        ]
        # Prediction: edge present 0-4, absent 5-9
        pred = [
            _pred_frame(i, {EK} if i <= 4 else set())
            for i in range(10)
        ]

        result = compute_termination_delay(gt, pred)
        assert result.num_evaluated == 1
        assert result.per_edge_delay[EK] == 1  # drops at frame 5, t_end=4
        assert result.per_edge_censored[EK] is False

    def test_never_drops_censored(self):
        """Method never drops edge → censored with delay = seq_len - t_end."""
        gt = [
            _frame(i, {EK: True} if i <= 4 else {EK: False})
            for i in range(10)
        ]
        # Prediction: edge present ALL frames
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_termination_delay(gt, pred)
        assert result.num_censored == 1
        assert result.per_edge_censored[EK] is True
        assert result.per_edge_delay[EK] == 10 - 4  # = 6

    def test_delayed_drop(self):
        """Method drops edge 3 frames after GT termination."""
        gt = [
            _frame(i, {EK: True} if i <= 3 else {EK: False})
            for i in range(10)
        ]
        # Prediction: present 0-6, absent 7-9
        pred = [
            _pred_frame(i, {EK} if i <= 6 else set())
            for i in range(10)
        ]

        result = compute_termination_delay(gt, pred)
        assert result.per_edge_delay[EK] == 7 - 3  # delay = 4
        assert result.per_edge_censored[EK] is False

    def test_no_gt_termination(self):
        """Edge alive for entire sequence → excluded from metric."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_termination_delay(gt, pred)
        assert result.num_evaluated == 0

    def test_frame_count_mismatch(self):
        gt = [_frame(i) for i in range(5)]
        pred = [_pred_frame(i) for i in range(3)]
        with pytest.raises(ValueError, match="Frame count mismatch"):
            compute_termination_delay(gt, pred)


# ---------------------------------------------------------------
# False Persistence Rate
# ---------------------------------------------------------------

class TestFalsePersistence:
    """Tests for False Persistence Rate metric."""

    def test_perfect_no_false_persistence(self):
        """Method correctly drops at termination → rate = 0.0."""
        gt = [
            _frame(i, {EK: True} if i <= 4 else {EK: False})
            for i in range(10)
        ]
        pred = [
            _pred_frame(i, {EK} if i <= 4 else set())
            for i in range(10)
        ]

        result = compute_false_persistence(gt, pred)
        assert result.per_edge_rate[EK] == pytest.approx(0.0)

    def test_complete_false_persistence(self):
        """Method never drops → rate = 1.0."""
        gt = [
            _frame(i, {EK: True} if i <= 4 else {EK: False})
            for i in range(10)
        ]
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_false_persistence(gt, pred)
        # 5 post-termination frames (5,6,7,8,9), all falsely persisted
        assert result.per_edge_rate[EK] == pytest.approx(1.0)

    def test_partial_false_persistence(self):
        """Method drops partway through post-termination."""
        gt = [
            _frame(i, {EK: True} if i <= 4 else {EK: False})
            for i in range(10)
        ]
        # Present 0-6, absent 7-9 (3/5 post-term frames falsely persisted)
        pred = [
            _pred_frame(i, {EK} if i <= 6 else set())
            for i in range(10)
        ]

        result = compute_false_persistence(gt, pred)
        # Post-termination: frames 5-9 (5 frames), persisted at 5,6 (2 frames)
        assert result.per_edge_rate[EK] == pytest.approx(2 / 5)

    def test_no_termination_excluded(self):
        """Edge alive entire sequence → excluded."""
        gt = [_frame(i, {EK: True}) for i in range(10)]
        pred = [_pred_frame(i, {EK}) for i in range(10)]

        result = compute_false_persistence(gt, pred)
        assert result.num_evaluated == 0

    def test_multiple_edges(self):
        """Aggregation across multiple edges."""
        # EK terminates at frame 3, EK2 terminates at frame 5
        gt = []
        for i in range(10):
            edges: dict[EdgeKey, bool] = {}
            edges[EK] = i <= 3
            edges[EK2] = i <= 5
            gt.append(_frame(i, edges))

        # EK never drops (false persistence = 1.0 for 6 post-term frames)
        # EK2 drops immediately (false persistence = 0.0)
        pred = []
        for i in range(10):
            present: set[EdgeKey] = {EK}  # EK always present
            if i <= 5:
                present.add(EK2)  # EK2 present until frame 5
            pred.append(_pred_frame(i, present))

        result = compute_false_persistence(gt, pred)
        assert result.per_edge_rate[EK] == pytest.approx(1.0)
        assert result.per_edge_rate[EK2] == pytest.approx(0.0)
        assert result.mean_rate == pytest.approx(0.5)
