"""Comprehensive tests for all Phase 1 temporal baselines.

Tests cover:
    - Protocol compliance (TemporalEdgeModel)
    - Edge creation / observation
    - Repeated observation
    - Missing observation / decay
    - Visible absence vs occluded absence (gated EMA)
    - Deterministic behavior
    - Reset behavior
    - Threshold behavior
    - All methods consume identical canonical input

Uses synthetic fixtures from tests/fixtures/.
"""

from __future__ import annotations

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, TrackerState, VisibilityState
from taorg.temporal.base import TemporalEdgeModel
from taorg.temporal.no_memory import NoMemory
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.ema import EMA
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.hmm import HMM
from tests.fixtures.synthetic_fixtures import (
    make_simple_video_frames,
    make_termination_video_frames,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_frame(
    frame_idx: int,
    observed: bool = True,
    subject_vis: VisibilityState = VisibilityState.VISIBLE,
    object_vis: VisibilityState = VisibilityState.VISIBLE,
    predicate_score: float = 0.85,
    video_id: str = "test_vid",
    fps: float = 30.0,
) -> FrameGraph:
    """Build a single-edge FrameGraph for testing."""
    timestamp = frame_idx / fps
    edge_key = EdgeKey(subject_track_id=1, object_track_id=2, predicate_id=0)

    subject = TrackObservation(
        video_id=video_id, frame_idx=frame_idx, timestamp=timestamp,
        track_id=1, class_id=0, bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        visibility_state=subject_vis,
    )
    obj = TrackObservation(
        video_id=video_id, frame_idx=frame_idx, timestamp=timestamp,
        track_id=2, class_id=1, bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        visibility_state=object_vis,
    )
    edge = EdgeObservation(
        edge_key=edge_key, video_id=video_id, frame_idx=frame_idx,
        timestamp=timestamp, predicate_score=predicate_score,
        observed=observed,
        subject_visibility=subject_vis,
        object_visibility=object_vis,
        source_model="test",
    )
    return FrameGraph(
        video_id=video_id, frame_idx=frame_idx, timestamp=timestamp,
        track_observations=[subject, obj],
        edge_observations=[edge],
    )


ALL_METHODS = [
    NoMemory,
    FixedWindow,
    EMA,
    GatedEMA,
    HMM,
]


# ---------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------

class TestProtocolCompliance:
    """Every baseline must satisfy the TemporalEdgeModel protocol."""

    @pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
    def test_implements_protocol(self, cls):
        model = cls()
        assert isinstance(model, TemporalEdgeModel)

    @pytest.mark.parametrize("cls", ALL_METHODS, ids=lambda c: c.__name__)
    def test_lifecycle(self, cls):
        """Basic lifecycle: reset -> step -> finalize -> state_dict."""
        model = cls()
        model.reset_video("test")
        frame = _make_frame(0)
        out = model.step(frame)
        assert isinstance(out, FrameGraph)
        model.finalize_video()
        state = model.state_dict()
        assert isinstance(state, dict)


# ---------------------------------------------------------------
# Canonical input consumption — all methods same input
# ---------------------------------------------------------------

class TestCanonicalInput:
    """All methods must consume identical FrameGraph sequences."""

    def test_all_methods_consume_same_fixtures(self):
        frames = make_simple_video_frames()
        for cls in ALL_METHODS:
            model = cls()
            model.reset_video("synthetic_001")
            for f in frames:
                out = model.step(f)
                assert isinstance(out, FrameGraph)
                assert out.video_id == f.video_id
                assert out.frame_idx == f.frame_idx
            model.finalize_video()

    def test_all_methods_consume_termination_fixtures(self):
        frames = make_termination_video_frames()
        for cls in ALL_METHODS:
            model = cls()
            model.reset_video("synthetic_term_001")
            for f in frames:
                out = model.step(f)
                assert isinstance(out, FrameGraph)
            model.finalize_video()


# ---------------------------------------------------------------
# NoMemory tests
# ---------------------------------------------------------------

class TestNoMemory:
    def test_passes_through_observed(self):
        model = NoMemory()
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True))
        assert len(out.edge_observations) == 1

    def test_drops_absent(self):
        model = NoMemory()
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=False))
        assert len(out.edge_observations) == 0

    def test_threshold_filters(self):
        model = NoMemory(score_threshold=0.9)
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True, predicate_score=0.5))
        assert len(out.edge_observations) == 0

    def test_reset_is_idempotent(self):
        model = NoMemory()
        model.reset_video("v1")
        model.step(_make_frame(0))
        model.reset_video("v2")
        out = model.step(_make_frame(0, observed=False))
        assert len(out.edge_observations) == 0


# ---------------------------------------------------------------
# FixedWindow tests
# ---------------------------------------------------------------

class TestFixedWindow:
    def test_persists_within_window(self):
        model = FixedWindow(window_size=3)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True))
        model.step(_make_frame(1, observed=False))
        out = model.step(_make_frame(2, observed=False))
        # Edge observed at frame 0, window covers [0,1,2], should persist
        assert len(out.edge_observations) == 1
        assert out.edge_observations[0].observed is False

    def test_drops_after_window(self):
        model = FixedWindow(window_size=2)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True))
        model.step(_make_frame(1, observed=False))
        model.step(_make_frame(2, observed=False))
        out = model.step(_make_frame(3, observed=False))
        # Window is [2,3], last observation at 0 — should be evicted
        assert len(out.edge_observations) == 0

    def test_reset_clears_history(self):
        model = FixedWindow(window_size=5)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True))
        model.reset_video("v2")
        out = model.step(_make_frame(0, observed=False))
        assert len(out.edge_observations) == 0

    def test_threshold_filters(self):
        model = FixedWindow(window_size=5, score_threshold=0.9)
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True, predicate_score=0.5))
        assert len(out.edge_observations) == 0


# ---------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------

class TestEMA:
    def test_observation_increases_score(self):
        model = EMA(alpha=0.5, decay=0.5, score_threshold=0.0)
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True, predicate_score=0.8))
        assert len(out.edge_observations) == 1
        score = out.edge_observations[0].predicate_score
        assert score == pytest.approx(0.4)  # 0.5 * 0.8 + 0.5 * 0.0

    def test_decay_on_absence(self):
        model = EMA(alpha=0.5, decay=0.5, score_threshold=0.0)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True, predicate_score=1.0))
        out = model.step(_make_frame(1, observed=False))
        # Score after obs: 0.5. After decay: 0.5 * 0.5 = 0.25
        score = out.edge_observations[0].predicate_score
        assert score == pytest.approx(0.25)

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        for _ in range(3):
            model = EMA(alpha=0.3, decay=0.9)
            model.reset_video("v1")
            frames = make_simple_video_frames()
            outputs = [model.step(f) for f in frames]
            scores = [
                [e.predicate_score for e in o.edge_observations]
                for o in outputs
            ]
        # Run again and compare
        model2 = EMA(alpha=0.3, decay=0.9)
        model2.reset_video("v1")
        outputs2 = [model2.step(f) for f in make_simple_video_frames()]
        scores2 = [
            [e.predicate_score for e in o.edge_observations]
            for o in outputs2
        ]
        assert scores == scores2

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            EMA(alpha=0.0)

    def test_invalid_decay_rejected(self):
        with pytest.raises(ValueError, match="decay"):
            EMA(decay=1.0)

    def test_reset_clears_state(self):
        model = EMA(alpha=0.5, decay=0.9, score_threshold=0.0)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True, predicate_score=1.0))
        model.reset_video("v2")
        out = model.step(_make_frame(0, observed=False))
        # After reset, no edge should exist
        assert len(out.edge_observations) == 0

    def test_threshold(self):
        model = EMA(alpha=0.5, decay=0.5, score_threshold=0.5)
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True, predicate_score=0.6))
        # score = 0.5 * 0.6 = 0.3, below 0.5 threshold
        assert len(out.edge_observations) == 0


# ---------------------------------------------------------------
# GatedEMA tests — the CRITICAL baseline
# ---------------------------------------------------------------

class TestGatedEMA:
    def test_visible_absence_decays_faster_than_occluded(self):
        """The central research-relevant test.

        With the same initial score, visible absence should produce a
        LOWER score than occluded absence after the same number of
        absent frames.
        """
        # Visible absence
        model_vis = GatedEMA(
            alpha=0.5, decay_visible=0.5, decay_occluded=0.95,
            score_threshold=0.0,
        )
        model_vis.reset_video("v1")
        model_vis.step(_make_frame(0, observed=True, predicate_score=1.0))
        # 3 frames of VISIBLE absence
        for i in range(1, 4):
            out_vis = model_vis.step(_make_frame(
                i, observed=False,
                subject_vis=VisibilityState.VISIBLE,
                object_vis=VisibilityState.VISIBLE,
            ))
        score_vis = out_vis.edge_observations[0].predicate_score

        # Occluded absence
        model_occ = GatedEMA(
            alpha=0.5, decay_visible=0.5, decay_occluded=0.95,
            score_threshold=0.0,
        )
        model_occ.reset_video("v1")
        model_occ.step(_make_frame(0, observed=True, predicate_score=1.0))
        # 3 frames of OCCLUDED absence
        for i in range(1, 4):
            out_occ = model_occ.step(_make_frame(
                i, observed=False,
                subject_vis=VisibilityState.VISIBLE,
                object_vis=VisibilityState.OCCLUDED_OR_UNRELIABLE,
            ))
        score_occ = out_occ.edge_observations[0].predicate_score

        # Key assertion: visible absence decays faster
        assert score_vis < score_occ, (
            f"Visible absence score ({score_vis:.4f}) should be less than "
            f"occluded absence score ({score_occ:.4f})"
        )

    def test_visible_decay_value(self):
        """Verify exact decay computation for visible absence."""
        model = GatedEMA(
            alpha=1.0, decay_visible=0.5, decay_occluded=0.95,
            score_threshold=0.0,
        )
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True, predicate_score=1.0))
        out = model.step(_make_frame(
            1, observed=False,
            subject_vis=VisibilityState.VISIBLE,
            object_vis=VisibilityState.VISIBLE,
        ))
        # score = 1.0 * 0.5 = 0.5
        assert out.edge_observations[0].predicate_score == pytest.approx(0.5)

    def test_occluded_decay_value(self):
        """Verify exact decay computation for occluded absence."""
        model = GatedEMA(
            alpha=1.0, decay_visible=0.5, decay_occluded=0.8,
            score_threshold=0.0,
        )
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True, predicate_score=1.0))
        out = model.step(_make_frame(
            1, observed=False,
            subject_vis=VisibilityState.VISIBLE,
            object_vis=VisibilityState.OCCLUDED_OR_UNRELIABLE,
        ))
        # score = 1.0 * 0.8 = 0.8
        assert out.edge_observations[0].predicate_score == pytest.approx(0.8)

    def test_observation_updates_score(self):
        model = GatedEMA(
            alpha=0.5, decay_visible=0.5, decay_occluded=0.95,
            score_threshold=0.0,
        )
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True, predicate_score=0.8))
        assert len(out.edge_observations) == 1
        assert out.edge_observations[0].predicate_score == pytest.approx(0.4)

    def test_reset_clears_state(self):
        model = GatedEMA(alpha=0.5, decay_visible=0.5, decay_occluded=0.95)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True, predicate_score=1.0))
        model.reset_video("v2")
        out = model.step(_make_frame(0, observed=False))
        assert len(out.edge_observations) == 0

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            GatedEMA(alpha=0.0)
        with pytest.raises(ValueError, match="decay_visible"):
            GatedEMA(decay_visible=1.0)
        with pytest.raises(ValueError, match="decay_occluded"):
            GatedEMA(decay_occluded=-0.1)

    def test_default_unknown_equals_occluded(self):
        model = GatedEMA(decay_occluded=0.8)
        assert model.decay_unknown == 0.8

    def test_deterministic(self):
        frames = make_simple_video_frames()
        results = []
        for _ in range(2):
            model = GatedEMA(alpha=0.3, decay_visible=0.7, decay_occluded=0.95)
            model.reset_video("synthetic_001")
            outputs = [model.step(f) for f in frames]
            scores = [
                [e.predicate_score for e in o.edge_observations]
                for o in outputs
            ]
            results.append(scores)
        assert results[0] == results[1]

    def test_occlusion_bridging_scenario(self):
        """Gated EMA should bridge the occlusion gap in the synthetic fixture.

        In make_simple_video_frames, the edge is observed frames 0-3,
        occluded frames 4-6, and observed again frames 7-9.
        With high decay_occluded, the edge should persist through the gap.
        """
        model = GatedEMA(
            alpha=0.3, decay_visible=0.5, decay_occluded=0.98,
            score_threshold=0.05,
        )
        model.reset_video("synthetic_001")
        frames = make_simple_video_frames()
        outputs = [model.step(f) for f in frames]

        # During occlusion (frames 4-6), edge should still be output
        for i in [4, 5, 6]:
            assert len(outputs[i].edge_observations) > 0, (
                f"Edge should persist during occlusion at frame {i}"
            )

    def test_termination_detection_scenario(self):
        """Gated EMA should eventually drop the edge on visible absence.

        In make_termination_video_frames, the edge is observed frames 0-4,
        then absent with BOTH objects VISIBLE frames 5-9.
        With low decay_visible, the edge should be dropped.
        """
        model = GatedEMA(
            alpha=0.3, decay_visible=0.3, decay_occluded=0.98,
            score_threshold=0.05,
        )
        model.reset_video("synthetic_term_001")
        frames = make_termination_video_frames()
        outputs = [model.step(f) for f in frames]

        # By frame 9, the edge should have decayed below threshold
        assert len(outputs[9].edge_observations) == 0, (
            "Edge should be dropped after sustained visible absence"
        )


# ---------------------------------------------------------------
# HMM tests
# ---------------------------------------------------------------

class TestHMM:
    def test_observation_increases_belief(self):
        model = HMM(prior_alive=0.5, score_threshold=0.0)
        model.reset_video("v1")
        out = model.step(_make_frame(0, observed=True))
        assert len(out.edge_observations) == 1
        belief = out.edge_observations[0].predicate_score
        # After seeing an observation with prior 0.5, belief should increase
        assert belief > 0.5

    def test_absence_decreases_belief(self):
        model = HMM(
            stay_alive_prob=0.9, emit_obs_alive=0.9, emit_obs_dead=0.05,
            prior_alive=0.5, score_threshold=0.0,
        )
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True))
        out = model.step(_make_frame(1, observed=False))
        belief = out.edge_observations[0].predicate_score

        # After seeing observed then absent, belief should decrease from peak
        model2 = HMM(
            stay_alive_prob=0.9, emit_obs_alive=0.9, emit_obs_dead=0.05,
            prior_alive=0.5, score_threshold=0.0,
        )
        model2.reset_video("v1")
        out_obs = model2.step(_make_frame(0, observed=True))
        peak = out_obs.edge_observations[0].predicate_score
        assert belief < peak

    def test_reset_clears_belief(self):
        model = HMM(score_threshold=0.0)
        model.reset_video("v1")
        model.step(_make_frame(0, observed=True))
        model.reset_video("v2")
        out = model.step(_make_frame(0, observed=False))
        assert len(out.edge_observations) == 0 or (
            out.edge_observations[0].predicate_score < 0.5
        )

    def test_deterministic(self):
        frames = make_simple_video_frames()
        results = []
        for _ in range(2):
            model = HMM()
            model.reset_video("synthetic_001")
            outputs = [model.step(f) for f in frames]
            scores = [
                [e.predicate_score for e in o.edge_observations]
                for o in outputs
            ]
            results.append(scores)
        assert results[0] == results[1]

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError, match="stay_alive_prob"):
            HMM(stay_alive_prob=1.0)
        with pytest.raises(ValueError, match="emit_obs_alive"):
            HMM(emit_obs_alive=0.0)
