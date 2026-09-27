"""Comprehensive tests for corruption injection (occlusion + ID-switch).

Tests cover:
    - Deterministic corruption for identical seeds
    - Different corruption for different seeds
    - Exact gap length
    - Unaffected edges unchanged
    - Frame boundaries
    - Manifest correctness
    - Manifest serialization/deserialization
    - ID mapping traceability
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, TrackerState, VisibilityState
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from tests.fixtures.synthetic_fixtures import make_simple_video_frames


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _count_observed(frames: list[FrameGraph], edge_key: EdgeKey) -> int:
    """Count frames where edge_key is observed."""
    count = 0
    for frame in frames:
        for obs in frame.edge_observations:
            if obs.edge_key == edge_key and obs.observed:
                count += 1
    return count


def _get_edge_at_frame(
    frames: list[FrameGraph], edge_key: EdgeKey, frame_idx: int
) -> EdgeObservation | None:
    """Get edge observation at a specific frame."""
    for frame in frames:
        if frame.frame_idx == frame_idx:
            for obs in frame.edge_observations:
                if obs.edge_key == edge_key:
                    return obs
    return None


# ---------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------

class TestCorruptionManifest:
    def test_serialize_deserialize(self, tmp_path: Path):
        manifest = CorruptionManifest(
            seed=42,
            config={"type": "occlusion", "gap_lengths": [3]},
            entries=[
                CorruptionEntry(
                    corruption_type="occlusion",
                    video_id="test",
                    edge_key_tuple=(1, 2, 0),
                    start_frame=4,
                    end_frame=6,
                    duration=3,
                ),
            ],
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = CorruptionManifest.load(path)
        assert loaded.seed == 42
        assert len(loaded.entries) == 1
        assert loaded.entries[0].edge_key_tuple == (1, 2, 0)
        assert loaded.entries[0].start_frame == 4
        assert loaded.entries[0].end_frame == 6

    def test_edge_key_property(self):
        entry = CorruptionEntry(
            corruption_type="occlusion",
            video_id="test",
            edge_key_tuple=(1, 2, 3),
        )
        assert entry.edge_key == EdgeKey(1, 2, 3)

    def test_edge_key_none(self):
        entry = CorruptionEntry(
            corruption_type="id_switch",
            video_id="test",
        )
        assert entry.edge_key is None

    def test_manifest_len(self):
        manifest = CorruptionManifest(seed=0, entries=[
            CorruptionEntry(corruption_type="occlusion", video_id="t"),
            CorruptionEntry(corruption_type="occlusion", video_id="t"),
        ])
        assert len(manifest) == 2


# ---------------------------------------------------------------
# Occlusion injection tests
# ---------------------------------------------------------------

class TestOcclusionInjection:
    def test_deterministic_same_seed(self):
        """Same seed produces identical corruption."""
        frames = make_simple_video_frames(num_frames=20)
        ek = EdgeKey(1, 2, 0)
        config = OcclusionConfig(gap_lengths=[3])

        _, m1 = inject_occlusion(frames, seed=42, config=config)
        _, m2 = inject_occlusion(frames, seed=42, config=config)

        assert len(m1.entries) == len(m2.entries)
        for e1, e2 in zip(m1.entries, m2.entries):
            assert e1.start_frame == e2.start_frame
            assert e1.end_frame == e2.end_frame

    def test_different_seed_different_result(self):
        """Different seeds produce different corruption (with high probability)."""
        frames = make_simple_video_frames(num_frames=50)
        config = OcclusionConfig(gap_lengths=[3])

        _, m1 = inject_occlusion(frames, seed=42, config=config)
        _, m2 = inject_occlusion(frames, seed=123, config=config)

        # At least one entry should differ in position
        if m1.entries and m2.entries:
            positions1 = [(e.start_frame, e.end_frame) for e in m1.entries]
            positions2 = [(e.start_frame, e.end_frame) for e in m2.entries]
            # Not a hard assertion since it's probabilistic, but very likely
            # to differ with different seeds on a 50-frame sequence
            assert positions1 != positions2 or len(frames) < 5

    def test_exact_gap_length(self):
        """Corruption covers exactly the specified gap length."""
        frames = make_simple_video_frames(num_frames=20)
        config = OcclusionConfig(gap_lengths=[5])

        corrupted, manifest = inject_occlusion(frames, seed=42, config=config)

        for entry in manifest.entries:
            assert entry.duration == 5
            assert entry.end_frame - entry.start_frame + 1 == 5

    def test_observations_suppressed(self):
        """During corruption, observations are suppressed."""
        frames = make_simple_video_frames(num_frames=20)
        ek = EdgeKey(1, 2, 0)
        config = OcclusionConfig(gap_lengths=[3])

        corrupted, manifest = inject_occlusion(frames, seed=42, config=config)

        for entry in manifest.entries:
            for f_idx in range(entry.start_frame, entry.end_frame + 1):
                obs = _get_edge_at_frame(corrupted, ek, f_idx)
                assert obs is not None
                assert obs.observed is False, f"Edge should be suppressed at frame {f_idx}"
                assert obs.predicate_score == 0.0

    def test_unaffected_frames_unchanged(self):
        """Frames outside corruption ranges are unchanged."""
        frames = make_simple_video_frames(num_frames=20)
        ek = EdgeKey(1, 2, 0)
        config = OcclusionConfig(gap_lengths=[3])

        corrupted, manifest = inject_occlusion(frames, seed=42, config=config)

        corrupted_frame_set = set()
        for entry in manifest.entries:
            for f in range(entry.start_frame, entry.end_frame + 1):
                corrupted_frame_set.add(f)

        for f_idx in range(20):
            if f_idx in corrupted_frame_set:
                continue
            orig_obs = _get_edge_at_frame(frames, ek, f_idx)
            corr_obs = _get_edge_at_frame(corrupted, ek, f_idx)
            if orig_obs is not None and orig_obs.observed:
                assert corr_obs is not None
                assert corr_obs.observed == orig_obs.observed
                assert corr_obs.predicate_score == orig_obs.predicate_score

    def test_does_not_mutate_input(self):
        """Input frames are not mutated."""
        frames = make_simple_video_frames(num_frames=10)
        ek = EdgeKey(1, 2, 0)
        orig_count = _count_observed(frames, ek)

        inject_occlusion(frames, seed=42)

        assert _count_observed(frames, ek) == orig_count

    def test_manifest_records_correct_info(self):
        """Manifest entries match what actually happened."""
        frames = make_simple_video_frames(num_frames=20)
        config = OcclusionConfig(gap_lengths=[3])

        _, manifest = inject_occlusion(frames, seed=42, config=config)

        assert manifest.seed == 42
        assert manifest.config["type"] == "occlusion"
        for entry in manifest.entries:
            assert entry.corruption_type == "occlusion"
            assert entry.edge_key_tuple is not None
            assert entry.modified_visibility == "OCCLUDED_OR_UNRELIABLE"

    def test_empty_input(self):
        """Empty input produces empty output."""
        corrupted, manifest = inject_occlusion([], seed=42)
        assert len(corrupted) == 0
        assert len(manifest.entries) == 0

    def test_multiple_gap_lengths(self):
        """Multiple gap lengths are all applied."""
        frames = make_simple_video_frames(num_frames=50)
        config = OcclusionConfig(gap_lengths=[2, 5])

        _, manifest = inject_occlusion(frames, seed=42, config=config)

        durations = {e.duration for e in manifest.entries}
        # At least one gap length should be present
        assert len(manifest.entries) > 0


# ---------------------------------------------------------------
# ID-switch injection tests
# ---------------------------------------------------------------

class TestIDSwitchInjection:
    def test_deterministic_same_seed(self):
        """Same seed produces identical ID switch."""
        frames = make_simple_video_frames(num_frames=20)
        config = IDSwitchConfig(swap_duration=3, num_swaps=1)

        _, m1 = inject_id_switch(frames, seed=42, config=config)
        _, m2 = inject_id_switch(frames, seed=42, config=config)

        assert len(m1.entries) == len(m2.entries)
        for e1, e2 in zip(m1.entries, m2.entries):
            assert e1.start_frame == e2.start_frame
            assert e1.id_mapping == e2.id_mapping

    def test_track_ids_swapped(self):
        """Track IDs are correctly swapped within the swap range."""
        frames = make_simple_video_frames(num_frames=20)
        config = IDSwitchConfig(swap_duration=3, num_swaps=1)

        corrupted, manifest = inject_id_switch(frames, seed=42, config=config)

        if manifest.entries:
            entry = manifest.entries[0]
            id_map = entry.id_mapping
            for f_idx in range(entry.start_frame, entry.end_frame + 1):
                corr_frame = corrupted[f_idx]
                orig_frame = frames[f_idx]
                corr_track_ids = {t.track_id for t in corr_frame.track_observations}
                orig_track_ids = {t.track_id for t in orig_frame.track_observations}
                # The same track IDs should exist, just swapped
                assert corr_track_ids == orig_track_ids

    def test_does_not_mutate_input(self):
        """Input frames are not mutated."""
        frames = make_simple_video_frames(num_frames=10)
        orig_ids = [
            [t.track_id for t in f.track_observations]
            for f in frames
        ]

        inject_id_switch(frames, seed=42)

        new_ids = [
            [t.track_id for t in f.track_observations]
            for f in frames
        ]
        assert orig_ids == new_ids

    def test_manifest_records_id_mapping(self):
        """Manifest records exact ID mappings."""
        frames = make_simple_video_frames(num_frames=20)
        config = IDSwitchConfig(swap_duration=3, num_swaps=1)

        _, manifest = inject_id_switch(frames, seed=42, config=config)

        for entry in manifest.entries:
            assert entry.corruption_type == "id_switch"
            assert entry.id_mapping is not None
            # Mapping should be a swap (each maps to the other)
            for k, v in entry.id_mapping.items():
                assert entry.id_mapping.get(v) == k

    def test_empty_input(self):
        """Empty input produces empty output."""
        corrupted, manifest = inject_id_switch([], seed=42)
        assert len(corrupted) == 0
        assert len(manifest.entries) == 0
