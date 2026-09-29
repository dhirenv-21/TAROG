"""Tests for the Phase 3 synthetic world and the new deterministic injectors."""

from __future__ import annotations

import pytest

from taorg.data.synthetic_world import SyntheticWorldConfig, generate_video, sample_lifetimes
from taorg.experiments.phase3.conditions import ConditionSpec, build_instance, derive_seed
from taorg.noise.flicker import FlickerConfig, inject_flicker
from taorg.noise.visibility_noise import VisibilityNoiseConfig, inject_visibility_noise
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.schemas.tracks import VisibilityState

OCC = {"num_gaps_per_edge": 1, "min_pre_gap_frames": 2, "min_post_gap_frames": 2, "max_edges": None}


def _obs_signature(frames):
    return [
        [(o.edge_key, o.observed, round(o.predicate_score, 12), o.subject_visibility,
          o.object_visibility) for o in f.edge_observations]
        for f in frames
    ]


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

class TestSyntheticWorld:
    def test_deterministic(self):
        a, la = generate_video(123)
        b, lb = generate_video(123)
        assert la == lb
        assert _obs_signature(a) == _obs_signature(b)

    def test_different_seeds_differ(self):
        assert sample_lifetimes(1, SyntheticWorldConfig()) != sample_lifetimes(2, SyntheticWorldConfig())

    def test_lifetime_constraints(self):
        cfg = SyntheticWorldConfig()
        for seed in range(50):
            for lt in sample_lifetimes(seed, cfg):
                assert 0 <= lt.start_frame <= cfg.start_frame_max
                if lt.terminates:
                    length = lt.end_frame - lt.start_frame + 1
                    assert cfg.min_alive_frames <= length <= cfg.max_alive_frames
                    assert cfg.num_frames - 1 - lt.end_frame >= cfg.min_post_termination_frames
                else:
                    assert lt.end_frame == cfg.num_frames - 1

    def test_gt_flags_match_lifetimes(self):
        frames, lts = generate_video(7)
        for lt in lts:
            for f in frames:
                rec = next(o for o in f.edge_observations if o.edge_key == lt.edge_key)
                assert rec.observed == (lt.start_frame <= f.frame_idx <= lt.end_frame)
                if rec.observed:
                    assert 0.05 <= rec.predicate_score <= 1.0
                else:
                    assert rec.predicate_score == 0.0

    def test_timestamps_in_seconds(self):
        frames, _ = generate_video(3, SyntheticWorldConfig(fps=30.0))
        assert frames[30].timestamp == pytest.approx(1.0)

    def test_termination_occlusion_keeps_pairing(self):
        base = SyntheticWorldConfig()
        occ = SyntheticWorldConfig(termination_occlusion_prob=1.0)
        la, lb = sample_lifetimes(11, base), sample_lifetimes(11, occ)
        assert [(l.edge_key, l.start_frame, l.end_frame) for l in la] == \
               [(l.edge_key, l.start_frame, l.end_frame) for l in lb]
        frames, lts = generate_video(11, occ)
        term = [l for l in lts if l.terminates]
        assert term and all(l.post_termination_occluded_frames > 0 for l in term)
        lt = term[0]
        rec = next(o for o in frames[lt.end_frame + 1].edge_observations if o.edge_key == lt.edge_key)
        assert not rec.observed
        assert rec.edge_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            SyntheticWorldConfig(num_frames=50)


# ---------------------------------------------------------------------------
# Flicker
# ---------------------------------------------------------------------------

class TestFlicker:
    def test_zero_prob_is_noop(self):
        frames, _ = generate_video(5)
        out, man = inject_flicker(frames, seed=1, config=FlickerConfig(0.0))
        assert _obs_signature(out) == _obs_signature(frames)
        assert len(man) == 0

    def test_full_prob_suppresses_all_observed_keeps_visibility(self):
        frames, _ = generate_video(5)
        out, man = inject_flicker(frames, seed=1, config=FlickerConfig(1.0))
        for f_in, f_out in zip(frames, out):
            for a, b in zip(f_in.edge_observations, f_out.edge_observations):
                assert not b.observed
                assert b.edge_visibility == a.edge_visibility
        n_obs = sum(o.observed for f in frames for o in f.edge_observations)
        assert sum(e.duration for e in man.entries) == n_obs

    def test_deterministic_and_input_not_mutated(self):
        frames, _ = generate_video(5)
        before = _obs_signature(frames)
        a, ma = inject_flicker(frames, seed=9, config=FlickerConfig(0.2))
        b, mb = inject_flicker(frames, seed=9, config=FlickerConfig(0.2))
        assert _obs_signature(a) == _obs_signature(b)
        assert ma.to_dict() == mb.to_dict()
        assert _obs_signature(frames) == before

    def test_manifest_counts_match(self):
        frames, _ = generate_video(5)
        out, man = inject_flicker(frames, seed=3, config=FlickerConfig(0.1))
        n_changed = sum(a.observed and not b.observed
                        for fa, fb in zip(frames, out)
                        for a, b in zip(fa.edge_observations, fb.edge_observations))
        assert n_changed == sum(e.duration for e in man.entries) > 0

    def test_does_not_touch_occluded_records(self):
        frames, _ = generate_video(5)
        occ, man = inject_occlusion(frames, seed=2, config=OcclusionConfig(gap_lengths=[5], **OCC))
        out, _ = inject_flicker(occ, seed=3, config=FlickerConfig(1.0))
        for e in man.entries:
            for f in range(e.start_frame, e.end_frame + 1):
                rec = next(o for o in out[f].edge_observations if o.edge_key == e.edge_key)
                assert rec.edge_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE


# ---------------------------------------------------------------------------
# Visibility noise
# ---------------------------------------------------------------------------

class TestVisibilityNoise:
    def test_default_is_noop(self):
        frames, _ = generate_video(5)
        out, man = inject_visibility_noise(frames, seed=1)
        assert _obs_signature(out) == _obs_signature(frames)
        assert len(man) == 0

    def test_occluded_to_visible(self):
        frames, _ = generate_video(5)
        occ, _ = inject_occlusion(frames, seed=2, config=OcclusionConfig(gap_lengths=[5], **OCC))
        out, _ = inject_visibility_noise(occ, seed=1, config=VisibilityNoiseConfig(occluded_to_visible_prob=1.0))
        for f in out:
            for o in f.edge_observations:
                assert o.edge_visibility == VisibilityState.VISIBLE

    def test_visible_to_occluded_only_absences(self):
        frames, _ = generate_video(5)
        out, man = inject_visibility_noise(frames, seed=1, config=VisibilityNoiseConfig(visible_to_occluded_prob=1.0))
        for f in out:
            for o in f.edge_observations:
                if o.observed:
                    assert o.edge_visibility == VisibilityState.VISIBLE
                else:
                    assert o.edge_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE
        assert all(e.original_visibility == "VISIBLE" for e in man.entries)

    def test_unknown(self):
        frames, _ = generate_video(5)
        out, _ = inject_visibility_noise(frames, seed=1, config=VisibilityNoiseConfig(to_unknown_prob=1.0))
        assert all(o.edge_visibility == VisibilityState.UNKNOWN
                   for f in out for o in f.edge_observations if not o.observed)

    def test_deterministic(self):
        frames, _ = generate_video(5)
        cfg = VisibilityNoiseConfig(0.3, 0.3, 0.1)
        a, ma = inject_visibility_noise(frames, seed=4, config=cfg)
        b, mb = inject_visibility_noise(frames, seed=4, config=cfg)
        assert _obs_signature(a) == _obs_signature(b)
        assert ma.to_dict() == mb.to_dict()

    def test_invalid_prob(self):
        with pytest.raises(ValueError):
            VisibilityNoiseConfig(occluded_to_visible_prob=1.5)


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

class TestInstances:
    def test_derive_seed_stable_and_distinct(self):
        assert derive_seed(1, "a", 2) == derive_seed(1, "a", 2)
        assert derive_seed(1, "a", 2) != derive_seed(1, "a", 3)
        assert derive_seed(1, "a", 2) != derive_seed(2, "a", 2)
        assert 0 <= derive_seed(123, "x") < 2 ** 31
        # SHA-256 derived, therefore identical across processes/platforms.
        assert derive_seed(0, "occlusion") == derive_seed(0, "occlusion")

    def test_build_instance_deterministic(self):
        spec = ConditionSpec(name="C", role="h1_primary", calibrate=True,
                             flicker_miss_prob=0.1, vis_occluded_to_visible=0.2)
        a = build_instance(spec, 5, 42, {}, OCC)
        b = build_instance(spec, 5, 42, {}, OCC)
        assert a.manifest_sha256 == b.manifest_sha256
        assert _obs_signature(a.input_frames) == _obs_signature(b.input_frames)

    def test_same_gt_across_conditions_and_gaps(self):
        c0 = ConditionSpec(name="C0", role="h1_primary", calibrate=True)
        r2 = ConditionSpec(name="R2", role="h1_robustness", calibrate=True, flicker_miss_prob=0.2)
        a = build_instance(c0, 3, 42, {}, OCC)
        b = build_instance(r2, 10, 42, {}, OCC)
        assert _obs_signature(a.gt_frames) == _obs_signature(b.gt_frames)
        assert a.manifest_sha256 != b.manifest_sha256

    def test_gap_zero_has_no_occlusion(self):
        spec = ConditionSpec(name="S2", role="stress", tuned_on="C0",
                             id_switch={"swap_duration": 5, "num_swaps": 1, "require_same_class": True})
        inst = build_instance(spec, 0, 42, {}, OCC)
        assert len(inst.occlusion_manifest.entries) == 0
        assert inst.manifests["id_switch"]["entries"]

    def test_occlusion_gap_length_exact(self):
        spec = ConditionSpec(name="C0", role="h1_primary", calibrate=True)
        inst = build_instance(spec, 10, 42, {}, OCC)
        assert inst.occlusion_manifest.entries
        assert all(e.duration == 10 for e in inst.occlusion_manifest.entries)

    def test_condition_validation(self):
        with pytest.raises(ValueError):
            ConditionSpec(name="X", role="bogus", calibrate=True)
        with pytest.raises(ValueError):
            ConditionSpec(name="X", role="stress")
