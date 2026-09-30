"""Tests for Phase 4 infrastructure: injectors, succession world, search, VA-HMM."""

from __future__ import annotations

import json
import logging
import math

import pytest

from taorg.data.synthetic_world import (
    EdgeLifetime, SyntheticWorldConfig, generate_video, render_video, sample_lifetimes,
)
from taorg.experiments.phase4 import pipeline as P
from taorg.experiments.phase4.conditions import add_successors, exclusivity_matrix
from taorg.experiments.phase4.search import sample_method
from taorg.noise.bbox_jitter import inject_bbox_jitter
from taorg.noise.bursty_visibility import BurstyVisibilityConfig, inject_bursty_visibility
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.va_hmm import VisibilityAwareHMM

logging.getLogger("taorg").setLevel(logging.WARNING)
V = VisibilityState
CFG = P.load_config(__import__("pathlib").Path("configs/experiments/phase4.yaml"))


def test_render_refactor_is_identical():
    cfg = SyntheticWorldConfig()
    frames, lts = generate_video(123, cfg)
    again = render_video(sample_lifetimes(123, cfg), 123, cfg, "synthetic_world_123")
    assert [[(o.edge_key, o.predicate_score, o.observed) for o in f.edge_observations] for f in frames] == \
           [[(o.edge_key, o.predicate_score, o.observed) for o in f.edge_observations] for f in again]


def _absent_frames(n: int, vis: VisibilityState) -> list[FrameGraph]:
    k = EdgeKey(1, 2, 0)
    return [FrameGraph("v", f, f / 30, [], [EdgeObservation(k, "v", f, f / 30, 0.0, observed=False,
                                                             object_visibility=vis)])
            for f in range(n)]


def test_bursty_visibility_rate_and_scope():
    cfg = BurstyVisibilityConfig(error_rate=0.2, mean_burst_frames=10)
    assert abs(cfg.p_enter / (cfg.p_enter + cfg.p_exit) - 0.2) < 1e-12
    frames = _absent_frames(20000, V.VISIBLE)
    out, man = inject_bursty_visibility(frames, 7, cfg)
    flipped = sum(o.edge_visibility == V.OCCLUDED_OR_UNRELIABLE for f in out for o in f.edge_observations)
    assert 0.15 < flipped / 20000 < 0.25
    mean_run = sum(e.duration for e in man.entries) / len(man.entries)
    assert 7 < mean_run < 13
    again, _ = inject_bursty_visibility(frames, 7, cfg)
    assert [o.edge_visibility for f in again for o in f.edge_observations] == \
           [o.edge_visibility for f in out for o in f.edge_observations]
    unk, _ = inject_bursty_visibility(_absent_frames(500, V.UNKNOWN), 7, cfg)
    assert all(o.edge_visibility == V.UNKNOWN for f in unk for o in f.edge_observations)


def test_bbox_jitter_tracks_only_and_noop():
    frames, _ = generate_video(5, SyntheticWorldConfig())
    j, _ = inject_bbox_jitter(frames, 1, 0.15)
    assert [f.edge_observations for f in j] == [f.edge_observations for f in frames]
    assert any(a.bbox_xyxy != b.bbox_xyxy for a, b in zip(j[3].track_observations, frames[3].track_observations))
    z, _ = inject_bbox_jitter(frames, 1, 0.0)
    assert all(a.bbox_xyxy == b.bbox_xyxy for f, g in zip(z, frames)
               for a, b in zip(f.track_observations, g.track_observations))
    with pytest.raises(ValueError):
        inject_bbox_jitter(frames, 1, -0.1)


def test_add_successors_properties():
    cfg = SyntheticWorldConfig()
    partners = CFG["predicate_partners"]
    for seed in range(3000, 3010):
        lts = sample_lifetimes(seed, cfg)
        out, recs = add_successors(lts, seed, cfg, 1.0, partners)
        keys = [lt.edge_key for lt in out]
        assert len(keys) == len(set(keys))
        by_key = {lt.edge_key: lt for lt in out}
        for r in recs:
            if "skipped" in r:
                continue
            new, old = by_key[EdgeKey(*r["to"])], by_key[EdgeKey(*r["from"])]
            assert new.start_frame == old.end_frame + 1 and new.end_frame == cfg.num_frames - 1
            assert new.edge_key.predicate_id == partners[old.edge_key.predicate_id]
        none, recs0 = add_successors(lts, seed, cfg, 0.0, partners)
        assert none == sorted(lts, key=lambda x: x.edge_key) and recs0 == []


def test_exclusivity_matrix_counts():
    a = EdgeLifetime(EdgeKey(1, 2, 0), 0, 49, True, 0.8)
    b = EdgeLifetime(EdgeKey(1, 2, 1), 50, 99, False, 0.8)    # immediate successor of 0
    c = EdgeLifetime(EdgeKey(1, 3, 2), 0, 20, True, 0.8)
    d = EdgeLifetime(EdgeKey(1, 3, 3), 40, 99, False, 0.8)    # later birth, not a succession
    E = exclusivity_matrix([[a, b, c, d]], 6)
    assert E[0][1] == 0.5 and E[2][3] == 0.0 and E[1][0] == 0.0


def test_sampler_deterministic_bounds_and_constraint():
    space = CFG["search"]["spaces"]["vhped_m2"]
    a = sample_method("vhped_m2", space, 50, 777, 0.01)
    b = sample_method("vhped_m2", space, 50, 777, 0.01)
    assert [c.params_json for c in a] == [c.params_json for c in b] and len(a) == 50
    for c in a:
        p = json.loads(c.params_json)
        assert 0.01 <= p["mu"] <= 10 and 0.05 <= p["threshold"] <= 0.95
        assert p["threshold"] > p["nu"] / (p["mu"] + p["nu"]) + 0.01
    g = sample_method("gated_ema", CFG["search"]["spaces"]["gated_ema"], 20, 777, 0.01)
    assert all(0.7 <= json.loads(c.params_json)["decay_occluded"] <= 0.999 for c in g)
    assert sample_method("ema", CFG["search"]["spaces"]["ema"], 5, 777, 0.0)[0].params_json != \
        sample_method("hmm", CFG["search"]["spaces"]["hmm"], 5, 777, 0.0)[0].params_json


def _seq(pattern: list[tuple[bool, VisibilityState]]) -> list[FrameGraph]:
    k = EdgeKey(1, 2, 0)
    return [FrameGraph("v", f, f / 30, [], [EdgeObservation(k, "v", f, f / 30, 0.8 if obs else 0.0,
                                                             observed=obs, object_visibility=vis)])
            for f, (obs, vis) in enumerate(pattern)]


def _run(model, frames):
    model.reset_video("v")
    return [len(model.step(f).edge_observations) for f in frames]


def test_va_hmm_uses_flags_probabilistically():
    m = VisibilityAwareHMM(p_stay=0.99, p_birth=0.001, prior=0.5, theta_vis=0.05, theta_occ=0.999,
                           theta_unk=0.5, false_pos=0.01, threshold=0.5)
    on = [(True, V.VISIBLE)] * 10
    occ = _run(m, _seq(on + [(False, V.OCCLUDED_OR_UNRELIABLE)] * 20))
    vis = _run(m, _seq(on + [(False, V.VISIBLE)] * 20))
    assert occ[-1] == 1                 # occluded absence ~ no evidence: held
    assert vis[11] == 0                 # visible absence: strong negative evidence
    assert sum(occ) > sum(vis)
    with pytest.raises(ValueError):
        VisibilityAwareHMM(p_stay=1.5)


def test_va_hmm_no_record_is_unknown():
    m = VisibilityAwareHMM(theta_unk=0.999, threshold=0.5, p_stay=0.999)
    frames = _seq([(True, V.VISIBLE)] * 10) + [FrameGraph("v", 10 + i, (10 + i) / 30, [], []) for i in range(5)]
    assert _run(m, frames)[-1] == 1


def test_phase4_instance_determinism():
    sp = P.specs(CFG)["F7_mix_occlusion_visnoise"]
    a = P._build(P._slim_cfg(CFG), sp, 10, 3001)
    b = P._build(P._slim_cfg(CFG), sp, 10, 3001)
    assert a.manifest_sha256 == b.manifest_sha256
    assert not math.isnan(len(a.input_frames))
    assert "bbox_jitter" in a.manifests and "visibility_noise" in a.manifests
