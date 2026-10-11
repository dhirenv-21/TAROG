"""Episode-study conditions and deterministic instance construction (prereg §3-§4).

Corruption order (registered): render -> box jitter -> temporary gaps -> hidden
termination -> misleading post flags -> boundary flicker -> background misses ->
background false detections -> ID switch -> i.i.d. flag noise -> bursty flag noise.

An instance holds the GT episodes (for metrics only), the clean GT frames, the
corrupted INPUT frames (the only thing a model sees) and all manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from taorg.data.episode_world import EpisodeWorldConfig, jitter_boxes, render, sample_world
from taorg.experiments.phase3.conditions import derive_seed
from taorg.noise.bursty_visibility import BurstyVisibilityConfig, inject_bursty_visibility
from taorg.noise.episode_corruptions import (
    inject_boundary_flicker, inject_false_detections, inject_gaps, inject_hidden_termination,
    inject_post_termination_flags,
)
from taorg.noise.flicker import FlickerConfig, inject_flicker
from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.visibility_noise import VisibilityNoiseConfig, inject_visibility_noise
from taorg.schemas.edges import EdgeKey, FrameGraph

VALID_ROLES = ("ni", "benefit", "transfer")


@dataclass(frozen=True)
class EpisodeCondition:
    """Declarative condition (configs/experiments/episodes.yaml)."""

    name: str
    role: str
    calibrate: bool
    world_overrides: dict = field(default_factory=dict)
    base_overrides: dict = field(default_factory=dict)
    hidden_termination: dict | None = None
    misleading_post: dict | None = None
    boundary_flicker: dict | None = None
    id_switch: dict | None = None
    visibility_noise: dict | None = None
    bursty_visibility: dict | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"{self.name}: role must be one of {VALID_ROLES}")
        if self.role == "transfer" and self.calibrate:
            raise ValueError(f"{self.name}: transfer conditions are never calibrated")

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "EpisodeCondition":
        """Build from the config mapping."""
        known = {"role", "calibrate", "world_overrides", "base_overrides", "hidden_termination",
                 "misleading_post", "boundary_flicker", "id_switch", "visibility_noise", "bursty_visibility"}
        extra = set(d) - known
        if extra:
            raise KeyError(f"{name}: unknown condition keys {sorted(extra)}")
        return cls(name=name, role=d["role"], calibrate=bool(d.get("calibrate", False)),
                   world_overrides=dict(d.get("world_overrides") or {}),
                   base_overrides=dict(d.get("base_overrides") or {}),
                   hidden_termination=d.get("hidden_termination"), misleading_post=d.get("misleading_post"),
                   boundary_flicker=d.get("boundary_flicker"), id_switch=d.get("id_switch"),
                   visibility_noise=d.get("visibility_noise"), bursty_visibility=d.get("bursty_visibility"))


@dataclass
class EpisodeInstance:
    """One paired evaluation unit."""

    instance_id: str
    condition: str
    seed: int
    fps: float
    gt_episodes: dict[EdgeKey, list[tuple[int, int]]]
    gt_frames: list[FrameGraph]
    input_frames: list[FrameGraph]
    occlusion_manifest: CorruptionManifest
    manifests: dict[str, Any]
    manifest_sha256: str


def world_config(world: dict, cond: EpisodeCondition) -> EpisodeWorldConfig:
    """Base world merged with the condition's overrides."""
    return EpisodeWorldConfig.from_dict({**world, **cond.world_overrides})


def build_instance(cond: EpisodeCondition, seed: int, world: dict, base: dict) -> EpisodeInstance:
    """Construct one instance deterministically (same seed -> identical instance)."""
    wcfg = world_config(world, cond)
    b = {**base, **cond.base_overrides}
    w = sample_world(seed, wcfg)
    vid = f"episode_world_{seed}"
    gt = render(w, vid)
    m: dict[str, Any] = {"succession": w.succession}
    frames, man = jitter_boxes(gt, derive_seed(seed, "box_jitter"), wcfg.box_jitter)
    m["box_jitter"] = man.to_dict()
    frames, occ = inject_gaps(frames, w.keys, derive_seed(seed, "gaps", json.dumps(b, sort_keys=True)),
                              float(b["gap_prob"]), int(b["gap_max_per_episode"]), tuple(b["gap_frames"]),
                              int(b["gap_min_episode"]), int(b["gap_margin"]))
    m["occlusion"] = occ.to_dict()
    if cond.hidden_termination:
        h = cond.hidden_termination
        frames, man = inject_hidden_termination(frames, w.keys, derive_seed(seed, "hidden_termination"),
                                                float(h["prob"]), tuple(h["pre_frames"]), tuple(h["post_frames"]))
        m["termination_occlusion"] = man.to_dict()
    if cond.misleading_post:
        h = cond.misleading_post
        frames, man = inject_post_termination_flags(frames, w.keys, derive_seed(seed, "misleading_post"),
                                                    float(h["prob"]), tuple(h["post_frames"]))
        m["post_termination_flags"] = man.to_dict()
    if cond.boundary_flicker:
        h = cond.boundary_flicker
        frames, man = inject_boundary_flicker(frames, w.keys, derive_seed(seed, "boundary_flicker"),
                                              int(h["window"]), float(h["miss_prob"]), float(h["false_prob"]))
        m["boundary_flicker"] = man.to_dict()
    if b["miss_prob"] > 0:
        frames, man = inject_flicker(frames, derive_seed(seed, "background_miss"),
                                     FlickerConfig(miss_prob=float(b["miss_prob"])))
        m["flicker"] = man.to_dict()
    if b["false_detection_prob"] > 0:
        frames, man = inject_false_detections(frames, w.keys, derive_seed(seed, "false_detection"),
                                              float(b["false_detection_prob"]))
        m["false_detection"] = man.to_dict()
    if cond.id_switch:
        frames, man = inject_id_switch(frames, derive_seed(seed, "id_switch"), IDSwitchConfig(**cond.id_switch))
        m["id_switch"] = man.to_dict()
    if cond.visibility_noise:
        v = cond.visibility_noise
        frames, man = inject_visibility_noise(frames, derive_seed(seed, "visibility_noise"), VisibilityNoiseConfig(
            occluded_to_visible_prob=float(v.get("occluded_to_visible_prob", 0.0)),
            visible_to_occluded_prob=float(v.get("visible_to_occluded_prob", 0.0)),
            to_unknown_prob=float(v.get("to_unknown_prob", 0.0))))
        m["visibility_noise"] = man.to_dict()
    if cond.bursty_visibility:
        bc = BurstyVisibilityConfig(**cond.bursty_visibility)
        frames, man = inject_bursty_visibility(frames, derive_seed(seed, "bursty_visibility"), bc)
        m["bursty_visibility"] = man.to_dict()
    gt_eps = {ke.edge_key: list(ke.episodes) for ke in w.keys}
    record = {"condition": cond.name, "seed": seed, "world": wcfg.to_dict(), "base": b,
              "episodes": {str(tuple(k)): v for k, v in sorted(gt_eps.items())}, "manifests": m}
    sha = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return EpisodeInstance(instance_id=f"{cond.name}|s{seed}", condition=cond.name, seed=seed, fps=wcfg.fps,
                           gt_episodes=gt_eps, gt_frames=gt, input_frames=frames, occlusion_manifest=occ,
                           manifests=m, manifest_sha256=sha)


def fit_transition(instances_gt: list[dict[EdgeKey, list[tuple[int, int]]]], num_frames: int,
                   num_predicates: int, before: int, after: int) -> list[list[float]]:
    """Transition statistic E (formulation §6) from GROUND-TRUTH episodes of FITTING data.

    E[r][r'] = n_succ(r -> r') / (n_term(r) + 1); a termination of r on a pair at
    frame b counts as succeeded by r' if an episode of r' on the same pair starts
    in [b - before, b + after].  Only terminations with b < num_frames - 1 count.
    """
    n_term = [0] * num_predicates
    n_succ = [[0] * num_predicates for _ in range(num_predicates)]
    for eps in instances_gt:
        starts: dict[tuple[int, int, int], list[int]] = {}
        for k, lst in eps.items():
            starts.setdefault((k.subject_track_id, k.object_track_id, k.predicate_id), []).extend(a for a, _ in lst)
        for k, lst in eps.items():
            for _, b in lst:
                if b >= num_frames - 1:
                    continue
                r = k.predicate_id
                n_term[r] += 1
                for r2 in range(num_predicates):
                    if r2 == r:
                        continue
                    st = starts.get((k.subject_track_id, k.object_track_id, r2), [])
                    if any(b - before <= a <= b + after for a in st):
                        n_succ[r][r2] += 1
    return [[n_succ[r][q] / (n_term[r] + 1) for q in range(num_predicates)] for r in range(num_predicates)]
