"""Phase 4 conditions and deterministic instance construction.

Corruption order (fixed, registered):
    world lifetimes -> predicate succession (H3 world) -> render -> bbox jitter
    -> occlusion -> ID switch -> flicker -> i.i.d. visibility noise
    -> bursty visibility noise.

The Phase 3 injectors and seed derivation are reused unchanged; only the
jitter, bursty-visibility and succession steps are new.  Only the occlusion
manifest defines Persistence-F1 intervals (frozen metric definition).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from taorg.data.synthetic_world import EdgeLifetime, render_video, sample_lifetimes
from taorg.experiments.phase3.conditions import Instance, derive_seed
from taorg.noise.bbox_jitter import inject_bbox_jitter
from taorg.noise.bursty_visibility import BurstyVisibilityConfig, inject_bursty_visibility
from taorg.noise.flicker import FlickerConfig, inject_flicker
from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.noise.visibility_noise import VisibilityNoiseConfig, inject_visibility_noise
from taorg.data.synthetic_world import SyntheticWorldConfig
from taorg.schemas.edges import EdgeKey

VALID_ROLES = ("h2_failure", "h2_noninferiority", "h3_benefit", "h3_noninferiority",
               "h4_source", "h4_shift")


@dataclass(frozen=True)
class P4ConditionSpec:
    """Declarative Phase 4 condition (see configs/experiments/phase4.yaml)."""

    name: str
    role: str
    description: str = ""
    gaps: tuple[int, ...] = (3, 10, 20)
    calibrate: bool = False
    tuned_on: str | None = None
    world_overrides: dict = field(default_factory=dict)
    occlusion_overrides: dict = field(default_factory=dict)
    flicker_miss_prob: float = 0.0
    vis_occluded_to_visible: float = 0.0
    vis_visible_to_occluded: float = 0.0
    vis_to_unknown: float = 0.0
    id_switch: dict | None = None
    bursty_visibility: dict | None = None
    successor_prob: float = 0.0
    fps_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"{self.name}: role must be in {VALID_ROLES}, got {self.role}")
        if not self.calibrate and not self.tuned_on:
            raise ValueError(f"{self.name}: set calibrate=true or tuned_on")

    @classmethod
    def from_dict(cls, name: str, d: dict, default_gaps: Iterable[int]) -> "P4ConditionSpec":
        """Build from a config mapping."""
        vis = d.get("visibility_noise", {}) or {}
        return cls(
            name=name, role=d["role"], description=d.get("description", ""),
            gaps=tuple(int(g) for g in d.get("gaps", default_gaps)),
            calibrate=bool(d.get("calibrate", False)), tuned_on=d.get("tuned_on"),
            world_overrides=dict(d.get("world_overrides", {}) or {}),
            occlusion_overrides=dict(d.get("occlusion_overrides", {}) or {}),
            flicker_miss_prob=float(d.get("flicker_miss_prob", 0.0)),
            vis_occluded_to_visible=float(vis.get("occluded_to_visible_prob", 0.0)),
            vis_visible_to_occluded=float(vis.get("visible_to_occluded_prob", 0.0)),
            vis_to_unknown=float(vis.get("to_unknown_prob", 0.0)),
            id_switch=d.get("id_switch"), bursty_visibility=d.get("bursty_visibility"),
            successor_prob=float(d.get("successor_prob", 0.0)),
            fps_scale=float(d.get("fps_scale", 1.0)),
        )

    @property
    def tuning_source(self) -> str:
        """Condition whose calibration selects this condition's parameters."""
        return self.name if self.calibrate else str(self.tuned_on)


def world_config(world: dict, spec: P4ConditionSpec) -> SyntheticWorldConfig:
    """Merge base world config with condition overrides."""
    merged = {**world, **spec.world_overrides}
    for k in ("score_mean_range", "termination_occlusion_frames"):
        if k in merged and isinstance(merged[k], list):
            merged[k] = tuple(merged[k])
    return SyntheticWorldConfig(**merged)


def add_successors(
    lifetimes: list[EdgeLifetime], seed: int, config: SyntheticWorldConfig,
    prob: float, partners: dict[int, int],
) -> tuple[list[EdgeLifetime], list[dict[str, Any]]]:
    """Predicate succession (H3 world).

    Each terminating edge (u, v, r) is, with probability ``prob``, followed by a
    new edge (u, v, partner(r)) born on the frame after r ends and alive until
    the last frame.  Skipped (and recorded) when the successor key already
    exists, since every key has at most one lifetime.  Draws are made for every
    lifetime in key order, unconditionally, from a dedicated stream.

    Returns:
        (lifetimes sorted by key, list of succession records).
    """
    rng = random.Random(derive_seed(seed, "successor"))
    keys = {lt.edge_key for lt in lifetimes}
    out = list(lifetimes)
    records: list[dict[str, Any]] = []
    last = config.num_frames - 1
    for lt in sorted(lifetimes, key=lambda x: x.edge_key):
        u = rng.random()
        mean = rng.uniform(*config.score_mean_range)
        if not (lt.terminates and u < prob) or lt.end_frame >= last:
            continue
        k = EdgeKey(lt.edge_key.subject_track_id, lt.edge_key.object_track_id,
                    int(partners[lt.edge_key.predicate_id]))
        rec = {"from": list(lt.edge_key), "to": list(k), "birth": lt.end_frame + 1}
        if k in keys:
            rec["skipped"] = "key_exists"
            records.append(rec)
            continue
        keys.add(k)
        out.append(EdgeLifetime(edge_key=k, start_frame=lt.end_frame + 1, end_frame=last,
                                terminates=False, score_mean=mean))
        records.append(rec)
    return sorted(out, key=lambda x: x.edge_key), records


def build_instance(
    spec: P4ConditionSpec, gap: int, seed: int, world: dict, occlusion_defaults: dict,
    jitter_std: float, partners: dict[int, int],
) -> Instance:
    """Construct one Phase 4 instance deterministically."""
    wcfg = world_config(world, spec)
    lifetimes = sample_lifetimes(seed, wcfg)
    manifests: dict[str, Any] = {}
    if spec.successor_prob > 0:
        lifetimes, recs = add_successors(lifetimes, seed, wcfg, spec.successor_prob, partners)
        manifests["successors"] = recs
    video_id = f"synthetic_world_{seed}"
    gt = render_video(lifetimes, seed, wcfg, video_id)

    frames, m = inject_bbox_jitter(gt, derive_seed(seed, "bbox_jitter", jitter_std), jitter_std)
    manifests["bbox_jitter"] = m.to_dict()
    occ_kwargs = {**occlusion_defaults, **spec.occlusion_overrides}
    occ_seed = derive_seed(seed, "occlusion", gap, json.dumps(occ_kwargs, sort_keys=True))
    if gap > 0:
        frames, occ = inject_occlusion(frames, seed=occ_seed,
                                       config=OcclusionConfig(gap_lengths=[gap], **occ_kwargs))
    else:
        occ = CorruptionManifest(seed=occ_seed, config={"type": "occlusion", "gap_lengths": []})
    manifests["occlusion"] = occ.to_dict()
    if spec.id_switch:
        frames, m = inject_id_switch(
            frames, seed=derive_seed(seed, "id_switch", json.dumps(spec.id_switch, sort_keys=True)),
            config=IDSwitchConfig(**spec.id_switch))
        manifests["id_switch"] = m.to_dict()
    if spec.flicker_miss_prob > 0:
        frames, m = inject_flicker(frames, seed=derive_seed(seed, "flicker", spec.flicker_miss_prob),
                                   config=FlickerConfig(miss_prob=spec.flicker_miss_prob))
        manifests["flicker"] = m.to_dict()
    if spec.vis_occluded_to_visible or spec.vis_visible_to_occluded or spec.vis_to_unknown:
        frames, m = inject_visibility_noise(
            frames, seed=derive_seed(seed, "visibility_noise", spec.vis_occluded_to_visible,
                                     spec.vis_visible_to_occluded, spec.vis_to_unknown),
            config=VisibilityNoiseConfig(occluded_to_visible_prob=spec.vis_occluded_to_visible,
                                         visible_to_occluded_prob=spec.vis_visible_to_occluded,
                                         to_unknown_prob=spec.vis_to_unknown))
        manifests["visibility_noise"] = m.to_dict()
    if spec.bursty_visibility:
        bcfg = BurstyVisibilityConfig(**spec.bursty_visibility)
        frames, m = inject_bursty_visibility(
            frames, derive_seed(seed, "bursty_visibility", bcfg.error_rate, bcfg.mean_burst_frames), bcfg)
        manifests["bursty_visibility"] = m.to_dict()

    record = {"condition": spec.name, "gap": gap, "seed": seed, "world": wcfg.to_dict(),
              "successor_prob": spec.successor_prob, "jitter": jitter_std, "manifests": manifests}
    sha = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return Instance(instance_id=f"{spec.name}|g{gap}|s{seed}", condition=spec.name, gap=gap,
                    seed=seed, gt_frames=gt, input_frames=frames, lifetimes=lifetimes,
                    occlusion_manifest=occ, manifests=manifests, manifest_sha256=sha)


def exclusivity_matrix(
    lifetime_sets: Iterable[list[EdgeLifetime]], num_predicates: int,
) -> list[list[float]]:
    """Cross-predicate exclusivity statistic E (formulation §9), from GT lifetimes.

    E[r][r'] = n_succ(r -> r') / (n_prior(r -> r') + 1), where n_prior counts
    births of r' on a pair that previously carried r (ended before the birth),
    and n_succ counts those births occurring within one frame after r ended.
    Computed on CALIBRATION ground truth only.
    """
    n_prior = [[0] * num_predicates for _ in range(num_predicates)]
    n_succ = [[0] * num_predicates for _ in range(num_predicates)]
    for lts in lifetime_sets:
        by_pair: dict[tuple[int, int], list[EdgeLifetime]] = {}
        for lt in lts:
            by_pair.setdefault((lt.edge_key.subject_track_id, lt.edge_key.object_track_id), []).append(lt)
        for group in by_pair.values():
            for new in group:
                rp = new.edge_key.predicate_id
                for old in group:
                    r = old.edge_key.predicate_id
                    if r == rp or not old.terminates or old.end_frame >= new.start_frame:
                        continue
                    n_prior[r][rp] += 1
                    if new.start_frame - old.end_frame <= 1:
                        n_succ[r][rp] += 1
    return [[n_succ[r][q] / (n_prior[r][q] + 1) for q in range(num_predicates)]
            for r in range(num_predicates)]


def with_gaps(spec: P4ConditionSpec, gaps: Iterable[int]) -> P4ConditionSpec:
    """Copy of a spec with different gaps (smoke runs only)."""
    return replace(spec, gaps=tuple(gaps))
