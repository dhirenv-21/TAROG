"""Phase 3 corruption conditions and deterministic instance construction.

An *instance* is one (condition, gap length, seed) triple: a ground-truth
synthetic video plus its corrupted input stream and all corruption manifests.
Every method configuration evaluated on an instance receives exactly the same
corrupted input — comparisons across methods are therefore paired.

Seed derivation:
    - The scenario (video) seed IS the instance seed, so the same seed yields
      the same underlying video in every condition and gap (pairing across
      conditions as well as methods).
    - Corruption sub-seeds are derived with SHA-256 (never Python's salted
      ``hash()``), so they are stable across processes and platforms.

Corruption order: occlusion (on clean GT) -> ID switch -> flicker ->
visibility noise.  Only the occlusion manifest defines Persistence-F1
intervals (frozen metric definition §2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from taorg.data.synthetic_world import EdgeLifetime, SyntheticWorldConfig, generate_video
from taorg.noise.flicker import FlickerConfig, inject_flicker
from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.noise.visibility_noise import VisibilityNoiseConfig, inject_visibility_noise
from taorg.schemas.edges import FrameGraph

VALID_ROLES = ("h1_primary", "h1_robustness", "sweep", "stress")


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a stable 31-bit sub-seed from a base seed and labels.

    Args:
        base_seed: Instance seed.
        *parts: Labels identifying the random component (e.g. "occlusion", gap).

    Returns:
        Non-negative integer < 2**31.
    """
    text = "|".join([str(base_seed), *[str(p) for p in parts]])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


@dataclass(frozen=True)
class ConditionSpec:
    """Declarative description of one corruption condition.

    Attributes:
        name: Condition identifier (e.g. "C0_oracle").
        role: One of VALID_ROLES.
        description: Human-readable description.
        gaps: Occlusion gap lengths (frames) evaluated in this condition.
            A gap of 0 means "no occlusion injected".
        calibrate: If True, methods are tuned on this condition's own
            calibration instances (condition-matched tuning).
        tuned_on: If ``calibrate`` is False, the condition whose selected
            parameters are reused.
        world_overrides: Overrides for SyntheticWorldConfig fields.
        occlusion_overrides: Overrides for OcclusionConfig fields.
        flicker_miss_prob: Flicker probability (0 = none).
        vis_occluded_to_visible: Visibility noise parameter.
        vis_visible_to_occluded: Visibility noise parameter.
        vis_to_unknown: Visibility noise parameter.
        id_switch: IDSwitchConfig kwargs, or None for no ID switch.
        sweep_axis: For sweep conditions, the name of the swept variable.
        sweep_value: For sweep conditions, the swept value.
    """

    name: str
    role: str
    description: str = ""
    gaps: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    calibrate: bool = False
    tuned_on: str | None = None
    world_overrides: dict = field(default_factory=dict)
    occlusion_overrides: dict = field(default_factory=dict)
    flicker_miss_prob: float = 0.0
    vis_occluded_to_visible: float = 0.0
    vis_visible_to_occluded: float = 0.0
    vis_to_unknown: float = 0.0
    id_switch: dict | None = None
    sweep_axis: str | None = None
    sweep_value: float | None = None

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Condition {self.name}: role must be in {VALID_ROLES}, got {self.role}")
        if not self.calibrate and not self.tuned_on:
            raise ValueError(
                f"Condition {self.name}: set calibrate=true or name a tuned_on condition"
            )

    @classmethod
    def from_dict(cls, name: str, d: dict) -> ConditionSpec:
        """Build a ConditionSpec from a config mapping."""
        vis = d.get("visibility_noise", {}) or {}
        return cls(
            name=name,
            role=d["role"],
            description=d.get("description", ""),
            gaps=tuple(int(g) for g in d["gaps"]),
            calibrate=bool(d.get("calibrate", False)),
            tuned_on=d.get("tuned_on"),
            world_overrides=dict(d.get("world_overrides", {}) or {}),
            occlusion_overrides=dict(d.get("occlusion_overrides", {}) or {}),
            flicker_miss_prob=float(d.get("flicker_miss_prob", 0.0)),
            vis_occluded_to_visible=float(vis.get("occluded_to_visible_prob", 0.0)),
            vis_visible_to_occluded=float(vis.get("visible_to_occluded_prob", 0.0)),
            vis_to_unknown=float(vis.get("to_unknown_prob", 0.0)),
            id_switch=d.get("id_switch"),
            sweep_axis=d.get("sweep_axis"),
            sweep_value=d.get("sweep_value"),
        )


@dataclass
class Instance:
    """One paired evaluation unit: GT video + corrupted input + manifests."""

    instance_id: str
    condition: str
    gap: int
    seed: int
    gt_frames: list[FrameGraph]
    input_frames: list[FrameGraph]
    lifetimes: list[EdgeLifetime]
    occlusion_manifest: CorruptionManifest
    manifests: dict[str, dict]
    manifest_sha256: str


def world_config_for(world: dict, spec: ConditionSpec) -> SyntheticWorldConfig:
    """Merge base world config with condition overrides."""
    merged = {**world, **spec.world_overrides}
    for k in ("score_mean_range", "termination_occlusion_frames"):
        if k in merged and isinstance(merged[k], list):
            merged[k] = tuple(merged[k])
    return SyntheticWorldConfig(**merged)


def build_instance(
    spec: ConditionSpec,
    gap: int,
    seed: int,
    world: dict,
    occlusion_defaults: dict,
) -> Instance:
    """Construct one instance deterministically.

    Args:
        spec: Condition specification.
        gap: Occlusion gap length in frames (0 = no occlusion).
        seed: Instance / scenario seed.
        world: Base SyntheticWorldConfig kwargs.
        occlusion_defaults: Base OcclusionConfig kwargs (minus gap_lengths).

    Returns:
        The constructed Instance.
    """
    wcfg = world_config_for(world, spec)
    gt, lifetimes = generate_video(seed, wcfg)
    frames = gt
    manifests: dict[str, dict] = {}

    occ_kwargs = {**occlusion_defaults, **spec.occlusion_overrides}
    occ_seed = derive_seed(seed, "occlusion", gap, json.dumps(occ_kwargs, sort_keys=True))
    if gap > 0:
        frames, occ_manifest = inject_occlusion(
            frames, seed=occ_seed, config=OcclusionConfig(gap_lengths=[gap], **occ_kwargs),
        )
    else:
        occ_manifest = CorruptionManifest(seed=occ_seed, config={"type": "occlusion", "gap_lengths": []})
    manifests["occlusion"] = occ_manifest.to_dict()

    if spec.id_switch:
        frames, m = inject_id_switch(
            frames, seed=derive_seed(seed, "id_switch", json.dumps(spec.id_switch, sort_keys=True)),
            config=IDSwitchConfig(**spec.id_switch),
        )
        manifests["id_switch"] = m.to_dict()
    if spec.flicker_miss_prob > 0:
        frames, m = inject_flicker(
            frames, seed=derive_seed(seed, "flicker", spec.flicker_miss_prob),
            config=FlickerConfig(miss_prob=spec.flicker_miss_prob),
        )
        manifests["flicker"] = m.to_dict()
    if spec.vis_occluded_to_visible or spec.vis_visible_to_occluded or spec.vis_to_unknown:
        vcfg = VisibilityNoiseConfig(
            occluded_to_visible_prob=spec.vis_occluded_to_visible,
            visible_to_occluded_prob=spec.vis_visible_to_occluded,
            to_unknown_prob=spec.vis_to_unknown,
        )
        frames, m = inject_visibility_noise(
            frames,
            seed=derive_seed(seed, "visibility_noise", spec.vis_occluded_to_visible,
                             spec.vis_visible_to_occluded, spec.vis_to_unknown),
            config=vcfg,
        )
        manifests["visibility_noise"] = m.to_dict()

    record = {
        "condition": spec.name, "gap": gap, "seed": seed,
        "world": wcfg.to_dict(), "manifests": manifests,
    }
    sha = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
    return Instance(
        instance_id=f"{spec.name}|g{gap}|s{seed}",
        condition=spec.name, gap=gap, seed=seed,
        gt_frames=gt, input_frames=frames, lifetimes=lifetimes,
        occlusion_manifest=occ_manifest, manifests=manifests, manifest_sha256=sha,
    )
