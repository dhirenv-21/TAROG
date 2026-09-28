"""Deterministic visibility-signal noise injection.

The existing occlusion injector produces a *perfect* visibility signal:
every suppressed frame is flagged OCCLUDED_OR_UNRELIABLE and every genuine
termination stays VISIBLE.  Under that oracle, visibility-gated methods are
advantaged by construction.  This injector degrades the visibility signal so
that robustness to an imperfect visibility estimator can be measured.

Only ABSENCE records (``observed=False``) are modified — the visibility of an
observed edge does not influence any Phase 1 method.

Per absence record, two uniform draws ``u1, u2`` are always made (keeping the
random stream aligned regardless of branch):
    1. if ``u1 < to_unknown_prob``: object visibility -> UNKNOWN
    2. elif edge visibility is OCCLUDED_OR_UNRELIABLE and
       ``u2 < occluded_to_visible_prob``: subject and object -> VISIBLE
       (occlusion not detected)
    3. elif edge visibility is VISIBLE and ``u2 < visible_to_occluded_prob``:
       object -> OCCLUDED_OR_UNRELIABLE (spurious occlusion flag)

Input frames are not mutated.  Every modification is recorded in the manifest
(``corruption_type="visibility_noise"``), grouped into contiguous runs per edge
and per (original, modified) visibility pair.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass
class VisibilityNoiseConfig:
    """Configuration for visibility-signal noise.

    Attributes:
        occluded_to_visible_prob: P(an occluded absence is reported VISIBLE).
        visible_to_occluded_prob: P(a visible absence is reported OCCLUDED).
        to_unknown_prob: P(an absence record's visibility becomes UNKNOWN).
    """

    occluded_to_visible_prob: float = 0.0
    visible_to_occluded_prob: float = 0.0
    to_unknown_prob: float = 0.0

    def __post_init__(self) -> None:
        for name in ("occluded_to_visible_prob", "visible_to_occluded_prob", "to_unknown_prob"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")


def inject_visibility_noise(
    frames: list[FrameGraph],
    seed: int,
    config: VisibilityNoiseConfig | None = None,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Corrupt the visibility flags of absence records.

    Args:
        frames: Input frame sequence (not mutated).
        seed: Random seed.
        config: Noise configuration (defaults are a no-op).

    Returns:
        (corrupted_frames, manifest).
    """
    if config is None:
        config = VisibilityNoiseConfig()
    rng = random.Random(seed)
    manifest = CorruptionManifest(
        seed=seed,
        config={
            "type": "visibility_noise",
            "occluded_to_visible_prob": config.occluded_to_visible_prob,
            "visible_to_occluded_prob": config.visible_to_occluded_prob,
            "to_unknown_prob": config.to_unknown_prob,
        },
    )
    V = VisibilityState
    changes: dict[tuple[EdgeKey, str, str], list[int]] = {}
    out: list[FrameGraph] = []
    for frame in frames:
        new_edges: list[EdgeObservation] = []
        for obs in frame.edge_observations:
            if obs.observed:
                new_edges.append(obs)
                continue
            u1 = rng.random()
            u2 = rng.random()
            orig = obs.edge_visibility
            subj, objv = obs.subject_visibility, obs.object_visibility
            if u1 < config.to_unknown_prob:
                objv = V.UNKNOWN
            elif orig == V.OCCLUDED_OR_UNRELIABLE and u2 < config.occluded_to_visible_prob:
                subj, objv = V.VISIBLE, V.VISIBLE
            elif orig == V.VISIBLE and u2 < config.visible_to_occluded_prob:
                objv = V.OCCLUDED_OR_UNRELIABLE
            if (subj, objv) == (obs.subject_visibility, obs.object_visibility):
                new_edges.append(obs)
                continue
            new_obs = EdgeObservation(
                edge_key=obs.edge_key, video_id=obs.video_id,
                frame_idx=obs.frame_idx, timestamp=obs.timestamp,
                predicate_score=obs.predicate_score, observed=False,
                subject_visibility=subj, object_visibility=objv,
                source_model=obs.source_model, source_checkpoint=obs.source_checkpoint,
            )
            new_edges.append(new_obs)
            key = (obs.edge_key, orig.name, new_obs.edge_visibility.name)
            changes.setdefault(key, []).append(frame.frame_idx)
        out.append(FrameGraph(
            video_id=frame.video_id, frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=list(frame.track_observations),
            edge_observations=new_edges,
        ))

    video_id = frames[0].video_id if frames else ""
    for (key, orig_name, new_name) in sorted(changes, key=lambda k: (k[0], k[1], k[2])):
        idxs = changes[(key, orig_name, new_name)]
        start = prev = idxs[0]
        runs = []
        for f in idxs[1:]:
            if f == prev + 1:
                prev = f
                continue
            runs.append((start, prev))
            start = prev = f
        runs.append((start, prev))
        for s, e in runs:
            manifest.entries.append(CorruptionEntry(
                corruption_type="visibility_noise", video_id=video_id,
                edge_key_tuple=(key.subject_track_id, key.object_track_id, key.predicate_id),
                start_frame=s, end_frame=e, duration=e - s + 1,
                original_visibility=orig_name, modified_visibility=new_name,
            ))
    logger.debug("Injected visibility noise: %d runs (seed=%d)", len(manifest.entries), seed)
    return out, manifest
