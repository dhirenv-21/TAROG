"""Deterministic base-model flicker injection.

Simulates the base VidSGG model *missing* a relationship in a frame while both
objects remain VISIBLE — i.e. a visible false negative, not an occlusion.
This is the observation-noise regime in which visibility-gated methods are
expected to be most vulnerable, because they interpret visible absence as
evidence of termination.

Key properties:
    - Deterministic under seed (local ``random.Random``; one draw per
      observed edge record, in frame order, so the stream is stable).
    - Only ``observed=True`` records are affected; occlusion-suppressed records
      (already ``observed=False``) are never touched.
    - Visibility is left unchanged (VISIBLE stays VISIBLE).
    - Ground truth is conceptually unchanged; the manifest records every
      suppressed frame, grouped into contiguous runs per edge.
    - Input frames are not mutated.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph

logger = logging.getLogger(__name__)


@dataclass
class FlickerConfig:
    """Configuration for flicker injection.

    Attributes:
        miss_prob: Per-frame probability that an observed edge is suppressed.
    """

    miss_prob: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.miss_prob <= 1.0):
            raise ValueError(f"miss_prob must be in [0, 1], got {self.miss_prob}")


def _group_runs(frames_by_edge: dict[EdgeKey, list[int]]) -> list[tuple[EdgeKey, int, int]]:
    """Group sorted frame indices per edge into inclusive contiguous runs."""
    runs: list[tuple[EdgeKey, int, int]] = []
    for key in sorted(frames_by_edge):
        idxs = sorted(frames_by_edge[key])
        start = prev = idxs[0]
        for f in idxs[1:]:
            if f == prev + 1:
                prev = f
                continue
            runs.append((key, start, prev))
            start = prev = f
        runs.append((key, start, prev))
    return runs


def inject_flicker(
    frames: list[FrameGraph],
    seed: int,
    config: FlickerConfig | None = None,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Suppress observed edges at random frames while keeping them VISIBLE.

    Args:
        frames: Input frame sequence (not mutated).
        seed: Random seed.
        config: Flicker configuration (defaults to miss_prob=0, i.e. no-op).

    Returns:
        (corrupted_frames, manifest) with ``corruption_type="flicker"`` entries.
    """
    if config is None:
        config = FlickerConfig()
    rng = random.Random(seed)
    manifest = CorruptionManifest(
        seed=seed, config={"type": "flicker", "miss_prob": config.miss_prob},
    )
    suppressed: dict[EdgeKey, list[int]] = {}
    out: list[FrameGraph] = []
    for frame in frames:
        new_edges: list[EdgeObservation] = []
        for obs in frame.edge_observations:
            if obs.observed and rng.random() < config.miss_prob:
                suppressed.setdefault(obs.edge_key, []).append(frame.frame_idx)
                new_edges.append(EdgeObservation(
                    edge_key=obs.edge_key, video_id=obs.video_id,
                    frame_idx=obs.frame_idx, timestamp=obs.timestamp,
                    predicate_score=0.0, observed=False,
                    subject_visibility=obs.subject_visibility,
                    object_visibility=obs.object_visibility,
                    source_model=obs.source_model,
                    source_checkpoint=obs.source_checkpoint,
                ))
            else:
                new_edges.append(obs)
        out.append(FrameGraph(
            video_id=frame.video_id, frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=list(frame.track_observations),
            edge_observations=new_edges,
        ))

    video_id = frames[0].video_id if frames else ""
    for key, start, end in _group_runs(suppressed) if suppressed else []:
        manifest.entries.append(CorruptionEntry(
            corruption_type="flicker", video_id=video_id,
            edge_key_tuple=(key.subject_track_id, key.object_track_id, key.predicate_id),
            start_frame=start, end_frame=end, duration=end - start + 1,
            original_visibility="VISIBLE", modified_visibility="VISIBLE",
        ))
    logger.debug("Injected flicker: %d runs (seed=%d)", len(manifest.entries), seed)
    return out, manifest
