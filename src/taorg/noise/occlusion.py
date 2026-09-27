"""Deterministic occlusion injection for controlled experiments.

Synthetically suppresses edge observations for specified frame ranges,
simulating observation failure (missed detection, occlusion) rather than
genuine relationship termination.

Key properties:
    - Deterministic under seed
    - Produces a CorruptionManifest recording exactly what happened
    - Does not alter ground-truth relationship state
    - Only suppresses observations and modifies visibility state
    - Unaffected edges remain unchanged

Corresponds to PRD FR-9 and Architecture §4.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest

logger = logging.getLogger(__name__)


@dataclass
class OcclusionConfig:
    """Configuration for occlusion injection.

    Attributes:
        gap_lengths: List of gap durations in frames to apply.
        num_gaps_per_edge: Number of gaps to inject per selected edge.
            If more gaps than possible non-overlapping positions, fewer
            are injected.
        target_edge_keys: Specific edges to corrupt. If None, edges are
            selected randomly from those present in the sequence.
        max_edges: Maximum number of edges to corrupt (when target_edge_keys
            is None). If None, corrupt all eligible edges.
        min_pre_gap_frames: Minimum frames of observation before a gap.
        min_post_gap_frames: Minimum frames of observation after a gap.
    """

    gap_lengths: list[int] = field(default_factory=lambda: [3])
    num_gaps_per_edge: int = 1
    target_edge_keys: Optional[list[EdgeKey]] = None
    max_edges: Optional[int] = None
    min_pre_gap_frames: int = 1
    min_post_gap_frames: int = 1


def inject_occlusion(
    frames: list[FrameGraph],
    seed: int,
    config: OcclusionConfig | None = None,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Apply deterministic occlusion corruption to a frame sequence.

    Suppresses edge observations for specified gap lengths, simulating
    observation failure.  The relationship ground truth is conceptually
    unchanged — only the observed signal is affected.

    Args:
        frames: Input frame sequence (will NOT be mutated).
        seed: Random seed for deterministic corruption.
        config: Corruption configuration. Uses defaults if None.

    Returns:
        Tuple of (corrupted_frames, manifest).
    """
    if config is None:
        config = OcclusionConfig()

    rng = random.Random(seed)
    corrupted = deepcopy(frames)
    manifest = CorruptionManifest(
        seed=seed,
        config={
            "type": "occlusion",
            "gap_lengths": config.gap_lengths,
            "num_gaps_per_edge": config.num_gaps_per_edge,
            "max_edges": config.max_edges,
            "min_pre_gap_frames": config.min_pre_gap_frames,
            "min_post_gap_frames": config.min_post_gap_frames,
        },
    )

    if not frames:
        return corrupted, manifest

    video_id = frames[0].video_id
    num_frames = len(frames)

    # Identify which edges are observed and in which frames
    edge_frame_map: dict[EdgeKey, list[int]] = {}
    for f_idx, frame in enumerate(frames):
        for obs in frame.edge_observations:
            if obs.observed:
                edge_frame_map.setdefault(obs.edge_key, []).append(f_idx)

    # Select target edges
    if config.target_edge_keys is not None:
        target_keys = [k for k in config.target_edge_keys if k in edge_frame_map]
    else:
        target_keys = list(edge_frame_map.keys())
        rng.shuffle(target_keys)
        if config.max_edges is not None:
            target_keys = target_keys[:config.max_edges]

    # For each target edge, inject gaps
    for edge_key in target_keys:
        observed_frames = sorted(edge_frame_map[edge_key])
        if not observed_frames:
            continue

        first_obs = observed_frames[0]
        last_obs = observed_frames[-1]

        for gap_length in config.gap_lengths:
            for _ in range(config.num_gaps_per_edge):
                # Find valid start positions: after min_pre_gap, before min_post_gap
                earliest_start = first_obs + config.min_pre_gap_frames
                latest_start = last_obs - gap_length - config.min_post_gap_frames + 1

                if earliest_start > latest_start:
                    logger.debug(
                        "Cannot place gap of length %d for edge %s: "
                        "insufficient space",
                        gap_length, edge_key,
                    )
                    continue

                gap_start = rng.randint(earliest_start, latest_start)
                gap_end = gap_start + gap_length - 1  # inclusive

                # Apply corruption to frames
                for f_idx in range(gap_start, min(gap_end + 1, num_frames)):
                    frame = corrupted[f_idx]
                    new_edges: list[EdgeObservation] = []
                    for obs in frame.edge_observations:
                        if obs.edge_key == edge_key and obs.observed:
                            # Suppress observation: mark as not observed,
                            # set visibility to OCCLUDED
                            new_edges.append(EdgeObservation(
                                edge_key=obs.edge_key,
                                video_id=obs.video_id,
                                frame_idx=obs.frame_idx,
                                timestamp=obs.timestamp,
                                predicate_score=0.0,
                                observed=False,
                                subject_visibility=obs.subject_visibility,
                                object_visibility=VisibilityState.OCCLUDED_OR_UNRELIABLE,
                                source_model=obs.source_model,
                                source_checkpoint=obs.source_checkpoint,
                            ))
                        else:
                            new_edges.append(obs)
                    frame.edge_observations = new_edges

                # Record in manifest
                manifest.entries.append(CorruptionEntry(
                    corruption_type="occlusion",
                    video_id=video_id,
                    edge_key_tuple=(
                        edge_key.subject_track_id,
                        edge_key.object_track_id,
                        edge_key.predicate_id,
                    ),
                    start_frame=gap_start,
                    end_frame=gap_end,
                    duration=gap_length,
                    original_visibility="VISIBLE",
                    modified_visibility="OCCLUDED_OR_UNRELIABLE",
                ))

    logger.info(
        "Injected %d occlusion gaps across %d edges (seed=%d)",
        len(manifest.entries), len(target_keys), seed,
    )
    return corrupted, manifest
