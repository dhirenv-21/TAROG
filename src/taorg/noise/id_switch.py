"""Deterministic ID-switch injection for controlled experiments.

Swaps track IDs between two tracks for a bounded frame range,
simulating tracker identity switches.

Key properties:
    - Deterministic under seed
    - Produces a CorruptionManifest recording exact ID mappings
    - Bounded to explicit frame ranges
    - Reversible from manifest
    - Does not create impossible scenes (swaps between compatible tracks only)

Corresponds to PRD FR-9.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest

logger = logging.getLogger(__name__)


@dataclass
class IDSwitchConfig:
    """Configuration for ID-switch injection.

    Attributes:
        swap_duration: Duration of each ID swap in frames.
        num_swaps: Number of swaps to inject.
        target_track_pairs: Specific (track_a, track_b) pairs to swap.
            If None, pairs are selected randomly from co-occurring tracks.
        min_pre_swap_frames: Minimum frames before swap starts.
        min_post_swap_frames: Minimum frames after swap ends.
        require_same_class: If True, only swap tracks with the same class_id.
    """

    swap_duration: int = 5
    num_swaps: int = 1
    target_track_pairs: Optional[list[tuple[int, int]]] = None
    min_pre_swap_frames: int = 2
    min_post_swap_frames: int = 2
    require_same_class: bool = False


def inject_id_switch(
    frames: list[FrameGraph],
    seed: int,
    config: IDSwitchConfig | None = None,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Apply deterministic ID-switch corruption to a frame sequence.

    Swaps the track IDs of two tracks for a bounded duration. All
    EdgeObservations referencing the swapped track IDs are updated
    accordingly within the swap range.

    Args:
        frames: Input frame sequence (will NOT be mutated).
        seed: Random seed for deterministic corruption.
        config: Corruption configuration. Uses defaults if None.

    Returns:
        Tuple of (corrupted_frames, manifest).
    """
    if config is None:
        config = IDSwitchConfig()

    rng = random.Random(seed)
    corrupted = deepcopy(frames)
    manifest = CorruptionManifest(
        seed=seed,
        config={
            "type": "id_switch",
            "swap_duration": config.swap_duration,
            "num_swaps": config.num_swaps,
            "require_same_class": config.require_same_class,
            "min_pre_swap_frames": config.min_pre_swap_frames,
            "min_post_swap_frames": config.min_post_swap_frames,
        },
    )

    if not frames:
        return corrupted, manifest

    video_id = frames[0].video_id
    num_frames = len(frames)

    # Build track co-occurrence: which tracks appear in which frames
    track_frames: dict[int, set[int]] = {}
    track_classes: dict[int, int] = {}
    for f_idx, frame in enumerate(frames):
        for track in frame.track_observations:
            track_frames.setdefault(track.track_id, set()).add(f_idx)
            track_classes[track.track_id] = track.class_id

    # Find valid pairs
    if config.target_track_pairs is not None:
        candidate_pairs = config.target_track_pairs
    else:
        all_track_ids = sorted(track_frames.keys())
        candidate_pairs = []
        for i, tid_a in enumerate(all_track_ids):
            for tid_b in all_track_ids[i + 1:]:
                if config.require_same_class:
                    if track_classes.get(tid_a) != track_classes.get(tid_b):
                        continue
                # Both must co-occur for sufficient frames
                co_frames = track_frames[tid_a] & track_frames[tid_b]
                if len(co_frames) >= (
                    config.swap_duration
                    + config.min_pre_swap_frames
                    + config.min_post_swap_frames
                ):
                    candidate_pairs.append((tid_a, tid_b))

    rng.shuffle(candidate_pairs)

    swaps_done = 0
    for track_a, track_b in candidate_pairs:
        if swaps_done >= config.num_swaps:
            break

        co_frames = sorted(track_frames[track_a] & track_frames[track_b])
        if len(co_frames) < (
            config.swap_duration
            + config.min_pre_swap_frames
            + config.min_post_swap_frames
        ):
            continue

        # Find valid start position
        earliest_start = co_frames[config.min_pre_swap_frames]
        latest_start_idx = len(co_frames) - config.swap_duration - config.min_post_swap_frames
        if latest_start_idx < config.min_pre_swap_frames:
            continue
        latest_start = co_frames[latest_start_idx]

        if earliest_start > latest_start:
            continue

        # Pick a random start within the co-occurrence range
        valid_starts = [f for f in co_frames if earliest_start <= f <= latest_start]
        if not valid_starts:
            continue

        swap_start = rng.choice(valid_starts)
        swap_end = swap_start + config.swap_duration - 1

        id_mapping = {track_a: track_b, track_b: track_a}

        # Apply swap to frames
        for f_idx in range(swap_start, min(swap_end + 1, num_frames)):
            frame = corrupted[f_idx]

            # Swap track observations
            new_tracks: list[TrackObservation] = []
            for track in frame.track_observations:
                if track.track_id in id_mapping:
                    new_tracks.append(TrackObservation(
                        video_id=track.video_id,
                        frame_idx=track.frame_idx,
                        timestamp=track.timestamp,
                        track_id=id_mapping[track.track_id],
                        class_id=track.class_id,
                        bbox_xyxy=track.bbox_xyxy,
                        detector_confidence=track.detector_confidence,
                        tracker_state=track.tracker_state,
                        visibility_state=track.visibility_state,
                    ))
                else:
                    new_tracks.append(track)
            frame.track_observations = new_tracks

            # Swap edge observations
            new_edges: list[EdgeObservation] = []
            for obs in frame.edge_observations:
                new_subj = id_mapping.get(
                    obs.edge_key.subject_track_id,
                    obs.edge_key.subject_track_id,
                )
                new_obj = id_mapping.get(
                    obs.edge_key.object_track_id,
                    obs.edge_key.object_track_id,
                )
                new_key = EdgeKey(
                    subject_track_id=new_subj,
                    object_track_id=new_obj,
                    predicate_id=obs.edge_key.predicate_id,
                )
                new_edges.append(EdgeObservation(
                    edge_key=new_key,
                    video_id=obs.video_id,
                    frame_idx=obs.frame_idx,
                    timestamp=obs.timestamp,
                    predicate_score=obs.predicate_score,
                    observed=obs.observed,
                    subject_visibility=obs.subject_visibility,
                    object_visibility=obs.object_visibility,
                    source_model=obs.source_model,
                    source_checkpoint=obs.source_checkpoint,
                ))
            frame.edge_observations = new_edges

        # Record in manifest
        manifest.entries.append(CorruptionEntry(
            corruption_type="id_switch",
            video_id=video_id,
            track_ids=(track_a, track_b),
            start_frame=swap_start,
            end_frame=swap_end,
            duration=config.swap_duration,
            original_visibility="N/A",
            modified_visibility="N/A",
            id_mapping=id_mapping,
        ))

        swaps_done += 1

    logger.info(
        "Injected %d ID switches (seed=%d)",
        len(manifest.entries), seed,
    )
    return corrupted, manifest
