"""Visibility misclassification and timestamp jitter for robustness studies.

Both act on the observation stream only (never on ground truth):
    flip_visibility       with probability p per (pair, frame) the pair's object
                          visibility label is swapped VISIBLE <-> OCCLUDED_OR_UNRELIABLE
                          (UNKNOWN is left alone); detections are unchanged.
    jitter_timestamps     replaces frame intervals by dt_n = base_dt * U(1 - j, 1 + j),
                          keeping frame order and frame indices (irregular sampling).
"""

from __future__ import annotations

import random
from dataclasses import replace

from taorg.schemas.edges import FrameGraph
from taorg.schemas.tracks import VisibilityState

VIS, OCC = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE


def flip_visibility(frames: list[FrameGraph], prob: float, seed: int) -> list[FrameGraph]:
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    rng = random.Random(seed)
    out = []
    for f in frames:
        flip = {}
        edges = []
        for o in f.edge_observations:
            if o.pair_key not in flip:
                flip[o.pair_key] = rng.random() < prob
            vis = o.object_visibility
            if flip[o.pair_key] and vis in (VIS, OCC):
                vis = OCC if vis == VIS else VIS
            edges.append(replace(o, object_visibility=vis))
        out.append(FrameGraph(f.video_id, f.frame_idx, f.timestamp,
                              track_observations=list(f.track_observations), edge_observations=edges))
    return out


def jitter_timestamps(frames: list[FrameGraph], jitter: float, seed: int,
                      fps: float) -> list[FrameGraph]:
    if not 0.0 <= jitter < 1.0:
        raise ValueError(f"jitter must be in [0, 1), got {jitter}")
    rng = random.Random(seed)
    out, ts = [], 0.0
    for i, f in enumerate(frames):
        if i:
            ts += (1.0 / fps) * rng.uniform(1.0 - jitter, 1.0 + jitter)
        out.append(FrameGraph(
            f.video_id, f.frame_idx, ts,
            track_observations=[replace(t, timestamp=ts) for t in f.track_observations],
            edge_observations=[replace(o, timestamp=ts) for o in f.edge_observations]))
    return out
