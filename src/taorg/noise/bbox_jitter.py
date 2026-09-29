"""Bounding-box centre jitter — Phase 4 (identity cue is never an oracle).

The synthetic world renders every track with a fixed box, so an ID switch
would appear as a perfectly detectable box jump.  To make identity cues
realistic, this injector adds i.i.d. Gaussian noise to every track's box
centre on every frame:

    c' = c + N(0, (std * diag)^2 I),   diag = box diagonal,

keeping the box size.  It modifies ``track_observations`` only (edge records
are untouched), and is applied in EVERY Phase 4 condition, so a method that
reads boxes pays for false identity alarms everywhere.

Deterministic given the seed (one stream, tracks in id order, frames in order).
"""

from __future__ import annotations

import math
import random

from taorg.noise.manifest import CorruptionManifest
from taorg.schemas.edges import FrameGraph
from taorg.schemas.tracks import TrackObservation


def inject_bbox_jitter(
    frames: list[FrameGraph], seed: int, std: float,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Jitter every track's box centre.

    Args:
        frames: Input frames (not mutated).
        seed: Random seed.
        std: Jitter standard deviation in units of the box diagonal (>= 0).

    Returns:
        (jittered frames, manifest recording the configuration).
    """
    if std < 0:
        raise ValueError(f"std must be >= 0, got {std}")
    rng = random.Random(seed)
    manifest = CorruptionManifest(seed=seed, config={"type": "bbox_jitter", "std": std})
    out: list[FrameGraph] = []
    for frame in frames:
        tracks: list[TrackObservation] = []
        for t in sorted(frame.track_observations, key=lambda t: t.track_id):
            x1, y1, x2, y2 = t.bbox_xyxy
            diag = math.hypot(x2 - x1, y2 - y1)
            dx = rng.gauss(0.0, std * diag) if std > 0 else 0.0
            dy = rng.gauss(0.0, std * diag) if std > 0 else 0.0
            tracks.append(TrackObservation(
                video_id=t.video_id, frame_idx=t.frame_idx, timestamp=t.timestamp,
                track_id=t.track_id, class_id=t.class_id,
                bbox_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                detector_confidence=t.detector_confidence, tracker_state=t.tracker_state,
                visibility_state=t.visibility_state, occlusion_ratio=t.occlusion_ratio))
        out.append(FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx,
                              timestamp=frame.timestamp, track_observations=tracks,
                              edge_observations=list(frame.edge_observations)))
    return out, manifest
