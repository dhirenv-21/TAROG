"""Temporally correlated (bursty) visibility-flag noise — Phase 4.

Phase 3 visibility noise flipped absence-record flags i.i.d. per frame.  Real
visibility estimators fail in *runs* (a tracker coasting through an occluder,
a detector losing an object for several frames).  This injector models the
error process per edge as a two-state Markov chain over frames:

    OK  -> ERR with probability p_enter per frame
    ERR -> OK  with probability p_exit  per frame

with ``p_exit = 1 / mean_burst_frames`` and ``p_enter`` chosen so that the
stationary error probability equals ``error_rate``:
``p_enter = error_rate * p_exit / (1 - error_rate)``.  The initial state is
drawn from the stationary distribution.  The chain evolves on every frame; it
only *acts* on absence records (``observed=False``): while in ERR, a VISIBLE
flag becomes OCCLUDED_OR_UNRELIABLE and an OCCLUDED flag becomes VISIBLE.
UNKNOWN flags and observed records are never modified.

Deterministic given the seed; one independent stream per edge key (sorted).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BurstyVisibilityConfig:
    """Configuration of bursty flag noise.

    Attributes:
        error_rate: Stationary probability that the flag channel is in error.
        mean_burst_frames: Mean length of an error run (frames).
    """

    error_rate: float = 0.2
    mean_burst_frames: float = 10.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.error_rate < 1.0):
            raise ValueError(f"error_rate must be in [0, 1), got {self.error_rate}")
        if self.mean_burst_frames < 1.0:
            raise ValueError(f"mean_burst_frames must be >= 1, got {self.mean_burst_frames}")

    @property
    def p_exit(self) -> float:
        """Per-frame probability of leaving the error state."""
        return 1.0 / self.mean_burst_frames

    @property
    def p_enter(self) -> float:
        """Per-frame probability of entering the error state."""
        return self.error_rate * self.p_exit / (1.0 - self.error_rate)


def inject_bursty_visibility(
    frames: list[FrameGraph], seed: int, config: BurstyVisibilityConfig,
) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Apply bursty flag errors to absence records.

    Args:
        frames: Input frames (not mutated).
        seed: Random seed.
        config: Noise configuration.

    Returns:
        (corrupted frames, manifest with one entry per contiguous flipped run).
    """
    V = VisibilityState
    keys = sorted({o.edge_key for f in frames for o in f.edge_observations})
    rngs = {k: random.Random(f"{seed}|{k.subject_track_id}|{k.object_track_id}|{k.predicate_id}")
            for k in keys}
    state = {k: rngs[k].random() < config.error_rate for k in keys}
    manifest = CorruptionManifest(seed=seed, config={
        "type": "bursty_visibility", "error_rate": config.error_rate,
        "mean_burst_frames": config.mean_burst_frames})
    flipped: dict[tuple[EdgeKey, str, str], list[int]] = {}
    out: list[FrameGraph] = []
    for fi, frame in enumerate(frames):
        if fi > 0:
            for k in keys:
                u = rngs[k].random()
                state[k] = (u >= config.p_exit) if state[k] else (u < config.p_enter)
        new_edges: list[EdgeObservation] = []
        for obs in frame.edge_observations:
            vis = obs.edge_visibility
            if obs.observed or not state.get(obs.edge_key, False) or vis == V.UNKNOWN:
                new_edges.append(obs)
                continue
            if vis == V.VISIBLE:
                subj, objv = obs.subject_visibility, V.OCCLUDED_OR_UNRELIABLE
            else:
                subj, objv = V.VISIBLE, V.VISIBLE
            new = EdgeObservation(
                edge_key=obs.edge_key, video_id=obs.video_id, frame_idx=obs.frame_idx,
                timestamp=obs.timestamp, predicate_score=obs.predicate_score, observed=False,
                subject_visibility=subj, object_visibility=objv,
                source_model=obs.source_model, source_checkpoint=obs.source_checkpoint)
            new_edges.append(new)
            flipped.setdefault((obs.edge_key, vis.name, new.edge_visibility.name), []).append(frame.frame_idx)
        out.append(FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx,
                              timestamp=frame.timestamp,
                              track_observations=list(frame.track_observations),
                              edge_observations=new_edges))
    video_id = frames[0].video_id if frames else ""
    for (key, a, b) in sorted(flipped):
        idx = flipped[(key, a, b)]
        start = prev = idx[0]
        for f in idx[1:] + [None]:
            if f is not None and f == prev + 1:
                prev = f
                continue
            manifest.entries.append(CorruptionEntry(
                corruption_type="bursty_visibility", video_id=video_id,
                edge_key_tuple=(key.subject_track_id, key.object_track_id, key.predicate_id),
                start_frame=start, end_frame=prev, duration=prev - start + 1,
                original_visibility=a, modified_visibility=b))
            if f is not None:
                start = prev = f
    logger.debug("Bursty visibility noise: %d runs (seed=%d)", len(manifest.entries), seed)
    return out, manifest
