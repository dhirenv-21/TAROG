"""Synthetic record streams for the Phase 5 eager-vs-lazy harness and benchmarks.

Each candidate key (s, o, r) follows a latent ON/OFF process with occlusion periods; keys can
retire permanently (stale keys accumulate, the regime the lazy filter targets). Records:
detections when ON and visible (with misses), VISIBLE absence records, OCCLUDED/UNKNOWN absence
records, or no record; false positives; optional frame-interval jitter and identity box jumps.
All randomness goes through one ``numpy.random.Generator`` seeded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState

V = VisibilityState


@dataclass(frozen=True)
class TraceConfig:
    """Generator parameters (per-frame probabilities unless stated)."""

    n_frames: int = 600
    n_keys: int = 200            # candidate keys over the video
    n_tracks: int = 30
    n_predicates: int = 6
    fps: float = 30.0
    p_birth: float = 0.002       # a dormant key starts its life
    p_on: float = 0.05           # OFF -> ON while alive
    p_off: float = 0.03          # ON -> OFF
    p_retire: float = 0.004      # alive -> retired forever
    p_occ: float = 0.02          # visible -> occluded
    p_unocc: float = 0.1         # occluded -> visible
    miss: float = 0.1
    false_pos: float = 0.002
    p_vis_record: float = 0.3    # an OFF, visible, alive key gets a VISIBLE absence record
    p_occ_record: float = 0.5    # an occluded key gets an OCCLUDED (else UNKNOWN 1/2, else none)
    p_jitter: float = 0.0        # this frame's interval differs from 1/fps
    p_jump: float = 0.0          # a track's box jumps (identity evidence)


def make_trace(cfg: TraceConfig, rng: np.random.Generator) -> list[FrameGraph]:
    """A list of FrameGraphs (frame_idx 0..n-1) following ``cfg``."""
    keys = []
    for _ in range(cfg.n_keys):
        s, o = rng.choice(cfg.n_tracks, 2, replace=False)
        keys.append(EdgeKey(int(s), int(o), int(rng.integers(cfg.n_predicates))))
    keys = list(dict.fromkeys(keys))
    n = len(keys)
    alive = np.zeros(n, bool)
    retired = np.zeros(n, bool)
    on = np.zeros(n, bool)
    occ = np.zeros(n, bool)
    boxes = {t: np.array([rng.uniform(0, 500), rng.uniform(0, 500), 40.0, 80.0]) for t in range(cfg.n_tracks)}
    frames, t = [], 0.0
    for f in range(cfg.n_frames):
        if f:
            t += (1.0 / cfg.fps) * (rng.uniform(0.5, 2.0) if rng.random() < cfg.p_jitter else 1.0)
        born = (~alive) & (~retired) & (rng.random(n) < cfg.p_birth)
        alive |= born
        on[born] = True
        ret = alive & (rng.random(n) < cfg.p_retire)
        alive &= ~ret
        retired |= ret
        on &= alive
        flip_on = alive & ~on & (rng.random(n) < cfg.p_on)
        flip_off = on & (rng.random(n) < cfg.p_off)
        on = (on | flip_on) & ~flip_off
        occ = np.where(occ, rng.random(n) >= cfg.p_unocc, rng.random(n) < cfg.p_occ) & alive
        edges = []
        u = rng.random(n)
        for i in range(n):
            k = keys[i]
            if alive[i] and on[i] and not occ[i] and u[i] >= cfg.miss:
                edges.append(EdgeObservation(k, "syn", f, t, float(rng.uniform(0.5, 1.0)), True))
            elif alive[i] and occ[i]:
                r = rng.random()
                if r < cfg.p_occ_record:
                    edges.append(EdgeObservation(k, "syn", f, t, 0.0, False, V.VISIBLE, V.OCCLUDED_OR_UNRELIABLE))
                elif r < cfg.p_occ_record + (1 - cfg.p_occ_record) / 2:
                    edges.append(EdgeObservation(k, "syn", f, t, 0.0, False, V.VISIBLE, V.UNKNOWN))
            elif alive[i] and not on[i] and rng.random() < cfg.p_vis_record:
                edges.append(EdgeObservation(k, "syn", f, t, 0.0, False))
            elif rng.random() < cfg.false_pos:
                edges.append(EdgeObservation(k, "syn", f, t, float(rng.uniform(0.5, 1.0)), True))
        tracks = []
        for tid, b in boxes.items():
            if cfg.p_jump and rng.random() < cfg.p_jump:
                b[:2] = rng.uniform(0, 500, 2)
            else:
                b[:2] += rng.normal(0, 1.0, 2)
            tracks.append(TrackObservation("syn", f, t, tid, 0, (b[0], b[1], b[0] + b[2], b[1] + b[3])))
        frames.append(FrameGraph("syn", f, t, tracks, edges))
    return frames
