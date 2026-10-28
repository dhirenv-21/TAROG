"""Parameter-recovery (identifiability) study: simulate from V-HPED, refit, compare.

The generative process is the model's own discrete-time process (no
misspecification), so this tests identifiability and the trainer, not realism:

    regime per pair: 2-state Markov chain VIS <-> OCC (switch prob per frame)
    alive per (pair, predicate): starts alive; alive -> dead w.p. 1 - exp(-h dt)
    detection: o ~ Bernoulli(1 - exp(-rho_v * lambda dt)) if alive,
                   Bernoulli(1 - exp(-rho_v * eps dt))     if dead
    traces / lambda / h exactly as dynamics.py (marks = 1)

Records for every predicate of every pair are emitted each frame, as the
synthetic benchmark does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.vhped import TrainerConfig, VHPEDParams, fit_vhped
from taorg.temporal.vhped import dynamics

VIS, OCC = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE


@dataclass
class SimConfig:
    num_pairs: int = 40
    num_frames: int = 150
    fps: float = 30.0
    p_vis_to_occ: float = 0.02
    p_occ_to_vis: float = 0.15
    force_first_detection: bool = False  # detect every relation at frame 0 (no pre-birth gap)


def simulate(params: VHPEDParams, cfg: SimConfig, seed: int, video_id: str) -> list[FrameGraph]:
    if params.use_marks:
        raise ValueError("simulate() emits unit marks; use params with use_marks=False")
    rng = random.Random(seed)
    num_p = params.num_predicates
    dt = 1.0 / cfg.fps
    traces = [np.zeros(num_p) for _ in range(cfg.num_pairs)]
    alive = [[True] * num_p for _ in range(cfg.num_pairs)]
    regime = [VIS] * cfg.num_pairs
    frames = []
    for t in range(cfg.num_frames):
        edges = []
        for n in range(cfg.num_pairs):
            if t:
                switch = cfg.p_vis_to_occ if regime[n] == VIS else cfg.p_occ_to_vis
                if rng.random() < switch:
                    regime[n] = OCC if regime[n] == VIS else VIS
            v_idx = 0 if regime[n] == VIS else 1
            traces[n] = dynamics.decay_traces(traces[n], float(params.beta[v_idx]), dt)
            lam = dynamics.intensity(traces[n], params)
            haz = dynamics.hazard(traces[n], params)
            rho = float(params.rho[v_idx])
            marks = np.zeros(num_p)
            for p in range(num_p):
                if t and alive[n][p] and rng.random() < -math.expm1(-haz[p] * dt):
                    alive[n][p] = False
                rate = lam[p] if alive[n][p] else params.eps_rate
                seen = rng.random() < -math.expm1(-rho * rate * dt)
                if t == 0 and cfg.force_first_detection:
                    seen = True
                marks[p] = 1.0 if seen else 0.0
                edges.append(EdgeObservation(
                    edge_key=EdgeKey(n + 1, 1000 + n, p), video_id=video_id, frame_idx=t,
                    timestamp=t * dt, predicate_score=1.0 if seen else 0.0, observed=seen,
                    subject_visibility=VIS, object_visibility=regime[n]))
            traces[n] = traces[n] + marks
        frames.append(FrameGraph(video_id, t, t * dt, edge_observations=edges))
    return frames


RECOVERED = ("mu", "alpha_diag", "eta", "beta_vis", "beta_occ", "rho_occ", "eps_rate")


def _summary(p: VHPEDParams) -> dict[str, float]:
    return {"mu": float(p.mu.mean()), "alpha_diag": float(np.diag(p.alpha).mean()),
            "eta": float(p.eta.mean()), "beta_vis": float(p.beta[0]),
            "beta_occ": float(p.beta[1]), "rho_occ": float(p.rho[1]),
            "eps_rate": float(p.eps_rate)}


def recovery_run(true: VHPEDParams, init: VHPEDParams, sim: SimConfig, n_videos: int,
                 seed: int, trainer: TrainerConfig) -> dict[str, Any]:
    videos = [simulate(true, sim, seed * 1000 + i, f"sim{seed}_{i}") for i in range(n_videos)]
    fit = fit_vhped(videos, init, TrainerConfig(**{**trainer.__dict__, "seed": seed}),
                    fps=sim.fps)
    t, f, s = _summary(true), _summary(fit.params), _summary(init)
    return {"seed": seed, "true": t, "fitted": f, "init": s,
            "abs_error": {k: abs(f[k] - t[k]) for k in RECOVERED},
            "init_abs_error": {k: abs(s[k] - t[k]) for k in RECOVERED},
            "nll": [fit.nll_history[0], fit.nll_history[-1]]}
