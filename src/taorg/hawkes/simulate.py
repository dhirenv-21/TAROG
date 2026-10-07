"""Synthetic relationship-event simulator (Ogata thinning) for tests and demonstrations.

Two predicates per ordered pair. Predicate 0 turns on/off with constant rates; the ONSET of
predicate 1 is excited by recent ONSETs of predicate 0 with strength ``alpha_cross`` and an
exponential kernel (rate ``beta`` per second). With alpha_cross = 0 the predicates are independent.
SYNTHETIC — says nothing about real videos.
"""

from __future__ import annotations

import math

import numpy as np

from taorg.hawkes.vidor import PairRecord, VideoRecord


def simulate(alpha_cross: float, seed: int, n_videos: int = 60, n_pairs: int = 4, fps: float = 30.0,
             T: int = 1800, beta: float = 1.0) -> list[VideoRecord]:
    """Simulate videos (frames at ``fps``, length T frames)."""
    rng = np.random.default_rng(seed)
    mu_on, mu_off = np.array([0.05, 0.02]), np.array([0.3, 0.3])
    vids = []
    for i in range(n_videos):
        pairs = {}
        for pk in range(n_pairs):
            t, t_end = 0.0, T / fps
            on, start = [False, False], [0, 0]
            hist: list[tuple[float, int]] = []
            rels: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
            while True:
                ub = mu_off.sum() + mu_on.sum() + alpha_cross * (1 + len(hist))
                t += rng.exponential(1 / ub)
                if t >= t_end:
                    break
                exc = alpha_cross * sum(math.exp(-beta * (t - tj)) for tj, c in hist if c == 0)
                lam = [0 if on[0] else mu_on[0], 0 if on[1] else mu_on[1] + exc,
                       mu_off[0] if on[0] else 0, mu_off[1] if on[1] else 0]
                u = rng.uniform(0, ub)
                cum = np.cumsum(lam)
                if u >= cum[-1]:
                    continue
                j = int(np.searchsorted(cum, u, side="right"))
                r = j % 2
                f = int(t * fps) + 1
                if j < 2:
                    on[r], start[r] = True, f
                    hist.append((f / fps, 2 * r))
                else:
                    on[r] = False
                    if f > start[r]:
                        rels[r].append((start[r], f))
            for r in (0, 1):
                if on[r] and T > start[r]:
                    rels[r].append((start[r], T))
            pairs[(pk, pk + 10)] = PairRecord((0, T), {r: iv for r, iv in rels.items() if iv})
        vids.append(VideoRecord(f"synthetic_{seed}_{i}", fps, T, pairs))
    return vids
