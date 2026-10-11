"""Phase 5H-1 Stage-0-generated hypotheses H-a..H-d (research/phase5h1_preregistration.md §8).

Model-free statistic for (source predicate s, target predicate g) on the same ordered pair:

    RR = [E_in / X_in] / [(E_tot - E_in) / (X_tot - X_in)]

X_tot = total on-time of g (all on-intervals, including censored ones; seconds);
W     = union over observed ONSET(s) times t_s of (t_s, t_s + window];
X_in  = |on-time of g  ∩  W|;
E_tot = observed OFFSET(g) events (end frame < T); E_in = those with time in W.
"""

from __future__ import annotations

import numpy as np

from taorg.hawkes.vidor import VideoRecord


def _union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def video_counts(v: VideoRecord, s: int, g: int, window: float) -> np.ndarray:
    """(E_in, X_in, E_tot, X_tot) of one video."""
    f = v.fps
    out = np.zeros(4)
    for pr in v.pairs.values():
        if g not in pr.rels:
            continue
        on = [(b / f, e / f) for b, e in pr.rels[g]]
        offs = [e / f for b, e in pr.rels[g] if e < v.T]
        W = _union([(b / f, b / f + window) for b, e in pr.rels.get(s, []) if b > 0])
        x_in = 0.0
        for a, b in on:
            for wa, wb in W:
                x_in += max(0.0, min(b, wb) - max(a, wa))
        e_in = sum(1 for t in offs if any(wa < t <= wb for wa, wb in W))
        out += (e_in, x_in, len(offs), sum(b - a for a, b in on))
    return out


def counts(videos: list[VideoRecord], s: int, g: int, window: float) -> np.ndarray:
    """(n_videos, 4) per-video counts."""
    return np.array([video_counts(v, s, g, window) for v in videos]).reshape(len(videos), 4)


def log_rr(c: np.ndarray) -> float:
    """log rate ratio from summed counts (E_in, X_in, E_tot, X_tot); nan if undefined."""
    e_in, x_in, e_tot, x_tot = c
    e_out, x_out = e_tot - e_in, x_tot - x_in
    if x_in <= 0 or x_out <= 0 or e_out <= 0:
        return float("nan")
    if e_in <= 0:
        return float("-inf")
    return float(np.log((e_in / x_in) / (e_out / x_out)))
