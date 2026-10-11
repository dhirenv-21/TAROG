"""Episode-level metrics (research/taorg_episode_preregistration.md §6, items 1-11).

Registered before any comparison.  The historical (frozen) metrics in
``research/metric_definitions.md`` are NOT changed; they are computed alongside
by the pipeline with the original implementations.

Inputs are pure data:
    gt:   key -> sorted, disjoint inclusive GT episodes [(a, b), ...]
    out:  per-frame set of output keys (len = number of frames)
Units: frames, converted to seconds with ``fps`` where stated.
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

NAN = float("nan")


def runs(on: Sequence[bool]) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive (start, end) intervals."""
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(on) - 1))
    return out


def iou(g: tuple[int, int], o: tuple[int, int]) -> float:
    """Temporal IoU of two inclusive frame intervals."""
    inter = min(g[1], o[1]) - max(g[0], o[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / (max(g[1], o[1]) - min(g[0], o[0]) + 1)


def match_episodes(G: Sequence[tuple[int, int]], O: Sequence[tuple[int, int]], thr: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching with IoU >= thr (descending IoU; ties: earlier g, then earlier o)."""
    cand = sorted(((iou(g, o), i, j) for i, g in enumerate(G) for j, o in enumerate(O) if iou(g, o) >= thr),
                  key=lambda x: (-x[0], x[1], x[2]))
    used_g, used_o, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_g and j not in used_o:
            used_g.add(i)
            used_o.add(j)
            pairs.append((i, j))
    return pairs


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else NAN


def episode_metrics(gt: Mapping[Hashable, Sequence[tuple[int, int]]], out: Sequence[set],
                    fps: float, *, iou_thr: float = 0.5, coverage_thr: float = 0.5,
                    min_dead: int = 10, sustained: int = 15) -> dict[str, float]:
    """All episode-study metrics of one video (definitions: prereg §6)."""
    n = len(out)
    gt_keys = set(gt)
    out_keys = set().union(*out) if out else set()
    keys = sorted(gt_keys | out_keys, key=lambda k: tuple(k) if isinstance(k, tuple) else (k,))
    on = {k: [k in out[f] for f in range(n)] for k in keys}
    alive = {}
    for k in keys:
        a = [False] * n
        for s, e in gt.get(k, ()):
            for f in range(s, e + 1):
                a[f] = True
        alive[k] = a

    # 1. frame-level
    tp = fp = fn = 0
    fpt_frames = ghost_frames = 0
    for k in keys:
        for f in range(n):
            o, g = on[k][f], alive[k][f]
            tp += o and g
            fp += o and not g
            fn += g and not o
            if o and not g:
                if k in gt_keys:
                    fpt_frames += 1
                else:
                    ghost_frames += 1
    fp_ = tp / (tp + fp) if tp + fp else 0.0
    fr_ = tp / (tp + fn) if tp + fn else 0.0
    frame_f1 = _f1(fp_, fr_)

    # 2. episode matching
    n_gt = n_out = n_match = 0
    missed = 0
    frag: list[float] = []
    frr_num = frr_den = 0
    react_frames = 0
    std: list[float] = []
    std_cens = 0
    rdd: list[float] = []
    rdd_cens = 0
    n_rec = bridged = 0
    onset: list[float] = []
    onset_cens = 0
    for k in keys:
        G = list(gt.get(k, ()))
        O = runs(on[k])
        n_gt += len(G)
        n_out += len(O)
        n_match += len(match_episodes(G, O, iou_thr))
        for idx, (a, b) in enumerate(G):
            cov = sum(on[k][a:b + 1]) / (b - a + 1)
            if cov < coverage_thr:
                missed += 1
            else:
                frag.append(sum(1 for (x, y) in O if not (y < a or x > b)) - 1)
            # onset / recurrence delay
            first = next((f for f in range(a, b + 1) if on[k][f]), None)
            if idx == 0:
                if first is None:
                    onset.append((b - a + 1) / fps)
                    onset_cens += 1
                else:
                    onset.append((first - a) / fps)
            else:
                n_rec += 1
                c = G[idx - 1][1] + 1
                if a > c and all(on[k][c:a]):
                    bridged += 1
                elif first is None:
                    rdd.append((b - a + 1) / fps)
                    rdd_cens += 1
                else:
                    rdd.append((first - a) / fps)
            # terminations
            if b >= n - 1:
                continue
            d = (G[idx + 1][0] - 1) if idx + 1 < len(G) else n - 1
            D = d - b
            if D < min_dead:
                continue
            frr_den += 1
            td = next((f for f in range(b + 1, d + 1) if not on[k][f]), None)
            if td is not None:
                extra = sum(on[k][td + 1:d + 1])
                react_frames += extra
                frr_num += extra > 0
            tstar = next((f for f in range(b + 1, d + 1)
                          if not any(on[k][f:min(f + sustained - 1, d) + 1])), None)
            if tstar is None:
                std.append((D + 1) / fps)
                std_cens += 1
            else:
                std.append((tstar - b) / fps)
    ep_p = n_match / n_out if n_out else 0.0
    ep_r = n_match / n_gt if n_gt else 0.0
    ep_f1 = _f1(ep_p, ep_r)
    return {
        "frame_precision": fp_, "frame_recall": fr_, "frame_f1": frame_f1,
        "episode_precision": ep_p, "episode_recall": ep_r, "episode_f1": ep_f1,
        "J": 0.5 * frame_f1 + 0.5 * ep_f1,
        "n_gt_episodes": float(n_gt), "n_output_runs": float(n_out),
        "missed_rate": missed / n_gt if n_gt else NAN,
        "fragmentation": _mean(frag),
        "frr": frr_num / frr_den if frr_den else NAN,
        "n_terminations": float(frr_den), "reactivation_frames": float(react_frames),
        "fpt_s": fpt_frames / fps, "ghost_time_s": ghost_frames / fps,
        "std_mean_s": _mean(std), "std_censored": float(std_cens),
        "rdd_mean_s": _mean(rdd), "rdd_censored": float(rdd_cens),
        "n_recurrences": float(n_rec), "bridge_rate": bridged / n_rec if n_rec else NAN,
        "onset_mean_s": _mean(onset), "onset_censored": float(onset_cens),
    }


def is_nan(x: float) -> bool:
    """True if x is a float NaN."""
    return isinstance(x, float) and math.isnan(x)
