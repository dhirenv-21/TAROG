"""Downstream-study outcomes (research/taorg_downstream_preregistration.md §6). Pure data in, numbers out.

Inputs per video:
    out[k]   bool array over frames: the method outputs key k
    lab[k]   dict of bool arrays: alive, takeover, gap, both_out (same length; index 0 may be unused)
Units: frames; seconds via fps.

O1 identity ghost time  IGT  = sum_k sum_t out & takeover / fps
O2 persistence in genuine gaps: pgg_num = sum out & gap, pgg_den = sum gap; FTT = (den - num) / fps
O3 stable termination delay at VISIBLE terminations (STD-g) and false reactivation (FRR-g)
O4 recurrence detection delay (RDD) and J: taorg.metrics.episodes.episode_metrics on alive runs
"""

from __future__ import annotations

from typing import Hashable, Mapping

import numpy as np

from taorg.metrics.episodes import episode_metrics, runs

MIN_DEAD = 10
SUSTAINED = 15


def video_outcomes(out: Mapping[Hashable, np.ndarray], lab: Mapping[Hashable, Mapping[str, np.ndarray]],
                   fps: float, first: int = 1) -> dict[str, float]:
    """All outcomes for one video; arrays are indexed by frame, frames ``first..len-1`` are scored."""
    igt = pnum = pden = 0
    std, std_cens, frr_num, frr_den = [], 0, 0, 0
    gt_ep: dict[Hashable, list[tuple[int, int]]] = {}
    keys = list(lab)
    n = None
    for k in keys:
        L = lab[k]
        o = np.asarray(out[k], bool)[first:]
        alive = np.asarray(L["alive"], bool)[first:]
        n = len(alive)
        igt += int((o & np.asarray(L["takeover"], bool)[first:]).sum())
        g = np.asarray(L["gap"], bool)[first:]
        pnum += int((o & g).sum())
        pden += int(g.sum())
        bo = np.asarray(L["both_out"], bool)[first:]
        ep = runs(alive.tolist())
        gt_ep[k] = ep
        for idx, (a, b) in enumerate(ep):
            if b >= n - 1:
                continue
            d = (ep[idx + 1][0] - 1) if idx + 1 < len(ep) else n - 1
            D = d - b
            if D < MIN_DEAD or not bo[b + 1]:
                continue
            frr_den += 1
            td = next((f for f in range(b + 1, d + 1) if not o[f]), None)
            if td is not None and o[td + 1:d + 1].any():
                frr_num += 1
            tstar = next((f for f in range(b + 1, d + 1) if not o[f:min(f + SUSTAINED - 1, d) + 1].any()), None)
            if tstar is None:
                std.append((D + 1) / fps)
                std_cens += 1
            else:
                std.append((tstar - b) / fps)
    n = n or 0
    outsets = [set() for _ in range(n)]
    for k in keys:
        o = np.asarray(out[k], bool)[first:]
        for f in np.where(o)[0]:
            outsets[f].add(k)
    em = episode_metrics(gt_ep, outsets, fps) if n else {}
    # RDD events = non-bridged recurrences (episode_metrics averages over exactly these)
    rdd_n = em["n_recurrences"] * (1.0 - em["bridge_rate"]) if n and em["n_recurrences"] else 0.0
    rdd_n = float(round(rdd_n))
    rdd_sum = float(em["rdd_mean_s"] * rdd_n) if rdd_n else 0.0
    return {"igt_s": igt / fps, "pgg_num": float(pnum), "pgg_den": float(pden), "ftt_s": (pden - pnum) / fps,
            "std_g_sum": float(sum(std)), "std_g_n": float(len(std)), "std_g_censored": float(std_cens),
            "frr_g_num": float(frr_num), "frr_g_den": float(frr_den),
            "rdd_sum": rdd_sum, "rdd_n": rdd_n,
            "J": em.get("J", np.nan), "frame_f1": em.get("frame_f1", np.nan), "episode_f1": em.get("episode_f1", np.nan),
            "onset_mean_s": em.get("onset_mean_s", np.nan), "n_frames": float(n), "n_keys": float(len(keys))}
