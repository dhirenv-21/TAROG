"""ORACLE (ground truth) for the observation audit — evaluation and labelled diagnostics ONLY.

Reads the MOT17 benchmark's manual annotations (gt/gt.txt; considered class-1 pedestrians)
and assigns each absence frame of each tracker track one category (audit §5):
    FALSE_TRACK, GONE, TAKEOVER, OCCLUDED_GAP, VISIBLE_MISS
Hold is correct iff the category is OCCLUDED_GAP or VISIBLE_MISS.
Nothing here may be passed to an inference-time method (except the labelled oracle variant).
"""

from __future__ import annotations

import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from taorg.adapters.mot17 import FrameTracks, iou

HOLD = ("OCCLUDED_GAP", "VISIBLE_MISS")
DROP = ("FALSE_TRACK", "GONE", "TAKEOVER")
VIS_THRESHOLD = 0.5
MATCH_IOU = 0.5


def load_gt(zip_path: Path, seq: str, length: int) -> dict[int, dict[int, tuple[tuple, float]]]:
    """frame -> {gt_id: (xyxy box, visibility)} for considered class-1 pedestrians."""
    with zipfile.ZipFile(zip_path) as z:
        a = np.loadtxt(io.BytesIO(z.read(f"train/MOT17-{seq}-FRCNN/gt/gt.txt")), delimiter=",", ndmin=2)
    a = a[(a[:, 6] == 1) & (a[:, 7] == 1)]
    gt: dict[int, dict[int, tuple[tuple, float]]] = {f: {} for f in range(1, length + 1)}
    for r in a:
        f, i = int(r[0]), int(r[1])
        if 1 <= f <= length:
            gt[f][i] = ((r[2], r[3], r[2] + r[4], r[3] + r[5]), float(r[8]))
    return gt


def match(frames: list[FrameTracks], gt: dict) -> dict[int, dict[int, int]]:
    """frame -> {tracker_id: gt_id} by Hungarian assignment on IoU >= MATCH_IOU."""
    out: dict[int, dict[int, int]] = {}
    for fr in frames:
        tids = list(fr.outputs)
        gids = list(gt[fr.frame])
        m: dict[int, int] = {}
        if tids and gids:
            M = np.array([[iou(fr.outputs[t][0], gt[fr.frame][g][0]) for g in gids] for t in tids])
            rows, cols = linear_sum_assignment(-M)
            for r, c in zip(rows, cols):
                if M[r, c] >= MATCH_IOU:
                    m[tids[r]] = gids[c]
        out[fr.frame] = m
    return out


def identity_of(matches: dict[int, dict[int, int]]) -> dict[int, int]:
    """g(X): majority ground-truth identity of each tracker track (absent => FALSE_TRACK)."""
    c: dict[int, Counter] = {}
    for m in matches.values():
        for t, g in m.items():
            c.setdefault(t, Counter())[g] += 1
    return {t: cnt.most_common(1)[0][0] for t, cnt in c.items()}


def categorise(x: int, frame: int, g: dict[int, int], matches: dict, gt: dict) -> str:
    """Oracle category of track x's absence at `frame` (audit §5)."""
    if x not in g:
        return "FALSE_TRACK"
    gid = g[x]
    if gid not in gt[frame]:
        return "GONE"
    if any(v == gid and t != x for t, v in matches[frame].items()):
        return "TAKEOVER"
    return "OCCLUDED_GAP" if gt[frame][gid][1] < VIS_THRESHOLD else "VISIBLE_MISS"
