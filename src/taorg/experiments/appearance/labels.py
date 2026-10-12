"""ORACLE labels for the appearance audit — evaluation only (prereg §3.2–§3.4).

Builds on taorg.experiments.observation.oracle (unchanged): Hungarian IoU >= 0.5 matching of
output detection boxes to considered ground-truth pedestrians; g(X) = majority identity.
Adds: the AMBIGUOUS purity rule, incident labels, and the common ground-truth-aligned unit.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from taorg.experiments.observation.oracle import categorise

PURITY_MIN = 2.0 / 3.0
INCIDENT_MAJORITY = 2.0 / 3.0
GAP_CATS = ("OCCLUDED_GAP", "VISIBLE_MISS")


def purity(matches: dict[int, dict[int, int]]) -> dict[int, float]:
    """Share of each track's matched frames that carry its majority ground-truth identity."""
    c: dict[int, Counter] = {}
    for m in matches.values():
        for t, g in m.items():
            c.setdefault(t, Counter())[g] += 1
    return {t: cnt.most_common(1)[0][1] / sum(cnt.values()) for t, cnt in c.items()}


def frame_label(x: int, frame: int, g: dict[int, int], pur: dict[int, float], matches: dict, gt: dict) -> str:
    """GAP / TAKEOVER / GONE / FALSE_TRACK / AMBIGUOUS for track x's missing record at frame."""
    cat = categorise(x, frame, g, matches, gt)
    if cat == "FALSE_TRACK":
        return cat
    if pur.get(x, 0.0) < PURITY_MIN:
        return "AMBIGUOUS"
    return "GAP" if cat in GAP_CATS else cat


def incidents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group missing-record rows (sorted by frame per track) into maximal consecutive runs.

    Each row needs 'unit', 'track', 'frame', 'label'. Returns incidents with 'rows' (indices),
    'label' (TAKEOVER / GAP / AMBIGUOUS / OTHER) and 'length'.
    """
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["unit"], rows[i]["track"], rows[i]["frame"]))
    out: list[dict[str, Any]] = []
    cur: list[int] = []

    def close() -> None:
        if not cur:
            return
        labs = Counter(rows[i]["label"] for i in cur)
        n = len(cur)
        k = labs["TAKEOVER"] + labs["GAP"]
        if k < 0.5 * n:
            lab = "OTHER"
        elif labs["TAKEOVER"] >= INCIDENT_MAJORITY * k:
            lab = "TAKEOVER"
        elif labs["GAP"] >= INCIDENT_MAJORITY * k:
            lab = "GAP"
        else:
            lab = "AMBIGUOUS"
        r0 = rows[cur[0]]
        out.append({"unit": r0["unit"], "track": r0["track"], "start": r0["frame"], "length": n,
                    "label": lab, "rows": list(cur)})

    for i in order:
        if cur:
            p = rows[cur[-1]]
            r = rows[i]
            if not (r["unit"] == p["unit"] and r["track"] == p["track"] and r["frame"] == p["frame"] + 1):
                close()
                cur = []
        cur.append(i)
    close()
    return out


def common_unit(matches: dict[int, dict[int, int]], gt: dict[int, dict]) -> dict[str, int]:
    """Ground-truth identity-frame states and ground-truth gap resolutions for one tracker (§3.4)."""
    inv = {f: {g: t for t, g in m.items()} for f, m in matches.items()}
    last_track: dict[int, int] = {}
    gap_open: dict[int, int] = {}
    c: Counter = Counter()
    for f in sorted(gt):
        for gid in gt[f]:
            t = inv.get(f, {}).get(gid)
            if gid not in last_track:
                if t is not None:
                    last_track[gid] = t
                continue
            if t is None:
                c["UNCOVERED"] += 1
                gap_open[gid] = gap_open.get(gid, 0) + 1
                continue
            if t == last_track[gid]:
                c["CONTINUED"] += 1
            else:
                c["SWITCHED"] += 1
            if gid in gap_open:
                c["GAP_BRIDGED" if t == last_track[gid] else "GAP_FRAGMENTED"] += 1
                del gap_open[gid]
            last_track[gid] = t
    c["GAP_ENDED"] += len(gap_open)
    return dict(c)
