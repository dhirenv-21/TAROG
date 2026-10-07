"""VidOR annotation loading and Stage 0 event extraction.

Implements the frozen rules of research/phase5h_stage0_preregistration.md §3.
All times are in seconds (frame / fps). Intervals are half-open [b, e) in frames.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Iterable

Interval = tuple[int, int]


@dataclass
class PairRecord:
    """One ordered pair (u, v) in one video."""

    window: Interval                        # [w0, w1) frames, hull of co-existence and relations
    rels: dict[int, list[Interval]] = field(default_factory=dict)   # predicate -> sorted intervals


@dataclass
class VideoRecord:
    """A video reduced to what Stage 0 needs."""

    vid: str
    fps: float
    T: int                                  # frame_count
    pairs: dict[tuple[int, int], PairRecord]
    cats: dict[int, str] = field(default_factory=dict)   # track id -> object category (Phase 5H-1)


def load_videos(root: str, split_dir: str, vocab: list[str] | None = None
                ) -> tuple[list[VideoRecord], list[str], dict[str, int]]:
    """Load one VidOR split.

    Args:
        root: data root (e.g. data/vidor).
        split_dir: 'training' or 'validation'.
        vocab: predicate vocabulary (sorted names); derived from this split if None.

    Returns:
        (videos sorted by id, vocabulary, stats). Relations with predicates outside
        the vocabulary are dropped and counted.
    """
    files = sorted(glob.glob(os.path.join(root, split_dir, "**", "*.json"), recursive=True))
    raw = [json.load(open(f, encoding="utf-8")) for f in files]
    if vocab is None:
        vocab = sorted({r["predicate"] for d in raw for r in d["relation_instances"]})
    index = {p: i for i, p in enumerate(vocab)}
    stats = {"videos": len(raw), "relations": 0, "dropped_unknown_predicate": 0,
             "relations_outside_coexistence": 0, "pairs": 0}
    out: list[VideoRecord] = []
    for d in raw:
        T = int(d["frame_count"])
        span: dict[int, list[int]] = {}
        for t, objs in enumerate(d["trajectories"]):
            for o in objs:
                s = span.setdefault(o["tid"], [t, t])
                s[1] = t
        rels: dict[tuple[int, int], dict[int, list[Interval]]] = {}
        for r in d["relation_instances"]:
            if r["predicate"] not in index:
                stats["dropped_unknown_predicate"] += 1
                continue
            stats["relations"] += 1
            b, e = int(r["begin_fid"]), min(int(r["end_fid"]), T)
            if e <= b:
                continue
            rels.setdefault((r["subject_tid"], r["object_tid"]), {}).setdefault(index[r["predicate"]], []).append((b, e))
        pairs: dict[tuple[int, int], PairRecord] = {}
        tids = sorted(span)
        for u in tids:
            for v in tids:
                if u == v:
                    continue
                lo, hi = max(span[u][0], span[v][0]), min(span[u][1], span[v][1]) + 1
                pr = rels.get((u, v), {})
                if hi <= lo and not pr:
                    continue
                w0, w1 = (lo, hi) if hi > lo else (T, 0)
                for ivs in pr.values():
                    for b, e in ivs:
                        if hi <= lo or b < lo or e > hi:
                            stats["relations_outside_coexistence"] += 1
                        w0, w1 = min(w0, b), max(w1, e)
                pairs[(u, v)] = PairRecord((w0, w1), {k: sorted(x) for k, x in pr.items()})
        for (u, v), pr in rels.items():             # relations on tracks without trajectories
            if (u, v) not in pairs:
                ivs = [iv for x in pr.values() for iv in x]
                pairs[(u, v)] = PairRecord((min(b for b, _ in ivs), max(e for _, e in ivs)),
                                           {k: sorted(x) for k, x in pr.items()})
        stats["pairs"] += len(pairs)
        cats = {int(o["tid"]): str(o["category"]) for o in d.get("subject/objects", [])}
        out.append(VideoRecord(str(d["video_id"]), float(d["fps"]), T, pairs, cats))
    out.sort(key=lambda v: v.vid)
    return out, vocab, stats


def circular_shift(videos: Iterable[VideoRecord], seed: int) -> list[VideoRecord]:
    """Null control (prereg §6): shift each (pair, predicate) interval set by one offset mod T.

    Wrapping intervals are split in two; the pair window is re-derived as the hull of the
    original co-existence window and the shifted intervals.
    """
    rng = random.Random(seed)
    out = []
    for v in videos:
        T = v.T
        pairs = {}
        for key in sorted(v.pairs):
            pr = v.pairs[key]
            new: dict[int, list[Interval]] = {}
            for r in sorted(pr.rels):
                d = rng.randrange(T)
                ivs = []
                for b, e in pr.rels[r]:
                    nb, ne = b + d, e + d
                    if nb >= T:
                        ivs.append((nb - T, ne - T))
                    elif ne > T:
                        ivs.extend([(nb, T), (0, ne - T)])
                    else:
                        ivs.append((nb, ne))
                ivs.sort()
                merged: list[list[int]] = []
                for b, e in ivs:                    # merge pieces that became adjacent/overlapping
                    if merged and b <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([b, e])
                new[r] = [(b, e) for b, e in merged]
            w0, w1 = pr.window
            for ivs in new.values():
                for b, e in ivs:
                    w0, w1 = min(w0, b), max(w1, e)
            pairs[key] = PairRecord((w0, w1), new)
        out.append(VideoRecord(v.vid, v.fps, T, pairs, dict(v.cats)))
    return out


@dataclass
class PairStream:
    """Events and risk sets of one pair (seconds). Types: 2r = ONSET(r), 2r+1 = OFFSET(r)."""

    events: list[tuple[float, int]]          # (time, type); includes PAIR_BIRTH as type = K
    window: tuple[float, float]
    on_risk: dict[int, list[tuple[float, float]]]       # predicate -> onset risk intervals (present predicates)
    off_risk: dict[int, list[tuple[float, float]]]      # predicate -> on-intervals
    ages: list[tuple[int, float, bool, bool]]           # (predicate, duration s, onset known, offset observed)


def pair_stream(v: VideoRecord, pr: PairRecord, K: int) -> PairStream:
    """Apply the frozen extraction rules (prereg §3) to one pair."""
    f = v.fps
    w0, w1 = pr.window
    ev: list[tuple[float, int]] = []
    if w0 > 0:
        ev.append((w0 / f, K))
    on_risk, off_risk, ages = {}, {}, []
    for r, ivs in pr.rels.items():
        on_iv, gaps, cur = [], [], w0
        for b, e in ivs:
            if b > 0:
                ev.append((b / f, 2 * r))
            if e < v.T:
                ev.append((e / f, 2 * r + 1))
            on_iv.append((b / f, e / f))
            if b > cur:
                gaps.append((cur / f, b / f))
            cur = max(cur, e)
            ages.append((r, (e - b) / f, b > 0, e < v.T))
        if w1 > cur:
            gaps.append((cur / f, w1 / f))
        on_risk[r] = gaps
        off_risk[r] = on_iv
    ev.sort(key=lambda x: (x[0], x[1]))
    return PairStream(ev, (w0 / f, w1 / f), on_risk, off_risk, ages)
