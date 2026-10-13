"""Relationship observations (oracle predictor on tracker outputs) and scoring labels (prereg §4, §6).

ANNOTATION ACCESS — this module is the only place that combines annotations with tracker outputs.
* ``build_stream`` produces the methods' ONLY relational input: at frame t it uses the
  annotations of frame t alone (matching + active relations), never a future frame.
* ``scoring_arrays`` produces oracle labels for scoring only.
Frames are 1-based (internal); annotation frame f maps to internal f + 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from taorg.adapters.mot17 import FrameTracks
from taorg.adapters.vidor import VidorVideo, active_relations
from taorg.experiments.appearance.labels import frame_label, purity
from taorg.experiments.observation.oracle import identity_of, match
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState


def gt_dict(v: VidorVideo) -> dict[int, dict[int, tuple[tuple, float]]]:
    """Internal-frame -> {person tid: (xyxy, 1.0)} (visibility is not annotated in VidOR)."""
    return {f + 1: {t: (b, 1.0) for t, b in v.boxes[f].items()} for f in range(v.frame_count)}


def predicate_ids(v: VidorVideo) -> dict[str, int]:
    """Per-video predicate index (sorted names); names are saved alongside ids."""
    return {p: i for i, p in enumerate(sorted({r[2] for r in v.relations}))}


@dataclass
class Stream:
    """The observation stream and its bookkeeping (frames 1..T)."""

    graphs: list[FrameGraph]
    keys: list[EdgeKey]                         # scored keys (observed at least once), first-seen order
    observed: dict[EdgeKey, np.ndarray]         # bool[T+1] (index 0 unused)
    matches: dict[int, dict[int, int]]          # frame -> {track: person tid}
    pred_names: dict[int, str]


def build_stream(v: VidorVideo, frames: list[FrameTracks]) -> Stream:
    """Oracle relation predictor on tracker outputs; visible-absence records when both endpoints output."""
    gt = gt_dict(v)
    matches = match(frames, gt)
    pid = predicate_ids(v)
    T = len(frames)
    known: list[EdgeKey] = []
    seen: set[EdgeKey] = set()
    obs_rows: dict[EdgeKey, np.ndarray] = {}
    graphs = []
    for fr in frames:
        f = fr.frame
        m = matches[f]
        inv = {g: t for t, g in m.items()}
        rel = active_relations(v, f - 1)
        now: set[EdgeKey] = set()
        for (i, j, p) in rel:
            if i in inv and j in inv and inv[i] != inv[j]:
                now.add(EdgeKey(inv[i], inv[j], pid[p]))
        for k in sorted(now):
            if k not in seen:
                seen.add(k)
                known.append(k)
                obs_rows[k] = np.zeros(T + 1, bool)
            obs_rows[k][f] = True
        ts = (f - 1) / v.fps
        eo = []
        for k in known:
            if k in now:
                eo.append(EdgeObservation(k, v.video_id, f, ts, 1.0, True))
            elif k.subject_track_id in fr.outputs and k.object_track_id in fr.outputs:
                eo.append(EdgeObservation(k, v.video_id, f, ts, 0.0, False, VisibilityState.VISIBLE,
                                          VisibilityState.VISIBLE))
        graphs.append(FrameGraph(video_id=v.video_id, frame_idx=f, timestamp=ts, track_observations=[],
                                 edge_observations=eo))
    return Stream(graphs, known, obs_rows, matches, {i: p for p, i in pid.items()})


def scoring_arrays(v: VidorVideo, frames: list[FrameTracks], s: Stream) -> dict[EdgeKey, dict[str, np.ndarray]]:
    """Per scored key, bool[T+1] arrays: observed, alive (alive*), takeover (an endpoint in TAKEOVER),
    gap (alive and not observed), both_out (both endpoints output), ambiguous (an endpoint AMBIGUOUS)."""
    gt = gt_dict(v)
    g = identity_of(s.matches)
    pur = purity(s.matches)
    T = len(frames)
    inv_name = {i: p for i, p in s.pred_names.items()}
    active = [set()] + [active_relations(v, f - 1) for f in range(1, T + 1)]
    lab_cache: dict[tuple[int, int], str] = {}

    def state(x: int, f: int) -> tuple[str, Any]:
        if x in frames[f - 1].outputs:
            return "OUT", s.matches[f].get(x)
        key = (x, f)
        if key not in lab_cache:
            lab_cache[key] = frame_label(x, f, g, pur, s.matches, gt)
        lab = lab_cache[key]
        return lab, (g.get(x) if lab == "GAP" else None)

    out: dict[EdgeKey, dict[str, np.ndarray]] = {}
    for k in s.keys:
        a = {n: np.zeros(T + 1, bool) for n in ("alive", "takeover", "gap", "both_out", "ambiguous")}
        p = inv_name[k.predicate_id]
        for f in range(1, T + 1):
            sx, ix = state(k.subject_track_id, f)
            sy, iy = state(k.object_track_id, f)
            a["both_out"][f] = sx == "OUT" and sy == "OUT"
            a["takeover"][f] = "TAKEOVER" in (sx, sy)
            a["ambiguous"][f] = "AMBIGUOUS" in (sx, sy)
            alive = ix is not None and iy is not None and (ix, iy, p) in active[f]
            a["alive"][f] = alive
            a["gap"][f] = alive and not s.observed[k][f]
        a["observed"] = s.observed[k]
        out[k] = a
    return out
