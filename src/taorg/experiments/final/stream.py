"""Final evaluation (research/taorg_final_preregistration.md): EGTR-predicted relationship stream on real
tracker identities, oracle scoring labels, and genuine-event outcomes.

Information separation:
* ``build_predicted_stream`` (INFERENCE): EGTR person relations + tracker outputs only. No annotations.
* ``scoring_labels`` (ORACLE, evaluation only): VidOR annotations + the appearance-audit oracle.
Frames are internal 1-based; the relationship layer runs on ``sampled`` frames only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from taorg.adapters.egtr import MAPPED
from taorg.adapters.mot17 import FrameTracks, iou
from taorg.adapters.vidor import VidorVideo, active_relations
from taorg.experiments.appearance.labels import frame_label, purity
from taorg.experiments.downstream.observations import gt_dict
from taorg.experiments.observation.oracle import identity_of, match
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

MATCH_IOU = 0.5


def load_sgg(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """frame -> (person boxes (N,4), obj scores (N,), pair scores (N,N,len(MAPPED)))."""
    z = np.load(path)
    frames, o = z["frames"], z["offsets"]                 # decompress each array once
    boxes, obj, pair = z["boxes"], z["obj_scores"], z["pair_scores"]
    out = {}
    pair_off = 0
    for i, f in enumerate(frames):
        n = int(o[i + 1] - o[i])
        sc = pair[pair_off:pair_off + n * n].reshape(n, n, len(MAPPED))
        pair_off += n * n
        out[int(f)] = (boxes[o[i]:o[i + 1]], obj[o[i]:o[i + 1]], sc)
    return out


def assign(boxes: np.ndarray, outputs: dict[int, tuple]) -> dict[int, int]:
    """EGTR person query -> tracker id, Hungarian on IoU >= MATCH_IOU (one-to-one)."""
    tids = list(outputs)
    if not len(boxes) or not tids:
        return {}
    M = np.array([[iou(tuple(b), outputs[t][0]) for t in tids] for b in boxes])
    r, c = linear_sum_assignment(-M)
    return {int(i): tids[j] for i, j in zip(r, c) if M[i, j] >= MATCH_IOU}


@dataclass
class PredStream:
    """Methods' input: graphs on sampled frames, plus bookkeeping for scoring."""

    graphs: list[FrameGraph]
    sampled: list[int]
    predicted_keys: list[EdgeKey]


def build_predicted_stream(video_id: str, fps: float, frames: list[FrameTracks],
                           sgg: dict[int, tuple], sampled: list[int], tau: float) -> PredStream:
    """INFERENCE: observed iff the EGTR triplet score >= tau for two matched tracker tracks; a known key
    whose two endpoints are both matched to EGTR person queries but which is not predicted gets a
    VISIBLE absence record (evaluated negative); otherwise no record (not evaluated)."""
    known: list[EdgeKey] = []
    seen: set[EdgeKey] = set()
    graphs = []
    for f in sampled:
        boxes, _, sc = sgg.get(f, (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 0, len(MAPPED)))))
        q2t = assign(boxes, frames[f - 1].outputs)
        evaluated = set(q2t.values())
        now: dict[EdgeKey, float] = {}
        for qa, x in q2t.items():
            for qb, y in q2t.items():
                if x == y:
                    continue
                for m in range(len(MAPPED)):
                    s = float(sc[qa, qb, m])
                    if s >= tau:
                        k = EdgeKey(x, y, m)
                        now[k] = max(s, now.get(k, 0.0))
        for k in sorted(now):
            if k not in seen:
                seen.add(k)
                known.append(k)
        ts = (f - 1) / fps
        eo = [EdgeObservation(k, video_id, f, ts, now[k], True) for k in sorted(now)]
        for k in known:
            if k not in now and k.subject_track_id in evaluated and k.object_track_id in evaluated:
                eo.append(EdgeObservation(k, video_id, f, ts, 0.0, False, VisibilityState.VISIBLE,
                                          VisibilityState.VISIBLE))
        graphs.append(FrameGraph(video_id=video_id, frame_idx=f, timestamp=ts, track_observations=[],
                                 edge_observations=eo))
    return PredStream(graphs, list(sampled), known)


def oracle_keys(v: VidorVideo, frames: list[FrameTracks], matches: dict, sampled: list[int]) -> list[EdgeKey]:
    """ORACLE: tracker keys whose annotated relation (mapped predicates) is active at some sampled frame."""
    out, seen = [], set()
    for f in sampled:
        inv = {g: t for t, g in matches[f].items()}
        for (i, j, p) in active_relations(v, f - 1):
            if p in MAPPED and i in inv and j in inv and inv[i] != inv[j]:
                k = EdgeKey(inv[i], inv[j], MAPPED.index(p))
                if k not in seen:
                    seen.add(k)
                    out.append(k)
    return out


def scoring_labels(v: VidorVideo, frames: list[FrameTracks], keys: list[EdgeKey], sampled: list[int]
                   ) -> tuple[dict[EdgeKey, dict[str, np.ndarray]], dict]:
    """ORACLE labels on the sampled frames (index 0 = dummy, 1..len(sampled)).

    alive: annotated relation between the endpoints' identities is active (identity = current GT match
           if output, majority identity if the track is in a genuine GAP, else none)
    takeover: an endpoint is in TAKEOVER;  gap: alive and an endpoint is in GAP (genuine observation gap)
    both_out: both endpoints output
    ended_after / started_at: the relation of the identities at this frame is inactive at the next /
           previous sampled frame (genuine termination / onset, not identity-matching flicker)
    """
    gt = gt_dict(v)
    matches = match(frames, gt)
    g = identity_of(matches)
    pur = purity(matches)
    act = {f: active_relations(v, f - 1) for f in sampled}
    cache: dict[tuple[int, int], tuple] = {}

    def state(x: int, f: int):
        key = (x, f)
        if key not in cache:
            if x in frames[f - 1].outputs:
                cache[key] = ("OUT", matches[f].get(x))
            else:
                lab = frame_label(x, f, g, pur, matches, gt)
                cache[key] = (lab, g.get(x) if lab == "GAP" else None)
        return cache[key]

    n = len(sampled)
    out = {}
    for k in keys:
        a = {nm: np.zeros(n + 1, bool) for nm in ("alive", "takeover", "gap", "both_out", "ended_after", "started_at")}
        p = MAPPED[k.predicate_id]
        for i, f in enumerate(sampled, start=1):
            sx, ix = state(k.subject_track_id, f)
            sy, iy = state(k.object_track_id, f)
            a["both_out"][i] = sx == "OUT" and sy == "OUT"
            a["takeover"][i] = "TAKEOVER" in (sx, sy)
            alive = ix is not None and iy is not None and (ix, iy, p) in act[f]
            a["alive"][i] = alive
            a["gap"][i] = alive and "GAP" in (sx, sy)
            if ix is not None and iy is not None:
                a["ended_after"][i] = i < n and (ix, iy, p) not in act[sampled[i]]
                a["started_at"][i] = i > 1 and (ix, iy, p) not in act[sampled[i - 2]]
        out[k] = a
    return out, matches
