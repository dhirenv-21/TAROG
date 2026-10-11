"""Episode-study corruption injectors (research/taorg_episode_preregistration.md §4).

All injectors are deterministic given the seed, never mutate their input, act on
the edge records of the rendered stream, and return a ``CorruptionManifest``.
They read ground-truth episodes (the simulator decides WHERE to corrupt); the
models never see the episodes or the manifests.

Manifest types:
    occlusion               temporary OCC gap inside an episode (frozen Persistence-F1 intervals)
    termination_occlusion   C2: detections suppressed + OCC around a genuine termination
    post_termination_flags  C3: dead records flagged OCC after a genuine termination
    boundary_flicker        C5: misses / false detections near episode boundaries
    false_detection         background confident false detections on dead records
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

from taorg.data.episode_world import KeyEpisodes
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

OCC = VisibilityState.OCCLUDED_OR_UNRELIABLE

# (frame, key) -> replacement record builder
Edit = Callable[[EdgeObservation], EdgeObservation]


def _rec(o: EdgeObservation, observed: bool, score: float, vis: VisibilityState | None = None) -> EdgeObservation:
    return EdgeObservation(edge_key=o.edge_key, video_id=o.video_id, frame_idx=o.frame_idx,
                           timestamp=o.timestamp, predicate_score=score, observed=observed,
                           subject_visibility=o.subject_visibility,
                           object_visibility=o.object_visibility if vis is None else vis,
                           source_model=o.source_model, source_checkpoint=o.source_checkpoint)


def _apply(frames: list[FrameGraph], edits: dict[tuple[int, EdgeKey], Edit]) -> list[FrameGraph]:
    out = []
    for fr in frames:
        new = [edits[(fr.frame_idx, o.edge_key)](o) if (fr.frame_idx, o.edge_key) in edits else o
               for o in fr.edge_observations]
        out.append(FrameGraph(video_id=fr.video_id, frame_idx=fr.frame_idx, timestamp=fr.timestamp,
                              track_observations=fr.track_observations, edge_observations=new))
    return out


def _entry(kind: str, video_id: str, key: EdgeKey, a: int, b: int, orig: str = "VISIBLE",
           mod: str = "OCCLUDED_OR_UNRELIABLE") -> CorruptionEntry:
    return CorruptionEntry(corruption_type=kind, video_id=video_id, edge_key_tuple=tuple(key),
                           start_frame=a, end_frame=b, duration=b - a + 1,
                           original_visibility=orig, modified_visibility=mod)


def _next_start(eps: tuple[tuple[int, int], ...], b: int, n: int) -> int:
    nxt = [a for a, _ in eps if a > b]
    return min(nxt) if nxt else n


def inject_gaps(frames: list[FrameGraph], keys: Iterable[KeyEpisodes], seed: int, prob: float,
                max_per_episode: int, length: tuple[int, int], min_episode: int, margin: int
                ) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Temporary OCC gaps inside episodes (relationship alive, detections suppressed)."""
    rng = random.Random(seed)
    vid = frames[0].video_id
    man = CorruptionManifest(seed=seed, config={"type": "occlusion", "prob": prob, "max_per_episode": max_per_episode,
                                                "length": list(length), "min_episode": min_episode, "margin": margin})
    edits: dict[tuple[int, EdgeKey], Edit] = {}
    for ke in keys:
        for a, b in ke.episodes:
            used: list[tuple[int, int]] = []
            for _ in range(max_per_episode):
                u, L, pos = rng.random(), rng.randint(*length), rng.random()
                if u >= prob or b - a + 1 < min_episode:
                    continue
                lo, hi = a + margin, b - margin - L + 1
                if hi < lo:
                    continue
                s = lo + min(int(pos * (hi - lo + 1)), hi - lo)
                e = s + L - 1
                if any(not (e + 1 < x or s - 1 > y) for x, y in used):
                    continue
                used.append((s, e))
                for f in range(s, e + 1):
                    edits[(f, ke.edge_key)] = lambda o: _rec(o, False, 0.0, OCC)
                man.entries.append(_entry("occlusion", vid, ke.edge_key, s, e))
    return _apply(frames, edits), man


def inject_hidden_termination(frames: list[FrameGraph], keys: Iterable[KeyEpisodes], seed: int, prob: float,
                              pre: tuple[int, int], post: tuple[int, int]
                              ) -> tuple[list[FrameGraph], CorruptionManifest]:
    """C2: the relation ends while hidden: suppress + flag OCC from end-pre+1 to end+post."""
    rng = random.Random(seed)
    n = len(frames)
    vid = frames[0].video_id
    man = CorruptionManifest(seed=seed, config={"type": "termination_occlusion", "prob": prob,
                                                "pre": list(pre), "post": list(post)})
    edits: dict[tuple[int, EdgeKey], Edit] = {}
    for ke in keys:
        for a, b in ke.episodes:
            u, p1, p2 = rng.random(), rng.randint(*pre), rng.randint(*post)
            if b >= n - 1 or u >= prob:
                continue
            s = max(a + 2, b - p1 + 1)
            e = min(b + p2, _next_start(ke.episodes, b, n) - 3, n - 1)
            if e <= b:
                continue
            for f in range(s, e + 1):
                edits[(f, ke.edge_key)] = lambda o: _rec(o, False, 0.0, OCC)
            man.entries.append(_entry("termination_occlusion", vid, ke.edge_key, s, e))
    return _apply(frames, edits), man


def inject_post_termination_flags(frames: list[FrameGraph], keys: Iterable[KeyEpisodes], seed: int, prob: float,
                                  post: tuple[int, int]) -> tuple[list[FrameGraph], CorruptionManifest]:
    """C3: after a genuine termination, the (correctly absent) records are flagged OCC."""
    rng = random.Random(seed)
    n = len(frames)
    vid = frames[0].video_id
    man = CorruptionManifest(seed=seed, config={"type": "post_termination_flags", "prob": prob, "post": list(post)})
    edits: dict[tuple[int, EdgeKey], Edit] = {}
    for ke in keys:
        for a, b in ke.episodes:
            u, p = rng.random(), rng.randint(*post)
            if b >= n - 1 or u >= prob:
                continue
            e = min(b + p, _next_start(ke.episodes, b, n) - 3, n - 1)
            if e <= b:
                continue
            for f in range(b + 1, e + 1):
                edits[(f, ke.edge_key)] = lambda o: _rec(o, False, 0.0, OCC)
            man.entries.append(_entry("post_termination_flags", vid, ke.edge_key, b + 1, e))
    return _apply(frames, edits), man


def inject_boundary_flicker(frames: list[FrameGraph], keys: Iterable[KeyEpisodes], seed: int, window: int,
                            miss_prob: float, false_prob: float) -> tuple[list[FrameGraph], CorruptionManifest]:
    """C5: near every episode boundary, alive records are missed (VISIBLE) and dead
    records are falsely detected (VISIBLE, score from the key's distribution).

    One uniform draw per (key, frame) in the window, plus one score draw, in
    key -> frame order, unconditionally (aligned stream).
    """
    rng = random.Random(seed)
    n = len(frames)
    vid = frames[0].video_id
    man = CorruptionManifest(seed=seed, config={"type": "boundary_flicker", "window": window,
                                                "miss_prob": miss_prob, "false_prob": false_prob})
    edits: dict[tuple[int, EdgeKey], Edit] = {}
    for ke in keys:
        bounds = set()
        for a, b in ke.episodes:
            if a > 0:
                bounds.add(a - 0.5)
            if b < n - 1:
                bounds.add(b + 0.5)
        near = sorted({f for x in bounds for f in range(int(x + 0.5) - window, int(x - 0.5) + window + 1)
                       if 0 <= f < n})
        for f in near:
            u = rng.random()
            sc = min(1.0, max(0.05, rng.gauss(ke.score_mean, 0.05)))
            alive = ke.alive(f)
            if alive and u < miss_prob:
                edits[(f, ke.edge_key)] = lambda o: _rec(o, False, 0.0, VisibilityState.VISIBLE)
                man.entries.append(_entry("boundary_flicker", vid, ke.edge_key, f, f, "OBSERVED", "MISSED"))
            elif not alive and u < false_prob:
                edits[(f, ke.edge_key)] = (lambda s: (lambda o: _rec(o, True, s, VisibilityState.VISIBLE)))(sc)
                man.entries.append(_entry("boundary_flicker", vid, ke.edge_key, f, f, "ABSENT", "FALSE_DETECTION"))
    return _apply(frames, edits), man


def inject_false_detections(frames: list[FrameGraph], keys: Iterable[KeyEpisodes], seed: int, prob: float
                            ) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Background confident false detections on GT-dead records (one draw per dead record).

    Only records that are currently absences are converted; flags are set VISIBLE
    (a detection is by convention a visible observation).
    """
    rng = random.Random(seed)
    vid = frames[0].video_id
    man = CorruptionManifest(seed=seed, config={"type": "false_detection", "prob": prob})
    by_key = {ke.edge_key: ke for ke in keys}
    edits: dict[tuple[int, EdgeKey], Edit] = {}
    for fr in frames:
        for o in fr.edge_observations:
            ke = by_key.get(o.edge_key)
            if ke is None or ke.alive(fr.frame_idx):
                continue
            u = rng.random()
            sc = min(1.0, max(0.05, rng.gauss(ke.score_mean, 0.05)))
            if u < prob and not o.observed:
                edits[(fr.frame_idx, o.edge_key)] = (lambda s: (lambda r: _rec(r, True, s, VisibilityState.VISIBLE)))(sc)
                man.entries.append(_entry("false_detection", vid, o.edge_key, fr.frame_idx, fr.frame_idx,
                                          "ABSENT", "FALSE_DETECTION"))
    return _apply(frames, edits), man
