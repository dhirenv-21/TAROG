"""Synthetic episode world: recurring relationships on moving tracks (episode study).

Specified in research/taorg_episode_preregistration.md §3.  Differences from
``taorg.data.synthetic_world`` (which is left untouched):
    - a key may have SEVERAL alive episodes (genuine recurrence);
    - optional succession: a terminated relation of predicate r on (s, o) spawns
      an episode of partner(r) on the same pair, with overlap or a short gap;
    - tracks MOVE (smooth random-walk velocity, reflecting arena), so that an
      identity cue must distinguish ordinary motion from identity switches.

Ground truth = the episode list per key.  The rendered GT frames carry clean
boxes and one record per key per frame (observed iff alive, VISIBLE flags).
Observed (input) boxes are jittered by :func:`jitter_boxes`.

Randomness: independent ``random.Random`` streams derived with SHA-256 from the
seed and a component label, so the same seed yields the same tracks and
episodes in every condition that does not override the corresponding world
parameters (paired comparisons across conditions).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from taorg.experiments.phase3.conditions import derive_seed
from taorg.noise.manifest import CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, TrackerState, VisibilityState


@dataclass(frozen=True)
class EpisodeWorldConfig:
    """World parameters (see the pre-registration for meanings; frames unless noted)."""

    num_frames: int = 360
    fps: float = 30.0
    num_subject_tracks: int = 2
    num_object_tracks: int = 4
    num_predicates: int = 6
    num_keys: int = 8
    first_start_max: int = 45
    on_frames: tuple[int, int] = (40, 120)
    off_frames: tuple[int, int] = (10, 60)
    p_recur: float = 0.5
    p_open: float = 0.2
    min_tail: int = 20
    p_succ: float = 0.4
    succ_offset: tuple[int, int] = (-8, 4)
    succ_clearance: int = 10
    predicate_partners: dict = field(default_factory=lambda: {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4})
    score_mean_range: tuple[float, float] = (0.6, 0.95)
    score_noise_std: float = 0.05
    score_min: float = 0.05
    box_width: tuple[float, float] = (50.0, 110.0)
    box_aspect: tuple[float, float] = (1.0, 1.8)
    arena: tuple[float, float, float, float] = (60.0, 60.0, 580.0, 420.0)
    v0: float = 60.0
    accel_std: float = 80.0
    v_max: float = 200.0
    box_jitter: float = 0.05

    def __post_init__(self) -> None:
        if self.num_frames < self.on_frames[0] + self.first_start_max + self.min_tail + 2:
            raise ValueError("num_frames too short for the configured episode lengths")
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
        if not (1 <= self.num_keys <= self.num_subject_tracks * self.num_object_tracks * self.num_predicates):
            raise ValueError("num_keys out of range")
        for name in ("p_recur", "p_open", "p_succ"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.on_frames[0] < 1 or self.on_frames[0] > self.on_frames[1]:
            raise ValueError("on_frames must satisfy 1 <= lo <= hi")
        if self.off_frames[0] < 1 or self.off_frames[0] > self.off_frames[1]:
            raise ValueError("off_frames must satisfy 1 <= lo <= hi")
        if set(self.predicate_partners) != set(range(self.num_predicates)):
            raise ValueError("predicate_partners must map every predicate")

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeWorldConfig":
        """Build from a (YAML) mapping, converting lists to tuples."""
        kw = dict(d)
        for k in ("on_frames", "off_frames", "succ_offset", "score_mean_range", "box_width",
                  "box_aspect", "arena"):
            if k in kw and isinstance(kw[k], list):
                kw[k] = tuple(kw[k])
        if "predicate_partners" in kw:
            kw["predicate_partners"] = {int(a): int(b) for a, b in kw["predicate_partners"].items()}
        return cls(**kw)

    def to_dict(self) -> dict:
        """JSON-serialisable dict."""
        d = asdict(self)
        d["predicate_partners"] = {str(k): v for k, v in self.predicate_partners.items()}
        return d


@dataclass(frozen=True)
class KeyEpisodes:
    """Ground-truth episodes of one key (inclusive frame intervals, sorted, disjoint)."""

    edge_key: EdgeKey
    episodes: tuple[tuple[int, int], ...]
    score_mean: float
    successor_of: tuple[int, int, int] | None = None

    def alive(self, f: int) -> bool:
        """True if the key is alive at frame f."""
        return any(a <= f <= b for a, b in self.episodes)


@dataclass
class EpisodeWorld:
    """One sampled world: GT episodes, clean track trajectories and succession records."""

    seed: int
    config: EpisodeWorldConfig
    keys: list[KeyEpisodes]
    track_boxes: dict[int, list[tuple[float, float, float, float]]]
    track_class: dict[int, int]
    succession: list[dict]


def _sample_episodes(rng: random.Random, cfg: EpisodeWorldConfig) -> list[tuple[int, int]]:
    last = cfg.num_frames - 1
    eps: list[tuple[int, int]] = []
    t = rng.randint(0, cfg.first_start_max)
    while True:
        length = rng.randint(*cfg.on_frames)
        recur = rng.random() < cfg.p_recur
        gap = rng.randint(*cfg.off_frames)
        open_last = rng.random() < cfg.p_open
        end = t + length - 1
        nxt = end + 1 + gap
        can_recur = recur and nxt + cfg.on_frames[0] - 1 <= last - cfg.min_tail
        if can_recur and end < last:
            eps.append((t, end))
            t = nxt
            continue
        if open_last:
            eps.append((t, last))
        else:
            end = min(end, last - cfg.min_tail)
            if end < t + cfg.on_frames[0] - 1:          # cannot fit a terminating episode
                end = last
            eps.append((t, end))
        return eps


def _free(eps: tuple[tuple[int, int], ...], a: int, b: int, clearance: int) -> bool:
    return all(b + clearance < x or a - clearance > y for x, y in eps)


def _tracks(seed: int, cfg: EpisodeWorldConfig) -> tuple[dict, dict]:
    rng = random.Random(derive_seed(seed, "episode_world", "tracks"))
    n_tr = cfg.num_subject_tracks + cfg.num_object_tracks
    x0, y0, x1, y1 = cfg.arena
    dt = 1.0 / cfg.fps
    boxes: dict[int, list] = {}
    cls: dict[int, int] = {}
    for tid in range(1, n_tr + 1):
        cls[tid] = 0 if tid <= cfg.num_subject_tracks else 1
        w = rng.uniform(*cfg.box_width)
        h = w * rng.uniform(*cfg.box_aspect)
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        vx, vy = rng.gauss(0.0, cfg.v0), rng.gauss(0.0, cfg.v0)
        seq = []
        for _ in range(cfg.num_frames):
            seq.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            vx += rng.gauss(0.0, cfg.accel_std * math.sqrt(dt))
            vy += rng.gauss(0.0, cfg.accel_std * math.sqrt(dt))
            sp = math.hypot(vx, vy)
            if sp > cfg.v_max:
                vx, vy = vx * cfg.v_max / sp, vy * cfg.v_max / sp
            cx, cy = cx + vx * dt, cy + vy * dt
            if cx < x0 or cx > x1:
                vx = -vx
                cx = min(max(cx, x0), x1)
            if cy < y0 or cy > y1:
                vy = -vy
                cy = min(max(cy, y0), y1)
        boxes[tid] = seq
    return boxes, cls


def sample_world(seed: int, cfg: EpisodeWorldConfig) -> EpisodeWorld:
    """Sample the GT episodes and clean trajectories of one video (deterministic)."""
    rng = random.Random(derive_seed(seed, "episode_world", "keys"))
    subjects = list(range(1, cfg.num_subject_tracks + 1))
    objects = list(range(cfg.num_subject_tracks + 1, cfg.num_subject_tracks + cfg.num_object_tracks + 1))
    all_keys = [EdgeKey(s, o, p) for s in subjects for o in objects for p in range(cfg.num_predicates)]
    keys = sorted(rng.sample(all_keys, cfg.num_keys))
    by_key: dict[EdgeKey, KeyEpisodes] = {}
    for k in keys:
        eps = _sample_episodes(rng, cfg)
        by_key[k] = KeyEpisodes(k, tuple(eps), rng.uniform(*cfg.score_mean_range))

    srng = random.Random(derive_seed(seed, "episode_world", "succession"))
    last = cfg.num_frames - 1
    records: list[dict] = []
    for k in keys:                                   # primary keys only, sorted
        for (a, b) in by_key[k].episodes:
            u = srng.random()
            off = srng.randint(*cfg.succ_offset)
            length = srng.randint(*cfg.on_frames)
            mean = srng.uniform(*cfg.score_mean_range)
            if b >= last or u >= cfg.p_succ:
                continue
            pk = EdgeKey(k.subject_track_id, k.object_track_id, int(cfg.predicate_partners[k.predicate_id]))
            start = max(0, b + off)
            end = min(start + length - 1, last)
            rec = {"from": list(k), "to": list(pk), "end_from": b, "start": start, "end": end}
            existing = by_key.get(pk)
            if existing is not None and not _free(existing.episodes, start, end, cfg.succ_clearance):
                rec["skipped"] = "conflict"
                records.append(rec)
                continue
            if existing is None:
                by_key[pk] = KeyEpisodes(pk, ((start, end),), mean, successor_of=tuple(k))
            else:
                eps = tuple(sorted(existing.episodes + ((start, end),)))
                by_key[pk] = KeyEpisodes(pk, eps, existing.score_mean, existing.successor_of)
            records.append(rec)
    boxes, cls = _tracks(seed, cfg)
    return EpisodeWorld(seed=seed, config=cfg, keys=[by_key[k] for k in sorted(by_key)],
                        track_boxes=boxes, track_class=cls, succession=records)


def render(world: EpisodeWorld, video_id: str) -> list[FrameGraph]:
    """Clean GT frames: true boxes, one record per key per frame (observed iff alive)."""
    cfg = world.config
    srng = random.Random(derive_seed(world.seed, "episode_world", "scores"))
    frames: list[FrameGraph] = []
    for f in range(cfg.num_frames):
        ts = f / cfg.fps
        tracks = [TrackObservation(video_id=video_id, frame_idx=f, timestamp=ts, track_id=tid,
                                   class_id=world.track_class[tid], bbox_xyxy=world.track_boxes[tid][f],
                                   detector_confidence=0.9, tracker_state=TrackerState.TRACKED,
                                   visibility_state=VisibilityState.VISIBLE)
                  for tid in sorted(world.track_boxes)]
        edges = []
        for ke in world.keys:
            alive = ke.alive(f)
            score = 0.0
            if alive:
                score = min(1.0, max(cfg.score_min, srng.gauss(ke.score_mean, cfg.score_noise_std)))
            edges.append(EdgeObservation(edge_key=ke.edge_key, video_id=video_id, frame_idx=f, timestamp=ts,
                                         predicate_score=score, observed=alive, source_model="episode_world"))
        frames.append(FrameGraph(video_id=video_id, frame_idx=f, timestamp=ts,
                                 track_observations=tracks, edge_observations=edges))
    return frames


def jitter_boxes(frames: list[FrameGraph], seed: int, std: float) -> tuple[list[FrameGraph], CorruptionManifest]:
    """Observed-box centre jitter N(0, std * diagonal) per axis (input frames not mutated)."""
    rng = random.Random(seed)
    man = CorruptionManifest(seed=seed, config={"type": "box_jitter", "std": std})
    out = []
    for fr in frames:
        tracks = []
        for t in fr.track_observations:
            x1, y1, x2, y2 = t.bbox_xyxy
            d = math.hypot(x2 - x1, y2 - y1)
            dx, dy = rng.gauss(0.0, std * d), rng.gauss(0.0, std * d)
            tracks.append(TrackObservation(
                video_id=t.video_id, frame_idx=t.frame_idx, timestamp=t.timestamp, track_id=t.track_id,
                class_id=t.class_id, bbox_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                detector_confidence=t.detector_confidence, tracker_state=t.tracker_state,
                visibility_state=t.visibility_state))
        out.append(FrameGraph(video_id=fr.video_id, frame_idx=fr.frame_idx, timestamp=fr.timestamp,
                              track_observations=tracks, edge_observations=list(fr.edge_observations)))
    return out, man
