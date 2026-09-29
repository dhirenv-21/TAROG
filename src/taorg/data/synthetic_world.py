"""Parametric synthetic video generator for controlled robustness experiments.

Generates ground-truth FrameGraph sequences whose relationship lifetimes are
known exactly.  Corruption (occlusion, flicker, visibility noise, ID switches)
is applied afterwards by the injectors in ``taorg.noise`` — this module never
hard-codes observation failures, with ONE exception documented below
(``termination_occlusion_prob``), which is a *scene* property rather than a
corruption.

World model (all assumptions are explicit and configurable):
    - ``num_frames`` frames at ``fps``; timestamps are ``frame_idx / fps`` seconds.
    - Subject tracks (class 0) and object tracks (class 1) are present and
      VISIBLE for the whole sequence.
    - ``num_edges`` distinct (subject, object, predicate) edges.  Each edge is
      alive on exactly ONE contiguous interval ``[start, end]``.
    - With probability ``terminating_prob`` an edge genuinely terminates
      before the sequence ends, leaving at least ``min_post_termination_frames``
      frames of visible absence afterwards (so termination metrics are defined).
      Otherwise it stays alive until the last frame.
    - While alive, the edge is observed every frame with a noisy predicate
      score ``clip(N(mu_e, score_noise_std), score_min, 1)`` where ``mu_e`` is
      drawn per edge from ``score_mean_range``.
    - Every edge has a record in every frame: ``observed=False`` records with
      ``predicate_score=0`` represent absence (before birth / after death).

Scene property (NOT corruption):
    ``termination_occlusion_prob`` — probability that a terminating edge's
    object becomes OCCLUDED_OR_UNRELIABLE for a random number of frames right
    after the relationship ends (e.g. a cup is carried out of view).  This is
    the fundamental ambiguity case for visibility-gated methods: the absence is
    genuine termination but the visibility signal says "occluded".
    Simplification: the occlusion is applied to that edge's record only, not to
    other edges sharing the object.

Randomness: a local ``random.Random(seed)`` — no global RNG state is touched.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, TrackerState, VisibilityState


@dataclass(frozen=True)
class SyntheticWorldConfig:
    """Configuration of the synthetic world.

    Attributes:
        num_frames: Sequence length in frames.
        fps: Frame rate used to derive timestamps (seconds).
        num_subject_tracks: Number of subject tracks (class 0).
        num_object_tracks: Number of object tracks (class 1).
        num_predicates: Size of the predicate vocabulary.
        num_edges: Number of distinct edges per video.
        start_frame_max: Latest possible birth frame of an edge.
        terminating_prob: Probability that an edge terminates before the end.
        min_alive_frames: Minimum alive-interval length of a terminating edge.
        max_alive_frames: Maximum alive-interval length of a terminating edge.
        min_post_termination_frames: Minimum visible-absence tail after a
            terminating edge ends.
        score_mean_range: Range of the per-edge mean predicate score.
        score_noise_std: Std of per-frame score noise around the edge mean.
        score_min: Lower clip for observed predicate scores.
        termination_occlusion_prob: Scene property; see module docstring.
        termination_occlusion_frames: (min, max) length in frames of the
            post-termination occlusion when it occurs.
    """

    num_frames: int = 300
    fps: float = 30.0
    num_subject_tracks: int = 2
    num_object_tracks: int = 4
    num_predicates: int = 6
    num_edges: int = 8
    start_frame_max: int = 60
    terminating_prob: float = 0.6
    min_alive_frames: int = 60
    max_alive_frames: int = 180
    min_post_termination_frames: int = 40
    score_mean_range: tuple[float, float] = (0.6, 0.95)
    score_noise_std: float = 0.05
    score_min: float = 0.05
    termination_occlusion_prob: float = 0.0
    termination_occlusion_frames: tuple[int, int] = (15, 40)

    def __post_init__(self) -> None:
        if self.num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {self.num_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        max_edges = self.num_subject_tracks * self.num_object_tracks * self.num_predicates
        if not (1 <= self.num_edges <= max_edges):
            raise ValueError(
                f"num_edges must be in [1, {max_edges}] for the configured "
                f"track/predicate counts, got {self.num_edges}"
            )
        if not (0.0 <= self.terminating_prob <= 1.0):
            raise ValueError(f"terminating_prob must be in [0, 1], got {self.terminating_prob}")
        if not (0.0 <= self.termination_occlusion_prob <= 1.0):
            raise ValueError(
                "termination_occlusion_prob must be in [0, 1], "
                f"got {self.termination_occlusion_prob}"
            )
        if self.min_alive_frames > self.max_alive_frames:
            raise ValueError("min_alive_frames must be <= max_alive_frames")
        latest_end = self.num_frames - 1 - self.min_post_termination_frames
        if self.start_frame_max + self.min_alive_frames - 1 > latest_end:
            raise ValueError(
                "Sequence too short: a terminating edge born at start_frame_max "
                "cannot satisfy min_alive_frames and min_post_termination_frames. "
                "Increase num_frames or reduce the other lengths."
            )
        lo, hi = self.termination_occlusion_frames
        if not (1 <= lo <= hi):
            raise ValueError("termination_occlusion_frames must satisfy 1 <= min <= max")

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of the configuration."""
        d = asdict(self)
        d["score_mean_range"] = list(self.score_mean_range)
        d["termination_occlusion_frames"] = list(self.termination_occlusion_frames)
        return d


@dataclass(frozen=True)
class EdgeLifetime:
    """Ground-truth lifetime of one edge.

    Attributes:
        edge_key: Edge identifier.
        start_frame: First alive frame (inclusive).
        end_frame: Last alive frame (inclusive).
        terminates: True if the edge ends before the last frame.
        score_mean: Mean observed predicate score.
        post_termination_occluded_frames: Number of frames after ``end_frame``
            during which the edge's object is OCCLUDED (scene property); 0 if none.
    """

    edge_key: EdgeKey
    start_frame: int
    end_frame: int
    terminates: bool
    score_mean: float
    post_termination_occluded_frames: int = 0


def sample_lifetimes(seed: int, config: SyntheticWorldConfig) -> list[EdgeLifetime]:
    """Sample the ground-truth edge lifetimes for one synthetic video.

    Args:
        seed: Scenario seed (fully determines the output).
        config: World configuration.

    Returns:
        Lifetimes sorted by edge key.
    """
    rng = random.Random(seed)
    subjects = list(range(1, config.num_subject_tracks + 1))
    objects = list(
        range(config.num_subject_tracks + 1,
              config.num_subject_tracks + config.num_object_tracks + 1)
    )
    all_keys = [
        EdgeKey(s, o, p)
        for s in subjects for o in objects for p in range(config.num_predicates)
    ]
    keys = sorted(rng.sample(all_keys, config.num_edges))

    last = config.num_frames - 1
    latest_end = last - config.min_post_termination_frames
    lifetimes: list[EdgeLifetime] = []
    for key in keys:
        # Every draw is made unconditionally, in a fixed order, so that two
        # configs differing only in termination_occlusion_prob produce the
        # SAME lifetimes for the same seed (paired comparison across conditions).
        start = rng.randint(0, config.start_frame_max)
        terminates = rng.random() < config.terminating_prob
        score_mean = rng.uniform(*config.score_mean_range)
        length_u = rng.random()
        occ_u = rng.random()
        occ_len = rng.randint(*config.termination_occlusion_frames)
        occ_frames = 0
        if terminates:
            max_len = min(config.max_alive_frames, latest_end - start + 1)
            span = max_len - config.min_alive_frames + 1
            length = config.min_alive_frames + min(int(length_u * span), span - 1)
            end = start + length - 1
            if occ_u < config.termination_occlusion_prob:
                occ_frames = min(occ_len, last - end)
        else:
            end = last
        lifetimes.append(EdgeLifetime(
            edge_key=key, start_frame=start, end_frame=end, terminates=terminates,
            score_mean=score_mean, post_termination_occluded_frames=occ_frames,
        ))
    return lifetimes


def generate_video(
    seed: int,
    config: SyntheticWorldConfig | None = None,
    video_id: str | None = None,
) -> tuple[list[FrameGraph], list[EdgeLifetime]]:
    """Generate one ground-truth synthetic video.

    Args:
        seed: Scenario seed.  The same (seed, config) always yields the same video.
        config: World configuration (defaults if None).
        video_id: Optional video identifier; defaults to ``synthetic_world_<seed>``.

    Returns:
        (frames, lifetimes): the clean ground-truth frame sequence and the
        per-edge ground-truth lifetimes used to build it.
    """
    if config is None:
        config = SyntheticWorldConfig()
    if video_id is None:
        video_id = f"synthetic_world_{seed}"

    lifetimes = sample_lifetimes(seed, config)
    # Separate stream for per-frame scores so lifetimes do not depend on
    # how many score draws are made.
    score_rng = random.Random(seed * 7919 + 1)

    track_ids = list(range(1, config.num_subject_tracks + config.num_object_tracks + 1))
    track_class = {
        tid: (0 if tid <= config.num_subject_tracks else 1) for tid in track_ids
    }
    track_bbox = {
        tid: (40.0 * tid, 30.0 * tid, 40.0 * tid + 60.0, 30.0 * tid + 90.0)
        for tid in track_ids
    }

    frames: list[FrameGraph] = []
    for f in range(config.num_frames):
        ts = f / config.fps
        tracks = [
            TrackObservation(
                video_id=video_id, frame_idx=f, timestamp=ts, track_id=tid,
                class_id=track_class[tid], bbox_xyxy=track_bbox[tid],
                detector_confidence=0.9, tracker_state=TrackerState.TRACKED,
                visibility_state=VisibilityState.VISIBLE,
            )
            for tid in track_ids
        ]
        edges: list[EdgeObservation] = []
        for lt in lifetimes:
            alive = lt.start_frame <= f <= lt.end_frame
            if alive:
                score = score_rng.gauss(lt.score_mean, config.score_noise_std)
                score = min(1.0, max(config.score_min, score))
            else:
                score = 0.0
            obj_vis = VisibilityState.VISIBLE
            if (
                not alive
                and lt.post_termination_occluded_frames > 0
                and lt.end_frame < f <= lt.end_frame + lt.post_termination_occluded_frames
            ):
                obj_vis = VisibilityState.OCCLUDED_OR_UNRELIABLE
            edges.append(EdgeObservation(
                edge_key=lt.edge_key, video_id=video_id, frame_idx=f, timestamp=ts,
                predicate_score=score, observed=alive,
                subject_visibility=VisibilityState.VISIBLE,
                object_visibility=obj_vis,
                source_model="synthetic_world",
            ))
        frames.append(FrameGraph(
            video_id=video_id, frame_idx=f, timestamp=ts,
            track_observations=tracks, edge_observations=edges,
        ))
    return frames, lifetimes
