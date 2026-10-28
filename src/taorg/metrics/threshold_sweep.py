"""Vectorised evaluation of the frozen metrics over many score thresholds.

Every temporal method's internal state is independent of its output
threshold, so running it once with score_threshold = 0 and filtering
`predicate_score >= t` reproduces a run at threshold t exactly (tested per
method).  This module computes the frozen metrics of
research/metric_definitions.md for all thresholds at once from per-edge
score arrays.  Tests assert equality with the reference implementations in
persistence.py, churn.py and termination.py.

Pooling across videos: TP/FP/FN are summed (micro); per-edge quantities
(termination delay, false persistence, churn) are pooled over all edges of
all videos and then averaged.

Shapes: K thresholds, E edges, T frames (frame_idx must equal list position).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from taorg.noise.manifest import CorruptionManifest
from taorg.schemas.edges import EdgeKey, FrameGraph


@dataclass
class VideoScores:
    """Per-video arrays for one method.

    Attributes:
        gt: GT-observed indicator, (E, T) bool.
        score: Output predicate_score, -inf where the edge was not output, (E, T).
        corrupted: Frames inside an occlusion interval of the manifest, (E, T) bool.
    """

    gt: np.ndarray
    score: np.ndarray
    corrupted: np.ndarray


def video_scores(
    gt_frames: list[FrameGraph],
    pred_frames: list[FrameGraph],
    manifest: CorruptionManifest | None = None,
) -> VideoScores:
    """Build score arrays for one video from GT frames and a threshold-0 run."""
    if len(gt_frames) != len(pred_frames):
        raise ValueError(f"Frame count mismatch: GT={len(gt_frames)}, pred={len(pred_frames)}")
    for i, f in enumerate(gt_frames):
        if f.frame_idx != i:
            raise ValueError("threshold_sweep requires frame_idx == position in the video")
    index: dict[EdgeKey, int] = {}
    for frames in (gt_frames, pred_frames):
        for f in frames:
            for o in f.edge_observations:
                if (frames is pred_frames or o.observed) and o.edge_key not in index:
                    index[o.edge_key] = len(index)
    e_n, t_n = len(index), len(gt_frames)
    gt = np.zeros((e_n, t_n), dtype=bool)
    score = np.full((e_n, t_n), -np.inf)
    corrupted = np.zeros((e_n, t_n), dtype=bool)
    for t, f in enumerate(gt_frames):
        for o in f.edge_observations:
            if o.observed:
                gt[index[o.edge_key], t] = True
    for t, f in enumerate(pred_frames):
        for o in f.edge_observations:
            e = index[o.edge_key]
            score[e, t] = max(score[e, t], o.predicate_score)
    if manifest is not None:
        for entry in manifest.entries:
            if entry.corruption_type != "occlusion" or entry.edge_key is None:
                continue
            e = index.get(entry.edge_key)
            if e is not None:
                corrupted[e, max(entry.start_frame, 0): min(entry.end_frame + 1, t_n)] = True
    return VideoScores(gt=gt, score=score, corrupted=corrupted)


@dataclass
class SweepAccumulator:
    """Accumulates metric sufficient statistics for K thresholds over videos."""

    thresholds: np.ndarray
    frame_tp: np.ndarray = field(init=False)
    frame_fp: np.ndarray = field(init=False)
    frame_fn: np.ndarray = field(init=False)
    pers_tp: np.ndarray = field(init=False)
    pers_fp: np.ndarray = field(init=False)
    pers_fn: np.ndarray = field(init=False)
    delays: list[np.ndarray] = field(init=False)
    censored: np.ndarray = field(init=False)
    fpr_sum: np.ndarray = field(init=False)
    churn_sum: np.ndarray = field(init=False)
    churn_edges: np.ndarray = field(init=False)
    n_terminating: int = field(init=False, default=0)
    n_fpr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.thresholds = np.asarray(self.thresholds, dtype=np.float64)
        k = len(self.thresholds)
        for name in ("frame_tp", "frame_fp", "frame_fn", "pers_tp", "pers_fp", "pers_fn",
                     "censored", "churn_edges"):
            setattr(self, name, np.zeros(k, dtype=np.int64))
        self.fpr_sum = np.zeros(k)
        self.churn_sum = np.zeros(k)
        self.delays = []

    def add(self, v: VideoScores) -> None:
        """Add one video's statistics."""
        e_n, t_n = v.gt.shape
        if e_n == 0 or t_n == 0:
            return
        pred = v.score[None] >= self.thresholds[:, None, None]  # (K, E, T)
        gt = v.gt[None]
        self.frame_tp += (pred & gt).sum((1, 2))
        self.frame_fp += (pred & ~gt).sum((1, 2))
        self.frame_fn += (~pred & gt).sum((1, 2))
        c = v.corrupted[None]
        self.pers_tp += (pred & gt & c).sum((1, 2))
        self.pers_fp += (pred & ~gt & c).sum((1, 2))
        self.pers_fn += (~pred & gt & c).sum((1, 2))

        frames = np.arange(t_n)
        has_gt = v.gt.any(1)
        last = np.where(has_gt, t_n - 1 - np.argmax(v.gt[:, ::-1], axis=1), -1)
        first = np.where(has_gt, np.argmax(v.gt, axis=1), -1)

        # Termination delay + false persistence (edges with GT end before last frame).
        term = has_gt & (last < t_n - 1)
        if term.any():
            t_end = last[term]
            post = frames[None, :] > t_end[:, None]  # (Et, T)
            p_term = pred[:, term]  # (K, Et, T)
            drop = ~p_term & post[None]
            dropped = drop.any(2)
            t_drop = np.argmax(drop, axis=2)
            delay = np.where(dropped, t_drop - t_end[None], t_n - t_end[None])
            self.delays.append(delay)
            self.censored += (~dropped).sum(1)
            self.n_terminating += int(term.sum())
            self.fpr_sum += ((p_term & post[None]).sum(2) / post.sum(1)[None]).sum(1)
            self.n_fpr += int(term.sum())

        # Churn: GT edges over their GT window; predicted-only edges count with 0.
        window = has_gt[:, None] & (frames[None] >= first[:, None]) & (frames[None] <= last[:, None])
        pairs = window[:, 1:] & window[:, :-1]
        flips = ((pred[..., 1:] != pred[..., :-1]) & pairs[None]).sum(2)  # (K, E)
        n_pairs = pairs.sum(1)
        rate = np.where(n_pairs > 0, flips / np.maximum(n_pairs, 1), 0.0)
        self.churn_sum += rate.sum(1)
        pred_only = (~has_gt)[None] & pred.any(2)
        self.churn_edges += int(has_gt.sum()) + pred_only.sum(1)

    def results(self) -> list[dict[str, float]]:
        """Metrics per threshold, in threshold order."""
        def f1(tp, fp, fn):
            p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
            r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
            return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-300), 0.0), r

        frame_f1, _ = f1(self.frame_tp, self.frame_fp, self.frame_fn)
        pers_f1, pers_rec = f1(self.pers_tp, self.pers_fp, self.pers_fn)
        delays = np.concatenate(self.delays, axis=1) if self.delays else None
        out = []
        for k, thr in enumerate(self.thresholds):
            d = np.sort(delays[k]) if delays is not None else np.array([])
            out.append({
                "threshold": float(thr),
                "frame_f1": float(frame_f1[k]),
                "persistence_f1": float(pers_f1[k]),
                "persistence_recall": float(pers_rec[k]),
                "termination_delay_mean": float(d.mean()) if d.size else 0.0,
                "termination_delay_median": float(d[len(d) // 2]) if d.size else 0.0,
                "termination_censored_frac": float(self.censored[k] / self.n_terminating)
                if self.n_terminating else 0.0,
                "false_persistence_rate": float(self.fpr_sum[k] / self.n_fpr) if self.n_fpr else 0.0,
                "churn": float(self.churn_sum[k] / self.churn_edges[k]) if self.churn_edges[k] else 0.0,
                "num_terminating_edges": self.n_terminating,
                "persistence_frames": int(self.pers_tp[k] + self.pers_fn[k]),
            })
        return out


def hypervolume(points: list[tuple[float, float]]) -> float:
    """Area dominated by (cost x, benefit y) points, reference (x=1, y=0).

    Used for the persistence-vs-false-persistence trade-off: x = false
    persistence rate (lower better), y = persistence F1 (higher better).
    Analysis aid only — not a frozen metric.
    """
    pts = sorted((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in points)
    area, best_y = 0.0, 0.0
    for i, (x, y) in enumerate(pts):
        best_y = max(best_y, y)
        x_next = pts[i + 1][0] if i + 1 < len(pts) else 1.0
        area += (x_next - x) * best_y
    return area
