"""Calibration of edge-alive scores: ECE and Brier score.

Definition frozen in research/metric_definitions.md §6 (frozen 2026-10-28,
before any calibration result was computed).

Evaluation units: every (edge, frame) for which the method outputs a score
when run with score_threshold = 0 (i.e. every edge it is tracking).  Label =
1 if the edge is observed in the uncorrupted GT at that frame, else 0.
Untracked (edge, frame) units are excluded — calibration says nothing about
recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from taorg.metrics.threshold_sweep import VideoScores


@dataclass
class CalibrationAccumulator:
    """Pools (score, label) units over videos.

    Attributes:
        num_bins: Equal-width bins on [0, 1] for ECE.
    """

    num_bins: int = 10
    _scores: list[np.ndarray] = field(default_factory=list)
    _labels: list[np.ndarray] = field(default_factory=list)

    def add(self, v: VideoScores) -> None:
        mask = np.isfinite(v.score)
        self._scores.append(np.clip(v.score[mask], 0.0, 1.0))
        self._labels.append(v.gt[mask].astype(np.float64))

    def result(self) -> dict[str, float]:
        if not self._scores:
            return {"ece": 0.0, "brier": 0.0, "num_units": 0}
        s = np.concatenate(self._scores)
        y = np.concatenate(self._labels)
        return {"ece": expected_calibration_error(s, y, self.num_bins),
                "brier": float(np.mean((s - y) ** 2)) if s.size else 0.0,
                "num_units": int(s.size)}


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) |mean(label_b) - mean(score_b)| over equal-width bins.

    Bin b covers [b/B, (b+1)/B); the last bin includes 1.0.  Empty input -> 0.
    """
    if scores.size == 0:
        return 0.0
    bins = np.minimum((scores * num_bins).astype(int), num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - scores[m].mean())
    return float(ece)
