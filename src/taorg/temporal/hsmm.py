"""Duration-aware (explicit-duration, semi-Markov) two-state filter — baseline B4.

research/taorg_episode_formulation.md §8.  Per key, the forward filter keeps
P(state, age) for state in {ON, OFF} and age bins of width DT_REF = 1/30 s up
to MAX_AGE_S (the last bin absorbs older ages).  Dwell times are Weibull with
(shape, scale [s]) per state; the per-bin leaving probability is the exact
Weibull conditional hazard over the bin, h(a) = 1 - exp(-[(t_{a+1}/l)^k - (t_a/l)^k]),
and the last bin uses the hazard at MAX_AGE_S.  Emissions are those of the
VA-HMM (same parameter ranges): detection 1 - theta_vis vs false_pos;
absence with flag F: theta_F vs 1 - false_pos ("no record" = UNK).
Output iff P(ON) > threshold.  A key is initialised at its first detection with
P(ON, age 0) = prior and P(OFF, age >= MAX_AGE_S) = 1 - prior.

Irregular frame intervals advance ceil(dt / DT_REF - 1e-9) bins (an
approximation stated in the formulation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

DT_REF = 1.0 / 30.0
MAX_AGE_S = 6.0


def weibull_bin_hazard(shape: float, scale: float, n_bins: int, width: float) -> np.ndarray:
    """Per-bin leaving probability of a Weibull dwell time (last bin: hazard at its left edge)."""
    t = np.arange(n_bins + 1) * width
    H = (t / scale) ** shape
    h = 1.0 - np.exp(-(H[1:] - H[:-1]))
    tl = (n_bins - 1) * width
    rate_last = (shape / scale) * (max(tl, width) / scale) ** (shape - 1.0)
    h[-1] = 1.0 - math.exp(-rate_last * width)
    return np.clip(h, 1e-12, 1.0 - 1e-12)


@dataclass
class DurationHSMM:
    """Explicit-duration ON/OFF HSMM forward filter with flag-conditioned emissions."""

    theta_vis: float = 0.05
    theta_occ: float = 0.95
    theta_unk: float = 0.5
    false_pos: float = 0.01
    threshold: float = 0.5
    prior: float = 0.5
    on_shape: float = 1.5
    on_scale: float = 3.0
    off_shape: float = 1.0
    off_scale: float = 5.0

    _on: dict[EdgeKey, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _off: dict[EdgeKey, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _t_prev: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for n in ("theta_vis", "theta_unk", "false_pos", "threshold", "prior"):
            v = getattr(self, n)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{n} must be in (0, 1), got {v}")
        if not 0.0 < self.theta_occ <= 1.0:
            raise ValueError("theta_occ must be in (0, 1]")
        for n in ("on_shape", "on_scale", "off_shape", "off_scale"):
            if getattr(self, n) <= 0:
                raise ValueError(f"{n} must be > 0")
        self._A = int(round(MAX_AGE_S / DT_REF))
        self._h_on = weibull_bin_hazard(self.on_shape, self.on_scale, self._A, DT_REF)
        self._h_off = weibull_bin_hazard(self.off_shape, self.off_scale, self._A, DT_REF)

    def reset_video(self, video_id: str) -> None:
        """Reset state."""
        self._on.clear()
        self._off.clear()
        self._t_prev = None

    def _advance(self, on: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        end_on = on * self._h_on
        end_off = off * self._h_off
        stay_on = on - end_on
        stay_off = off - end_off
        new_on = np.empty_like(on)
        new_off = np.empty_like(off)
        new_on[1:] = stay_on[:-1]
        new_on[-1] += stay_on[-1]
        new_off[1:] = stay_off[:-1]
        new_off[-1] += stay_off[-1]
        new_on[0] = end_off.sum()
        new_off[0] = end_on.sum()
        return new_on, new_off

    def _lik(self, o: EdgeObservation | None) -> tuple[float, float]:
        if o is not None and o.observed:
            return 1.0 - self.theta_vis, self.false_pos
        vis = o.edge_visibility if o is not None else VisibilityState.UNKNOWN
        th = {VisibilityState.VISIBLE: self.theta_vis,
              VisibilityState.OCCLUDED_OR_UNRELIABLE: self.theta_occ}.get(vis, self.theta_unk)
        return th, 1.0 - self.false_pos

    def p_on(self, key: EdgeKey) -> float:
        """Current P(ON) of a maintained key (NaN if not maintained)."""
        on = self._on.get(key)
        return math.nan if on is None else float(on.sum())

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Predict (age shift + Weibull transitions), update with emissions, threshold."""
        dt = DT_REF if self._t_prev is None else max(frame.timestamp - self._t_prev, 1e-9)
        self._t_prev = frame.timestamp
        n_sub = max(1, math.ceil(dt / DT_REF - 1e-9))
        records = {o.edge_key: o for o in frame.edge_observations}
        for k, o in records.items():
            if o.observed and k not in self._on:
                on = np.zeros(self._A)
                off = np.zeros(self._A)
                on[0] = self.prior
                off[-1] = 1.0 - self.prior
                self._on[k], self._off[k] = on, off
        out = []
        for k in list(self._on):
            on, off = self._on[k], self._off[k]
            for _ in range(n_sub):
                on, off = self._advance(on, off)
            l1, l0 = self._lik(records.get(k))
            on = on * l1
            off = off * l0
            tot = on.sum() + off.sum()
            if not (tot > 0 and math.isfinite(tot)):
                raise FloatingPointError(f"HSMM normalisation failed for {k}")
            on /= tot
            off /= tot
            self._on[k], self._off[k] = on, off
            p = float(on.sum())
            if p > self.threshold:
                o = records.get(k)
                out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                           timestamp=frame.timestamp, predicate_score=p,
                                           observed=bool(o is not None and o.observed), source_model="hsmm"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """No-op."""

    def state_dict(self) -> dict:
        """Configuration."""
        return {"method": "hsmm", "theta_vis": self.theta_vis, "theta_occ": self.theta_occ,
                "theta_unk": self.theta_unk, "false_pos": self.false_pos, "threshold": self.threshold,
                "prior": self.prior, "on_shape": self.on_shape, "on_scale": self.on_scale,
                "off_shape": self.off_shape, "off_scale": self.off_scale, "active_edges": len(self._on)}
