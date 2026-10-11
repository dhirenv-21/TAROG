"""Relationship-episode estimator — eager reference (episode-study ladder P1-P4).

Implements research/taorg_episode_formulation.md exactly (frozen at the
episode-study pre-registration, commit 42b61ea).

Per key k:
    IDLE    CUSUM  C <- max(0, C + log L~)       birth when C >= h + h_r exp(-(t - t_close)/tau_r)
    ACTIVE  death filter (Z = 0 absorbing within the episode):
                p- = p exp(-mu_n dt);  p = p- L~1 / (p- L~1 + (1 - p-) L~0)
            terminate (absorbing; state discarded) when p < end_threshold
Output: k iff ACTIVE after the frame.

Observation model: detection (1 - m) vs f; absence with flag F: theta_F vs (1 - f),
theta_VIS = m, theta_F = m + a_F (1 - f - m) for F in {OCC, UNK}  (=> every absence has
likelihood ratio <= 1; Lemma 1: no relapse without detections).
Identity (P2+): L~ = w L + (1 - w) with w from taorg.temporal.identity.
Hazard (P3/P4): mu_n = mu (1 + kappa_c c_n + kappa_h eta_n) with c_n from the
estimator's own ACTIVE set of the same pair after frame n-1 and eta_n from its own
birth times (exponential kernel tau_h).  E is a fitted parameter (calibration GT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.identity import DT_REF, IdentityReliability

P_MIN, P_MAX = 1e-12, 1.0 - 1e-12
P_BIRTH = 0.5          # fresh episode belief before the birth frame's update (formulation §4)


@dataclass
class _KeyState:
    active: bool = False
    p: float = 0.0
    cusum: float = 0.0
    t_close: Optional[float] = None
    episode: int = 0          # index of the current / last episode (1-based once born)


@dataclass
class EpisodeEstimator:
    """Episode estimator (rung chosen by the feature switches)."""

    miss: float = 0.05
    false_pos: float = 0.01
    a_occ: float = 0.9
    a_unk: float = 0.5
    mu: float = 0.3
    end_threshold: float = 0.2
    birth_threshold: float = 5.0
    refractory_boost: float = 0.0
    refractory_tau: float = 0.5
    identity: bool = False
    sigma_id: float = 0.1
    r_id: float = 3.0
    context: bool = False
    kappa_c: float = 0.0
    history: bool = False
    kappa_h: float = 0.0
    tau_h: float = 1.0
    transition: Optional[list[list[float]]] = None
    rung: str = "custom"

    _s: dict[EdgeKey, _KeyState] = field(default_factory=dict, init=False, repr=False)
    _t_prev: Optional[float] = field(default=None, init=False, repr=False)
    _births: dict[tuple[int, int], list[tuple[float, int]]] = field(default_factory=dict, init=False, repr=False)
    _id: Optional[IdentityReliability] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.miss < 1.0 and 0.0 < self.false_pos < 1.0):
            raise ValueError("miss and false_pos must be in (0, 1)")
        if self.miss >= 1.0 - self.false_pos:
            raise ValueError("require miss < 1 - false_pos (detections must be evidence for existence)")
        for n in ("a_occ", "a_unk"):
            if not 0.0 <= getattr(self, n) <= 1.0:
                raise ValueError(f"{n} must be in [0, 1]")
        if self.mu <= 0 or self.refractory_tau <= 0 or self.tau_h <= 0:
            raise ValueError("mu, refractory_tau and tau_h must be > 0")
        if not 0.0 < self.end_threshold <= 0.5:
            raise ValueError("end_threshold must be in (0, 0.5]")
        if self.birth_threshold <= 0 or self.refractory_boost < 0:
            raise ValueError("birth_threshold must be > 0 and refractory_boost >= 0")
        if self.kappa_c < 0 or self.kappa_h < 0:
            raise ValueError("kappa_c and kappa_h must be >= 0")
        if (self.context or self.history) and self.transition is None:
            raise ValueError("context/history rungs require the fitted transition statistic")
        m, f = self.miss, self.false_pos
        self._theta = {VisibilityState.VISIBLE: m,
                       VisibilityState.OCCLUDED_OR_UNRELIABLE: m + self.a_occ * (1.0 - f - m),
                       VisibilityState.UNKNOWN: m + self.a_unk * (1.0 - f - m)}
        if self.identity:
            self._id = IdentityReliability(sigma_id=self.sigma_id, r_id=self.r_id)

    # ------------------------------------------------------------------ helpers
    def reset_video(self, video_id: str) -> None:
        """Reset all state for a new video."""
        self._s.clear()
        self._births.clear()
        self._t_prev = None
        if self._id is not None:
            self._id.reset()

    def likelihoods(self, o: Optional[EdgeObservation], w: float = 1.0) -> tuple[float, float]:
        """(L1, L0) of one record, identity-tempered by w (formulation §2)."""
        if o is not None and o.observed:
            l1, l0 = 1.0 - self.miss, self.false_pos
        else:
            vis = o.edge_visibility if o is not None else VisibilityState.UNKNOWN
            l1, l0 = self._theta[vis], 1.0 - self.false_pos
        if w < 1.0:
            l1, l0 = w * l1 + (1.0 - w), w * l0 + (1.0 - w)
        return l1, l0

    def _hazard(self, k: EdgeKey, t: float, active_prev: dict[tuple[int, int], list[int]]) -> float:
        mult = 1.0
        if self.context or self.history:
            row = self.transition[k.predicate_id]
            pair = (k.subject_track_id, k.object_track_id)
            if self.context and self.kappa_c > 0:
                c = sum(row[r2] for r2 in active_prev.get(pair, ()) if r2 != k.predicate_id)
                mult += self.kappa_c * c
            if self.history and self.kappa_h > 0:
                eta = sum(row[r2] * math.exp(-(t - tb) / self.tau_h)
                          for tb, r2 in self._births.get(pair, ()) if r2 != k.predicate_id and tb < t)
                mult += self.kappa_h * eta
        return self.mu * mult

    def belief(self, key: EdgeKey) -> float:
        """Within-episode existence belief (NaN if IDLE / unknown)."""
        s = self._s.get(key)
        return s.p if (s is not None and s.active) else math.nan

    def cusum(self, key: EdgeKey) -> float:
        """Birth statistic of an IDLE key (NaN if ACTIVE / unknown)."""
        s = self._s.get(key)
        return s.cusum if (s is not None and not s.active) else math.nan

    def episode_index(self, key: EdgeKey) -> int:
        """Number of episodes born so far for a key (0 if none)."""
        s = self._s.get(key)
        return 0 if s is None else s.episode

    # ------------------------------------------------------------------ step
    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame (eager: every maintained key is updated)."""
        t = frame.timestamp
        dt = DT_REF if self._t_prev is None else max(t - self._t_prev, 1e-9)
        self._t_prev = t
        if self._id is not None:
            self._id.update(frame.track_observations, dt)
        records = {o.edge_key: o for o in frame.edge_observations}
        for k, o in records.items():
            if o.observed and k not in self._s:
                self._s[k] = _KeyState()
        active_prev: dict[tuple[int, int], list[int]] = {}
        if self.context:
            for k, s in self._s.items():
                if s.active:
                    active_prev.setdefault((k.subject_track_id, k.object_track_id), []).append(k.predicate_id)
        new_births: list[tuple[EdgeKey, float]] = []
        out: list[EdgeObservation] = []
        for k, s in self._s.items():
            o = records.get(k)
            w = self._id.edge(k) if self._id is not None else 1.0
            l1, l0 = self.likelihoods(o, w)
            if s.active:
                pm = s.p * math.exp(-self._hazard(k, t, active_prev) * dt)
                p = pm * l1 / (pm * l1 + (1.0 - pm) * l0)
                s.p = min(P_MAX, max(P_MIN, p))
                if s.p < self.end_threshold:          # termination: absorbing for this episode
                    s.active, s.p, s.cusum, s.t_close = False, 0.0, 0.0, t
            else:
                s.cusum = max(0.0, s.cusum + math.log(l1 / l0))
                h_eff = self.birth_threshold
                if s.t_close is not None and self.refractory_boost > 0:
                    h_eff += self.refractory_boost * math.exp(-(t - s.t_close) / self.refractory_tau)
                if s.cusum >= h_eff:                  # birth of a NEW episode, fresh state
                    p = P_BIRTH * l1 / (P_BIRTH * l1 + (1.0 - P_BIRTH) * l0)
                    s.active, s.p, s.cusum, s.episode = True, min(P_MAX, max(P_MIN, p)), 0.0, s.episode + 1
                    new_births.append((k, t))
                    if s.p < self.end_threshold:       # cannot happen with end_threshold <= 0.5 (Λ_det > 1)
                        s.active, s.t_close = False, t
            if s.active:
                out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                           timestamp=t, predicate_score=s.p,
                                           observed=bool(o is not None and o.observed),
                                           source_model=f"episode_{self.rung}"))
        if self.history:
            for k, tb in new_births:
                self._births.setdefault((k.subject_track_id, k.object_track_id), []).append((tb, k.predicate_id))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=t,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """No-op."""

    def state_dict(self) -> dict[str, Any]:
        """Configuration and state size."""
        return {"method": f"episode_{self.rung}", "miss": self.miss, "false_pos": self.false_pos,
                "a_occ": self.a_occ, "a_unk": self.a_unk, "mu": self.mu, "end_threshold": self.end_threshold,
                "birth_threshold": self.birth_threshold, "refractory_boost": self.refractory_boost,
                "refractory_tau": self.refractory_tau, "identity": self.identity, "sigma_id": self.sigma_id,
                "r_id": self.r_id, "context": self.context, "kappa_c": self.kappa_c, "history": self.history,
                "kappa_h": self.kappa_h, "tau_h": self.tau_h, "maintained_keys": len(self._s)}


def episode_p1(**p: Any) -> EpisodeEstimator:
    """P1: episode estimator."""
    return EpisodeEstimator(**p, rung="p1")


def episode_p2(**p: Any) -> EpisodeEstimator:
    """P2: + identity reliability."""
    return EpisodeEstimator(**p, identity=True, rung="p2")


def episode_p3(**p: Any) -> EpisodeEstimator:
    """P3: P2 + simple transition context."""
    return EpisodeEstimator(**p, identity=True, context=True, rung="p3")


def episode_p4(**p: Any) -> EpisodeEstimator:
    """P4: P3 + event-history context."""
    return EpisodeEstimator(**p, identity=True, context=True, history=True, rung="p4")
