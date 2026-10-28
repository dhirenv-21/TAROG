"""IA-VHPED: identity-aware V-HPED (eager) and its lazy variant.

Per pair and frame (research/derivations/identity_aware.md §5):

    P(o | past) = zeta * prod_p c_p + (1 - zeta) * prod_p Q(o_p)
    w           = zeta * prod_p c_p / P(o | past)
    a_p         = w * post_p + (1 - w) * abar_p      abar = a_pred (tracked) | b (born)
    traces     += w * marks

where c_p, post_p are V-HPED's one-step evidence and posterior and
Q(o=1) = 1 - exp(-rho_v * phi * dt).  With zeta = 1 this is exactly V-HPED.

The V-HPED baseline in temporal/vhped/ is not modified; this class only
overrides the per-pair update hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from taorg.schemas.edges import EdgeObservation, FrameGraph, PairKey
from taorg.temporal.vhped import dynamics
from taorg.temporal.vhped.alive import filter_step
from taorg.temporal.vhped.lazy import LazyVHPED
from taorg.temporal.vhped.model import VHPED, _PairState
from taorg.temporal.vhped.visibility import pair_regime, thinning, trace_decay
from taorg.temporal.vhped_ia.identity import TrackIdentityFilter
from taorg.temporal.vhped_ia.params import IAParams


@dataclass
class IAVHPED(VHPED):
    """Identity-aware V-HPED (eager reference)."""

    ia: IAParams | None = None
    _identity: TrackIdentityFilter = field(default_factory=TrackIdentityFilter, init=False,
                                           repr=False)
    _now: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ia is None:
            num_p = self.params.num_predicates if self.params is not None else 26
            self.ia = IAParams.default(num_p)
        if self.params is None:
            self.params = self.ia.base
        elif self.params is not self.ia.base:
            raise ValueError("Pass V-HPED parameters through `ia.base`, not `params`")
        super().__post_init__()
        self._identity.recovery_rate = self.ia.recovery_rate
        self._identity.calibration = self.ia.jump_calibration

    def reset_video(self, video_id: str) -> None:
        super().reset_video(video_id)
        self._identity.reset()

    def step(self, frame: FrameGraph) -> FrameGraph:
        self._identity.update(frame)
        self._now = frame.timestamp
        return super().step(frame)

    def pair_identity(self, pk: PairKey) -> float:
        return self._identity.pair(pk, self._now)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d.update(method="vhped_ia", ia=self.ia.to_dict())
        return d

    def _update_pair(self, state: _PairState, pair_obs: list[EdgeObservation], dt: float,
                     pk: PairKey | None = None) -> bool:
        if pk is None:
            raise ValueError("IAVHPED needs the pair key for identity weighting")
        p, ia = self.params, self.ia
        num_p = p.num_predicates
        regime = pair_regime(pair_obs)
        rho = thinning(regime, p)

        observed = np.zeros(num_p, dtype=bool)
        marks = np.zeros(num_p)
        for obs in pair_obs:
            if obs.observed:
                k = obs.edge_key.predicate_id
                observed[k] = True
                marks[k] = max(marks[k], obs.predicate_score if p.use_marks else 1.0)

        state.traces = dynamics.decay_traces(state.traces, trace_decay(regime, p), dt)
        lam = dynamics.intensity(state.traces, p)
        haz = dynamics.hazard(state.traces, p)

        born = observed & ~state.tracked
        a_prev = np.where(born, p.prior_alive, state.belief)
        state.tracked |= observed
        idx = np.flatnonzero(state.tracked)
        w = 1.0
        if idx.size:
            post, log_c = filter_step(a_prev[idx], observed[idx], lam[idx], haz[idx], rho=rho,
                                      eps_rate=p.eps_rate, rebirth_rate=p.rebirth_rate, dt=dt)
            rate = haz[idx] + p.rebirth_rate
            pi = np.where(rate > 0, p.rebirth_rate / np.where(rate > 0, rate, 1.0), 0.0)
            pred = pi + (a_prev[idx] - pi) * np.exp(-rate * dt)

            zeta = self.pair_identity(pk)
            q1 = -np.expm1(-rho * ia.foreign_rate * dt)
            log_q = np.where(observed[idx], np.log(max(q1, 1e-300)), np.log1p(-q1))
            with np.errstate(divide="ignore"):
                log_a = np.log(zeta) + log_c.sum()
                log_b = np.log1p(-zeta) + log_q.sum()
            log_p = np.logaddexp(log_a, log_b)
            w = float(np.exp(log_a - log_p)) if ia.posterior_weight else zeta
            a_bar = np.where(born[idx], ia.foreign_birth, pred) if ia.identity_births else pred
            state.belief[idx] = w * post + (1.0 - w) * a_bar
            self.log_likelihood += float(log_p)

        state.traces = state.traces + w * marks

        if self.prune_below > 0.0:
            drop = state.tracked & (state.belief < self.prune_below)
            state.tracked &= ~drop
            state.belief[drop] = 0.0
            if not state.tracked.any():
                return False
        return True


@dataclass
class LazyIAVHPED(IAVHPED, LazyVHPED):
    """IA-VHPED on the lazy wake-up heap.

    Exact under the V-HPED lazy conditions (rho_unk = 0, gamma = 0): an
    untouched pair's update does not depend on zeta (all likelihood factors
    are 1), see identity_aware.md §7.  Identity beliefs are still updated for
    every track present in a frame.
    """
