"""V-HPED — Visibility-aware Hawkes-Process Edge Dynamics (eager reference).

Implements the TemporalEdgeModel protocol.  Every active pair is updated
every frame (eager); the lazy/expiry-heap version is Phase 5 and must be
checked against this class.

Per frame, for each active subject–object pair:
    1. decay the pair's excitation traces with beta_v for its regime v
    2. lambda = softplus(mu + e @ alpha), h = softplus(eta + e @ gamma)
    3. alive/dead filter step with visibility-thinned emissions
    4. add this frame's events (marks) to the traces

Output predicate_score is the alive probability a_p.  An edge is emitted
iff a_p >= score_threshold (threshold domain = alive probability).

Method name in configs/plots: "V-HPED"
See research/derivations/vhped_model.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph, PairKey
from taorg.temporal.vhped import dynamics
from taorg.temporal.vhped.alive import filter_step
from taorg.temporal.vhped.params import VHPEDParams
from taorg.temporal.vhped.visibility import pair_regime, thinning, trace_decay

logger = logging.getLogger(__name__)

# Hand-set reference parameters (H3 "no MLE" condition).  Mirrors the
# `vhped` section of configs/temporal/vhped.yaml; a test keeps them in sync.
HAND_SET_CONFIG: dict[str, Any] = {
    "num_predicates": 26,
    "mu": 5.0,
    "alpha_self": 7.0,
    "alpha_cross": 0.0,
    "eta": -0.5,
    "gamma_self": 0.0,
    "gamma_cross": 0.0,
    "beta_vis": 3.0,
    "beta_occ": 0.3,
    "beta_unk": 1.0,
    "rho_occ": 0.05,
    "rho_unk": 0.5,
    "eps_rate": 1.0,
    "rebirth_rate": 0.0,
    "prior_alive": 0.5,
    "use_marks": True,
}


@dataclass
class _PairState:
    """Mutable per-pair state.  Arrays have shape (P,)."""

    traces: np.ndarray
    belief: np.ndarray
    tracked: np.ndarray  # bool: predicate has been born on this pair


@dataclass
class VHPED:
    """V-HPED temporal edge model (eager reference implementation).

    Configuration:
        params: Model parameters.  Defaults to HAND_SET_CONFIG.
        score_threshold: Keep an edge iff P(alive) >= this value.
        fps: Used only for the first frame's interval (1 / fps seconds).
        prune_below: Untrack a predicate when its belief falls below this.
            Set to 0.0 for exact equivalence with likelihood.py.
    """

    params: VHPEDParams | None = None
    score_threshold: float = 0.5
    fps: float = 30.0
    prune_below: float = 1e-6

    _video_id: str = field(default="", init=False, repr=False)
    _pairs: dict[PairKey, _PairState] = field(default_factory=dict, init=False, repr=False)
    _last_timestamp: float | None = field(default=None, init=False, repr=False)
    log_likelihood: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = VHPEDParams.from_config(HAND_SET_CONFIG)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be in [0, 1], got {self.score_threshold}")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if not 0.0 <= self.prune_below < 1.0:
            raise ValueError(f"prune_below must be in [0, 1), got {self.prune_below}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VHPED:
        """Build from a config mapping with model params + runtime keys."""
        params = VHPEDParams.from_config(cfg)
        params = params.apply_ablation(
            use_visibility=cfg.get("use_visibility", True),
            use_excitation=cfg.get("use_excitation", True),
            use_cross_predicate=cfg.get("use_cross_predicate", True),
        )
        return cls(
            params=params,
            score_threshold=float(cfg.get("score_threshold", 0.5)),
            fps=float(cfg.get("fps", 30.0)),
            prune_below=float(cfg.get("prune_below", 1e-6)),
        )

    # ------------------------------------------------------------------
    # TemporalEdgeModel protocol
    # ------------------------------------------------------------------

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._pairs.clear()
        self._last_timestamp = None
        self.log_likelihood = 0.0

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame; return edges whose alive probability passes the threshold."""
        dt = self._interval(frame.timestamp)
        by_pair, observed_map, absent_map = self._parse_frame(frame)

        for pk, obs_list in by_pair.items():
            if pk not in self._pairs and any(o.observed for o in obs_list):
                self._pairs[pk] = self._new_pair()

        dead_pairs = [
            pk for pk, state in self._pairs.items()
            if not self._update_pair(state, by_pair.get(pk, []), dt, pk=pk)
        ]
        for pk in dead_pairs:
            del self._pairs[pk]

        return FrameGraph(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=frame.track_observations,
            edge_observations=self._emit(frame, observed_map, absent_map),
        )

    def _parse_frame(
        self, frame: FrameGraph
    ) -> tuple[dict[PairKey, list[EdgeObservation]], dict[EdgeKey, EdgeObservation],
               dict[EdgeKey, EdgeObservation]]:
        """Group a frame's records by pair; validate predicate ids."""
        num_p = self.params.num_predicates
        by_pair: dict[PairKey, list[EdgeObservation]] = {}
        observed_map: dict[EdgeKey, EdgeObservation] = {}
        absent_map: dict[EdgeKey, EdgeObservation] = {}
        for obs in frame.edge_observations:
            if obs.edge_key.predicate_id >= num_p or obs.edge_key.predicate_id < 0:
                raise ValueError(
                    f"predicate_id {obs.edge_key.predicate_id} outside V-HPED vocabulary "
                    f"[0, {num_p}); set num_predicates in configs/temporal/vhped.yaml"
                )
            by_pair.setdefault(obs.pair_key, []).append(obs)
            (observed_map if obs.observed else absent_map)[obs.edge_key] = obs
        return by_pair, observed_map, absent_map

    def _new_pair(self) -> _PairState:
        num_p = self.params.num_predicates
        return _PairState(
            traces=np.zeros(num_p),
            belief=np.zeros(num_p),
            tracked=np.zeros(num_p, dtype=bool),
        )

    def _update_pair(self, state: _PairState, pair_obs: list[EdgeObservation], dt: float,
                     pk: PairKey | None = None) -> bool:
        """One eager frame step for a pair.  Returns False if the pair is now empty.

        `pk` is unused here; subclasses (identity-aware V-HPED) need it.
        """
        params = self.params
        num_p = params.num_predicates
        regime = pair_regime(pair_obs)

        observed = np.zeros(num_p, dtype=bool)
        marks = np.zeros(num_p)
        for obs in pair_obs:
            if obs.observed:
                p = obs.edge_key.predicate_id
                observed[p] = True
                marks[p] = max(marks[p], obs.predicate_score if params.use_marks else 1.0)

        state.traces = dynamics.decay_traces(state.traces, trace_decay(regime, params), dt)
        lam = dynamics.intensity(state.traces, params)
        haz = dynamics.hazard(state.traces, params)

        born = observed & ~state.tracked
        state.belief[born] = params.prior_alive
        state.tracked |= observed
        idx = np.flatnonzero(state.tracked)
        if idx.size:
            post, log_ev = filter_step(
                state.belief[idx], observed[idx], lam[idx], haz[idx],
                rho=thinning(regime, params),
                eps_rate=params.eps_rate,
                rebirth_rate=params.rebirth_rate,
                dt=dt,
            )
            state.belief[idx] = post
            self.log_likelihood += float(log_ev.sum())

        state.traces = state.traces + marks

        if self.prune_below > 0.0:
            drop = state.tracked & (state.belief < self.prune_below)
            state.tracked &= ~drop
            state.belief[drop] = 0.0
            if not state.tracked.any():
                return False
        return True

    def finalize_video(self) -> None:
        """Finalize current video."""
        logger.debug(
            "V-HPED video %s: log-likelihood %.4f, %d pairs active at end",
            self._video_id, self.log_likelihood, len(self._pairs),
        )

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {
            "method": "vhped",
            "params": self.params.to_dict(),
            "score_threshold": self.score_threshold,
            "fps": self.fps,
            "prune_below": self.prune_below,
            "active_pairs": len(self._pairs),
            "log_likelihood": self.log_likelihood,
        }

    # ------------------------------------------------------------------
    # Inspection helpers (tests, analysis)
    # ------------------------------------------------------------------

    def alive_probability(self, edge_key: EdgeKey) -> float:
        """Current P(alive) for an edge; 0.0 if untracked."""
        state = self._pairs.get(PairKey(edge_key.subject_track_id, edge_key.object_track_id))
        if state is None or not state.tracked[edge_key.predicate_id]:
            return 0.0
        return float(state.belief[edge_key.predicate_id])

    def current_intensity(self, pair: PairKey) -> np.ndarray:
        """Intensity lambda (1/s) from the pair's current traces, shape (P,)."""
        state = self._pairs.get(pair)
        traces = np.zeros(self.params.num_predicates) if state is None else state.traces
        return dynamics.intensity(traces, self.params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _interval(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            dt = 1.0 / self.fps
        else:
            dt = timestamp - self._last_timestamp
            if dt <= 0.0:
                raise ValueError(
                    f"Non-increasing timestamp {timestamp} after {self._last_timestamp} "
                    f"in video {self._video_id}; frames must be in time order"
                )
        self._last_timestamp = timestamp
        return dt

    def _emit(
        self,
        frame: FrameGraph,
        observed_map: dict[EdgeKey, EdgeObservation],
        absent_map: dict[EdgeKey, EdgeObservation],
    ) -> list[EdgeObservation]:
        output: list[EdgeObservation] = []
        for pk, state in self._pairs.items():
            for p in np.flatnonzero(state.tracked):
                belief = float(state.belief[p])
                if belief < self.score_threshold:
                    continue
                key = EdgeKey(pk.subject_track_id, pk.object_track_id, int(p))
                src = observed_map.get(key) or absent_map.get(key)
                if src is not None:
                    output.append(EdgeObservation(
                        edge_key=key,
                        video_id=frame.video_id,
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                        predicate_score=belief,
                        observed=src.observed,
                        subject_visibility=src.subject_visibility,
                        object_visibility=src.object_visibility,
                        source_model=src.source_model if src.observed else "vhped",
                        source_checkpoint=src.source_checkpoint if src.observed else "",
                    ))
                else:
                    output.append(EdgeObservation(
                        edge_key=key,
                        video_id=frame.video_id,
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                        predicate_score=belief,
                        observed=False,
                        source_model="vhped",
                    ))
        return output
