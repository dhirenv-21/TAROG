"""Simple 2-state Hidden Markov Model temporal baseline (scaffold).

Models each edge as having two hidden states: ALIVE and DEAD.
Observations (edge present/absent) are emissions from the hidden state.

This is a SCAFFOLD implementation.  The transition and emission
probabilities are hand-set rather than learned from data.  Full
Baum-Welch parameter estimation can be added in a later phase if
this baseline proves useful for comparison.

Method name in configs/plots: "HMM"

Mathematical model:
    Hidden states:  S = {ALIVE, DEAD}
    Observations:   O = {observed, absent}

    Transition matrix A:
        A[ALIVE -> ALIVE] = stay_alive_prob
        A[ALIVE -> DEAD]  = 1 - stay_alive_prob
        A[DEAD  -> ALIVE] = resurrect_prob
        A[DEAD  -> DEAD]  = 1 - resurrect_prob

    Emission matrix B:
        B[ALIVE, observed] = emit_obs_alive
        B[ALIVE, absent]   = 1 - emit_obs_alive
        B[DEAD, observed]  = emit_obs_dead
        B[DEAD, absent]    = 1 - emit_obs_dead

    Forward variable (unnormalized belief):
        At each frame, compute P(state | observations_so_far)
        using the standard HMM forward step.

    alive_score = P(ALIVE | observations)

This baseline is valuable because it provides a principled probabilistic
comparison point that does NOT use visibility information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)


@dataclass
class HMM:
    """Simple 2-state HMM temporal baseline.

    Configuration:
        stay_alive_prob: P(ALIVE_t | ALIVE_{t-1}).
        resurrect_prob: P(ALIVE_t | DEAD_{t-1}).  Usually very low.
        emit_obs_alive: P(observed | ALIVE).  High = edge usually seen when alive.
        emit_obs_dead: P(observed | DEAD).  Low = edge rarely seen when dead.
        prior_alive: Initial P(ALIVE) when an edge is first seen.
        score_threshold: Minimum P(ALIVE) to output an edge.
    """

    stay_alive_prob: float = 0.9
    resurrect_prob: float = 0.01
    emit_obs_alive: float = 0.9
    emit_obs_dead: float = 0.05
    prior_alive: float = 0.5
    score_threshold: float = 0.3

    # Internal state: P(ALIVE | observations_so_far) for each edge
    _video_id: str = field(default="", init=False, repr=False)
    _belief: dict[EdgeKey, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for name, val in [
            ("stay_alive_prob", self.stay_alive_prob),
            ("resurrect_prob", self.resurrect_prob),
            ("emit_obs_alive", self.emit_obs_alive),
            ("emit_obs_dead", self.emit_obs_dead),
            ("prior_alive", self.prior_alive),
        ]:
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {val}")

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._belief.clear()

    def _forward_step(
        self, prior_alive: float, observed: bool
    ) -> float:
        """One step of the HMM forward algorithm.

        Computes P(ALIVE_t | o_{1:t}) from P(ALIVE_{t-1} | o_{1:t-1}).

        Args:
            prior_alive: P(ALIVE_{t-1} | o_{1:t-1}).
            observed: Whether the edge was observed at time t.

        Returns:
            P(ALIVE_t | o_{1:t}).
        """
        prior_dead = 1.0 - prior_alive

        # Predict step: P(state_t | o_{1:t-1})
        pred_alive = (
            self.stay_alive_prob * prior_alive
            + self.resurrect_prob * prior_dead
        )
        pred_dead = (
            (1.0 - self.stay_alive_prob) * prior_alive
            + (1.0 - self.resurrect_prob) * prior_dead
        )

        # Emission likelihood
        if observed:
            lik_alive = self.emit_obs_alive
            lik_dead = self.emit_obs_dead
        else:
            lik_alive = 1.0 - self.emit_obs_alive
            lik_dead = 1.0 - self.emit_obs_dead

        # Update step: P(state_t | o_{1:t})
        unnorm_alive = lik_alive * pred_alive
        unnorm_dead = lik_dead * pred_dead
        total = unnorm_alive + unnorm_dead

        if total < 1e-30:
            # Numerical safeguard
            return 0.5

        return unnorm_alive / total

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Process one frame using HMM forward filtering.

        Args:
            frame: Input frame with observations.

        Returns:
            Frame with edges filtered by HMM belief.
        """
        # Collect which edges are observed this frame
        observed_map: dict[EdgeKey, EdgeObservation] = {}
        for obs in frame.edge_observations:
            if obs.observed:
                observed_map[obs.edge_key] = obs

        # Update beliefs for observed edges
        for edge_key, obs in observed_map.items():
            old_belief = self._belief.get(edge_key, self.prior_alive)
            self._belief[edge_key] = self._forward_step(old_belief, True)

        # Update beliefs for absent edges (already tracked)
        dead_keys: list[EdgeKey] = []
        for edge_key in list(self._belief.keys()):
            if edge_key in observed_map:
                continue
            self._belief[edge_key] = self._forward_step(
                self._belief[edge_key], False
            )
            # Prune negligible beliefs
            if self._belief[edge_key] < 1e-8:
                dead_keys.append(edge_key)

        for k in dead_keys:
            del self._belief[k]

        # Build output: edges with P(ALIVE) >= threshold
        output_edges: list[EdgeObservation] = []
        for edge_key, belief in self._belief.items():
            if belief < self.score_threshold:
                continue
            if edge_key in observed_map:
                orig = observed_map[edge_key]
                output_edges.append(EdgeObservation(
                    edge_key=edge_key,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    predicate_score=belief,
                    observed=True,
                    subject_visibility=orig.subject_visibility,
                    object_visibility=orig.object_visibility,
                    source_model=orig.source_model,
                    source_checkpoint=orig.source_checkpoint,
                ))
            else:
                output_edges.append(EdgeObservation(
                    edge_key=edge_key,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    predicate_score=belief,
                    observed=False,
                    source_model="hmm",
                ))

        return FrameGraph(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=frame.track_observations,
            edge_observations=output_edges,
        )

    def finalize_video(self) -> None:
        """Finalize current video."""
        pass

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {
            "method": "hmm",
            "stay_alive_prob": self.stay_alive_prob,
            "resurrect_prob": self.resurrect_prob,
            "emit_obs_alive": self.emit_obs_alive,
            "emit_obs_dead": self.emit_obs_dead,
            "prior_alive": self.prior_alive,
            "score_threshold": self.score_threshold,
            "active_edges": len(self._belief),
        }
