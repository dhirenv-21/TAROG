"""Visibility-aware HMM temporal baseline (Phase 4 rung M1).

Mandatory Phase 4 baseline (research/vhped_formulation.md §8).  It receives
exactly the visibility information available to V-HPED (the flag of each
absence record, or "no record"), but uses it only through per-flag emission
likelihoods in a discrete-time, per-frame, two-state HMM.  Its role is to
answer: "is any V-HPED improvement caused merely by using visibility
probabilistically?"

Model (per edge, per frame):
    Hidden Z in {0 (not existing), 1 (existing)}.
    Transition: P(Z_t=1 | Z_{t-1}=1) = p_stay,  P(Z_t=1 | Z_{t-1}=0) = p_birth.
    Emission, detection D and flag F of the record:
        P(D=1 | Z=1)          = 1 - theta_F   (theta_F for the flag the absence would carry)
        P(D=0 | Z=1, F)       = theta_F,  F in {VIS, OCC, UNK}; "no record" uses UNK
        P(D=1 | Z=0)          = false_pos,   P(D=0 | Z=0) = 1 - false_pos
    For a detected record the flag is VISIBLE, so its likelihood is
    (1 - theta_VIS) vs false_pos.
    Output: edge emitted iff P(Z_t=1 | history) > threshold.

A key starts being tracked when first detected, with P(Z=1) = prior before the
update.  Parameters are per frame (no frame-rate invariance by design).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState


@dataclass
class VisibilityAwareHMM:
    """Discrete-time 2-state HMM with flag-conditioned absence likelihoods."""

    p_stay: float = 0.99
    p_birth: float = 0.001
    prior: float = 0.5
    theta_vis: float = 0.05
    theta_occ: float = 0.95
    theta_unk: float = 0.5
    false_pos: float = 0.01
    threshold: float = 0.5

    _video_id: str = field(default="", init=False, repr=False)
    _belief: dict[EdgeKey, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("p_stay", "p_birth", "prior", "theta_vis", "theta_occ", "theta_unk",
                     "false_pos", "threshold"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0) and not (name == "theta_occ" and v == 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {v}")

    def reset_video(self, video_id: str) -> None:
        """Reset internal state for a new video."""
        self._video_id = video_id
        self._belief.clear()

    def _theta(self, vis: VisibilityState) -> float:
        if vis == VisibilityState.VISIBLE:
            return self.theta_vis
        if vis == VisibilityState.OCCLUDED_OR_UNRELIABLE:
            return self.theta_occ
        return self.theta_unk

    def _step_one(self, p: float, detected: bool, theta: float) -> float:
        pred = self.p_stay * p + self.p_birth * (1.0 - p)
        if detected:
            l1, l0 = 1.0 - self.theta_vis, self.false_pos
        else:
            l1, l0 = theta, 1.0 - self.false_pos
        a, b = l1 * pred, l0 * (1.0 - pred)
        tot = a + b
        return a / tot if tot > 0 else pred

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Filter one frame and emit edges with belief above threshold."""
        records: dict[EdgeKey, EdgeObservation] = {o.edge_key: o for o in frame.edge_observations}
        for k, o in records.items():
            if o.observed and k not in self._belief:
                self._belief[k] = self.prior
        out: list[EdgeObservation] = []
        for k in list(self._belief):
            o = records.get(k)
            if o is not None and o.observed:
                p = self._step_one(self._belief[k], True, 0.0)
            else:
                vis = o.edge_visibility if o is not None else VisibilityState.UNKNOWN
                p = self._step_one(self._belief[k], False, self._theta(vis))
            self._belief[k] = p
            if p > self.threshold:
                out.append(EdgeObservation(
                    edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp, predicate_score=p,
                    observed=bool(o is not None and o.observed), source_model="va_hmm"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx,
                          timestamp=frame.timestamp, track_observations=frame.track_observations,
                          edge_observations=out)

    def finalize_video(self) -> None:
        """Finalize current video (no-op)."""

    def state_dict(self) -> dict:
        """Serializable configuration and state size."""
        return {"method": "va_hmm", "p_stay": self.p_stay, "p_birth": self.p_birth,
                "prior": self.prior, "theta_vis": self.theta_vis, "theta_occ": self.theta_occ,
                "theta_unk": self.theta_unk, "false_pos": self.false_pos,
                "threshold": self.threshold, "active_edges": len(self._belief)}
