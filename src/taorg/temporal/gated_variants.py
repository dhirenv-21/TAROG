"""GatedEMA variants for the episode study (baselines B2 and B5).

B2 GatedEMA + hysteresis: the unchanged GatedEMA score; the output turns ON when
    score >= on_threshold and OFF when score < off_ratio * on_threshold, and keeps
    its previous state in between.
B5 GatedEMA + identity: GatedEMA whose updates are tempered by the identity
    reliability w of the shared module (taorg.temporal.identity):
        detection:  s <- s + w alpha (x - s)
        absence:    s <- s * decay**w
    w = 1 recovers GatedEMA exactly.

GatedEMA itself (taorg.temporal.gated_ema) is NOT modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.identity import DT_REF, IdentityReliability


@dataclass
class GatedEMAHysteresis:
    """B2: GatedEMA score with a two-threshold (hysteresis) output decision."""

    alpha: float = 0.3
    decay_visible: float = 0.7
    decay_occluded: float = 0.95
    decay_unknown: float = -1.0
    score_threshold: float = 0.3          # = theta_on
    off_ratio: float = 0.5                # theta_off = off_ratio * theta_on

    _inner: GatedEMA = field(init=False, repr=False)
    _on: set = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.off_ratio < 1.0:
            raise ValueError("off_ratio must be in (0, 1)")
        if self.score_threshold <= 0:
            raise ValueError("score_threshold (theta_on) must be > 0")
        self._inner = GatedEMA(alpha=self.alpha, decay_visible=self.decay_visible,
                               decay_occluded=self.decay_occluded, decay_unknown=self.decay_unknown,
                               score_threshold=0.0)

    def reset_video(self, video_id: str) -> None:
        """Reset state."""
        self._inner.reset_video(video_id)
        self._on.clear()

    def step(self, frame: FrameGraph) -> FrameGraph:
        """GatedEMA update, then hysteresis on the maintained scores."""
        inner = self._inner.step(frame)                 # threshold 0 -> every maintained key
        on_t, off_t = self.score_threshold, self.off_ratio * self.score_threshold
        out = []
        for o in inner.edge_observations:
            k, s = o.edge_key, o.predicate_score
            if k in self._on:
                if s < off_t:
                    self._on.discard(k)
            elif s >= on_t:
                self._on.add(k)
            if k in self._on:
                out.append(o)
        alive = {o.edge_key for o in inner.edge_observations}
        self._on &= alive                                   # keys pruned by GatedEMA (< 1e-8) are off
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """No-op."""

    def state_dict(self) -> dict:
        """Configuration."""
        return {"method": "gated_ema_hyst", **self._inner.state_dict(), "theta_on": self.score_threshold,
                "off_ratio": self.off_ratio, "on_edges": len(self._on)}


@dataclass
class GatedEMAIdentity(GatedEMA):
    """B5: GatedEMA with identity-tempered updates (shared identity module)."""

    sigma_id: float = 0.1
    r_id: float = 3.0
    _id: IdentityReliability = field(init=False, repr=False)
    _t_prev: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._id = IdentityReliability(sigma_id=self.sigma_id, r_id=self.r_id)

    def reset_video(self, video_id: str) -> None:
        """Reset state."""
        super().reset_video(video_id)
        self._id.reset()
        self._t_prev = None

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Identity update, then GatedEMA with tempered evidence."""
        dt = DT_REF if self._t_prev is None else max(frame.timestamp - self._t_prev, 1e-9)
        self._t_prev = frame.timestamp
        self._id.update(frame.track_observations, dt)
        records: dict[EdgeKey, EdgeObservation] = {o.edge_key: o for o in frame.edge_observations}
        for k, o in records.items():
            if o.observed:
                w = self._id.edge(k)
                old = self._edge_scores.get(k, 0.0)
                self._edge_scores[k] = old + w * self.alpha * (o.predicate_score - old)
        dead = []
        for k in list(self._edge_scores):
            o = records.get(k)
            if o is not None and o.observed:
                continue
            vis = o.edge_visibility if o is not None else VisibilityState.UNKNOWN
            w = self._id.edge(k)
            self._edge_scores[k] *= self._get_decay_for_visibility(vis) ** w
            if self._edge_scores[k] < 1e-8:
                dead.append(k)
        for k in dead:
            del self._edge_scores[k]
        out = []
        for k, s in self._edge_scores.items():
            if s < self.score_threshold or s <= 0.0:
                continue
            o = records.get(k)
            out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                       timestamp=frame.timestamp, predicate_score=s,
                                       observed=bool(o is not None and o.observed), source_model="gated_ema_id"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          track_observations=frame.track_observations, edge_observations=out)

    def state_dict(self) -> dict:
        """Configuration."""
        d = super().state_dict()
        d.update(method="gated_ema_id", sigma_id=self.sigma_id, r_id=self.r_id)
        return d
