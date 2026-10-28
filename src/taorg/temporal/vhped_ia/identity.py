"""Per-track identity belief zeta (research/derivations/identity_aware.md §3, §11).

zeta_i = P(label i refers to its home object | kappa_{<= n}).  Per frame, for
every track present: recovery over the elapsed interval, then the jump toggle
    zeta <- (1 - J) * zeta + J * (1 - zeta),     J = P(jump | kappa).
A track's first appearance sets zeta = 1.  Tracks absent from a frame are not
updated (their recovery is applied when next queried).  kappa = None -> no jump.

Jump probability (IA-v2, §11): a tracker score is not a calibrated probability, so
    J = sigmoid(a + b * logit(1 - kappa))          (learned calibration, b >= 0)
With `calibration=None` the raw v1 mapping J = 1 - kappa is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from taorg.schemas.edges import FrameGraph, PairKey

KAPPA_CLIP = 1e-6


def jump_probability(kappa: float | None, calibration: tuple[float, float] | None) -> float:
    """P(label re-assigned at this frame | tracker score kappa)."""
    if kappa is None:
        return 0.0
    if calibration is None:
        return 1.0 - kappa
    if kappa >= 1.0:
        return 0.0
    a, b = calibration
    k = min(max(kappa, KAPPA_CLIP), 1.0 - KAPPA_CLIP)
    x = a + b * (math.log1p(-k) - math.log(k))  # a + b * logit(1 - kappa)
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


@dataclass
class TrackIdentityFilter:
    """Maintains zeta for every track label of the current video."""

    recovery_rate: float = 0.0
    calibration: tuple[float, float] | None = None
    _zeta: dict[int, float] = field(default_factory=dict)
    _t: dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._zeta.clear()
        self._t.clear()

    def _recovered(self, track: int, t: float) -> float:
        z = self._zeta[track]
        if self.recovery_rate > 0.0:
            z = 1.0 - (1.0 - z) * math.exp(-self.recovery_rate * (t - self._t[track]))
        return z

    def update(self, frame: FrameGraph) -> None:
        """Apply this frame's identity scores (call before any pair update)."""
        t = frame.timestamp
        for obs in frame.track_observations:
            kappa = obs.identity_confidence
            if kappa is not None and not 0.0 <= kappa <= 1.0:
                raise ValueError(f"identity_confidence {kappa} outside [0, 1] "
                                 f"(track {obs.track_id}, frame {frame.frame_idx})")
            if obs.track_id not in self._zeta:
                self._zeta[obs.track_id] = 1.0
            else:
                z = self._recovered(obs.track_id, t)
                j = jump_probability(kappa, self.calibration)
                self._zeta[obs.track_id] = (1.0 - j) * z + j * (1.0 - z)
            self._t[obs.track_id] = t

    def track(self, track: int, t: float) -> float:
        """zeta of a track at time t (1.0 for labels never seen with a TrackObservation)."""
        if track not in self._zeta:
            return 1.0
        return self._recovered(track, t)

    def pair(self, pk: PairKey, t: float) -> float:
        """Pair identity belief zeta_ij = zeta_i * zeta_j (independent tracks)."""
        return self.track(pk.subject_track_id, t) * self.track(pk.object_track_id, t)
