"""Per-track identity reliability from constant-velocity prediction residuals.

research/taorg_episode_formulation.md §5.  Shared VERBATIM by the episode ladder
(P2+) and by the GatedEMA+identity baseline (B5), so that "the same identity
mechanism" is literally the same code.

Per track: centre c, diagonal D, velocity v (px/s), reliability u in [0, 1].
    recovery   u <- 1 - (1 - u) exp(-r_id dt)
    residual   d = |c - (c_prev + v dt)| / D
    g0(d)      2-D Gaussian density, sigma = sigma_id * sqrt(max(1, dt/dt_ref))
    g1(d)      uniform density on a disc of radius D_MAX (diagonals)
    switch     s = h g1 / (h g1 + (1 - h) g0),  h = 1 - exp(-H dt)
    update     u <- u (1 - s);  v updated only if s < 0.5;  c_prev <- c
Edge reliability w(s, o) = u(s) u(o).  Inputs: track boxes and timestamps only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey
from taorg.schemas.tracks import TrackObservation

SWITCH_HAZARD = 0.05      # H, 1/s
D_MAX = 3.0               # box diagonals
VEL_GAIN = 0.5            # beta
DT_REF = 1.0 / 30.0


@dataclass
class IdentityReliability:
    """Identity-reliability tracker (see module docstring)."""

    sigma_id: float = 0.1
    r_id: float = 3.0
    _c: dict[int, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)
    _v: dict[int, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)
    _u: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sigma_id <= 0 or self.r_id <= 0:
            raise ValueError("sigma_id and r_id must be > 0")

    def reset(self) -> None:
        """Forget all tracks."""
        self._c.clear()
        self._v.clear()
        self._u.clear()

    def switch_posterior(self, d: float, dt: float) -> float:
        """Posterior probability that a residual d (diagonals) came from an identity switch."""
        sig = self.sigma_id * math.sqrt(max(1.0, dt / DT_REF))
        h = 1.0 - math.exp(-SWITCH_HAZARD * dt)
        g0 = math.exp(-0.5 * (d / sig) ** 2) / (2.0 * math.pi * sig * sig)
        g1 = 1.0 / (math.pi * D_MAX * D_MAX) if d <= D_MAX else 0.0
        num = h * g1
        den = num + (1.0 - h) * g0
        if den <= 0.0:
            return 1.0          # both densities vanish: residual far beyond any ordinary motion
        return num / den

    def update(self, tracks: list[TrackObservation], dt: float) -> None:
        """Advance all track reliabilities by dt and process this frame's boxes."""
        rec = math.exp(-self.r_id * dt)
        for k in self._u:
            self._u[k] = 1.0 - (1.0 - self._u[k]) * rec
        for t in tracks:
            x1, y1, x2, y2 = t.bbox_xyxy
            c = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
            diag = math.hypot(x2 - x1, y2 - y1)
            prev = self._c.get(t.track_id)
            if prev is None or diag <= 0:
                self._c[t.track_id] = c
                self._v.setdefault(t.track_id, (0.0, 0.0))
                self._u.setdefault(t.track_id, 1.0)
                continue
            v = self._v[t.track_id]
            px, py = prev[0] + v[0] * dt, prev[1] + v[1] * dt
            d = math.hypot(c[0] - px, c[1] - py) / diag
            s = self.switch_posterior(d, dt)
            self._u[t.track_id] = self._u[t.track_id] * (1.0 - s)
            if s < 0.5:
                ox, oy = (c[0] - prev[0]) / dt, (c[1] - prev[1]) / dt
                self._v[t.track_id] = (v[0] + VEL_GAIN * (ox - v[0]), v[1] + VEL_GAIN * (oy - v[1]))
            self._c[t.track_id] = c

    def track(self, tid: int) -> float:
        """Reliability of one track (1.0 if unknown)."""
        return self._u.get(tid, 1.0)

    def edge(self, key: EdgeKey) -> float:
        """w(k) = u(subject) * u(object)."""
        return self.track(key.subject_track_id) * self.track(key.object_track_id)
