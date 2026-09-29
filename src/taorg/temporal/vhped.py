"""V-HPED eager reference implementation (Phase 4 rungs M2-M6).

Implements ``research/vhped_formulation.md`` exactly (frozen at the Phase 4
pre-registration, commit bad0ef9).  Name is provisional: formulation §9
concludes that this is a continuous-time factorial HMM filter with
identity-tempered likelihoods and optional cross-predicate evidence coupling,
not a Hawkes process.

Per edge e, the exact filter keeps the joint posterior x over
(Z, V) in {00, 01, 10, 11} (Z = relationship exists, V = observable):

    predict:  x' = x P(dt),   P(dt) = expm(Q dt)   (Q: CTMC generator, rates in 1/s)
    update:   x  ∝ L_eff^(dt/dt_ref) ⊙ x'
    output:   p = x[10] + x[11];  emit iff p > threshold

Rungs (feature switches; see ``make_rung``):
    M2  flags trusted as exact (r_ov = r_vo = 1e-6), eta = 0, no identity, no cross
    M3  + tuned flag confusion r_ov, r_vo
    M4  + occlusion-onset termination coupling eta
    M5  + identity reliability (box-jump switch evidence, mixture likelihood)
    M6  + cross-predicate exclusivity evidence (E learned from calibration GT)

This is the EAGER reference: every maintained edge is updated every frame.
``propagate_constant`` gives the closed-form n-step update used to check the
lazy-materialisation equivalence (formulation §10); the lazy/heap mechanism
itself is Phase 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.linalg import expm

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

# State order (Z, V): 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)
EPS_OCC = 1e-3          # P(detect | exists, not observable), fixed (formulation §4)
FLAG_EXACT = 1e-6       # M2 flag confusion ("trusted as exact")
SWITCH_HAZARD = 0.03    # identity switch hazard, 1/s (formulation §7)
D_MAX = 2.0             # support of the switch displacement density (box diagonals)


def generator(mu: float, nu: float, gamma_o: float, gamma_v: float, eta: float = 0.0) -> np.ndarray:
    """CTMC generator Q over (Z, V) with optional onset coupling eta (M4).

    Q[i, j] is the rate of i -> j (i != j); rows sum to zero.
    """
    Q = np.zeros((4, 4))
    # Z transitions (V unchanged)
    Q[0, 2] = nu
    Q[1, 3] = nu
    Q[2, 0] = mu
    Q[3, 1] = mu
    # V transitions (Z unchanged)
    Q[0, 1] = gamma_v
    Q[2, 3] = gamma_v
    Q[1, 0] = gamma_o
    Q[3, 2] = (1.0 - eta) * gamma_o
    # M4: termination coinciding with loss of view, (1,1) -> (0,0)
    Q[3, 0] = eta * gamma_o
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def stationary(Q: np.ndarray) -> np.ndarray:
    """Stationary distribution pi of Q (pi Q = 0, sum pi = 1)."""
    A = np.vstack([Q.T, np.ones(Q.shape[0])])
    b = np.zeros(Q.shape[0] + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def emission(kind: str, miss: float, false_pos: float, r_ov: float, r_vo: float) -> tuple[float, ...]:
    """Per-state likelihood of one record (formulation §4 table).

    ``kind``: 'det' (detected), 'vis', 'occ' (absence with that flag), 'unk'
    (absence with UNKNOWN flag, or no record at all).
    """
    m, f, e = miss, false_pos, EPS_OCC
    if kind == "det":
        return (f, f, e, 1.0 - m)
    if kind == "vis":
        return ((1 - f) * r_ov, (1 - f) * (1 - r_vo), (1 - e) * r_ov, m * (1 - r_vo))
    if kind == "occ":
        return ((1 - f) * (1 - r_ov), (1 - f) * r_vo, (1 - e) * (1 - r_ov), m * r_vo)
    if kind == "unk":
        return (1 - f, 1 - f, 1 - e, m)
    raise ValueError(kind)


def propagate_constant(x: Sequence[float], P: np.ndarray, L: Sequence[float], n: int) -> np.ndarray:
    """Closed-form n-step filter with constant evidence L (lazy materialisation).

    Equivalent in exact arithmetic to n eager steps ``x <- normalize(L * (x P))``
    because normalisation commutes with the linear map: result ∝ x (P diag(L))^n.
    Uses repeated squaring, with rescaling to avoid underflow.
    """
    M = P @ np.diag(np.asarray(L, dtype=float))
    v = np.asarray(x, dtype=float)
    R = np.eye(4)
    B = M.copy()
    k = n
    while k > 0:
        if k & 1:
            R = R @ B
            R /= R.max()
        B = B @ B
        B /= B.max()
        k >>= 1
    out = v @ R
    return out / out.sum()


@dataclass
class VHPED:
    """Continuous-time visibility-aware edge filter (eager reference).

    Rates are per second; ``threshold`` must exceed nu/(mu+nu) for unobserved
    edges to expire (enforced by the Phase 4 sampler, not here).
    """

    mu: float = 0.5
    nu: float = 0.01
    gamma_o: float = 0.5
    gamma_v: float = 3.0
    miss: float = 0.05
    false_pos: float = 0.01
    threshold: float = 0.5
    r_ov: float = FLAG_EXACT
    r_vo: float = FLAG_EXACT
    eta: float = 0.0
    identity: bool = False
    sigma_id: float = 0.3
    r_id: float = 3.0
    cross: bool = False
    w_x: float = 0.0
    exclusivity: Optional[list[list[float]]] = None
    dt_ref: float = 1.0 / 30.0
    rung: str = "custom"

    _x: dict[EdgeKey, list[float]] = field(default_factory=dict, init=False, repr=False)
    _P: dict[float, tuple[tuple[float, ...], ...]] = field(default_factory=dict, init=False, repr=False)
    _t_prev: Optional[float] = field(default=None, init=False, repr=False)
    _u: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _box: dict[int, tuple[float, float, float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for n in ("mu", "nu", "gamma_o", "gamma_v"):
            if getattr(self, n) <= 0:
                raise ValueError(f"{n} must be > 0")
        for n in ("miss", "false_pos", "threshold", "r_ov", "r_vo"):
            v = getattr(self, n)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{n} must be in (0, 1), got {v}")
        if not (0.0 <= self.eta < 1.0):
            raise ValueError("eta must be in [0, 1)")
        if not (0.0 <= self.w_x <= 1.0):
            raise ValueError("w_x must be in [0, 1]")
        if self.cross and self.exclusivity is None:
            raise ValueError("cross-predicate evidence requires the exclusivity matrix")
        self._Q = generator(self.mu, self.nu, self.gamma_o, self.gamma_v, self.eta)
        self._pi = tuple(float(v) for v in stationary(self._Q))
        self._L = {k: emission(k, self.miss, self.false_pos, self.r_ov, self.r_vo)
                   for k in ("det", "vis", "occ", "unk")}

    # -- helpers --------------------------------------------------------------
    def _trans(self, dt: float) -> tuple[tuple[float, ...], ...]:
        key = round(dt, 9)
        P = self._P.get(key)
        if P is None:
            M = expm(self._Q * dt)
            P = tuple(tuple(float(v) for v in row) for row in M)
            self._P[key] = P
        return P

    def reset_video(self, video_id: str) -> None:
        """Reset all state for a new video."""
        self._x.clear()
        self._t_prev = None
        self._u.clear()
        self._box.clear()

    def belief(self, key: EdgeKey) -> float:
        """Current P(Z=1) of a maintained edge (NaN if not maintained)."""
        x = self._x.get(key)
        return math.nan if x is None else x[2] + x[3]

    def _update_identity(self, frame: FrameGraph, dt: float) -> None:
        h = 1.0 - math.exp(-SWITCH_HAZARD * dt)
        rec = math.exp(-self.r_id * dt)
        c0 = math.sqrt(2.0 / math.pi) / self.sigma_id
        for k in list(self._u):
            self._u[k] = 1.0 - (1.0 - self._u[k]) * rec
        for t in frame.track_observations:
            x1, y1, x2, y2 = t.bbox_xyxy
            cx, cy, diag = 0.5 * (x1 + x2), 0.5 * (y1 + y2), math.hypot(x2 - x1, y2 - y1)
            prev = self._box.get(t.track_id)
            self._box[t.track_id] = (cx, cy, diag)
            u = self._u.get(t.track_id, 1.0)
            if prev is not None and diag > 0:
                d = math.hypot(cx - prev[0], cy - prev[1]) / diag
                g0 = c0 * math.exp(-0.5 * (d / self.sigma_id) ** 2)
                g1 = 1.0 / D_MAX if d <= D_MAX else 0.0
                s = h * g1 / (h * g1 + (1.0 - h) * g0) if (h * g1 + (1.0 - h) * g0) > 0 else 1.0
                u *= (1.0 - s)
            self._u[t.track_id] = u

    # -- main step -------------------------------------------------------------
    def step(self, frame: FrameGraph) -> FrameGraph:
        """Eager filter step over all maintained edges."""
        dt = self.dt_ref if self._t_prev is None else max(frame.timestamp - self._t_prev, 1e-9)
        self._t_prev = frame.timestamp
        a = dt / self.dt_ref
        temper = abs(a - 1.0) > 1e-9
        P = self._trans(dt)
        if self.identity:
            self._update_identity(frame, dt)
        records: dict[EdgeKey, EdgeObservation] = {o.edge_key: o for o in frame.edge_observations}
        observed_by_pair: dict[tuple[int, int], list[int]] = {}
        for k, o in records.items():
            if o.observed:
                observed_by_pair.setdefault((k.subject_track_id, k.object_track_id), []).append(k.predicate_id)
                if k not in self._x:
                    self._x[k] = list(self._pi)
        out: list[EdgeObservation] = []
        Ls = self._L
        for k, x in self._x.items():
            o = records.get(k)
            if o is None:
                L = Ls["unk"]
            elif o.observed:
                L = Ls["det"]
            else:
                vis = o.edge_visibility
                L = Ls["vis"] if vis == VisibilityState.VISIBLE else (
                    Ls["occ"] if vis == VisibilityState.OCCLUDED_OR_UNRELIABLE else Ls["unk"])
            if self.identity:
                w = self._u.get(k.subject_track_id, 1.0) * self._u.get(k.object_track_id, 1.0)
                if w < 1.0:
                    L = tuple(w * l + (1.0 - w) for l in L)
            if temper:
                L = tuple(l ** a for l in L)
            x0, x1, x2, x3 = x
            p0 = x0 * P[0][0] + x1 * P[1][0] + x2 * P[2][0] + x3 * P[3][0]
            p1 = x0 * P[0][1] + x1 * P[1][1] + x2 * P[2][1] + x3 * P[3][1]
            p2 = x0 * P[0][2] + x1 * P[1][2] + x2 * P[2][2] + x3 * P[3][2]
            p3 = x0 * P[0][3] + x1 * P[1][3] + x2 * P[2][3] + x3 * P[3][3]
            c = 1.0
            if self.cross and self.w_x > 0:
                others = observed_by_pair.get((k.subject_track_id, k.object_track_id))
                if others:
                    row = self.exclusivity[k.predicate_id]
                    for rp in others:
                        if rp != k.predicate_id:
                            c *= (1.0 - self.w_x * row[rp])
                    if temper:
                        c = c ** a
            n0, n1, n2, n3 = L[0] * p0, L[1] * p1, c * L[2] * p2, c * L[3] * p3
            tot = n0 + n1 + n2 + n3
            if tot > 0 and math.isfinite(tot):
                x[0], x[1], x[2], x[3] = n0 / tot, n1 / tot, n2 / tot, n3 / tot
            else:  # numerical safeguard: keep the prediction
                s = p0 + p1 + p2 + p3
                x[0], x[1], x[2], x[3] = p0 / s, p1 / s, p2 / s, p3 / s
            prob = x[2] + x[3]
            if prob > self.threshold:
                out.append(EdgeObservation(
                    edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp, predicate_score=prob,
                    observed=bool(o is not None and o.observed), source_model=f"vhped_{self.rung}"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """Finalize current video (no-op)."""

    def state_dict(self) -> dict[str, Any]:
        """Configuration and state size."""
        return {"method": f"vhped_{self.rung}", "mu": self.mu, "nu": self.nu, "gamma_o": self.gamma_o,
                "gamma_v": self.gamma_v, "miss": self.miss, "false_pos": self.false_pos,
                "threshold": self.threshold, "r_ov": self.r_ov, "r_vo": self.r_vo, "eta": self.eta,
                "identity": self.identity, "sigma_id": self.sigma_id, "r_id": self.r_id,
                "cross": self.cross, "w_x": self.w_x, "active_edges": len(self._x)}


def vhped_m2(**p: Any) -> VHPED:
    """M2: continuous-time (Z, V) dynamics, flags trusted as exact."""
    return VHPED(**p, rung="m2")


def vhped_m3(**p: Any) -> VHPED:
    """M3: M2 + tuned flag confusion (uncertain visibility)."""
    return VHPED(**p, rung="m3")


def vhped_m4(**p: Any) -> VHPED:
    """M4: M3 + occlusion-onset termination coupling eta."""
    return VHPED(**p, rung="m4")


def vhped_m5(**p: Any) -> VHPED:
    """M5: M4 + identity reliability."""
    return VHPED(**p, identity=True, rung="m5")


def vhped_m6(**p: Any) -> VHPED:
    """M6: M5 + cross-predicate exclusivity evidence."""
    return VHPED(**p, identity=True, cross=True, rung="m6")
