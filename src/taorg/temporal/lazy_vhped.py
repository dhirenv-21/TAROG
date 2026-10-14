"""Lazy state maintenance for the V-HPED filter (Phase 5; research/derivations/lazy_maintenance.md).

The eager reference (``taorg.temporal.vhped.VHPED``) updates every key ever observed in every frame
and never prunes. ``LazyVHPED`` produces the same emitted edges and scores (within a declared
floating-point tolerance) while touching only keys whose evidence or decision can change:

* ACTIVE keys are updated exactly as in the eager filter (same arithmetic, same order of operations).
* A key that receives the constant "no-record" evidence L_unk (with identity weight 1 and no
  cross-predicate factor) evolves as x_{n+1} ∝ x_n M, M = P(Δ) diag(L_unk^a), a strictly positive
  matrix. By Birkhoff's contraction theorem the Hilbert projective distance d_n = d_H(x_n, x*) to
  the Perron fixed point x* never increases. Since odds(p_n) <= odds(p*) e^{d_n}, a key with
  odds(p*) e^{d_n} <= odds(threshold) can NEVER be emitted again under that evidence, even though
  p_n need not be monotone. Such a key is FROZEN: its posterior is retained (recurrence and
  reactivation remain exact) and it costs nothing per frame.
* A not-yet-certified key gets a CERTIFICATION DEADLINE n1 (from per-regime block contraction
  coefficients) in a versioned min-heap (``ExpiryQueue``); it stays active until the deadline, is
  re-verified from its actual state at pop time, and is frozen or rescheduled. Evidence changes
  cancel the deadline (generation invalidation).
* A frozen key is MATERIALISED in closed form (``propagate_constant``) when it receives any record,
  when another predicate is observed on its pair under cross-predicate coupling (M6), when the frame
  interval changes (the regime M changes), or when its belief is read.
* Identity reliability (M5) makes every key's evidence time-varying: no key is frozen (eager
  fallback), which is exact by construction.

The heap schedules certification, not deletion: no posterior is ever discarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.expiry_queue import ExpiryQueue
from taorg.temporal.vhped import VHPED, propagate_constant

SAFETY = 1e-9            # absolute margin on the log-odds certificate
DIRECT_STEPS = 16        # materialise spans this short by direct stepping (eager arithmetic)
MAX_BLOCK_LOG2 = 14      # block contraction coefficients for m = 1, 2, 4, ..., 2^14 frames
RECHECK = 1024           # conservative re-evaluation horizon when no finite deadline exists
MIN_QUIET = 15           # cost policy: a key is frozen only after >= MIN_QUIET constant-evidence frames


def projective_diameter(M: np.ndarray) -> float:
    """Birkhoff projective diameter Δ(M) = max log(M_ik M_jl / (M_il M_jk)) of a positive matrix."""
    if (M <= 0).any():
        return math.inf
    L = np.log(M)
    # max over i,j,k,l of (L_ik - L_il) - (L_jk - L_jl)
    D = L[:, :, None] - L[:, None, :]            # [i, k, l] = L_ik - L_il
    return float((D.max(axis=0) - D.min(axis=0)).max())


def hilbert_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Hilbert projective distance between positive vectors (inf if a component is 0)."""
    if (x <= 0).any() or (y <= 0).any():
        return math.inf
    r = np.log(x) - np.log(y)
    return float(r.max() - r.min())


def distance_to_fixed_point(x: list[float], cert: "Certificate") -> float:
    """Pure-Python Hilbert distance of a 4-state posterior to the certificate's fixed point."""
    if x[0] <= 0 or x[1] <= 0 or x[2] <= 0 or x[3] <= 0:
        return math.inf
    ls = cert.log_x_star
    r0, r1, r2, r3 = (math.log(x[0]) - ls[0], math.log(x[1]) - ls[1], math.log(x[2]) - ls[2],
                      math.log(x[3]) - ls[3])
    return max(r0, r1, r2, r3) - min(r0, r1, r2, r3)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class Certificate:
    """Constant no-record evidence regime for one frame interval."""

    P: np.ndarray                     # P(Δ) = expm(Q Δ)
    M: np.ndarray                     # P(Δ) diag(L_unk^a)
    L: tuple[float, ...]
    x_star: np.ndarray                # Perron (left) fixed point, sums to 1
    log_x_star: tuple[float, ...]
    p_star: float
    target: Optional[float]           # log(odds τ / odds p*) - SAFETY; None if p* >= τ (no freezing)
    kappas: tuple[tuple[int, float], ...]   # (block length m, κ(M^m)) with κ < 1


def make_certificate(P: np.ndarray, L: tuple[float, ...], threshold: float) -> Certificate:
    """Fixed point, certificate target and block contraction coefficients for M = P diag(L)."""
    M = np.asarray(P, float) @ np.diag(np.asarray(L, float))
    w, V = np.linalg.eig(M.T)
    v = np.real(V[:, int(np.argmax(np.real(w)))])
    v = np.abs(v) / np.abs(v).sum()
    p_star = float(v[2] + v[3])
    target = (logit(threshold) - logit(p_star) - SAFETY) if p_star < threshold else None
    kap = []
    B = M.copy()
    for e in range(MAX_BLOCK_LOG2 + 1):
        k = math.tanh(projective_diameter(B) / 4.0)
        if k < 1.0:
            kap.append((2 ** e, k))
        B = B @ B
        B /= B.max()
    return Certificate(np.asarray(P, float), M, L, v, tuple(math.log(c) for c in v), p_star, target, tuple(kap))


def steps_to_certify(cert: Certificate, d0: float) -> Optional[int]:
    """Smallest n (over block bounds) with d_n <= target guaranteed; 0 if already; None if unknown."""
    if cert.target is None or cert.target <= 0:
        return None
    if d0 <= cert.target:
        return 0
    if not math.isfinite(d0):
        return 1                                   # one positive step makes the state positive
    best = None
    for m, k in cert.kappas:
        if k <= 0.0:
            n = m
        else:
            blocks = math.ceil(math.log(cert.target / d0) / math.log(k))
            n = m * max(1, blocks)
        best = n if best is None else min(best, n)
    return best


@dataclass
class LazyVHPED(VHPED):
    """V-HPED with certified lazy maintenance; emitted edges equal the eager reference's."""

    _frozen: dict[EdgeKey, tuple[list[float], int, float]] = field(default_factory=dict, init=False, repr=False)
    _pair_frozen: dict[tuple[int, int], set] = field(default_factory=dict, init=False, repr=False)
    _queue: ExpiryQueue = field(default_factory=ExpiryQueue, init=False, repr=False)
    _certs: dict[float, Certificate] = field(default_factory=dict, init=False, repr=False)
    _s: int = field(default=0, init=False, repr=False)
    _regime: Optional[float] = field(default=None, init=False, repr=False)
    stats: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._reset_lazy()

    def reset_video(self, video_id: str) -> None:
        """Reset all state and counters for a new video."""
        super().reset_video(video_id)
        self._reset_lazy()

    def _reset_lazy(self) -> None:
        self._frozen.clear()
        self._pair_frozen.clear()
        self._queue.reset()
        self._s = 0
        self._regime = None
        self.stats = {k: 0 for k in ("steps", "key_updates", "emitted", "certify_calls", "deadlines",
                                     "freezes", "freeze_rejected_at_pop", "mat_record", "mat_pair",
                                     "mat_regime", "mat_read", "mat_direct", "mat_closed_form",
                                     "max_active", "max_frozen", "identity_fallback_steps")}

    # -- certificates -------------------------------------------------------------
    def _cert(self, dt_key: float, P, a: float, temper: bool) -> Certificate:
        c = self._certs.get(dt_key)
        if c is None:
            L = self._L["unk"]
            if temper:
                L = tuple(v ** a for v in L)
            c = make_certificate(np.asarray(P, float), L, self.threshold)
            self._certs[dt_key] = c
        return c

    # -- frozen-key bookkeeping ----------------------------------------------------
    def _freeze(self, k: EdgeKey, x: list[float]) -> None:
        self._frozen[k] = (list(x), self._s, self._regime)
        self._pair_frozen.setdefault((k.subject_track_id, k.object_track_id), set()).add(k)
        del self._x[k]
        self.stats["freezes"] += 1

    def _advance(self, x: list[float], n: int, cert: Certificate) -> list[float]:
        """n constant no-record steps: direct (eager arithmetic) if short, closed form otherwise."""
        if n <= 0:
            return list(x)
        if n <= DIRECT_STEPS:
            self.stats["mat_direct"] += 1
            M = cert.M
            v = list(x)
            for _ in range(n):
                y = [v[0] * M[0][j] + v[1] * M[1][j] + v[2] * M[2][j] + v[3] * M[3][j] for j in range(4)]
                t = y[0] + y[1] + y[2] + y[3]
                v = [y[0] / t, y[1] / t, y[2] / t, y[3] / t]
            return v
        self.stats["mat_closed_form"] += 1
        return [float(v) for v in propagate_constant(x, cert.P, cert.L, n)]

    def _materialize(self, k: EdgeKey, upto: int, reason: str) -> list[float]:
        """Bring a frozen key to 'after step upto' and make it active again."""
        x, s0, regime = self._frozen.pop(k)
        self._pair_frozen[(k.subject_track_id, k.object_track_id)].discard(k)
        x = self._advance(x, upto - s0, self._certs[regime])
        self._x[k] = x
        self.stats[f"mat_{reason}"] += 1
        return x

    def belief(self, key: EdgeKey) -> float:
        """Current P(Z=1); frozen keys are evaluated without changing the maintained state."""
        if key in self._frozen:
            x, s0, regime = self._frozen[key]
            self.stats["mat_read"] += 1
            v = self._advance(x, self._s - s0, self._certs[regime])
            return v[2] + v[3]
        return super().belief(key)

    # -- main step -----------------------------------------------------------------
    def step(self, frame: FrameGraph) -> FrameGraph:
        """One frame; emitted edges equal the eager reference's (see module docstring)."""
        dt = self.dt_ref if self._t_prev is None else max(frame.timestamp - self._t_prev, 1e-9)
        self._t_prev = frame.timestamp
        a = dt / self.dt_ref
        temper = abs(a - 1.0) > 1e-9
        P = self._trans(dt)
        dt_key = round(dt, 9)
        self._s += 1
        s = self._s
        self.stats["steps"] += 1
        # a change of frame interval changes M: frozen spans end here (materialise under the old regime)
        if self._regime is not None and dt_key != self._regime:
            for k in list(self._frozen):
                self._materialize(k, s - 1, "regime")
            self._queue.cancel_all()
        self._regime = dt_key
        cert = self._cert(dt_key, P, a, temper)
        if self.identity:
            self._update_identity(frame, dt)
            self.stats["identity_fallback_steps"] += 1
        records: dict[EdgeKey, EdgeObservation] = {o.edge_key: o for o in frame.edge_observations}
        observed_by_pair: dict[tuple[int, int], list[int]] = {}
        for k, o in records.items():
            if k in self._frozen and (o.observed or o.edge_visibility != VisibilityState.UNKNOWN
                                      or self.identity):
                self._materialize(k, s - 1, "record")
            if o.observed:
                observed_by_pair.setdefault((k.subject_track_id, k.object_track_id), []).append(k.predicate_id)
                if k not in self._x and k not in self._frozen:
                    self._x[k] = list(self._pi)
        if self.cross and self.w_x > 0:
            for pair in observed_by_pair:
                for k in list(self._pair_frozen.get(pair, ())):
                    self._materialize(k, s - 1, "pair")
        out: list[EdgeObservation] = []
        Ls = self._L
        constant_keys = []
        for k, x in self._x.items():
            o = records.get(k)
            if o is None:
                L = Ls["unk"]
                const = True
            elif o.observed:
                L = Ls["det"]
                const = False
            else:
                vis = o.edge_visibility
                L = Ls["vis"] if vis == VisibilityState.VISIBLE else (
                    Ls["occ"] if vis == VisibilityState.OCCLUDED_OR_UNRELIABLE else Ls["unk"])
                const = vis == VisibilityState.UNKNOWN
            if self.identity:
                w = self._u.get(k.subject_track_id, 1.0) * self._u.get(k.object_track_id, 1.0)
                if w < 1.0:
                    L = tuple(w * l + (1.0 - w) for l in L)
                const = False
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
                    if c != 1.0:
                        const = False
            n0, n1, n2, n3 = L[0] * p0, L[1] * p1, c * L[2] * p2, c * L[3] * p3
            tot = n0 + n1 + n2 + n3
            if tot > 0 and math.isfinite(tot):
                x[0], x[1], x[2], x[3] = n0 / tot, n1 / tot, n2 / tot, n3 / tot
            else:  # numerical safeguard: keep the prediction (as in the eager reference)
                sm = p0 + p1 + p2 + p3
                x[0], x[1], x[2], x[3] = p0 / sm, p1 / sm, p2 / sm, p3 / sm
                const = False
            prob = x[2] + x[3]
            if prob > self.threshold:
                out.append(EdgeObservation(
                    edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp, predicate_score=prob,
                    observed=bool(o is not None and o.observed), source_model=f"vhped_{self.rung}"))
            if const:
                constant_keys.append(k)
            elif k in self._queue:
                self._queue.cancel(k)
        self.stats["key_updates"] += len(self._x)
        self.stats["emitted"] += len(out)
        # certification (constant evidence only; never under identity reliability)
        if not self.identity and cert.target is not None:
            for k in constant_keys:
                if k in self._queue:
                    continue
                self.stats["certify_calls"] += 1
                n1 = steps_to_certify(cert, distance_to_fixed_point(self._x[k], cert))
                n1 = RECHECK if n1 is None else max(n1, MIN_QUIET)       # cost policy only
                self._queue.schedule(k, float(s + n1))
                self.stats["deadlines"] += 1
            for k in self._queue.pop_due(float(s)):
                if k not in self._x:
                    continue
                d = distance_to_fixed_point(self._x[k], cert)
                if d <= cert.target:
                    self._freeze(k, self._x[k])
                else:                                   # verification failed: conservative reschedule
                    self.stats["freeze_rejected_at_pop"] += 1
                    n1 = steps_to_certify(cert, d)
                    self._queue.schedule(k, float(s + (n1 if n1 else RECHECK)))
        self.stats["max_active"] = max(self.stats["max_active"], len(self._x))
        self.stats["max_frozen"] = max(self.stats["max_frozen"], len(self._frozen))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          track_observations=frame.track_observations, edge_observations=out)

    def counters(self) -> dict[str, int]:
        """Operation counters plus current sizes and heap instrumentation."""
        return {**self.stats, "active": len(self._x), "frozen": len(self._frozen), **self._queue.counters()}

    def state_dict(self) -> dict[str, Any]:
        """Configuration plus lazy counters."""
        return {**super().state_dict(), "lazy": True, **self.counters()}
