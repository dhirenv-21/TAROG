"""Run-outcome (cure-model) termination rule for GatedEMA + hysteresis.

research/taorg_next_method_formulation.md.  Births are B2's (GatedEMA score >= theta_on);
only the TERMINATION decision during an absence run of an ON key is replaced:

    stay ON at absence bin l of a run with onset flag F  iff  q_F(l) >= tau

Modes (all use only the input stream; the prior is fitted on calibration INPUT streams):
    "km"        q_F(l) = 1 - S_F(H) / S_F(l),  S_F = shrunk Kaplan-Meier survival of
                re-detection times of absence runs (mixture-cure posterior, nonparametric)
    "static"    same rule with the prior hazards only (no online learning)          [ablation]
    "pooled"    "km" with a single stratum (flags ignored)                           [ablation]
    "geometric" q_F(l) = pi r^l / (pi r^l + 1 - pi), r = 1 - p_F; p_F, pi_F learned online
                with Beta pseudo-counts (adaptive detection-profile analogue)        [comparator]

Learning population: absence runs that START while the key is ON; a run is followed
(regardless of later output) until re-detection (event at its length) or H bins
(censored => counted as 'no return within H').  Run length is counted in sampling bins
of width DT_REF = 1/30 s: l = round((t - t_last_detection) / DT_REF).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.gated_ema import GatedEMA

DT_REF = 1.0 / 30.0
STRATA = ("VISIBLE", "OCCLUDED_OR_UNRELIABLE", "UNKNOWN")
MODES = ("km", "static", "pooled", "geometric")


@dataclass
class _Run:
    stratum: str
    t_start: float           # time of the last detection before the run
    last_bin: int = 0        # highest absence bin already counted at risk
    done: bool = False


@dataclass
class HysteresisRunPosterior:
    """B2 births + run-outcome termination rule (see module docstring)."""

    alpha: float = 0.3
    decay_visible: float = 0.7
    decay_occluded: float = 0.95
    decay_unknown: float = -1.0
    score_threshold: float = 0.3          # theta_on (B2 birth threshold)
    tau: float = 0.5                      # risk threshold on the gap posterior
    kappa: float = 50.0                   # prior strength (pseudo-runs at risk)
    horizon_s: float = 3.0                # H: runs without re-detection within H are 'no return'
    mode: str = "km"
    prior: Optional[dict[str, Any]] = None
    off_ratio: float = 0.5                # B2's theta_off = off_ratio * theta_on, used for VISIBLE frames
    visible_b2: bool = True               # VISIBLE absences keep B2's own off-rule (score < theta_off)

    _inner: GatedEMA = field(init=False, repr=False)
    _on: set = field(default_factory=set, init=False, repr=False)
    _last_det: dict = field(default_factory=dict, init=False, repr=False)
    _runs: dict = field(default_factory=dict, init=False, repr=False)
    _Y: dict = field(default_factory=dict, init=False, repr=False)
    _d: dict = field(default_factory=dict, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not 0.0 < self.tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        if self.kappa <= 0 or self.horizon_s <= 0 or self.score_threshold <= 0:
            raise ValueError("kappa, horizon_s and score_threshold must be > 0")
        if self.prior is None:
            raise ValueError("a prior fitted on calibration input streams is required")
        self._H = int(round(self.horizon_s / DT_REF))
        if not 0.0 < self.off_ratio < 1.0:
            raise ValueError("off_ratio must be in (0, 1)")
        base = ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN") if self.visible_b2 else STRATA
        self._strata = ("ALL",) if self.mode == "pooled" else base
        for s in self._strata:
            if s not in self.prior["h0"]:
                raise ValueError(f"prior lacks stratum {s}")
            if len(self.prior["h0"][s]) < self._H:
                raise ValueError("prior hazard shorter than the horizon")
        self._inner = GatedEMA(alpha=self.alpha, decay_visible=self.decay_visible,
                               decay_occluded=self.decay_occluded, decay_unknown=self.decay_unknown,
                               score_threshold=0.0)

    # ----------------------------------------------------------- estimation
    def reset_video(self, video_id: str) -> None:
        """Forget all state, including the online estimates (per-video adaptation)."""
        self._inner.reset_video(video_id)
        self._on.clear()
        self._last_det.clear()
        self._runs.clear()
        self._Y = {s: [0.0] * (self._H + 1) for s in self._strata}
        self._d = {s: [0.0] * (self._H + 1) for s in self._strata}
        self._cache.clear()

    def _stratum(self, flag: str) -> str:
        return "ALL" if self.mode == "pooled" else flag

    def hazard(self, s: str) -> list[float]:
        """Shrunk per-bin return hazard h(1..H) of stratum s (index 0 unused)."""
        h0 = self.prior["h0"][s]
        if self.mode == "static":
            return [0.0] + [min(1.0 - 1e-9, max(1e-9, h0[l - 1])) for l in range(1, self._H + 1)]
        Y, d, k = self._Y[s], self._d[s], self.kappa
        return [0.0] + [min(1.0 - 1e-9, max(1e-9, (d[l] + k * h0[l - 1]) / (Y[l] + k))) for l in range(1, self._H + 1)]

    def gap_posterior(self, s: str, l: int) -> float:
        """q_s(l) = P(run returns within H | no return in its first l bins)."""
        if l >= self._H:
            return 0.0
        if l <= 0:
            return 1.0
        key = (s, l)
        if key in self._cache:
            return self._cache[key]
        if self.mode == "geometric":
            g = self.prior["geom"][s]
            Y, d, k = self._Y[s], self._d[s], self.kappa
            ret = sum(d[1:])
            frames = sum(Y[1:])
            p = (ret + k * g["p"]) / (frames + k)                 # per-bin return prob, pooled over bins
            n_ret, n_cens = ret, self._Y[s][self._H] - self._d[s][self._H]
            pi = (n_ret + k * g["pi"]) / (n_ret + max(0.0, n_cens) + k)
            a = pi * (1.0 - p) ** l
            q = a / (a + 1.0 - pi)
        else:
            h = self.hazard(s)
            S_l = 1.0
            for j in range(1, l + 1):
                S_l *= 1.0 - h[j]
            S_H = S_l
            for j in range(l + 1, self._H + 1):
                S_H *= 1.0 - h[j]
            q = max(0.0, 1.0 - S_H / S_l)
        self._cache[key] = q
        return q

    # ----------------------------------------------------------- step
    def step(self, frame: FrameGraph) -> FrameGraph:
        """B2 score update; births by theta_on; terminations by the run posterior."""
        t = frame.timestamp
        inner = self._inner.step(frame)
        scores = {o.edge_key: o.predicate_score for o in inner.edge_observations}
        records = {o.edge_key: o for o in frame.edge_observations}
        self._cache.clear()
        # 1. advance learning runs (at-risk counts) and close runs ended by a detection
        for k, run in list(self._runs.items()):
            o = records.get(k)
            l = int(round((t - run.t_start) / DT_REF))
            if o is not None and o.observed:
                L = min(run.last_bin + 1, self._H)
                if not run.done and L <= self._H and run.last_bin < self._H:
                    self._d[run.stratum][L] += 1.0
                    self._Y[run.stratum][L] += 1.0
                del self._runs[k]
                continue
            if run.done:
                continue
            # count bins last_bin+1 .. min(l, H) at risk with no event (absent through them)
            for b in range(run.last_bin + 1, min(l, self._H) + 1):
                self._Y[run.stratum][b] += 1.0
            run.last_bin = max(run.last_bin, min(l, self._H))
            if l >= self._H:
                run.done = True
        self._cache.clear()
        # 2. decisions
        out: list[EdgeObservation] = []
        for k in set(scores) | set(self._on):
            o = records.get(k)
            det = o is not None and o.observed
            if det:
                self._last_det[k] = t
                if k not in self._on and scores.get(k, 0.0) >= self.score_threshold:
                    self._on.add(k)
            elif k in self._on:
                flag_now = o.edge_visibility.name if o is not None else "UNKNOWN"
                if self.visible_b2 and flag_now == "VISIBLE":
                    # visible absence: B2's own decision (score already decayed through the run)
                    if scores.get(k, 0.0) < self.off_ratio * self.score_threshold:
                        self._on.discard(k)
                    if k in self._on:
                        out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                                   timestamp=t, predicate_score=scores.get(k, 0.0), observed=False,
                                                   source_model=f"b2_runpost_{self.mode}"))
                    continue
                if k not in self._runs:
                    flag = flag_now
                    self._runs[k] = _Run(self._stratum(flag), self._last_det.get(k, t - DT_REF))
                    l = int(round((t - self._runs[k].t_start) / DT_REF))
                    run = self._runs[k]
                    for b in range(1, min(l, self._H) + 1):
                        self._Y[run.stratum][b] += 1.0
                    run.last_bin = min(l, self._H)
                    run.done = l >= self._H
                    self._cache.clear()
                run = self._runs[k]
                l = int(round((t - run.t_start) / DT_REF))
                if self.gap_posterior(run.stratum, l) < self.tau:
                    self._on.discard(k)
            if k in self._on:
                out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                           timestamp=t, predicate_score=scores.get(k, 0.0), observed=det,
                                           source_model=f"b2_runpost_{self.mode}"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=t,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """No-op."""

    def state_dict(self) -> dict[str, Any]:
        """Configuration."""
        return {"method": f"b2_runpost_{self.mode}", "tau": self.tau, "kappa": self.kappa,
                "horizon_s": self.horizon_s, "score_threshold": self.score_threshold,
                "on_edges": len(self._on)}


def fit_prior(streams: list[list[FrameGraph]], horizon_s: float = 3.0) -> dict[str, Any]:
    """Prior hazards from CALIBRATION INPUT streams (observable only; no ground truth).

    Population: every absence run that follows a detection of a key (output-independent).
    Returns per-stratum shrinkage targets h0[l-1], l = 1..H (+ 'ALL'), and geometric
    targets {p, pi} per stratum.
    """
    H = int(round(horizon_s / DT_REF))
    Y = {s: [0.0] * (H + 1) for s in STRATA + ("ALL",)}
    d = {s: [0.0] * (H + 1) for s in STRATA + ("ALL",)}
    n_run = {s: 0 for s in STRATA + ("ALL",)}
    for frames in streams:
        last: dict[EdgeKey, float] = {}
        runs: dict[EdgeKey, list] = {}
        for fr in frames:
            t = fr.timestamp
            recs = {o.edge_key: o for o in fr.edge_observations}
            keys = set(recs) | set(last)
            for k in keys:
                o = recs.get(k)
                if o is not None and o.observed:
                    if k in runs and runs[k][2] < H:
                        L = runs[k][2] + 1
                        for s in (runs[k][0], "ALL"):
                            d[s][L] += 1
                            Y[s][L] += 1
                    runs.pop(k, None)
                    last[k] = t
                elif k in last:
                    if k not in runs:
                        flag = o.edge_visibility.name if o is not None else "UNKNOWN"
                        runs[k] = [flag, last[k], 0]
                        n_run[flag] += 1
                        n_run["ALL"] += 1
                    r = runs[k]
                    l = int(round((t - r[1]) / DT_REF))
                    for b in range(r[2] + 1, min(l, H) + 1):
                        for s in (r[0], "ALL"):
                            Y[s][b] += 1
                    r[2] = max(r[2], min(l, H))
    h0, geom = {}, {}
    for s in Y:
        hs, last_h = [], 0.05
        for l in range(1, H + 1):
            if Y[s][l] > 0:
                last_h = d[s][l] / Y[s][l]
            hs.append(min(0.999, max(1e-4, last_h)))
        h0[s] = hs
        ret = sum(d[s][1:])
        geom[s] = {"p": min(0.999, max(1e-4, ret / max(1.0, sum(Y[s][1:])))),
                   "pi": min(0.999, max(1e-3, ret / max(1.0, n_run[s])))}
    return {"horizon_s": horizon_s, "h0": h0, "geom": geom, "n_runs": n_run}
