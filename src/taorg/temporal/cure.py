"""Survival-calibrated hold for NON-VISIBLE absences, added to GatedEMA + hysteresis (B2).

research/taorg_next_method_formulation.md (frozen with the next-method pre-registration).

The decision is B2's exactly, except on frames where an ON key has a NON-VISIBLE absence
(flag OCCLUDED_OR_UNRELIABLE or UNKNOWN, or no record).  On those frames the key stays ON
iff a hold rule says so; the run clock l counts sampling bins since the key's last detection:

    l = round((t - t_last_detection) / DT_REF),  DT_REF = 1/30 s

Hold rules (``mode``):
    "km"        q_F(l) >= tau,  q_F(l) = 1 - S_F(H)/S_F(l)          (mixture-cure posterior; S_F a
                shrunk Kaplan-Meier survival of re-detection times, learned ONLINE per video)
    "static"    same with the prior hazards only (no online learning)            [ablation]
    "pooled"    "km" with one stratum for OCC and UNK                            [ablation]
    "geometric" q_F(l) = pi r^l/(pi r^l + 1 - pi), r = 1 - p_F, with p_F, pi_F learned online
                with Beta-type pseudo-counts (adaptive detection-profile analogue) [comparator]
    "buffer"    l <= N_F  (fixed per-flag coasting buffer, tuned)                [comparator]

Run (learning unit): starts at the first non-visible absence frame of an ON key after a
detection; stratum F = that frame's flag; continues (independently of later output) until
the next detection (event at bin L) or until l >= H (censored, 'no return within H').
Only input-stream information is used; the prior is fitted on CALIBRATION input streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.temporal.gated_ema import GatedEMA

DT_REF = 1.0 / 30.0
NONVIS = ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN")
MODES = ("km", "static", "pooled", "geometric", "buffer")


def flag_of(o: Optional[EdgeObservation]) -> str:
    """Visibility class of an absence record ('UNKNOWN' if there is no record)."""
    return "UNKNOWN" if o is None else o.edge_visibility.name


def bins(t: float, t0: float) -> int:
    """Run length in sampling bins of DT_REF."""
    return int(round((t - t0) / DT_REF))


class RunCounts:
    """Online right-censored re-detection-time counts per stratum (Kaplan-Meier risk sets).

    Y[s][l]: runs at risk at bin l (absent through bin l-1); d[s][l]: re-detections at bin l.
    Bins 1..H.
    """

    def __init__(self, strata: Iterable[str], H: int):
        self.H = H
        self.Y = {s: [0.0] * (H + 1) for s in strata}
        self.d = {s: [0.0] * (H + 1) for s in strata}

    def at_risk(self, s: str, b_from: int, b_to: int) -> None:
        """A run absent through bins b_from..b_to (no event there)."""
        Y = self.Y[s]
        for b in range(max(1, b_from), min(b_to, self.H) + 1):
            Y[b] += 1.0

    def event(self, s: str, b: int) -> None:
        """Re-detection at bin b (at risk there, and an event)."""
        if 1 <= b <= self.H:
            self.Y[s][b] += 1.0
            self.d[s][b] += 1.0


@dataclass
class _Run:
    stratum: str
    t0: float               # time of the last detection before the run
    counted: int = 0        # bins already added to the risk set
    closed: bool = False    # reached H (censored: no return within H)


@dataclass
class B2SurvivalHold:
    """B2 (GatedEMA + hysteresis) with a hold rule for non-visible absences."""

    alpha: float = 0.3
    decay_visible: float = 0.7
    decay_occluded: float = 0.95
    decay_unknown: float = -1.0
    score_threshold: float = 0.3          # theta_on
    off_ratio: float = 0.5                # theta_off = off_ratio * theta_on
    mode: str = "km"
    tau: float = 0.5
    kappa: float = 50.0
    n_occ: int = 30
    n_unk: int = 30
    horizon_s: float = 3.0
    prior: Optional[dict[str, Any]] = None

    _inner: GatedEMA = field(init=False, repr=False)
    _on: set = field(default_factory=set, init=False, repr=False)
    _last_det: dict = field(default_factory=dict, init=False, repr=False)
    _runs: dict = field(default_factory=dict, init=False, repr=False)
    _counts: Optional[RunCounts] = field(default=None, init=False, repr=False)
    _qcache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not 0.0 < self.off_ratio < 1.0 or self.score_threshold <= 0:
            raise ValueError("require 0 < off_ratio < 1 and score_threshold > 0")
        if not 0.0 < self.tau < 1.0 or self.kappa <= 0 or self.horizon_s <= 0:
            raise ValueError("require 0 < tau < 1, kappa > 0, horizon_s > 0")
        if self.n_occ < 0 or self.n_unk < 0:
            raise ValueError("buffer lengths must be >= 0")
        self.n_occ, self.n_unk = int(round(self.n_occ)), int(round(self.n_unk))
        self._H = int(round(self.horizon_s / DT_REF))
        self._strata = ("ALL",) if self.mode == "pooled" else NONVIS
        if self.mode != "buffer":
            if self.prior is None:
                raise ValueError("this mode needs a prior fitted on calibration input streams")
            if abs(float(self.prior["horizon_s"]) - self.horizon_s) > 1e-9:
                raise ValueError("prior horizon differs from horizon_s")
            for s in self._strata:
                if s not in self.prior["h0"] or len(self.prior["h0"][s]) != self._H:
                    raise ValueError(f"prior hazard for stratum {s} missing or of wrong length")
        self._inner = GatedEMA(alpha=self.alpha, decay_visible=self.decay_visible,
                               decay_occluded=self.decay_occluded, decay_unknown=self.decay_unknown,
                               score_threshold=0.0)

    # ------------------------------------------------------------------ estimation
    def reset_video(self, video_id: str) -> None:
        """Reset all state, including the online estimates (adaptation is per video)."""
        self._inner.reset_video(video_id)
        self._on.clear()
        self._last_det.clear()
        self._runs.clear()
        self._counts = RunCounts(self._strata, self._H)
        self._qcache.clear()

    def _stratum(self, flag: str) -> str:
        return "ALL" if self.mode == "pooled" else flag

    def hazards(self, s: str) -> list[float]:
        """Per-bin re-detection hazard h(1..H) (index 0 unused), shrunk toward the prior."""
        h0 = self.prior["h0"][s]
        if self.mode == "static":
            return [0.0] + [min(1 - 1e-12, max(1e-12, x)) for x in h0]
        Y, d, k = self._counts.Y[s], self._counts.d[s], self.kappa
        return [0.0] + [min(1 - 1e-12, max(1e-12, (d[l] + k * h0[l - 1]) / (Y[l] + k)))
                        for l in range(1, self._H + 1)]

    def gap_posterior(self, s: str, l: int) -> float:
        """q_s(l): posterior that a run absent through bin l is re-detected within H."""
        if l >= self._H:
            return 0.0
        if l <= 0:
            return 1.0
        key = (s, l)
        if key in self._qcache:
            return self._qcache[key]
        if self.mode == "geometric":
            g = self.prior["geom"][s]
            Y, d, k = self._counts.Y[s], self._counts.d[s], self.kappa
            n_ret = sum(d[1:])
            p = (n_ret + k * g["p"]) / (sum(Y[1:]) + k)
            n_cens = Y[self._H] - d[self._H]
            pi = (n_ret + k * g["pi"]) / (n_ret + n_cens + k)
            a = pi * (1.0 - p) ** l
            q = a / (a + 1.0 - pi)
        else:
            h = self.hazards(s)
            S_l = 1.0
            for j in range(1, l + 1):
                S_l *= 1.0 - h[j]
            S_H = S_l
            for j in range(l + 1, self._H + 1):
                S_H *= 1.0 - h[j]
            q = max(0.0, 1.0 - S_H / S_l)
        self._qcache[key] = q
        return q

    def _hold(self, flag: str, l: int) -> bool:
        if self.mode == "buffer":
            return l <= (self.n_occ if flag == "OCCLUDED_OR_UNRELIABLE" else self.n_unk)
        return self.gap_posterior(self._stratum(flag), l) >= self.tau

    # ------------------------------------------------------------------ step
    def step(self, frame: FrameGraph) -> FrameGraph:
        """B2 score update, run bookkeeping, B2 decisions overridden on non-visible absences."""
        t = frame.timestamp
        inner = self._inner.step(frame)
        scores = {o.edge_key: o.predicate_score for o in inner.edge_observations}
        records = {o.edge_key: o for o in frame.edge_observations}
        on_t, off_t = self.score_threshold, self.off_ratio * self.score_threshold
        learn = self.mode in ("km", "pooled", "geometric")
        # 1. learning bookkeeping for open runs (output-independent once a run has started)
        for k, run in list(self._runs.items()):
            o = records.get(k)
            if o is not None and o.observed:
                if learn and not run.closed:
                    self._counts.event(run.stratum, bins(t, run.t0))
                del self._runs[k]
                continue
            if run.closed:
                continue
            l = bins(t, run.t0)
            if learn:
                self._counts.at_risk(run.stratum, run.counted + 1, l)
            run.counted = max(run.counted, min(l, self._H))
            run.closed = l >= self._H
        self._qcache.clear()
        # 2. decisions
        out: list[EdgeObservation] = []
        for k in sorted(set(scores) | self._on):
            o = records.get(k)
            det = o is not None and o.observed
            if det:
                self._last_det[k] = t
            flag = None if det else flag_of(o)
            if k in self._on and not det and flag in NONVIS:
                if k not in self._runs:
                    t0 = self._last_det.get(k, t - DT_REF)
                    l0 = bins(t, t0)
                    run = _Run(self._stratum(flag), t0, counted=min(l0, self._H), closed=l0 >= self._H)
                    if learn:
                        self._counts.at_risk(run.stratum, 1, l0)
                        self._qcache.clear()
                    self._runs[k] = run
                if not self._hold(flag, bins(t, self._runs[k].t0)):
                    self._on.discard(k)
            else:                                              # B2's decision, unchanged
                s = scores.get(k)
                if s is None:
                    self._on.discard(k)                         # pruned by GatedEMA
                elif k in self._on:
                    if s < off_t:
                        self._on.discard(k)
                elif s >= on_t:
                    self._on.add(k)
            if k in self._on:
                out.append(EdgeObservation(edge_key=k, video_id=frame.video_id, frame_idx=frame.frame_idx,
                                           timestamp=t, predicate_score=scores.get(k, 0.0), observed=det,
                                           source_model=f"b2_hold_{self.mode}"))
        return FrameGraph(video_id=frame.video_id, frame_idx=frame.frame_idx, timestamp=t,
                          track_observations=frame.track_observations, edge_observations=out)

    def finalize_video(self) -> None:
        """No-op."""

    def state_dict(self) -> dict[str, Any]:
        """Configuration and state size."""
        return {"method": f"b2_hold_{self.mode}", "tau": self.tau, "kappa": self.kappa, "n_occ": self.n_occ,
                "n_unk": self.n_unk, "horizon_s": self.horizon_s, "on_edges": len(self._on)}


def fit_prior(streams: Iterable[list[FrameGraph]], horizon_s: float = 3.0) -> dict[str, Any]:
    """Prior hazards from CALIBRATION INPUT streams (observable data only, no ground truth).

    Population (output-independent): every run that starts at the first non-visible absence
    frame of a key after one of its detections; stratum = that frame's flag (+ 'ALL').
    Returns h0[s] (H hazards for bins 1..H) and geometric targets {p, pi} per stratum.
    Bins without data inherit the previous bin's hazard (first default 0.05).
    """
    H = int(round(horizon_s / DT_REF))
    strata = NONVIS + ("ALL",)
    C = RunCounts(strata, H)
    n_runs = {s: 0 for s in strata}
    for frames in streams:
        last: dict[EdgeKey, float] = {}
        runs: dict[EdgeKey, _Run] = {}
        for fr in frames:
            t = fr.timestamp
            recs = {o.edge_key: o for o in fr.edge_observations}
            for k in sorted(set(recs) | set(last)):
                o = recs.get(k)
                if o is not None and o.observed:
                    r = runs.pop(k, None)
                    if r is not None and not r.closed:
                        for s in (r.stratum, "ALL"):
                            C.event(s, bins(t, r.t0))
                    last[k] = t
                    continue
                if k not in last:
                    continue
                r = runs.get(k)
                if r is None:
                    f = flag_of(o)
                    if f not in NONVIS:
                        continue
                    r = _Run(f, last[k])
                    runs[k] = r
                    n_runs[f] += 1
                    n_runs["ALL"] += 1
                if r.closed:
                    continue
                l = bins(t, r.t0)
                for s in (r.stratum, "ALL"):
                    C.at_risk(s, r.counted + 1, l)
                r.counted = max(r.counted, min(l, H))
                r.closed = l >= H
    h0, geom = {}, {}
    for s in strata:
        hs, prev = [], 0.05
        for l in range(1, H + 1):
            if C.Y[s][l] > 0:
                prev = C.d[s][l] / C.Y[s][l]
            hs.append(min(0.999, max(1e-4, prev)))
        h0[s] = hs
        n_ret = sum(C.d[s][1:])
        geom[s] = {"p": min(0.999, max(1e-4, n_ret / max(1.0, sum(C.Y[s][1:])))),
                   "pi": min(0.999, max(1e-3, n_ret / max(1.0, n_runs[s])))}
    return {"horizon_s": horizon_s, "h0": h0, "geom": geom, "n_runs": n_runs}


def km_survival(h: list[float]) -> list[float]:
    """S(l) = prod_{j<=l} (1 - h(j)) for l = 0..H (h[0] unused)."""
    S = [1.0]
    for j in range(1, len(h)):
        S.append(S[-1] * (1.0 - h[j]))
    return S
