"""Phase 5H-1 point-process GLM (multiplicative-intensity relational event model).

    lambda_{p,k}(t) = R_{p,k}(t) * exp(eta_k + w_k . x_{p,k}(t))

All covariates are piecewise constant in t (research/phase5h1_preregistration.md §4), so the
compensator is an exact sum over segments and the log-likelihood of target k equals a Poisson
regression log-likelihood:  sum_i [ y_i z_i - d_i exp(z_i) ]  over at-risk segments i with
exposure d_i (seconds) and event count y_i in {0, 1}.

Segments of a pair are the intervals (a, b] between consecutive change points; an event at
time b belongs to the segment ending at b and is predicted from features valid on (a, b]
(strict history: events at exactly b are not in the segment's history).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp

from taorg.hawkes.vidor import PairRecord, VideoRecord

BLOCKS = ("HET", "STATE", "HIST")


@dataclass
class Layout:
    """Column layout of the common design matrix."""

    vocab: list[str]
    cats: list[str]
    lag_edges: list[float]
    elapsed_edges: list[float]
    count_bins: list[int]
    active_bins: list[int]
    dur_edges: list[float]
    sources: list[int]                         # modelled source types (K-indices) for HIST
    offsets: dict[str, int] = field(default_factory=dict)
    size: int = 0

    def __post_init__(self) -> None:
        o = 0
        for name, n in (("subj", len(self.cats) + 1), ("obj", len(self.cats) + 1),
                        ("elapsed", len(self.elapsed_edges) + 1), ("count", len(self.count_bins)),
                        ("active", len(self.active_bins)), ("STATE", len(self.vocab)),
                        ("HIST", len(self.sources) * (len(self.lag_edges) - 1))):
            self.offsets[name] = o
            o += n
        self.size = o
        self._src = {c: i for i, c in enumerate(self.sources)}
        self._cat = {c: i for i, c in enumerate(self.cats)}

    @property
    def n_dur(self) -> int:
        return len(self.dur_edges) + 2           # bins + never/unknown

    def block_cols(self, block: str) -> np.ndarray:
        if block == "HET":
            return np.arange(0, self.offsets["STATE"])
        if block == "STATE":
            return np.arange(self.offsets["STATE"], self.offsets["HIST"])
        if block == "HIST":
            return np.arange(self.offsets["HIST"], self.size)
        raise ValueError(block)

    def hist_col(self, src_type: int, lag_bin: int) -> int | None:
        i = self._src.get(src_type)
        return None if i is None else self.offsets["HIST"] + i * (len(self.lag_edges) - 1) + lag_bin

    def cat_index(self, c: str | None) -> int:
        return self._cat.get(c, len(self.cats))     # last slot = unseen/unknown category


def _bin(x: float, edges: Sequence[float]) -> int:
    """Index of x among half-open bins (-inf, e0], (e0, e1], ... (strict upper boundaries by bisect)."""
    return bisect.bisect_left(list(edges), x)


def _count_bin(n: int, lowers: Sequence[int]) -> int:
    return bisect.bisect_right(list(lowers), n) - 1


@dataclass
class Design:
    """Segments, common features and per-target at-risk rows."""

    layout: Layout
    targets: list[int]                          # target types (K-indices)
    seg_video: np.ndarray                       # (S,)
    seg_dur: np.ndarray                         # (S,) seconds
    X: sp.csr_matrix                            # (S, layout.size)
    rows: dict[int, np.ndarray]                 # target -> at-risk segment indices
    dur: dict[int, np.ndarray]                  # target -> duration-bin index per row
    y: dict[int, np.ndarray]                    # target -> event indicator per row
    n_videos: int


def pair_segments(v: VideoRecord, key: tuple[int, int], pr: PairRecord, lay: Layout, K: int,
                  targets: set[int]) -> tuple[list[float], list[float], list[dict[int, float]],
                                              dict[int, list[tuple[int, int, int]]]]:
    """Segments of one pair: (starts, ends, feature dicts, per-target (local seg, dur bin, y))."""
    f = v.fps
    w0, w1 = pr.window[0] / f, pr.window[1] / f
    if w1 <= w0:
        return [], [], [], {}
    ivs = {r: [(b / f, e / f, b > 0, e < v.T) for b, e in ints] for r, ints in pr.rels.items()}
    events: list[tuple[float, int]] = []
    for r, lst in ivs.items():
        for b, e, kb, ke in lst:
            if kb:
                events.append((b, 2 * r))
            if ke:
                events.append((e, 2 * r + 1))
    events.sort()
    cps = {w0, w1}
    for t, _ in events:
        cps.add(t)
        cps.update(t + d for d in lay.lag_edges[1:])
    cps.update(w0 + d for d in lay.elapsed_edges)
    for lst in ivs.values():
        for b, e, kb, ke in lst:
            cps.update((b, e))
            if kb:
                cps.update(b + d for d in lay.dur_edges)
            if ke:
                cps.update(e + d for d in lay.dur_edges)
    pts = sorted(c for c in cps if w0 <= c <= w1)
    ev_at: dict[float, list[int]] = {}
    for t, c in events:
        ev_at.setdefault(t, []).append(c)
    subj = lay.offsets["subj"] + lay.cat_index(v.cats.get(key[0]))
    obj = lay.offsets["obj"] + lay.cat_index(v.cats.get(key[1]))
    starts, ends, feats = [], [], []
    per_t: dict[int, list[tuple[int, int, int]]] = {}
    ev_times = [t for t, _ in events]
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            continue
        m = 0.5 * (a + b)
        x: dict[int, float] = {subj: 1.0, obj: 1.0}
        x[lay.offsets["elapsed"] + _bin(m - w0, lay.elapsed_edges)] = 1.0
        n_prior = bisect.bisect_right(ev_times, a)
        x[lay.offsets["count"] + _count_bin(n_prior, lay.count_bins)] = 1.0
        on = {}
        for r, lst in ivs.items():
            st = next(((b0, e0, kb, ke) for b0, e0, kb, ke in lst if b0 < m < e0), None)
            if st is not None:
                on[r] = st
        x[lay.offsets["active"] + _count_bin(len(on), lay.active_bins)] = 1.0
        for r in on:
            x[lay.offsets["STATE"] + r] = 1.0
        for t, c in events[:n_prior]:
            lag = m - t
            if lag <= lay.lag_edges[-1]:
                col = lay.hist_col(c, _bin(lag, lay.lag_edges) - 1)
                if col is not None:
                    x[col] = 1.0
        li = len(starts)
        starts.append(a)
        ends.append(b)
        feats.append(x)
        here = ev_at.get(b, [])
        for r in range(len(lay.vocab)):
            on_t, off_t = 2 * r, 2 * r + 1
            if r in on and off_t in targets:
                b0, e0, kb, ke = on[r]
                d = 1 + _bin(m - b0, lay.dur_edges) if kb else 0          # 0 = unknown age
                per_t.setdefault(off_t, []).append((li, d, int(off_t in here)))
            elif r not in on and on_t in targets:
                prev = [e0 for b0, e0, kb, ke in ivs.get(r, []) if ke and e0 <= a]
                d = 1 + _bin(m - max(prev), lay.dur_edges) if prev else 0  # 0 = never on before
                if r in ivs or d or on_t in here:
                    per_t.setdefault(on_t, []).append((li, d, int(on_t in here)))
    return starts, ends, feats, per_t


def build_design(videos: list[VideoRecord], lay: Layout, targets: Sequence[int]) -> Design:
    """Build the design for a list of videos (order defines video indices)."""
    K = 2 * len(lay.vocab)
    tset = set(targets)
    onset_targets = [k for k in targets if k % 2 == 0]
    seg_video, seg_dur, rows_i, cols_i = [], [], [], []
    rows: dict[int, list[int]] = {k: [] for k in targets}
    dur: dict[int, list[int]] = {k: [] for k in targets}
    y: dict[int, list[int]] = {k: [] for k in targets}
    for vi, v in enumerate(videos):
        for key, pr in v.pairs.items():
            starts, ends, feats, per_t = pair_segments(v, key, pr, lay, K, tset)
            base = len(seg_video)
            for li, (a, b, x) in enumerate(zip(starts, ends, feats)):
                seg_video.append(vi)
                seg_dur.append(b - a)
                for c in x:
                    rows_i.append(base + li)
                    cols_i.append(c)
            present_on = set(per_t)
            for k, lst in per_t.items():
                for li, d, yy in lst:
                    rows[k].append(base + li)
                    dur[k].append(d)
                    y[k].append(yy)
            # onset targets of predicates absent from the pair: whole window at risk, "never" bin
            for k in onset_targets:
                if k not in present_on and (k // 2) not in pr.rels:
                    for li in range(len(starts)):
                        rows[k].append(base + li)
                        dur[k].append(0)
                        y[k].append(0)
    S = len(seg_video)
    X = sp.csr_matrix((np.ones(len(rows_i)), (rows_i, cols_i)), shape=(S, lay.size))
    return Design(lay, list(targets), np.array(seg_video, dtype=np.int64), np.array(seg_dur),
                  X, {k: np.array(v, dtype=np.int64) for k, v in rows.items()},
                  {k: np.array(v, dtype=np.int64) for k, v in dur.items()},
                  {k: np.array(v, dtype=np.float64) for k, v in y.items()}, len(videos))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

RUNGS: dict[str, tuple[str, ...]] = {
    "G0": (), "G1": ("HET",), "G2": ("HET", "DUR"), "G3": ("HET", "DUR", "STATE"),
    "G4x": ("HET", "DUR", "HIST"), "G4": ("HET", "DUR", "STATE", "HIST"), "G4+": ("HET", "DUR", "STATE", "HIST"),
}


def target_columns(lay: Layout, rung: str, k: int) -> np.ndarray:
    """Common-design columns used by target k under a rung (own predicate excluded from STATE/HIST)."""
    blocks = RUNGS[rung]
    r = k // 2
    cols: list[int] = []
    if "HET" in blocks:
        cols += lay.block_cols("HET").tolist()
    if "STATE" in blocks:
        cols += [c for c in lay.block_cols("STATE") if c - lay.offsets["STATE"] != r]
    if "HIST" in blocks:
        nl = len(lay.lag_edges) - 1
        for i, src in enumerate(lay.sources):
            if src // 2 != r:
                cols += [lay.offsets["HIST"] + i * nl + m for m in range(nl)]
    return np.array(cols, dtype=np.int64)


@dataclass
class TargetFit:
    """Fitted parameters of one target."""

    k: int
    cols: np.ndarray
    eta: float
    w: np.ndarray            # weights for cols
    v: np.ndarray            # duration-bin weights (n_dur) or empty


def _target_matrix(d: Design, k: int, cols: np.ndarray, use_dur: bool) -> sp.csr_matrix:
    rows = d.rows[k]
    M = d.X[rows][:, cols]
    if use_dur:
        n = d.layout.n_dur
        D = sp.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), d.dur[k])), shape=(len(rows), n))
        M = sp.hstack([M, D], format="csr")
    return M


def fit_target(d: Design, k: int, rung: str, ridge: float, max_iter: int = 500,
               rows_mask: np.ndarray | None = None) -> TargetFit:
    """Ridge-penalised Poisson regression for one target (exact point-process MLE + penalty)."""
    cols = target_columns(d.layout, rung, k)
    use_dur = "DUR" in RUNGS[rung]
    M = _target_matrix(d, k, cols, use_dur)
    expo = d.seg_dur[d.rows[k]]
    yy = d.y[k]
    if rows_mask is not None:
        M, expo, yy = M[rows_mask], expo[rows_mask], yy[rows_mask]
    p = M.shape[1]
    n_ev = yy.sum()
    eta0 = math.log(max(n_ev, 0.5) / max(expo.sum(), 1e-12))
    hist_lo = None
    if rung == "G4+":
        nh = sum(1 for c in cols if c >= d.layout.offsets["HIST"])
        hist_lo = (len(cols) - nh, len(cols))

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta, w = theta[0], theta[1:]
        z = eta + M @ w
        mu = expo * np.exp(np.minimum(z, 50.0))
        val = float(mu.sum() - yy @ z + 0.5 * ridge * w @ w)
        r = mu - yy
        g = np.concatenate([[r.sum()], M.T @ r + ridge * w])
        return val, g
    bounds = [(None, None)] * (1 + p)
    if hist_lo is not None:
        for j in range(hist_lo[0], hist_lo[1]):
            bounds[1 + j] = (0.0, None)
    x0 = np.concatenate([[eta0], np.zeros(p)])
    res = so.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7})
    w = res.x[1:]
    nd = d.layout.n_dur if use_dur else 0
    return TargetFit(k, cols, float(res.x[0]), w[: p - nd], w[p - nd:] if nd else np.zeros(0))


def target_ll_per_video(d: Design, fit: TargetFit, use_dur: bool) -> np.ndarray:
    """Per-video held-out log-likelihood contribution of one target."""
    k = fit.k
    rows = d.rows[k]
    z = fit.eta + d.X[rows][:, fit.cols] @ fit.w
    if use_dur:
        z = z + fit.v[d.dur[k]]
    contrib = d.y[k] * z - d.seg_dur[rows] * np.exp(np.minimum(z, 50.0))
    out = np.zeros(d.n_videos)
    np.add.at(out, d.seg_video[rows], contrib)
    return out


def fit_rung(d: Design, rung: str, ridge: float, max_iter: int = 500) -> dict[int, TargetFit]:
    return {k: fit_target(d, k, rung, ridge, max_iter) for k in d.targets}


def ll_rung(d: Design, fits: dict[int, TargetFit], rung: str) -> np.ndarray:
    use_dur = "DUR" in RUNGS[rung]
    return sum(target_ll_per_video(d, fits[k], use_dur) for k in d.targets)


def n_events_per_video(d: Design) -> np.ndarray:
    out = np.zeros(d.n_videos)
    for k in d.targets:
        np.add.at(out, d.seg_video[d.rows[k]], d.y[k])
    return out
