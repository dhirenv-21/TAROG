"""Sufficient statistics for exponential-kernel Hawkes likelihoods, decomposed per video.

For a shared decay beta, the log-likelihood of every model in Stage 0 is linear in
(mu_k, alpha_k) inside the log at event times and inside the compensator:

    LL = sum_e log(mu_k(e) + alpha_k(e) . x_e)  -  sum_k [ mu_k T_k + alpha_k . I_k ]

x_e[c]  = sum_{j: t_j < t_e, col(j)=c, same pair} exp(-beta (t_e - t_j))        (strict history)
I_k[c]  = sum_pairs sum_{j: col c} int_{risk_k(p) ∩ (t_j, inf)} exp(-beta (s - t_j)) ds
T_k     = total risk time of type k.

Columns c = 0..K-1 are modelled types (2r ONSET, 2r+1 OFFSET); column K is PAIR_BIRTH.
Onset risk of a predicate absent from a pair is the whole pair window, so onset
statistics are stored as (window totals) + (per present predicate corrections).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from taorg.hawkes.vidor import PairStream, VideoRecord, pair_stream


def interval_integral(t: np.ndarray, cols: np.ndarray, a: float, b: float, beta: float,
                      out: np.ndarray) -> None:
    """Add int_{[a,b) ∩ (t_j, inf)} exp(-beta (s - t_j)) ds to out[col_j] for every j."""
    m = t < b
    if not m.any():
        return
    tt = t[m]
    lo = np.maximum(a, tt)
    np.add.at(out, cols[m], (np.exp(-beta * (lo - tt)) - np.exp(-beta * (b - tt))) / beta)


def event_features(t: np.ndarray, cols: np.ndarray, beta: float, ncol: int) -> np.ndarray:
    """x_i[c] = sum_{j: t_j < t_i, col_j = c} exp(-beta (t_i - t_j)) (dense, O(m^2))."""
    D = t[:, None] - t[None, :]
    W = np.where(D > 0, np.exp(-beta * np.where(D > 0, D, 0.0)), 0.0)
    X = np.zeros((len(t), ncol))
    for c in np.unique(cols):
        X[:, c] = W[:, cols == c].sum(axis=1)
    return X


@dataclass
class HawkesData:
    """Per-video sufficient statistics for one beta."""

    beta: float
    K: int                       # modelled types (2 * predicates)
    n_videos: int
    ev_video: np.ndarray         # (n_events,)
    ev_type: np.ndarray          # (n_events,)
    X: sp.csr_matrix             # (n_events, K+1)
    TW: np.ndarray               # (n_videos,) summed pair-window lengths
    J: np.ndarray                # (n_videos, K+1) window integrals of excitation
    dT: np.ndarray               # (n_videos, R) onset risk-time corrections (present predicates)
    dJ: list[sp.csr_matrix]      # R x (n_videos, K+1) onset integral corrections
    Toff: np.ndarray             # (n_videos, R) offset risk time
    Ioff: list[sp.csr_matrix]    # R x (n_videos, K+1) offset integrals

    @property
    def R(self) -> int:
        return self.K // 2


def build(videos: list[VideoRecord], R: int, beta: float, streams: list[list[PairStream]] | None = None
          ) -> HawkesData:
    """Compute sufficient statistics for all videos (in order)."""
    K = 2 * R
    ncol = K + 1
    nv = len(videos)
    ev_video, ev_type, rows = [], [], []
    TW = np.zeros(nv)
    J = np.zeros((nv, ncol))
    dT = np.zeros((nv, R))
    Toff = np.zeros((nv, R))
    dJ_rows = [[] for _ in range(R)]
    Ioff_rows = [[] for _ in range(R)]
    for vi, v in enumerate(videos):
        ps_list = streams[vi] if streams is not None else [pair_stream(v, pr, K) for pr in v.pairs.values()]
        for ps in ps_list:
            a, b = ps.window
            TW[vi] += b - a
            if not ps.events:
                continue
            t = np.array([e[0] for e in ps.events])
            c = np.array([e[1] for e in ps.events])
            jw = np.zeros(ncol)
            interval_integral(t, c, a, b, beta, jw)
            J[vi] += jw
            mod = c < K
            if mod.any():
                X = event_features(t, c, beta, ncol)[mod]
                ev_video.extend([vi] * int(mod.sum()))
                ev_type.extend(c[mod].tolist())
                rows.append(sp.csr_matrix(X))
            for r, gaps in ps.on_risk.items():
                acc = np.zeros(ncol)
                for ga, gb in gaps:
                    interval_integral(t, c, ga, gb, beta, acc)
                dT[vi, r] += sum(gb - ga for ga, gb in gaps) - (b - a)
                dJ_rows[r].append((vi, acc - jw))
            for r, ons in ps.off_risk.items():
                acc = np.zeros(ncol)
                for oa, ob in ons:
                    interval_integral(t, c, oa, ob, beta, acc)
                Toff[vi, r] += sum(ob - oa for oa, ob in ons)
                Ioff_rows[r].append((vi, acc))

    def to_csr(items: list[tuple[int, np.ndarray]]) -> sp.csr_matrix:
        M = np.zeros((nv, ncol))
        for vi, vec in items:
            M[vi] += vec
        return sp.csr_matrix(M)
    Xall = sp.vstack(rows).tocsr() if rows else sp.csr_matrix((0, ncol))
    return HawkesData(beta, K, nv, np.array(ev_video, dtype=np.int64), np.array(ev_type, dtype=np.int64), Xall,
                      TW, J, dT, [to_csr(x) for x in dJ_rows], Toff, [to_csr(x) for x in Ioff_rows])


def type_totals(d: HawkesData, videos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T_k, I_k) summed over a subset of videos: shapes (K,), (K, K+1)."""
    K, R = d.K, d.R
    T = np.zeros(K)
    I = np.zeros((K, K + 1))
    tw, jv = d.TW[videos].sum(), d.J[videos].sum(axis=0)
    for r in range(R):
        T[2 * r] = tw + d.dT[videos, r].sum()
        I[2 * r] = jv + np.asarray(d.dJ[r][videos].sum(axis=0)).ravel()
        T[2 * r + 1] = d.Toff[videos, r].sum()
        I[2 * r + 1] = np.asarray(d.Ioff[r][videos].sum(axis=0)).ravel()
    return T, I
