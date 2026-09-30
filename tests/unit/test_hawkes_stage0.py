"""Stage 0 Hawkes machinery: eager-reference likelihood, extraction rules, null shift, recovery."""

from __future__ import annotations

import math
import random

import numpy as np

from taorg.hawkes.dataset import build, type_totals
from taorg.hawkes.models import (
    HawkesParams, dur_fit, dur_ll_per_video, dur_stats, fit, ll_per_video, support,
)
from taorg.hawkes.vidor import PairRecord, VideoRecord, circular_shift, pair_stream

R = 3
K = 2 * R


def toy_videos(seed: int, n: int = 4) -> list[VideoRecord]:
    rng = random.Random(seed)
    vids = []
    for i in range(n):
        T = rng.randint(200, 400)
        pairs = {}
        for pk in range(3):
            w0 = rng.choice([0, rng.randint(1, 30)])
            w1 = T - rng.choice([0, rng.randint(1, 30)])
            rels = {}
            for r in range(R):
                if rng.random() < 0.6:
                    ivs, cur = [], w0 + rng.randint(0, 20) * (rng.random() < 0.7)
                    while cur < w1 - 5 and len(ivs) < 3:
                        e = min(w1, cur + rng.randint(5, 60))
                        ivs.append((cur, e))
                        cur = e + rng.randint(3, 50)
                    if ivs:
                        rels[r] = ivs
            pairs[(pk, pk + 10)] = PairRecord((w0, w1), rels)
        vids.append(VideoRecord(f"v{i}", 30.0, T, pairs))
    return vids


def brute_ll(v: VideoRecord, p: HawkesParams, grid: int = 20000) -> float:
    """Eager O(N^2) reference: direct intensity sums + fine numerical compensator."""
    total = 0.0
    for pr in v.pairs.values():
        ps = pair_stream(v, pr, K)
        ev = ps.events

        def lam(k: int, s: float) -> float:
            return p.mu[k] + sum(p.A[k, c] * math.exp(-p.beta * (s - tj)) for tj, c in ev if tj < s)
        for t, c in ev:
            if c < K:
                total += math.log(lam(c, t))
        a, b = ps.window
        for k in range(K):
            r, is_off = k // 2, k % 2 == 1
            if is_off:
                risk = ps.off_risk.get(r, [])
            else:
                risk = ps.on_risk.get(r, [(a, b)])
            for lo, hi in risk:
                if hi <= lo:
                    continue
                n = max(50, int(grid * (hi - lo) / 15.0))
                s = np.linspace(lo, hi, n + 1)
                mid = 0.5 * (s[1:] + s[:-1])
                total -= sum(lam(k, x) for x in mid) * (hi - lo) / n
    return total


def test_fast_likelihood_matches_eager_reference():
    vids = toy_videos(1, n=2)
    rng = np.random.default_rng(0)
    for beta in (0.5, 3.0):
        d = build(vids, R, beta)
        p = HawkesParams(rng.uniform(0.01, 0.3, K), rng.uniform(0, 0.5, (K, K + 1)), beta)
        fast = ll_per_video(d, p)
        for vi, v in enumerate(vids):
            ref = brute_ll(v, p, grid=4000)
            assert abs(fast[vi] - ref) < 2e-3 * max(1.0, abs(ref)), (fast[vi], ref)


def test_extraction_rules_censoring_and_birth():
    v = VideoRecord("x", 10.0, 100, {(1, 2): PairRecord((5, 100), {0: [(0, 20), (40, 100)], 1: [(10, 30)]})})
    ps = pair_stream(v, v.pairs[(1, 2)], K)
    types = [c for _, c in ps.events]
    assert K in types                          # PAIR_BIRTH at w0 = 5 > 0
    assert types.count(0) == 1                 # ONSET(0) only at 40 (0 is left-censored)
    assert types.count(1) == 1                 # OFFSET(0) only at 20 (100 = T right-censored)
    assert ps.on_risk[0] == [(2.0, 4.0)]       # between 20 and 40 frames (window starts at 5 < 20)
    assert ps.off_risk[0] == [(0.0, 2.0), (4.0, 10.0)]


def test_circular_shift_preserves_total_on_time_and_is_deterministic():
    vids = toy_videos(2)
    a, b = circular_shift(vids, 7), circular_shift(vids, 7)
    for v0, v1, v2 in zip(vids, a, b):
        for key in v0.pairs:
            for r, ivs in v0.pairs[key].rels.items():
                on0 = sum(e - s for s, e in ivs)
                on1 = sum(e - s for s, e in v1.pairs[key].rels[r])
                assert on0 == on1
                assert v1.pairs[key].rels[r] == v2.pairs[key].rels[r]


def test_totals_consistency_and_hk0_closed_form():
    vids = toy_videos(3)
    d = build(vids, R, 1.0)
    allv = np.arange(len(vids))
    T, I = type_totals(d, allv)
    p = fit(d, allv, "HK0", 0.0, 1e-10)
    for k in range(K):
        n = int((d.ev_type == k).sum())
        if n:
            assert abs(p.mu[k] - n / T[k]) < 1e-9
    streams = [[pair_stream(v, pr, K) for pr in v.pairs.values()] for v in vids]
    s = dur_stats(streams, R, [0.0, 1.0, 4.0], d)
    on, hz = dur_fit(s, allv, 1e-10)
    has = np.array([(d.ev_type == 2 * r).any() for r in range(R)])
    assert np.allclose(on[has], p.mu[0::2][has], rtol=1e-9)   # DUR onset part == HK0 onset
    assert np.isfinite(dur_ll_per_video(s, on, hz)).all()
    assert support("HK1", K)[0].sum() == 3 and support("HK2", K).all()


def simulate(alpha_cross: float, seed: int, n_videos: int = 60) -> list[VideoRecord]:
    """Two predicates; ONSET(1) is excited by ONSET(0) on the same pair (Ogata thinning)."""
    rng = np.random.default_rng(seed)
    fps, T = 30.0, 1800
    beta, mu_on, mu_off = 1.0, np.array([0.05, 0.02]), np.array([0.3, 0.3])
    vids = []
    for i in range(n_videos):
        pairs = {}
        for pk in range(4):
            t, Tend = 0.0, T / fps
            on = [False, False]
            start = [0.0, 0.0]
            hist: list[tuple[float, int]] = []
            rels = {0: [], 1: []}
            while True:
                ub = sum(mu_off) + sum(mu_on) + alpha_cross * (1 + len(hist))
                t += rng.exponential(1 / ub)
                if t >= Tend:
                    break
                exc = alpha_cross * sum(math.exp(-beta * (t - tj)) for tj, c in hist if c == 0)
                lam = [0 if on[0] else mu_on[0], 0 if on[1] else mu_on[1] + exc,
                       mu_off[0] if on[0] else 0, mu_off[1] if on[1] else 0]
                u = rng.uniform(0, ub)
                cum = np.cumsum(lam)
                if u >= cum[-1]:
                    continue
                j = int(np.searchsorted(cum, u, side="right"))
                r = j % 2
                f = int(t * fps) + 1
                if j < 2:
                    on[r], start[r] = True, f
                    hist.append((f / fps, 2 * r))
                else:
                    on[r] = False
                    if f > start[r]:
                        rels[r].append((start[r], f))
            for r in (0, 1):
                if on[r] and T > start[r]:
                    rels[r].append((start[r], T))
            pairs[(pk, pk + 10)] = PairRecord((0, T), {r: iv for r, iv in rels.items() if iv})
        vids.append(VideoRecord(f"s{i}", fps, T, pairs))
    return vids


def test_simulation_recovers_cross_excitation_and_null_gain():
    Rr = 2
    train, test = simulate(0.6, 1), simulate(0.6, 2, 30)
    d_tr, d_te = build(train, Rr, 1.0), build(test, Rr, 1.0)
    allv = np.arange(len(train))
    p1 = fit(d_tr, allv, "HK1", 0.0, 1e-10)
    p2 = fit(d_tr, allv, "HK2", 0.0, 1e-10)
    assert p2.A[2, 0] > 0.25                     # ONSET(1) <- ONSET(0)
    gain = (ll_per_video(d_te, p2) - ll_per_video(d_te, p1)).sum()
    assert gain > 0
    null_tr, null_te = simulate(0.0, 3), simulate(0.0, 4, 30)
    n_tr, n_te = build(null_tr, Rr, 1.0), build(null_te, Rr, 1.0)
    q1 = fit(n_tr, np.arange(len(null_tr)), "HK1", 0.0, 1e-10)
    q2 = fit(n_tr, np.arange(len(null_tr)), "HK2", 0.0, 1e-10)
    null_gain = (ll_per_video(n_te, q2) - ll_per_video(n_te, q1)).sum()
    assert gain > null_gain + 5.0
