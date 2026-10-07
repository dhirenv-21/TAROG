"""Phase 5H-1 GLM: segment likelihood vs an eager reference, valid intensity, constraints."""

from __future__ import annotations

import bisect
import math

import numpy as np

from taorg.hawkes.glm import (
    Layout, RUNGS, TargetFit, build_design, fit_target, ll_rung, fit_rung, n_events_per_video,
    target_columns, target_ll_per_video,
)
from taorg.hawkes.vidor import PairRecord, VideoRecord

VOCAB = ["a", "b", "c"]
K = 6


def layout() -> Layout:
    return Layout(VOCAB, ["person", "cup"], [0.0, 0.5, 2.0, 10.0], [2.0, 10.0], [0, 1, 2, 3, 5, 9], [0, 1, 2, 3],
                  [1.0, 4.0, 16.0], sources=list(range(K)))


def toy() -> list[VideoRecord]:
    p1 = PairRecord((0, 300), {0: [(0, 60), (120, 200)], 1: [(55, 130)], 2: [(210, 300)]})
    p2 = PairRecord((30, 280), {1: [(40, 90)]})
    p3 = PairRecord((10, 250), {})
    return [VideoRecord("v0", 10.0, 300, {(1, 2): p1, (2, 1): p2, (1, 3): p3}, {1: "person", 2: "cup", 3: "dog"})]


def ref_features(v, key, pr, lay, t):
    """Eager reference: covariates and per-target risk/dur bin at time t (strict history)."""
    f = v.fps
    w0, w1 = pr.window[0] / f, pr.window[1] / f
    ivs = {r: [(b / f, e / f, b > 0, e < v.T) for b, e in x] for r, x in pr.rels.items()}
    ev = sorted([(b, 2 * r) for r, x in ivs.items() for b, e, kb, ke in x if kb]
                + [(e, 2 * r + 1) for r, x in ivs.items() for b, e, kb, ke in x if ke])
    hist = [(s, c) for s, c in ev if s < t]
    x = {lay.offsets["subj"] + lay.cat_index(v.cats.get(key[0])): 1.0,
         lay.offsets["obj"] + lay.cat_index(v.cats.get(key[1])): 1.0}
    x[lay.offsets["elapsed"] + bisect.bisect_left(lay.elapsed_edges, t - w0)] = 1.0
    x[lay.offsets["count"] + bisect.bisect_right(lay.count_bins, len(hist)) - 1] = 1.0
    on = {r: st for r, lst in ivs.items() for st in lst if st[0] < t <= st[1]}
    x[lay.offsets["active"] + bisect.bisect_right(lay.active_bins, len(on)) - 1] = 1.0
    for r in on:
        x[lay.offsets["STATE"] + r] = 1.0
    for s, c in hist:
        lag = t - s
        if 0 < lag <= 10.0:
            x[lay.hist_col(c, bisect.bisect_left(lay.lag_edges, lag) - 1)] = 1.0
    risk = {}
    for r in range(len(VOCAB)):
        if r in on:
            b0, e0, kb, ke = on[r]
            risk[2 * r + 1] = 1 + bisect.bisect_left(lay.dur_edges, t - b0) if kb else 0
        else:
            prev = [e0 for b0, e0, kb, ke in ivs.get(r, []) if ke and e0 < t]
            risk[2 * r] = 1 + bisect.bisect_left(lay.dur_edges, t - max(prev)) if prev else 0
    return x, risk, ev, (w0, w1)


def ref_ll(v, lay, fits, rung, n=40000):
    total = 0.0
    use_dur = "DUR" in RUNGS[rung]

    def z(fit, x, dbin):
        val = fit.eta + sum(x.get(c, 0.0) * w for c, w in zip(fit.cols, fit.w))
        return val + (fit.v[dbin] if use_dur else 0.0)
    for key, pr in v.pairs.items():
        x0, r0, ev, (w0, w1) = ref_features(v, key, pr, lay, (pr.window[0] + 1e-9) / v.fps)
        for t, c in ev:
            if c in fits and w0 < t <= w1:
                x, risk, _, _ = ref_features(v, key, pr, lay, t)
                total += z(fits[c], x, risk[c])
        grid = np.linspace(w0, w1, n + 1)
        mids = 0.5 * (grid[1:] + grid[:-1])
        dt = (w1 - w0) / n
        for t in mids:
            x, risk, _, _ = ref_features(v, key, pr, lay, t)
            for k, dbin in risk.items():
                if k in fits:
                    total -= math.exp(z(fits[k], x, dbin)) * dt
    return total


def test_segment_likelihood_matches_eager_reference():
    lay = layout()
    vids = toy()
    d = build_design(vids, lay, list(range(K)))
    rng = np.random.default_rng(1)
    for rung in ("G1", "G2", "G3", "G4"):
        fits = {}
        for k in range(K):
            cols = target_columns(lay, rung, k)
            fits[k] = TargetFit(k, cols, float(rng.normal(-1, 0.3)), rng.normal(0, 0.4, len(cols)),
                                rng.normal(0, 0.4, lay.n_dur))
        fast = ll_rung(d, fits, rung).sum()
        ref = ref_ll(vids[0], lay, fits, rung, n=6000)
        assert abs(fast - ref) < 2e-3 * max(1.0, abs(ref)), (rung, fast, ref)


def test_events_and_risk_sets_counted_once():
    lay = layout()
    d = build_design(toy(), lay, list(range(K)))
    # events: ON(a)@12s, OFF(a)@6s, OFF(a)@20s, ON(b)@5.5s, OFF(b)@13s, ON(c)@21s; pair2: ON(b)@4, OFF(b)@9
    assert n_events_per_video(d)[0] == 8
    for k in range(K):
        assert len(np.unique(d.rows[k])) == len(d.rows[k])


def test_valid_intensity_under_extreme_inhibition_and_positive_only():
    lay = layout()
    d = build_design(toy(), lay, list(range(K)))
    k = 1
    cols = target_columns(lay, "G4", k)
    fit = TargetFit(k, cols, 0.0, np.full(len(cols), -200.0), np.zeros(lay.n_dur))
    ll = target_ll_per_video(d, fit, True)
    assert np.isfinite(ll).all()                                # exp link: intensity > 0, likelihood finite
    pos = fit_target(d, k, "G4+", ridge=1.0)
    hist = pos.cols >= lay.offsets["HIST"]
    assert (pos.w[hist] >= -1e-12).all()


def test_fit_is_reproducible_and_beats_base_on_training_data():
    lay = layout()
    d = build_design(toy(), lay, list(range(K)))
    a, b = fit_rung(d, "G2", 1.0), fit_rung(d, "G2", 1.0)
    assert all(np.allclose(a[k].w, b[k].w) and a[k].eta == b[k].eta for k in a)
    g0 = ll_rung(d, fit_rung(d, "G0", 1.0), "G0").sum()
    g2 = ll_rung(d, a, "G2").sum()
    assert g2 >= g0 - 1e-6


def test_leakage_guard_and_split_disjointness():
    import pytest
    from pathlib import Path
    from taorg.hawkes.phase5h1 import LeakageError, guard, load_config, split_indices
    from taorg.hawkes.stage0 import load_config as s0load
    with pytest.raises(LeakageError):
        guard({"SMOKE_FIT", "EVAL_A"}, allow_eval=False)
    guard({"EVAL_A"}, allow_eval=True)
    cfg = load_config(Path("configs/experiments/phase5h1.yaml"))
    idx = split_indices(7000, cfg, s0load(Path("configs/experiments/phase5h_stage0.yaml")))
    sets = {k: set(v.tolist()) for k, v in idx.items()}
    assert not sets["TRAIN"] & sets["SELECT"] and not sets["TRAIN"] & sets["EVAL_A"]
    assert not sets["SELECT"] & sets["EVAL_A"]
    assert len(sets["SELECT"]) == len(sets["EVAL_A"]) == 700 and len(sets["TRAIN"]) == 5600
    assert sets["SMOKE_FIT"] <= sets["TRAIN"] and sets["SMOKE_CHECK"] <= sets["TRAIN"]
    assert not sets["SMOKE_FIT"] & sets["SMOKE_CHECK"]
