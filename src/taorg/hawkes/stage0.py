"""Phase 5H Stage 0 pipeline (research/phase5h_stage0_preregistration.md).

Stages: data -> descriptive (FIT only) -> DEV selection -> TEST evaluation -> C1 null ->
C4 stability -> decision. All artifacts go to results/phase5h/stage0/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats as sps

from taorg.experiments.phase3.pipeline import finish_stage_metadata, new_stage_metadata
from taorg.hawkes.dataset import HawkesData, build
from taorg.hawkes.decision import decide_stage0
from taorg.hawkes.models import (
    HawkesParams, dur_fit, dur_ll_per_video, dur_stats, fit, ll_per_video,
)
from taorg.hawkes.vidor import VideoRecord, circular_shift, load_videos, pair_stream

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def streams_of(videos: list[VideoRecord], K: int) -> list:
    return [[pair_stream(v, pr, K) for pr in v.pairs.values()] for v in videos]


def splits(videos: list[VideoRecord], cfg: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """DEV / FIT / folds (indices into the sorted training list)."""
    s = cfg["splits"]
    ids = list(range(len(videos)))
    random.Random(s["shuffle_seed"]).shuffle(ids)
    dev = np.array(sorted(ids[: s["dev_size"]]))
    fitv = sorted(ids[s["dev_size"]:])
    f2 = list(fitv)
    random.Random(s["fold_seed"]).shuffle(f2)
    folds = [np.array(sorted(f2[i::s["n_folds"]])) for i in range(s["n_folds"])]
    return dev, np.array(fitv), folds


def boot_stats(diff_v: np.ndarray, ev_v: np.ndarray, R: int, rng: np.random.Generator, ci: float) -> dict[str, float]:
    """Mean ΔLL per event = sum(diff_v) / sum(ev_v), bootstrap over videos (ratio estimator)."""
    n = len(diff_v)
    idx = rng.integers(0, n, size=(R, n))
    reps = diff_v[idx].sum(1) / ev_v[idx].sum(1)
    lo, hi = np.quantile(reps, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(diff_v.sum() / ev_v.sum()), "ci_lo": float(lo), "ci_hi": float(hi)}


def cross_mask(K: int) -> np.ndarray:
    """(K, K+1) mask of cross-relation cells (different predicate; excludes PAIR_BIRTH)."""
    M = np.zeros((K, K + 1), dtype=bool)
    for k in range(K):
        for c in range(K):
            M[k, c] = (k // 2) != (c // 2)
    return M


def stability(fold_params: list[HawkesParams], K: int, branching_min: float, min_folds: int
              ) -> tuple[int, float, np.ndarray]:
    """S (cells present in >= min_folds folds) and mean pairwise Spearman of cross alphas."""
    m = cross_mask(K)
    vecs = [p.A[m] for p in fold_params]
    present = np.array([(v / p.beta) >= branching_min for v, p in zip(vecs, fold_params)])
    counts = present.sum(0)
    S = int((counts >= min_folds).sum())
    rhos = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            r = sps.spearmanr(vecs[i], vecs[j]).correlation
            rhos.append(0.0 if r is None or math.isnan(r) else float(r))
    return S, float(np.mean(rhos)), counts


def run(cfg_path: Path, out: Path, workers: int = 1) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    out.mkdir(parents=True, exist_ok=True)
    meta = new_stage_metadata("phase5h_stage0", cfg, False)
    t0 = time.time()
    dc, mc = cfg["data"], cfg["models"]
    train, vocab, st_tr = load_videos(dc["root"], dc["train_dir"])
    test, _, st_te = load_videos(dc["root"], dc["test_dir"], vocab)
    R = len(vocab)
    K = 2 * R
    dev, fitv, folds = splits(train, cfg)
    logger.info("Loaded %d train (%d dev, %d fit) and %d test videos; %d predicates", len(train), len(dev),
                len(fitv), len(test), R)
    s_tr, s_te = streams_of(train, K), streams_of(test, K)
    summary: dict[str, Any] = {"vocab": vocab, "load_stats": {"train": st_tr, "test": st_te},
                               "n_dev": len(dev), "n_fit": len(fitv), "n_test": len(test),
                               "fold_sizes": [len(f) for f in folds]}

    # ---- descriptive (FIT only; never decisive) ----
    summary["descriptive"] = descriptive(train, s_tr, fitv, vocab, K)

    # ---- DEV selection ----
    data_tr: dict[float, HawkesData] = {}
    sel_rows = []
    best: dict[str, tuple[float, float, float, HawkesParams]] = {}
    for beta in mc["beta_grid"]:
        tb = time.time()
        data_tr[beta] = build(train, R, beta, s_tr)
        logger.info("  built train stats beta=%g (%.0fs)", beta, time.time() - tb)
        for model in ("HK1", "HK2"):
            for pen in mc["penalty_grid"]:
                p = fit(data_tr[beta], fitv, model, pen, mc["mu_min"])
                dev_ll = float(ll_per_video(data_tr[beta], p)[dev].sum())
                sel_rows.append({"model": model, "beta": beta, "penalty": pen, "dev_ll": dev_ll})
                if model not in best or dev_ll > best[model][0]:
                    best[model] = (dev_ll, beta, pen, p)
                logger.info("    %s beta=%g pen=%g dev_ll=%.1f", model, beta, pen, dev_ll)
    summary["selection"] = sel_rows
    summary["selected"] = {m: {"beta": b[1], "penalty": b[2], "dev_ll": b[0]} for m, b in best.items()}
    (out / "selection.json").write_text(json.dumps({"rows": sel_rows, "selected": summary["selected"]}, indent=1),
                                        encoding="utf-8")
    b1, pen1, P1 = best["HK1"][1], best["HK1"][2], best["HK1"][3]
    b2, pen2, P2 = best["HK2"][1], best["HK2"][2], best["HK2"][3]
    P0 = fit(data_tr[b1], fitv, "HK0", 0.0, mc["mu_min"])
    lagw = np.diff(summary["descriptive"]["peri_event_top20"][0]["lag_bins_s"]) if summary["descriptive"]["peri_event_top20"] else []
    for e in summary["descriptive"]["peri_event_top20"]:
        r = vocab.index(e["target"][3:-1])
        e["baseline_onsets_per_source_event_by_lag"] = (P0.mu[2 * r] * np.asarray(lagw)).round(5).tolist()

    # ---- TEST (once) ----
    d_te1, d_te2 = build(test, R, b1, s_te), (build(test, R, b2, s_te) if b2 != b1 else None)
    d_te2 = d_te2 or d_te1
    ev_te = np.bincount(d_te1.ev_video, minlength=len(test)).astype(float)
    ll = {"HK0": ll_per_video(d_te1, HawkesParams(P0.mu, P0.A, b1)),
          "HK1": ll_per_video(d_te1, P1), "HK2": ll_per_video(d_te2, P2)}
    ds_tr = dur_stats(s_tr, R, mc["duration_bins_s"], data_tr[b1])
    ds_te = dur_stats(s_te, R, mc["duration_bins_s"], d_te1)
    on, hz = dur_fit(ds_tr, fitv, mc["mu_min"])
    ll["DUR"] = dur_ll_per_video(ds_te, on, hz)
    bc = cfg["bootstrap"]
    rng = np.random.default_rng(bc["seed"])
    n_ev = int(ev_te.sum())
    test_res = {m: float(v.sum() / n_ev) for m, v in ll.items()}
    g_cross = boot_stats(ll["HK2"] - ll["HK1"], ev_te, bc["resamples"], rng, bc["ci_level"])
    g_dur = boot_stats(ll["HK2"] - ll["DUR"], ev_te, bc["resamples"], rng, bc["ci_level"])
    g_base = boot_stats(ll["HK2"] - ll["HK0"], ev_te, bc["resamples"], rng, bc["ci_level"])
    g_self = boot_stats(ll["HK1"] - ll["HK0"], ev_te, bc["resamples"], rng, bc["ci_level"])
    g_dur1 = boot_stats(ll["HK1"] - ll["DUR"], ev_te, bc["resamples"], rng, bc["ci_level"])
    summary["test"] = {"n_events": n_ev, "ll_per_event": test_res, "G_cross": g_cross, "G_dur": g_dur,
                       "G_base": g_base, "G_self": g_self, "HK1_minus_DUR": g_dur1}
    np.savez_compressed(out / "params.npz", HK0_mu=P0.mu, HK1_mu=P1.mu, HK1_A=P1.A, HK2_mu=P2.mu, HK2_A=P2.A,
                        dur_on=on, dur_hz=hz, beta1=b1, beta2=b2)
    logger.info("TEST: %s  G_cross=%s", test_res, g_cross)

    # ---- C1 null ----
    fit_videos = [train[i] for i in fitv]
    null_g = []
    for seed in cfg["null"]["c1_seeds"]:
        tn = time.time()
        sf, stt = circular_shift(fit_videos, seed), circular_shift(test, seed + 100000)
        ssf, sst = streams_of(sf, K), streams_of(stt, K)
        idx = np.arange(len(sf))
        f1, e1 = build(sf, R, b1, ssf), build(stt, R, b1, sst)
        f2, e2 = (build(sf, R, b2, ssf), build(stt, R, b2, sst)) if b2 != b1 else (f1, e1)
        q1, q2 = fit(f1, idx, "HK1", pen1, mc["mu_min"]), fit(f2, idx, "HK2", pen2, mc["mu_min"])
        nev = len(e1.ev_type)
        g = float((ll_per_video(e2, q2).sum() - ll_per_video(e1, q1).sum()) / nev)
        null_g.append(g)
        logger.info("  C1 null seed %d: G_cross_null=%.5f (%.0fs)", seed, g, time.time() - tn)
    summary["c1_null"] = null_g

    # ---- C4 stability ----
    stc = cfg["stability"]
    real_fold = [fit(data_tr[b2], f, "HK2", pen2, mc["mu_min"]) for f in folds]
    S_real, rho_real, counts = stability(real_fold, K, stc["branching_min"], stc["min_folds_present"])
    S_null, rho_null = [], []
    pos = {v: i for i, v in enumerate(fitv)}
    for seed in cfg["null"]["c4_seeds"]:
        sf = circular_shift(fit_videos, seed)
        f2 = build(sf, R, b2, streams_of(sf, K))
        fp = [fit(f2, np.array([pos[v] for v in f]), "HK2", pen2, mc["mu_min"]) for f in folds]
        s_, r_, _ = stability(fp, K, stc["branching_min"], stc["min_folds_present"])
        S_null.append(s_)
        rho_null.append(r_)
        logger.info("  C4 null seed %d: S=%d rho=%.3f", seed, s_, r_)
    summary["c4"] = {"S_real": S_real, "rho_real": rho_real, "S_null": S_null, "rho_null": rho_null}
    np.save(out / "fold_presence_counts.npy", counts)
    np.savez_compressed(out / "fold_alphas.npz", *[p.A for p in real_fold])

    # ---- decision ----
    dec = decide_stage0(g_cross, null_g, g_dur, g_base, g_self["mean"], S_real, S_null, rho_real, rho_null,
                        cfg["decision"]["c3_ratio"])
    dec["c3_descriptive"] = {str(r): (dec["excess"] >= r * g_self["mean"]) for r in cfg["decision"]["c3_descriptive_ratios"]}
    summary["decision"] = dec
    summary["top_cross_cells"] = top_cells(P2, counts, vocab, K)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    finish_stage_metadata(meta, out, {"elapsed_s": time.time() - t0,
                                                               "label": dec["label"]})
    return summary


def top_cells(P: HawkesParams, counts: np.ndarray, vocab: list[str], K: int, n: int = 25) -> list[dict]:
    m = cross_mask(K)
    idx = np.argwhere(m)
    br = P.A[m] / P.beta
    order = np.argsort(-br)[:n]
    name = lambda c: "PAIR_BIRTH" if c == K else f"{'ON' if c % 2 == 0 else 'OFF'}({vocab[c // 2]})"
    return [{"target": name(int(idx[i][0])), "source": name(int(idx[i][1])), "alpha": float(P.A[m][i]),
             "branching": float(br[i]), "folds_present": int(counts[i])} for i in order]


def descriptive(train, s_tr, fitv, vocab, K) -> dict[str, Any]:
    """Counts, durations, censoring, peri-event onset rates (FIT only)."""
    R = len(vocab)
    n_on, n_off = np.zeros(R), np.zeros(R)
    durs = [[] for _ in range(R)]
    cens = np.zeros(R)
    for vi in fitv:
        for ps in s_tr[vi]:
            for _, c in ps.events:
                if c < K:
                    (n_on if c % 2 == 0 else n_off)[c // 2] += 1
            for r, d, known, obs in ps.ages:
                durs[r].append(d)
                cens[r] += (not known) or (not obs)
    per_pred = [{"predicate": vocab[r], "onsets": int(n_on[r]), "offsets": int(n_off[r]), "intervals": len(durs[r]),
                 "median_duration_s": float(np.median(durs[r])) if durs[r] else math.nan,
                 "censored_fraction": float(cens[r] / len(durs[r])) if durs[r] else math.nan} for r in range(R)]
    # Peri-event: ONSET(r) rate within lag bins after an event of r' (same pair) vs overall onset rate.
    lags = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    pair_counts: dict[tuple[int, int], np.ndarray] = {}
    trig_n: dict[int, int] = {}
    for vi in fitv:
        for ps in s_tr[vi]:
            ev = [(t, c) for t, c in ps.events if c < K]
            for i, (ti, ci) in enumerate(ev):
                trig_n[ci] = trig_n.get(ci, 0) + 1
                for tj, cj in ev[i + 1:]:
                    dt = tj - ti
                    if dt <= 0 or dt >= lags[-1]:
                        continue
                    if cj % 2 == 0 and cj // 2 != ci // 2:
                        key = (ci, cj)
                        arr = pair_counts.setdefault(key, np.zeros(len(lags) - 1))
                        arr[np.searchsorted(lags, dt, side="right") - 1] += 1
    tot = sorted(pair_counts.items(), key=lambda kv: -kv[1].sum())[:20]
    name = lambda c: f"{'ON' if c % 2 == 0 else 'OFF'}({vocab[c // 2]})"
    peri = [{"source": name(s), "target": name(t), "n_source_events": trig_n[s],
             "onsets_per_source_event_by_lag": (arr / trig_n[s]).round(5).tolist(), "lag_bins_s": lags}
            for (s, t), arr in tot]
    return {"per_predicate": per_pred, "peri_event_top20": peri}
