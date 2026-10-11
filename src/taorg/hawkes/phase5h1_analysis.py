"""Phase 5H-1 analysis of saved artifacts (no fitting). Implements prereg §5-§9 with amendment A3.

Reads checkpoints under the run root and the committed freeze; writes results/phase5h1/full/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from taorg.hawkes.decision1 import decide_phase5h1, family_passes
from taorg.hawkes.families import FAMILIES, family_of
from taorg.hawkes.phase5h1 import load_config
from taorg.hawkes.phase5h1_run import (
    CFG, FROZEN_DIR, MAIN_RUNGS, NULL_RUNGS, VARIANTS, _fit_spec, _frozen, _layout, _meta, _sha_file, ds_name,
)
from taorg.hawkes.runner import Checkpoints, code_hash

OUT = Path("results/phase5h1/full")


def _boot_idx(n: int, R: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(R, n))


def ratio_stats(diff_v: np.ndarray, ev_v: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    """Per-event Δ = Σ diff / Σ events with percentile 95% CI over video resamples (paired)."""
    reps = diff_v[idx].sum(1) / ev_v[idx].sum(1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"mean": float(diff_v.sum() / ev_v.sum()), "ci_lo": float(lo), "ci_hi": float(hi),
            "boot_sd": float(reps.std(ddof=1)), "frac_videos_positive": float((diff_v > 0).mean()),
            "n_videos": int(len(diff_v)), "n_events": float(ev_v.sum())}


class Data:
    """Per-target per-video EVAL log-likelihoods of all frozen models."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.m = _meta(run)
        self.cfg = load_config(CFG)
        self.code = code_hash()
        self.fz = _frozen()
        self.targets = self.m["targets"]
        self.vocab = self.m["layout"]["vocab"]
        self.nA, self.nB = self.m["n_videos"]["EVAL_A"], self.m["n_videos"]["EVAL_B"]
        ck = Checkpoints(run / "ckpt" / "evaluate")
        fz_sha = _sha_file(FROZEN_DIR / "params.npz")
        self.ll: dict[str, np.ndarray] = {}
        ev = None
        for rung in MAIN_RUNGS + VARIANTS:
            L = np.zeros((len(self.targets), self.nA + self.nB))
            E = np.zeros_like(L)
            for i, k in enumerate(self.targets):
                for part, sl in (("EVAL_A", slice(0, self.nA)), ("EVAL_B", slice(self.nA, None))):
                    rec = ck.get({"stage": "evaluate", "k": k, "rung": rung, "part": part, "frozen": fz_sha,
                                  "code": self.code, "layout_sha": self.m["layout_sha"]})
                    if rec is None:
                        raise RuntimeError(f"missing evaluation {rung} {k} {part}")
                    with ck.arrays(rec["task_id"]) as a:
                        L[i, sl], E[i, sl] = a["ll"], a["ev"]
            self.ll[rung] = L
            if ev is None:
                ev = E
            elif not np.array_equal(ev, E):
                raise RuntimeError("event counts differ between models")
        self.ev = ev
        R = int(self.cfg["bootstrap"]["resamples"])
        s = int(self.cfg["bootstrap"]["seed"])
        n = self.nA + self.nB
        self.idx = {"EVAL": _boot_idx(n, R, s), "EVAL_A": _boot_idx(self.nA, R, s), "EVAL_B": _boot_idx(self.nB, R, s)}
        self.sl = {"EVAL": slice(0, n), "EVAL_A": slice(0, self.nA), "EVAL_B": slice(self.nA, n)}
        self.fam_rows = {F: np.array([i for i, k in enumerate(self.targets) if family_of(self.vocab[k // 2]) == F],
                                     dtype=int) for F in FAMILIES}

    def delta(self, a: str, b: str, vset: str = "EVAL", rows: np.ndarray | None = None) -> dict[str, float]:
        r = slice(None) if rows is None else rows
        sl = self.sl[vset]
        diff = (self.ll[a][r] - self.ll[b][r]).sum(0)[sl]
        ev = self.ev[r].sum(0)[sl]
        return ratio_stats(diff, ev, self.idx[vset])

    def per_event(self, rung: str, vset: str = "EVAL") -> float:
        sl = self.sl[vset]
        return float(self.ll[rung][:, sl].sum() / self.ev[:, sl].sum())


def null_deltas(d: Data) -> dict[str, Any]:
    """Per replicate: Δ(G4x,G2), Δ(G4,G3), per-family Δ_F(G4,G3), event counts (null EVAL)."""
    ck = Checkpoints(d.run / "ckpt" / "nulls")
    sel = d.fz["selected_ridge"]
    nc = d.cfg["null_control"]
    out: dict[str, Any] = {"replicates": []}
    for kind, seeds in (("circ", nc["circ_seeds"]), ("jit", nc["jitter_seeds"])):
        for s in seeds:
            tr, evn = ds_name("TRAIN", [kind, s]), ds_name("EVAL", [kind, s])
            tot = {r: np.zeros(len(d.targets)) for r in NULL_RUNGS}
            evk = np.zeros(len(d.targets))
            nonconv = 0
            for i, k in enumerate(d.targets):
                for r in NULL_RUNGS:
                    rec = ck.get(_fit_spec("nulls", tr, k, {"rung": r, "ridge": sel[r]}, [evn], d.code, d.m))
                    if rec is None:
                        raise RuntimeError(f"missing null fit {kind}{s} {r} {k}")
                    nonconv += int(not rec["diag"]["success"])
                    with ck.arrays(rec["task_id"]) as a:
                        tot[r][i] = float(a[f"ll__{evn}"].sum())
                        evk[i] = float(a[f"ev__{evn}"].sum())
            rep = {"kind": kind, "seed": s, "n_events": float(evk.sum()),
                   "d_cross": float((tot["G4x"] - tot["G2"]).sum() / evk.sum()),
                   "d_time": float((tot["G4"] - tot["G3"]).sum() / evk.sum()), "non_converged": nonconv,
                   "families": {}}
            for F, rows in d.fam_rows.items():
                if len(rows) and evk[rows].sum() > 0:
                    rep["families"][F] = float((tot["G4"][rows] - tot["G3"][rows]).sum() / evk[rows].sum())
            out["replicates"].append(rep)
    return out


def _spearman_mean(W: np.ndarray) -> float:
    rhos = []
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            r = sps.spearmanr(W[i], W[j]).correlation
            rhos.append(0.0 if r is None or math.isnan(r) else float(r))
    return float(np.mean(rhos))


def fold_weights(d: Data, null: list | None) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """(5, n_cells) G4 HIST weights per fold; cells = (target k, common column)."""
    ck = Checkpoints(d.run / "ckpt" / "stability")
    lay = _layout(d.m)
    lam = d.fz["selected_ridge"]["G4"]
    cells: list[tuple[int, int]] = []
    W = []
    for f in range(5):
        row = []
        for k in d.targets:
            rec = ck.get(_fit_spec("stability", ds_name("TRAIN", null), k, {"rung": "G4", "ridge": lam, "groups": [f]},
                                   [], d.code, d.m))
            if rec is None:
                raise RuntimeError(f"missing fold fit {null} f={f} k={k}")
            with ck.arrays(rec["task_id"]) as a:
                cols, w = a["cols"], a["w"]
                h = cols >= lay.offsets["HIST"]
                row.append(w[h])
                if f == 0:
                    cells += [(k, int(c)) for c in cols[h]]
        W.append(np.concatenate(row))
    return np.array(W), cells


def _tstats(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = W.mean(0)
    sd = W.std(0, ddof=1)
    agree = (W > 0).all(0) | (W < 0).all(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / math.sqrt(W.shape[0])), np.nan)
    return mean, sd, agree, t


def stability(d: Data) -> dict[str, Any]:
    lay = _layout(d.m)
    Wr, cells = fold_weights(d, None)
    mean, sd, agree, t = _tstats(Wr)
    nulls = []
    for s in d.cfg["null_control"]["fold_null_seeds"]:
        Wn, cn = fold_weights(d, ["jit", s])
        assert cn == cells
        nulls.append((s, Wn) + _tstats(Wn))
    absr = np.abs(t)
    cand = np.sort(np.unique(absr[agree & np.isfinite(absr)]))
    tau, fdr_at = None, None
    scan = []
    for c in cand:
        real = int((agree & (absr >= c)).sum())
        nd = float(np.mean([int((ag & (np.abs(tt) >= c)).sum()) for _, _, _, _, ag, tt in nulls]))
        fdr = nd / real if real else math.nan
        scan.append((float(c), real, nd, fdr))
        if tau is None and real > 0 and fdr <= d.cfg["decision"]["fdr"]:
            tau, fdr_at = float(c), fdr
    disc = agree & (absr >= tau) if tau is not None else np.zeros(len(cells), dtype=bool)
    rho_real = _spearman_mean(Wr)
    rho_null = [_spearman_mean(Wn) for _, Wn, *_ in nulls]
    # support: registered (source events with target at risk) + exposure / target events with cell active
    sck = Checkpoints(d.run / "ckpt" / "stability")
    srec = sck.get({"stage": "stability", "metric": "support", "code": d.code, "layout_sha": d.m["layout_sha"]})
    with sck.arrays(srec["task_id"]) as a:
        S, st = a["support"], list(a["targets"])
    nl = len(lay.lag_edges) - 1

    def cell_info(i: int) -> dict[str, Any]:
        k, col = cells[i]
        j = col - lay.offsets["HIST"]
        src = lay.sources[j // nl]
        name = lambda c: f"{'ON' if c % 2 == 0 else 'OFF'}({d.vocab[c // 2]})"
        return {"target": name(k), "source": name(src), "lag_bin": ["(0,0.5]", "(0.5,2]", "(2,10]"][j % nl],
                "mean_w": float(mean[i]), "sd_w": float(sd[i]), "t": float(t[i]),
                "sign_agree": bool(agree[i]), "support_source_events": float(S[st.index(k), src]),
                "class": f"{'ON' if src % 2 == 0 else 'OFF'}>{'ON' if k % 2 == 0 else 'OFF'}",
                "target_family": family_of(d.vocab[k // 2]), "fold_w": Wr[:, i].round(4).tolist()}
    order = np.argsort(-np.nan_to_num(absr, nan=-1))
    top = [cell_info(i) for i in order[:40] if agree[i]]
    discoveries = [cell_info(i) for i in np.flatnonzero(disc)]
    fam_counts = {}
    for x in discoveries:
        fam_counts[x["target_family"]] = fam_counts.get(x["target_family"], 0) + 1
    cls_counts = {}
    for x in discoveries:
        cls_counts[x["class"]] = cls_counts.get(x["class"], 0) + 1
    return {"n_cells": len(cells), "n_sign_agree_real": int(agree.sum()),
            "n_sign_agree_null": [int(n[4].sum()) for n in nulls], "tau": tau, "fdr_at_tau": fdr_at,
            "n_discoveries": int(disc.sum()), "rho_real": rho_real, "rho_null": rho_null,
            "fdr_scan_head": scan[:5], "fdr_scan_tail": scan[-5:],
            "fdr_scan_min": min((x[3] for x in scan if not math.isnan(x[3])), default=None),
            "discoveries": discoveries, "discoveries_by_family": fam_counts, "discoveries_by_class": cls_counts,
            "top_sign_agreeing_cells": top}


def hypotheses(d: Data) -> dict[str, Any]:
    from taorg.hawkes.hyp1 import log_rr
    h = d.cfg["hypotheses"]
    ck = Checkpoints(d.run / "ckpt" / "hypotheses")

    def spec(rep: int) -> dict:
        return {"stage": "hypotheses", "rep": rep, "seed": h["null_seed"], "J": d.cfg["null_control"]["jitter_s"],
                "window": h["window_s"], "pairs": h["pairs"], "code": d.code}
    obs = ck.get(spec(-1))
    idx = d.idx["EVAL"]
    res = []
    with ck.arrays(obs["task_id"]) as a:
        per_video = {key: a[key] for key in a.files}
    null = {key: [] for key in per_video}
    for rep in range(int(h["null_reps"])):
        rec = ck.get(spec(rep))
        if rec is None:
            raise RuntimeError(f"missing hypothesis replicate {rep}")
        with ck.arrays(rec["task_id"]) as a:
            for key in per_video:
                null[key].append(log_rr(a[key]))
    for name, (s, g) in zip(("H-a", "H-b", "H-c", "H-d"), h["pairs"]):
        key = f"{s}>{g}"
        c = per_video[key]
        tot = c.sum(0)
        lr = log_rr(tot)
        boots = np.array([log_rr(c[i].sum(0)) for i in idx])
        fin = boots[np.isfinite(boots)]
        nl = np.array(null[key])
        p = (1 + np.sum(np.abs(nl) >= abs(lr))) / (len(nl) + 1) if math.isfinite(lr) else math.nan
        res.append({"hypothesis": name, "source_onset": s, "target_offset": g, "E_in": float(tot[0]),
                    "X_in_s": float(tot[1]), "E_tot": float(tot[2]), "X_tot_s": float(tot[3]),
                    "log_rr": lr, "rr": math.exp(lr) if math.isfinite(lr) else None,
                    "ci_lo": float(np.quantile(fin, 0.025)) if len(fin) else None,
                    "ci_hi": float(np.quantile(fin, 0.975)) if len(fin) else None,
                    "boot_nonfinite": int((~np.isfinite(boots)).sum()), "null_median": float(np.nanmedian(nl)),
                    "null_max_abs": float(np.nanmax(np.abs(nl))), "p_perm": float(p),
                    "underpowered": bool(tot[0] < h["min_events"])})
    ps = [(r["p_perm"], i) for i, r in enumerate(res) if math.isfinite(r["p_perm"])]
    ps.sort()
    m = len(ps)
    run = 0.0
    for j, (p, i) in enumerate(ps):
        run = max(run, min(1.0, (m - j) * p))
        res[i]["p_holm"] = run
        res[i]["significant_holm_0.05"] = run <= 0.05 and not res[i]["underpowered"]
    return {"window_s": h["window_s"], "null_reps": int(h["null_reps"]), "results": res}


def next_event(d: Data) -> dict[str, Any]:
    ck = Checkpoints(d.run / "ckpt" / "evaluate")
    fz_sha = _sha_file(FROZEN_DIR / "params.npz")
    arrs = {}
    for part in ("EVAL_A", "EVAL_B"):
        rec = ck.get({"stage": "evaluate", "metric": "next_event", "part": part, "frozen": fz_sha, "code": d.code})
        with ck.arrays(rec["task_id"]) as a:
            arrs[part] = {key: a[key] for key in a.files}
    cat = {key: np.concatenate([arrs["EVAL_A"][key], arrs["EVAL_B"][key]]) for key in arrs["EVAL_A"]}
    n = cat["n"]
    out = {"n_events": float(n.sum()),
           "per_rung": {r: {"logloss": float(cat[f"{r}|logloss"].sum() / n.sum()),
                            "top1": float(cat[f"{r}|top1"].sum() / n.sum())} for r in MAIN_RUNGS}}
    for a, b in (("G4", "G3"), ("G4x", "G2"), ("G2", "G1")):
        out[f"logloss_{a}_minus_{b}"] = ratio_stats(cat[f"{a}|logloss"] - cat[f"{b}|logloss"], n, d.idx["EVAL"])
        out[f"top1_{a}_minus_{b}"] = ratio_stats(cat[f"{a}|top1"] - cat[f"{b}|top1"], n, d.idx["EVAL"])
    return out


def costs(run: Path) -> dict[str, Any]:
    att = [json.loads(x) for x in (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    ends = [a for a in att if a.get("event") == "end"]
    by = {}
    for a in ends:
        st = a["job"].split(":")[0]
        b = by.setdefault(st, {"jobs": 0, "cpu_s": 0.0, "max_peak_rss_gb": 0.0, "failed": 0})
        b["jobs"] += 1
        b["cpu_s"] += float(a.get("elapsed_s") or 0)
        b["max_peak_rss_gb"] = max(b["max_peak_rss_gb"], float(a.get("peak_rss_gb") or 0))
        b["failed"] += int(a["status"] != "complete")
    inv = [json.loads(x) for x in (run / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"by_job_type": by, "total_job_cpu_h": sum(b["cpu_s"] for b in by.values()) / 3600,
            "invocations": inv}


def analyze(run: Path) -> dict[str, Any]:
    d = Data(run)
    cfg = d.cfg
    res: dict[str, Any] = {"LABEL": "Phase 5H-1 registered comparison - EXPLORATORY evidence class (prereg §2)",
                           "frozen_commit": json.loads((run / "data" / "EVAL_UNLOCK.json").read_text())["frozen_commit"],
                           "selected_ridge": d.fz["selected_ridge"], "code_hash": d.code}
    # sample sizes and support
    evA, evB = d.ev[:, d.sl["EVAL_A"]].sum(1), d.ev[:, d.sl["EVAL_B"]].sum(1)
    name = lambda k: f"{'ON' if k % 2 == 0 else 'OFF'}({d.vocab[k // 2]})"
    res["sample"] = {"n_videos": {"TRAIN": d.m["n_videos"]["TRAIN"], "SELECT": d.m["n_videos"]["SELECT"],
                                  "EVAL_A": d.nA, "EVAL_B": d.nB},
                     "n_targets": len(d.targets), "eval_events": float(d.ev.sum()),
                     "eval_events_A": float(evA.sum()), "eval_events_B": float(evB.sum()),
                     "per_target": [{"type": name(k), "family": family_of(d.vocab[k // 2]), "eval_events_A": float(a),
                                     "eval_events_B": float(b)} for k, a, b in zip(d.targets, evA, evB)],
                     "family_events": {F: float(d.ev[r].sum()) for F, r in d.fam_rows.items()}}
    res["ll_per_event"] = {v: {r: d.per_event(r, v) for r in MAIN_RUNGS + VARIANTS} for v in ("EVAL", "EVAL_A", "EVAL_B")}
    comps = [("G1", "G0"), ("G2", "G1"), ("G3", "G2"), ("G4x", "G2"), ("G4", "G3"), ("G4", "G4x"), ("G4+", "G4"),
             ("G4+", "G3"), ("G4", "G2")]
    res["comparisons"] = {f"{a}-{b}": {v: d.delta(a, b, v) for v in ("EVAL", "EVAL_A", "EVAL_B")} for a, b in comps}
    res["ablations"] = {f"{r}-G3": {v: d.delta(r, "G3", v) for v in ("EVAL", "EVAL_A", "EVAL_B")} for r in VARIANTS}
    nd = null_deltas(d)
    res["nulls"] = nd
    reps = nd["replicates"]
    null_cross = [x["d_cross"] for x in reps]
    null_time = [x["d_time"] for x in reps]
    fams = {}
    for F, rows in d.fam_rows.items():
        if not len(rows):
            continue
        fams[F] = {"d_a": d.delta("G4", "G3", "EVAL_A", rows), "d_b": d.delta("G4", "G3", "EVAL_B", rows),
                   "d_all": d.delta("G4", "G3", "EVAL", rows), "null": [x["families"][F] for x in reps],
                   "dur_gain": d.delta("G2", "G1", "EVAL", rows)["mean"], "n_targets": int(len(rows)),
                   "cross_all": d.delta("G4x", "G2", "EVAL", rows)}
        fams[F]["passes"] = family_passes(fams[F], cfg["decision"]["practical_ratio"])
        fams[F]["null_max"], fams[F]["null_med"] = max(fams[F]["null"]), float(np.median(fams[F]["null"]))
    res["families"] = fams
    stab = stability(d)
    res["stability"] = stab
    dec = decide_phase5h1(res["comparisons"]["G4x-G2"]["EVAL"], null_cross, res["comparisons"]["G4-G3"]["EVAL"],
                          null_time, res["comparisons"]["G2-G1"]["EVAL"]["mean"], stab["n_discoveries"],
                          stab["rho_real"], stab["rho_null"], fams, cfg["decision"]["practical_ratio"])
    dec["null_cross_max"], dec["null_cross_med"] = max(null_cross), float(np.median(null_cross))
    dec["null_time_max"], dec["null_time_med"] = max(null_time), float(np.median(null_time))
    dec["k3_threshold"] = cfg["decision"]["practical_ratio"] * res["comparisons"]["G2-G1"]["EVAL"]["mean"]
    for kind in ("circ", "jit"):
        dec[f"null_{kind}_cross_max"] = max(x["d_cross"] for x in reps if x["kind"] == kind)
        dec[f"null_{kind}_time_max"] = max(x["d_time"] for x in reps if x["kind"] == kind)
    res["decision"] = dec
    res["hypotheses"] = hypotheses(d)
    res["next_event"] = next_event(d)
    res["costs"] = costs(run)
    fd = d.fz.get("non_converged_or_nonfinite", [])
    res["fit_diagnostics"] = {"frozen_non_converged_or_nonfinite": len(fd), "ridge_at_grid_boundary":
                              d.fz["at_grid_boundary"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res
