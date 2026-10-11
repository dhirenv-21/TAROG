"""Registered next-method figures and predefined failure cases (prereg §9); artifacts only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.episodes.analysis import Data  # noqa: E402
from taorg.experiments.episodes.conditions import build_instance  # noqa: E402
from taorg.experiments.episodes.evaluation import build_model, run_model  # noqa: E402
from taorg.experiments.episodes.pipeline import conditions, read_csv_gz, seeds  # noqa: E402
from taorg.experiments.episodes.plots import SHORT, _save, _setup  # noqa: E402
from taorg.experiments.next_method.pipeline import load_prior, with_prior  # noqa: E402
from taorg.experiments.phase3.stats import descriptive  # noqa: E402

STYLE = {"gated_ema_hyst": ("#E69F00", "s"), "b2_hold_buffer": ("#56B4E9", "^"),
         "b2_hold_geometric": ("#009E73", "v"), "b2_hold_static": ("#999999", "D"),
         "b2_hold_pooled": ("#CC79A7", "*"), "b2_hold_km": ("#0072B2", "P"),
         "ref_hsmm": ("#555555", "1"), "ref_episode": ("#D55E00", "x"), "ref_va_hmm": ("#BBBBBB", "2")}
CONTRASTS = (("B2", "vs B2"), ("BUF", "vs fixed buffer"), ("GEO", "vs geometric adaptive"),
             ("STATIC", "vs own static prior"), ("POOLED", "vs pooled stratum"))


def _lab(cfg, v):
    return cfg["method_labels"].get(v, cfg["method_labels"].get(v.replace("ref_", ""), v))


def run_figures(cfg: dict[str, Any], out: Path) -> list[str]:
    """Fig. 1-5."""
    _setup()
    data = Data(read_csv_gz(out / "raw" / "evaluation.csv.gz"), seeds(cfg, "evaluation", False))
    an = json.loads((out / "aggregated" / "analysis.json").read_text(encoding="utf-8"))
    paired = {}
    with open(out / "aggregated" / "paired.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            paired[(r["contrast"], r["condition"], r["metric"])] = r
    conds = list(cfg["conditions"])
    pd_ = out / "plots"
    files = []
    # Fig 1: dJ of K1 vs each comparator
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cols = ["#E69F00", "#56B4E9", "#009E73", "#999999", "#CC79A7"]
    mk = ["s", "^", "v", "D", "*"]
    for j, (k, name) in enumerate(CONTRASTS):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            r = paired[(k, c, "J")]
            m, a, b = float(r["mean_diff"]), float(r["ci_lo"]), float(r["ci_hi"])
            xs.append(i - 0.3 + 0.15 * j)
            ys.append(m)
            lo.append(m - a)
            hi.append(b - m)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk[j], color=cols[j], ms=4, lw=0.8, capsize=1.5, label=f"K1 {name}")
    ax.axhline(0, color="black", lw=0.6)
    for g in (0.02, -0.02):
        ax.axhline(g, color="#888888", lw=0.5, ls="--")
    for i in range(len(conds) - 1):
        ax.axvline(i + 0.5, color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([SHORT[c] for c in conds], rotation=30, ha="right")
    ax.set_ylabel("ΔJ (95% CI)")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Fig. 1 — K1 vs comparators (dashed: ±0.02 practical margin); decision {an['gate']['decision']}",
                 y=1.12, fontsize=9)
    files += _save(fig, pd_ / "fig1_delta_J")
    # Fig 2: gap recall + FPT
    show = cfg["methods_order"] + ["ref_hsmm", "ref_episode"]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, metric, yl in ((axes[0], "gap_recall", "gap recall"), (axes[1], "fpt_s", "false-persistence time (s/video)")):
        w = 0.8 / len(show)
        for j, v in enumerate(show):
            col, m_ = STYLE[v]
            xs, ys, e1, e2 = [], [], [], []
            for i, c in enumerate(conds):
                d = descriptive(list(data.get(c, v, metric)))
                if d["n"] == 0:
                    continue
                xs.append(i - 0.4 + w * (j + 0.5))
                ys.append(d["mean"])
                e1.append(0 if math.isnan(d["ci_lo"]) else d["mean"] - d["ci_lo"])
                e2.append(0 if math.isnan(d["ci_hi"]) else d["ci_hi"] - d["mean"])
            ax.errorbar(xs, ys, yerr=[e1, e2], fmt=m_, color=col, ms=4, lw=0.7, capsize=1.2, label=_lab(cfg, v))
        ax.set_ylabel(yl)
        for i in range(len(conds) - 1):
            ax.axvline(i + 0.5, color="#EEEEEE", lw=0.6, zorder=0)
    axes[1].set_xticks(range(len(conds)))
    axes[1].set_xticklabels([SHORT[c] for c in conds], rotation=30, ha="right")
    axes[0].legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig. 2 — Persistence through gaps vs false persistence", y=1.1, fontsize=9)
    files += _save(fig, pd_ / "fig2_persistence_fpt")
    # Fig 3: ablation + latency
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for j, (k, name) in enumerate(CONTRASTS[2:]):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            r = paired[(k, c, "J")]
            m, a, b = float(r["mean_diff"]), float(r["ci_lo"]), float(r["ci_hi"])
            xs.append(i - 0.2 + 0.2 * j)
            ys.append(m)
            lo.append(m - a)
            hi.append(b - m)
        axes[0].errorbar(xs, ys, yerr=[lo, hi], fmt=mk[2 + j], color=cols[2 + j], ms=4, lw=0.8, capsize=1.5, label=name)
    axes[0].axhline(0, color="black", lw=0.6)
    for g in (0.02, -0.02):
        axes[0].axhline(g, color="#888888", lw=0.5, ls="--")
    axes[0].set_xticks(range(len(conds)))
    axes[0].set_xticklabels([SHORT[c] for c in conds], rotation=30, ha="right", fontsize=6)
    axes[0].set_ylabel("ΔJ of K1 (95% CI)")
    axes[0].legend(frameon=False, fontsize=6)
    lat = an["latency_ms"]
    ms = cfg["methods_order"]
    axes[1].barh(range(len(ms)), [lat[m] for m in ms], color=[STYLE[m][0] for m in ms], edgecolor="black", lw=0.3)
    axes[1].set_yticks(range(len(ms)))
    axes[1].set_yticklabels([_lab(cfg, m) for m in ms], fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ms / frame")
    fig.suptitle("Fig. 3 — Attribution: online learning vs static prior, pooling, geometric durations; cost",
                 y=1.03, fontsize=9)
    fig.tight_layout()
    files += _save(fig, pd_ / "fig3_ablation_cost")
    # Fig 4: hold horizons (static vs online at video end), first 10 evaluation seeds
    files += _horizons(cfg, out)
    # Fig 5: decision matrix
    res = an["pairwise"]
    keys = [k for k, _ in CONTRASTS]
    M = np.array([[1 if res[k][c]["win"] else (-1 if res[k][c]["loss"] else 0) for k in keys] for c in conds], float)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#D55E00", "#F2F2F2", "#0072B2"]), vmin=-1, vmax=1, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(keys)):
            ax.text(j, i, {1: "W", -1: "L", 0: "·"}[int(M[i, j])], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in CONTRASTS], rotation=30, ha="right")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([SHORT[c] for c in conds])
    ax.set_title(f"Fig. 5 — K1 outcomes; decision {an['gate']['decision']}, contribution "
                 f"{an['gate']['algorithm_contribution']}", fontsize=8)
    fig.tight_layout()
    files += _save(fig, pd_ / "fig5_decision_matrix")
    return files


def _lstar(model, s: str) -> int:
    L = 0
    for l in range(1, model._H):
        if model.gap_posterior(s, l) >= model.tau:
            L = l
        else:
            break
    return L


def _horizons(cfg, out: Path) -> list[str]:
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    prior = load_prior(out)
    conds = conditions(cfg)
    rows = []
    for c, cond in conds.items():
        for s in seeds(cfg, "evaluation", False)[:10]:
            inst = build_instance(cond, s, cfg["world"], cfg["base_corruption"])
            for m in ("b2_hold_km", "b2_hold_static"):
                model = build_model(m, with_prior(m, frozen[m]["params"], cfg, prior))
                run_model(model, inst.input_frames, "v")
                for st in ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN"):
                    rows.append((c, m, st, _lstar(model, st)))
    (out / "aggregated").mkdir(parents=True, exist_ok=True)
    with open(out / "aggregated" / "hold_horizons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "method", "stratum", "L_star_bins"])
        w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    cl = list(conds)
    for ax, st in zip(axes, ("OCCLUDED_OR_UNRELIABLE", "UNKNOWN")):
        for j, m in enumerate(("b2_hold_static", "b2_hold_km")):
            vals = [[r[3] / 30.0 for r in rows if r[0] == c and r[1] == m and r[2] == st] for c in cl]
            pos = np.arange(len(cl)) + (-0.18 if j == 0 else 0.18)
            ax.boxplot(vals, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": STYLE[m][0], "alpha": 0.6}, medianprops={"color": "black"})
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels([SHORT[c] for c in cl], rotation=40, ha="right", fontsize=6)
        ax.set_title("OCC-onset runs" if st.startswith("OCC") else "UNKNOWN / no-record runs", fontsize=8)
    axes[0].set_ylabel("hold horizon L* (s) at video end")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc=STYLE["b2_hold_static"][0]), plt.Rectangle((0, 0), 1, 1, fc=STYLE["b2_hold_km"][0])],
                   ["static prior", "online (K1)"], frameon=False, fontsize=6)
    fig.suptitle("Fig. 4 — Does online learning move the hold horizon per regime? (10 evaluation seeds)", y=1.02, fontsize=9)
    fig.tight_layout()
    return _save(fig, out / "plots" / "fig4_hold_horizons")


def run_failures(cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """Predefined rule: K1 vs B2 per-seed J regret in PRIMARY conditions and C6."""
    from taorg.experiments.episodes.failures import _plot
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    prior = load_prior(out)
    ev = seeds(cfg, "evaluation", False)
    data = Data(read_csv_gz(out / "raw" / "evaluation.csv.gz"), ev)
    cr = cfg["criteria"]
    T, B = cr["treatment"], cr["baseline"]
    index = []
    cfg_l = {**cfg, "method_labels": cfg["method_labels"]}
    for c in cr["primary_conditions"] + ["C6_id_switch"]:
        reg = data.get(c, B, "J") - data.get(c, T, "J")
        pos = sorted((r, s) for r, s in zip(reg, ev) if np.isfinite(r) and r > 0)
        rec = {"condition": c, "n_positive_regret": len(pos), "mean_regret": float(np.nanmean(reg))}
        if not pos:
            rec["result"] = "No failure observed"
        else:
            rec["cases"] = {}
            for tag, (r, s) in {"worst": pos[-1], "median": pos[(len(pos) - 1) // 2]}.items():
                inst = build_instance(conditions(cfg)[c], s, cfg["world"], cfg["base_corruption"])
                outs = {}
                for m in (B, T):
                    pred = run_model(build_model(m, with_prior(m, frozen[m]["params"], cfg, prior)), inst.input_frames, "v")
                    outs[m] = [{o.edge_key for o in fr.edge_observations} for fr in pred]
                from taorg.experiments.episodes import plots as EP
                for m in (B, T):
                    EP.STYLE.setdefault(m, (STYLE[m][0], STYLE[m][1]))
                files = _plot(cfg_l, inst, outs, f"{c} seed {s}: J regret of K1 vs B2 = {r:.3f} ({tag})",
                              out / "failures" / f"{c}__{tag}")
                rec["cases"][tag] = {"seed": s, "regret_J": float(r), "files": files}
        index.append(rec)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    (out / "failures" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index
