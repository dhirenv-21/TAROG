"""Registered episode-study figures (prereg §9), generated only from saved artifacts.

Colour-blind-safe Okabe-Ito palette; every method also has its own marker, so colour is
never the only encoding (Design.md).  Mean ± 95% t-interval over the 50 evaluation seeds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.episodes.analysis import Data  # noqa: E402
from taorg.experiments.episodes.pipeline import read_csv_gz, seeds  # noqa: E402
from taorg.experiments.phase3.stats import descriptive  # noqa: E402

STYLE = {
    "gated_ema": ("#000000", "o"), "gated_ema_hyst": ("#E69F00", "s"), "va_hmm": ("#56B4E9", "^"),
    "hsmm": ("#009E73", "v"), "gated_ema_id": ("#F0E442", "D"), "episode": ("#0072B2", "P"),
    "episode_id": ("#D55E00", "X"), "episode_ctx": ("#CC79A7", "*"), "episode_hist": ("#999999", "h"),
    "no_memory": ("#BBBBBB", "."),
}
SHOW = ["gated_ema", "gated_ema_hyst", "va_hmm", "hsmm", "gated_ema_id", "episode", "episode_id", "episode_ctx",
        "episode_hist"]
SHORT = {"C1_gaps": "C1 gaps", "C2_hidden_termination": "C2 hidden term.", "C3_misleading_post": "C3 misleading",
         "C4_recurrence": "C4 recurrence", "C5_boundary_flicker": "C5 bnd. flicker", "C6_id_switch": "C6 ID swap",
         "C7_motion": "C7 motion", "S0_mixed": "S0 mixed", "S1_mixed_flagnoise": "S1 flag noise",
         "S2_mixed_unknown": "S2 UNKNOWN", "S3_mixed_bursty": "S3 bursty"}


def _setup() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300})


def _save(fig, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(path.with_suffix(".pdf")), str(path.with_suffix(".png"))]


def _dots(ax, data: Data, conds: list[str], metric: str, labels: dict, methods=SHOW, ylabel: str = "") -> None:
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        col, mk = STYLE[m]
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            d = descriptive(list(data.get(c, m, metric)))
            if d["n"] == 0:
                continue
            xs.append(i - 0.4 + w * (j + 0.5))
            ys.append(d["mean"])
            lo.append(d["mean"] - d["ci_lo"] if not math.isnan(d["ci_lo"]) else 0)
            hi.append(d["ci_hi"] - d["mean"] if not math.isnan(d["ci_hi"]) else 0)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5,
                    mec="black" if m == "gated_ema_id" else col, mew=0.4, label=labels[m])
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([SHORT.get(c, c) for c in conds], rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric)
    for i in range(len(conds) - 1):
        ax.axvline(i + 0.5, color="#EEEEEE", lw=0.6, zorder=0)


def run_figures(cfg: dict[str, Any], out: Path) -> list[str]:
    """All registered figures."""
    _setup()
    rows = read_csv_gz(out / "raw" / "evaluation.csv.gz")
    data = Data(rows, seeds(cfg, "evaluation", False))
    an = json.loads((out / "aggregated" / "analysis.json").read_text(encoding="utf-8"))
    lab = cfg["method_labels"]
    conds = list(cfg["conditions"])
    pdir = out / "plots"
    files: list[str] = []

    # 1 persistence
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    _dots(axes[0], data, conds, "gap_recall", lab, ylabel="gap recall (frames kept\nthrough temporary gaps)")
    _dots(axes[1], data, conds, "J", lab, ylabel="J = ½ frame-F1 + ½ episode-F1")
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Fig. 1 — Persistence through temporary gaps and overall relationship quality", y=1.08, fontsize=9)
    files += _save(fig, pdir / "fig1_persistence")

    # 2 false reactivation
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    _dots(axes[0], data, conds, "frr", lab, ylabel="false-reactivation rate\n(per evaluable termination)")
    _dots(axes[1], data, conds, "fpt_s", lab, ylabel="false-persistence time\n(s per 12-s video)")
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Fig. 2 — False reactivation and false persistence", y=1.08, fontsize=9)
    files += _save(fig, pdir / "fig2_false_reactivation")

    # 3 recurrence / onset / termination delay
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    _dots(axes[0], data, conds, "rdd_mean_s", lab, ylabel="recurrence-detection\ndelay (s)")
    _dots(axes[1], data, conds, "bridge_rate", lab, ylabel="bridged recurrences\n(merged episodes)")
    _dots(axes[2], data, conds, "std_mean_s", lab, ylabel="sustained termination\ndelay (s)")
    axes[0].legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Fig. 3 — Recurrence detection, episode merging and termination", y=1.06, fontsize=9)
    files += _save(fig, pdir / "fig3_recurrence_delay")

    # 4 identity
    idc = ["C6_id_switch", "C7_motion", "S0_mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    _dots(axes[0], data, ["C6_id_switch"], "ghost_time_s", lab, ylabel="ghost output (s per video)")
    _dots(axes[1], data, ["C6_id_switch"], "swap_recall", lab, ylabel="swap recall (true keys)")
    _dots(axes[2], data, idc, "J", lab, ylabel="J")
    axes[0].legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.05), frameon=False)
    fig.suptitle("Fig. 4 — Identity-switch ghosts and identity false alarms (C7 = fast motion, no swaps)",
                 y=1.16, fontsize=9)
    fig.tight_layout()
    files += _save(fig, pdir / "fig4_identity_ghosts")

    # 5 ablation costs: ladder ΔJ + latency
    paired = {}
    import csv
    with open(out / "aggregated" / "paired.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            paired[(r["condition"], r["treatment"], r["comparator"], r["metric"])] = r
    cal = cfg["criteria"]["benefit_conditions"] + cfg["criteria"]["ni_conditions"]
    cal = [c for c in conds if c in cal]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
    steps = [("episode_id", "episode", "P2−P1 identity"), ("episode_ctx", "episode_id", "P3−P2 context"),
             ("episode_hist", "episode_ctx", "P4−P3 history")]
    for j, (x, y, name) in enumerate(steps):
        col, mk = STYLE[x]
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(cal):
            r = paired.get((c, x, y, "J"))
            if r is None:
                continue
            m, a, b = float(r["mean_diff"]), float(r["ci_lo"]), float(r["ci_hi"])
            xs.append(i - 0.25 + 0.25 * j)
            ys.append(m)
            lo.append(m - a)
            hi.append(b - m)
        axes[0].errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5, label=name)
    axes[0].axhline(0, color="black", lw=0.6)
    for g in (0.02, -0.02):
        axes[0].axhline(g, color="#888888", lw=0.5, ls="--")
    axes[0].set_xticks(range(len(cal)))
    axes[0].set_xticklabels([SHORT[c] for c in cal], rotation=30, ha="right")
    axes[0].set_ylabel("ΔJ vs rung below (95% CI)")
    axes[0].legend(frameon=False, loc="best")
    lat = an["latency_ms"]
    ms = cfg["methods_order"]
    axes[1].barh(range(len(ms)), [lat[m] for m in ms], color=[STYLE[m][0] for m in ms], edgecolor="black", lw=0.3)
    axes[1].set_yticks(range(len(ms)))
    axes[1].set_yticklabels([lab[m] for m in ms])
    axes[1].set_xlabel("ms / frame (one machine)")
    axes[1].invert_yaxis()
    fig.suptitle("Fig. 5 — Ablation: what each added mechanism buys (left, dashed = ±0.02 practical margin) "
                 "and costs (right)", y=1.03, fontsize=9)
    fig.tight_layout()
    files += _save(fig, pdir / "fig5_ablation_costs")

    # 6 decision matrix
    comps = cfg["criteria"]["comparators"] + [cfg["criteria"]["identity_comparator"]]
    res = an["pairwise_results"]
    M = np.full((len(conds), len(comps)), np.nan)
    for i, c in enumerate(conds):
        for j, y in enumerate(comps):
            r = res.get(c, {}).get(y)
            if r is not None:
                M[i, j] = 1 if r["win"] else (-1 if r["loss"] else 0)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    cmap = matplotlib.colors.ListedColormap(["#D55E00", "#F2F2F2", "#0072B2"])
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(comps)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, {1: "W", -1: "L", 0: "·"}[int(M[i, j])], ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#999999")
        ax.text(len(comps) - 0.35, i, an["cells"][conds[i]], va="center", fontsize=7)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([lab[y] for y in comps], rotation=30, ha="right")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([SHORT[c] for c in conds])
    ax.set_title(f"Fig. 6 — {lab[an['T']]} vs comparators (W/L/·); decision: "
                 f"{an['study_decision']['decision']}", fontsize=8.5)
    fig.tight_layout()
    files += _save(fig, pdir / "fig6_decision_matrix")
    return files
