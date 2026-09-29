"""Phase 4 figures, generated only from saved aggregated artifacts.

Style follows Design.md and the Phase 3 figures (Okabe-Ito palette + markers
+ line styles, fixed method order, units on axes, 95% CIs, PDF + PNG).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taorg.experiments.phase3.plots import _setup  # noqa: E402
from taorg.experiments.phase3.pipeline import read_csv_gz  # noqa: E402

logger = logging.getLogger(__name__)
KEY = ("fixed_window", "hmm", "gated_ema", "va_hmm", "vhped_m2", "vhped_m3", "vhped_m4", "vhped_m5", "vhped_m6")
STYLE = {
    "no_memory": dict(color="#7F7F7F", marker="o", linestyle=":"),
    "fixed_window": dict(color="#E69F00", marker="s", linestyle="-."),
    "ema": dict(color="#56B4E9", marker="^", linestyle="--"),
    "hmm": dict(color="#009E73", marker="v", linestyle=(0, (3, 1, 1, 1))),
    "gated_ema": dict(color="#0072B2", marker="D", linestyle="-"),
    "va_hmm": dict(color="#CC79A7", marker="P", linestyle="-"),
    "vhped_m2": dict(color="#F0E442", marker="x", linestyle="--"),
    "vhped_m3": dict(color="#D55E00", marker="*", linestyle=":"),
    "vhped_m4": dict(color="#999933", marker="h", linestyle="-."),
    "vhped_m5": dict(color="#000000", marker="o", linestyle="-"),
    "vhped_m6": dict(color="#882255", marker="d", linestyle="--"),
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def run_plots(cfg: dict[str, Any], out: Path) -> None:
    """Write Phase 4 figures to results/phase4/plots/."""
    _setup()
    agg = out / "aggregated"
    desc = pd.read_csv(agg / "descriptive.csv")
    lab = cfg["method_labels"]
    P = out / "plots"
    B = int(cfg["primary_budget_frames"])
    d5 = desc[desc.budget == B]
    fails = [c for c, d in cfg["conditions"].items() if d["role"] in ("h2_failure", "h2_noninferiority")]

    # Fig 1: F1 / delay / churn vs gap at the primary budget, all H2 conditions.
    for metric, ylab, fname in (("persistence_f1", "Edge Persistence F1", "fig1_f1_vs_gap"),
                                ("term_delay_mean", "Termination delay (frames)", "fig2_delay_vs_gap"),
                                ("churn_mean", "Relationship churn rate", "fig3_churn_vs_gap")):
        fig, axes = plt.subplots(2, 5, figsize=(11, 4.2), sharex=True)
        for ax, cond in zip(axes.flat, fails):
            dc = d5[d5.condition == cond]
            for m in KEY:
                r = dc[dc.method == m].sort_values("gap")
                if r.empty:
                    continue
                ax.errorbar(r.gap, r[f"{metric}__mean"], yerr=[r[f"{metric}__mean"] - r[f"{metric}__ci_lo"],
                                                                r[f"{metric}__ci_hi"] - r[f"{metric}__mean"]],
                            capsize=2, markersize=3.5, linewidth=1.1, label=lab[m], **STYLE[m])
            ax.set_title(cond, fontsize=7.5)
            ax.set_xscale("log")
            ax.set_xticks([3, 10, 20])
            ax.set_xticklabels(["3", "10", "20"])
        for ax in axes.flat[len(fails):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab)
        for ax in axes[1]:
            ax.set_xlabel("Occlusion gap (frames, 30 fps)")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.text(0.99, 1.0, f"budget = {B} frames; mean ± 95% CI over 30 videos", ha="right", fontsize=6.5)
        fig.tight_layout()
        _save(fig, P / fname)

    # Fig 4: frontiers at gap 10 (evaluation means of calibration Pareto sets).
    ev = []
    for g in ("baseline", "proposed"):
        p = out / "raw" / f"evaluation_{g}.csv.gz"
        if p.exists():
            ev.append(pd.DataFrame(read_csv_gz(p)))
    ev = pd.concat(ev)
    for c in ("persistence_f1", "term_delay_mean", "diag_swap_recall", "diag_ghost_edge_frames", "gap"):
        ev[c] = ev[c].astype(float)
    par: dict[str, Any] = {}
    for g in ("baseline", "proposed"):
        p = out / "configs" / f"pareto_{g}.json"
        if p.exists():
            for cond, v in json.loads(p.read_text(encoding="utf-8")).items():
                for fk, by in v.items():
                    par.setdefault(cond, {}).setdefault(fk, {}).update(by)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
    for ax, cond in zip(axes.flat, fails):
        fk = "identity" if cond in cfg["identity_conditions"] else "default"
        f = cfg["frontiers"][fk]
        e = ev[(ev.condition == cond) & (ev.gap == 10)]
        means = e.groupby("config_id")[[f["quality"], f["cost"]]].mean()
        for m in ("gated_ema", "va_hmm", "vhped_m5", "hmm", "fixed_window"):
            ids = [i for i in par.get(cond, {}).get(fk, {}).get(m, []) if i in means.index]
            if not ids:
                continue
            pts = means.loc[ids].sort_values(f["cost"])
            ax.step(pts[f["cost"]], pts[f["quality"]].cummax(), where="post", label=lab[m],
                    color=STYLE[m]["color"], linestyle=STYLE[m]["linestyle"], linewidth=1.1)
        ax.set_title(f"{cond} (gap 10)", fontsize=7.5)
        ax.set_xlim(f["cost_min"], f["cost_ref"])
        if fk == "default":
            ax.set_xscale("log")
            ax.set_xlabel("Termination delay (frames)")
            ax.set_ylabel("Persistence F1")
        else:
            ax.set_xlabel("Ghost edge-frames / video")
            ax.set_ylabel("Swap-interval recall")
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(fails):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.99, 1.0, "calibration Pareto sets evaluated on held-out videos", ha="right", fontsize=6.5)
    fig.tight_layout()
    _save(fig, P / "fig4_frontiers_gap10")

    # Fig 5: H4 shift, primary-budget points tuned on SC only.
    shifts = ["SC_low_noise"] + [c for c, d in cfg["conditions"].items() if d["role"] == "h4_shift"]
    dsh = d5[d5.condition.isin(shifts)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, (metric, ylab) in zip(axes, (("persistence_f1", "Edge Persistence F1 (mean over gaps)"),
                                         ("delay_s", "Termination delay (s)"),
                                         ("churn_mean", "Churn rate"))):
        x = np.arange(len(shifts))
        for j, m in enumerate(("gated_ema", "va_hmm", "vhped_m5")):
            vals = [dsh[(dsh.condition == c) & (dsh.method == m)][f"{metric}__mean"].mean() for c in shifts]
            ax.bar(x + (j - 1) * 0.27, vals, width=0.27, label=lab[m], color=STYLE[m]["color"])
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("_")[0] for s in shifts], rotation=0)
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False)
    fig.text(0.99, 1.0, "H4: all methods tuned once on SC (low noise); never retuned", ha="right", fontsize=6.5)
    fig.tight_layout()
    _save(fig, P / "fig5_h4_shift")

    # Fig 6: ablation ΔHV per step on its target conditions.
    dif = pd.read_csv(agg / "hv_differences.csv")
    ab = dif[dif.comparison.str.match(r"^A\d$")]
    if not ab.empty:
        fig, ax = plt.subplots(figsize=(11, 3.2))
        tags = sorted(ab.comparison.unique())
        conds = [c for c in cfg["conditions"] if cfg["conditions"][c].get("calibrate")]
        w = 0.8 / max(1, len(tags))
        x = np.arange(len(conds))
        for j, t in enumerate(tags):
            vals = [ab[(ab.comparison == t) & (ab.condition == c)].mean_diff.mean() for c in conds]
            ax.bar(x + j * w - 0.4 + w / 2, vals, width=w, label=f"{t}")
        ax.axhline(0.02, color="k", linewidth=0.6, linestyle=":")
        ax.axhline(-0.02, color="k", linewidth=0.6, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([c.split("_")[0] for c in conds])
        ax.set_ylabel("ΔHV (mean over gaps)")
        ax.legend(frameon=False, ncol=6, fontsize=6.5)
        fig.text(0.99, 1.0, "A0 M1−M0, A1 M2−M1, A2 M3−M2, A3 M4−M3, A4 M5−M4, A5 M6−M5; dotted: ±0.02 margin",
                 ha="right", fontsize=6.5)
        fig.tight_layout()
        _save(fig, P / "fig6_ablation")
    logger.info("Phase 4 plots written to %s", P)
