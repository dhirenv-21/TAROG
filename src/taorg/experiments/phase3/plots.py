"""Phase 3 figures, generated exclusively from saved aggregated artifacts.

Style follows Design.md: colorblind-safe (Okabe-Ito) palette combined with
distinct markers and line styles, fixed method order, paper method names,
axis labels with units, 95% CIs over evaluation seeds, PDF + PNG export.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

METHOD_ORDER = ("no_memory", "fixed_window", "ema", "gated_ema", "hmm")
STYLE = {
    "no_memory": dict(color="#7F7F7F", marker="o", linestyle=":"),
    "fixed_window": dict(color="#E69F00", marker="s", linestyle="-."),
    "ema": dict(color="#56B4E9", marker="^", linestyle="--"),
    "gated_ema": dict(color="#0072B2", marker="D", linestyle="-"),
    "hmm": dict(color="#009E73", marker="v", linestyle=(0, (3, 1, 1, 1))),
}
METRIC_LABEL = {
    "persistence_f1": "Edge Persistence F1",
    "false_pers_mean": "False persistence rate",
    "term_delay_mean": "Termination delay (frames)",
    "churn_mean": "Relationship churn rate",
    "diag_spurious_rate": "Spurious edge-frame rate (diag.)",
    "diag_clean_alive_recall": "Clean-period recall (diag.)",
}
GAP_LABEL = "Occlusion gap length (frames, 30 fps)"
STRESS_SHORT = {
    "S1_termination_under_occlusion": "S1 term.+occ.",
    "S2_id_switch_d5": "S2 ID-sw 5f",
    "S2_id_switch_d20": "S2 ID-sw 20f",
    "S3_repeated_occlusion": "S3 repeated",
    "S4_unknown_all": "S4 unk. 100%",
    "S4_unknown_half": "S4 unk. 50%",
    "S5_long_absence": "S5 long gap",
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def _save(fig: plt.Figure, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(path.with_suffix(".pdf")), str(path.with_suffix(".png"))]


def _gap_axis(ax: plt.Axes, gaps: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(gaps)
    ax.set_xticklabels([str(g) for g in gaps])
    ax.minorticks_off()
    ax.set_xlabel(GAP_LABEL)


def metric_vs_gap(desc: pd.DataFrame, labels: dict[str, str], metric: str,
                  conditions: list[str], budget: int, title_of: dict[str, str]) -> plt.Figure:
    """One panel per condition: metric mean ± 95% CI vs gap, one curve per method."""
    fig, axes = plt.subplots(1, len(conditions), figsize=(2.45 * len(conditions), 2.2),
                             sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        sub = desc[(desc.condition == cond) & (desc.budget == budget)
                   & (desc.selection_kind != "mismatched_C0")]
        gaps = sorted(sub.gap.unique())
        for i, m in enumerate(METHOD_ORDER):
            r = sub[sub.method == m].sort_values("gap")
            if r.empty:
                continue
            x = r.gap.to_numpy() * (1 + 0.03 * (i - 2))  # small dodge on log axis
            y = r[f"{metric}__mean"].to_numpy()
            lo = y - r[f"{metric}__ci_lo"].to_numpy()
            hi = r[f"{metric}__ci_hi"].to_numpy() - y
            ax.errorbar(x, y, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], label=labels[m],
                        markersize=3.5, linewidth=1.1, capsize=1.5, **STYLE[m])
        _gap_axis(ax, gaps)
        ax.set_title(title_of.get(cond, cond))
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel(METRIC_LABEL.get(metric, metric))
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.text(0.99, 0.99, f"budget = {budget} frames; mean ± 95% CI over seeds",
             ha="right", va="top", fontsize=6, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def pareto_figure(front: pd.DataFrame, desc: pd.DataFrame, labels: dict[str, str],
                  conditions: list[str], gap: int, budgets: list[int],
                  title_of: dict[str, str]) -> plt.Figure:
    """Rows: F1 vs delay, F1 vs FPR.  Cols: conditions.  Lines: eval frontiers."""
    fig, axes = plt.subplots(2, len(conditions), figsize=(2.45 * len(conditions), 4.3),
                             squeeze=False, sharey=True)
    for j, cond in enumerate(conditions):
        for i, (xkey, xlabel) in enumerate([("term_delay_mean", "Termination delay (frames)"),
                                            ("false_pers_mean", "False persistence rate")]):
            ax = axes[i][j]
            for m in METHOD_ORDER:
                f = front[(front.condition == cond) & (front.gap == gap) & (front.method == m)]
                f = f.sort_values(xkey)
                if f.empty:
                    continue
                st = STYLE[m]
                # Empirical frontier: running max of F1 as cost increases.
                xs, ys = f[xkey].to_numpy(), np.maximum.accumulate(f.persistence_f1.to_numpy())
                ax.step(xs, ys, where="post", color=st["color"], linestyle=st["linestyle"],
                        linewidth=1.0, label=labels[m])
                sel = desc[(desc.condition == cond) & (desc.gap == gap) & (desc.method == m)
                           & (desc.selection_kind == "matched")]
                for b in budgets:
                    r = sel[sel.budget == b]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.errorbar(r[f"{xkey}__mean"], r["persistence_f1__mean"],
                                xerr=[[r[f"{xkey}__mean"] - r[f"{xkey}__ci_lo"]],
                                      [r[f"{xkey}__ci_hi"] - r[f"{xkey}__mean"]]],
                                yerr=[[r["persistence_f1__mean"] - r["persistence_f1__ci_lo"]],
                                      [r["persistence_f1__ci_hi"] - r["persistence_f1__mean"]]],
                                color=st["color"], marker=st["marker"], markersize=3.5,
                                linewidth=0.7, capsize=1.2)
            if xkey == "term_delay_mean":
                ax.set_xscale("log")
                for b in budgets:
                    ax.axvline(b, color="#BBBBBB", linewidth=0.6, linestyle=":", zorder=0)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25, linewidth=0.5)
            if i == 0:
                ax.set_title(f"{title_of.get(cond, cond)} — gap {gap}")
        axes[0][0].set_ylabel("Edge Persistence F1")
        axes[1][0].set_ylabel("Edge Persistence F1")
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.99, 0.995, "lines: calibration-Pareto configs evaluated on held-out seeds; "
             "markers: budget-selected points (±95% CI); dotted: budgets",
             ha="right", va="top", fontsize=5.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def sweep_figure(desc: pd.DataFrame, labels: dict[str, str], cfg: dict[str, Any],
                 gap: int, budget: int) -> plt.Figure:
    """F1 / delay / churn vs noise level for the visibility and flicker sweeps."""
    conds = cfg["conditions"]
    primary = next(n for n, c in conds.items() if c["role"] == "h1_primary")
    axes_def = {
        "visibility_noise": [(primary, 0.0)] + sorted(
            [(n, c.get("sweep_value", (c.get("visibility_noise") or {}).get("occluded_to_visible_prob")))
             for n, c in conds.items()
             if c.get("sweep_axis") == "visibility_noise"
             or (c["role"] == "h1_robustness" and c.get("visibility_noise"))],
            key=lambda t: t[1]),
        "flicker": [(primary, 0.0)] + sorted(
            [(n, c.get("sweep_value", c.get("flicker_miss_prob")))
             for n, c in conds.items()
             if c.get("sweep_axis") == "flicker"
             or (c["role"] == "h1_robustness" and c.get("flicker_miss_prob"))],
            key=lambda t: t[1]),
    }
    metrics = ["persistence_f1", "term_delay_mean", "churn_mean"]
    fig, axes = plt.subplots(len(axes_def), len(metrics), figsize=(7.3, 4.2), squeeze=False)
    for i, (axis_name, pts) in enumerate(axes_def.items()):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            for m in METHOD_ORDER:
                xs, ys, lo, hi = [], [], [], []
                for cond, lvl in pts:
                    r = desc[(desc.condition == cond) & (desc.gap == gap) & (desc.budget == budget)
                             & (desc.method == m) & (desc.selection_kind == "matched")]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    xs.append(lvl)
                    ys.append(r[f"{metric}__mean"])
                    lo.append(r[f"{metric}__mean"] - r[f"{metric}__ci_lo"])
                    hi.append(r[f"{metric}__ci_hi"] - r[f"{metric}__mean"])
                ax.errorbar(xs, ys, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], label=labels[m],
                            markersize=3, linewidth=1.0, capsize=1.2, **STYLE[m])
            ax.set_xlabel("Visibility-flag error rate ρ" if axis_name == "visibility_noise"
                          else "Flicker (visible miss) rate q")
            ax.set_ylabel(METRIC_LABEL[metric])
            ax.grid(alpha=0.25, linewidth=0.5)
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.99, 0.995, f"gap = {gap} frames, budget = {budget} frames, condition-matched tuning; "
             "mean ± 95% CI", ha="right", va="top", fontsize=6, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def stress_figure(desc: pd.DataFrame, labels: dict[str, str], stress: list[str],
                  budget: int) -> plt.Figure:
    """Grouped bars (mean over the condition's gaps, per-seed CI) for stress conditions."""
    metrics = ["persistence_f1", "term_delay_mean", "false_pers_mean", "churn_mean",
               "diag_spurious_rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.3, 2.6), squeeze=False)
    width = 0.16
    for j, metric in enumerate(metrics):
        ax = axes[0][j]
        for k, m in enumerate(METHOD_ORDER):
            vals, errs = [], []
            for cond in stress:
                r = desc[(desc.condition == cond) & (desc.budget == budget) & (desc.method == m)]
                v = r[f"{metric}__mean"].to_numpy(dtype=float)
                vals.append(np.nanmean(v) if v.size and not np.all(np.isnan(v)) else np.nan)
                sd = r[f"{metric}__sd"].to_numpy(dtype=float)
                n = r[f"{metric}__n"].to_numpy(dtype=float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    se = np.sqrt(np.nanmean(sd ** 2) / np.nanmean(n)) if sd.size else np.nan
                errs.append(1.96 * se)
            x = np.arange(len(stress)) + (k - 2) * width
            ax.bar(x, vals, width, yerr=np.nan_to_num(errs), color=STYLE[m]["color"],
                   edgecolor="black", linewidth=0.3, label=labels[m],
                   hatch={"no_memory": "", "fixed_window": "//", "ema": "..",
                          "gated_ema": "", "hmm": "xx"}[m], error_kw=dict(lw=0.5, capsize=1))
        ax.set_xticks(np.arange(len(stress)))
        ax.set_xticklabels([STRESS_SHORT.get(s, s[:6]) for s in stress], rotation=90)
        ax.set_title(METRIC_LABEL.get(metric, metric), fontsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.99, 0.995, f"stress conditions, C0-tuned operating points, budget = {budget} frames; "
             "mean over the condition's gaps", ha="right", va="top", fontsize=6, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def run_plots(cfg: dict[str, Any], out_dir: Path) -> list[str]:
    """Generate all Phase 3 figures from ``aggregated/``."""
    _setup()
    agg, pdir = out_dir / "aggregated", out_dir / "plots"
    desc = pd.read_csv(agg / "descriptive_selected.csv")
    front = pd.read_csv(agg / "frontier_points.csv")
    labels = cfg["method_labels"]
    conds = cfg["conditions"]
    primary = next(n for n, c in conds.items() if c["role"] == "h1_primary")
    robust = [n for n, c in conds.items() if c["role"] == "h1_robustness"]
    stress = [n for n, c in conds.items() if c["role"] == "stress"]
    h1 = [primary, *robust]
    title_of = {primary: "C0: oracle visibility",
                "R1_visnoise_0.2": "R1: visibility-flag noise ρ=0.2",
                "R2_flicker_0.1": "R2: base-model flicker q=0.1"}
    B = int(cfg["primary_budget_frames"])
    written: list[str] = []
    for letter, metric in [("A", "persistence_f1"), ("B", "false_pers_mean"),
                           ("C", "term_delay_mean"), ("D", "churn_mean")]:
        written += _save(metric_vs_gap(desc, labels, metric, h1, B, title_of),
                         pdir / f"fig{letter}_{metric}_vs_gap")
        for b in cfg["budgets_frames"]:
            written += _save(metric_vs_gap(desc, labels, metric, h1, int(b), title_of),
                             pdir / "supplementary" / f"fig{letter}_{metric}_vs_gap_budget{b}")
    rep_gap = int(cfg["representative_gap_frames"])
    for gap in cfg["gap_lengths"]:
        fig = pareto_figure(front, desc, labels, h1, int(gap), [int(b) for b in cfg["budgets_frames"]],
                            title_of)
        name = f"figE_pareto_gap{gap}"
        written += _save(fig, (pdir if gap == rep_gap else pdir / "supplementary") / name)
    written += _save(sweep_figure(desc, labels, cfg, gap=rep_gap, budget=B), pdir / "figF_noise_sweeps")
    written += _save(stress_figure(desc, labels, stress, B), pdir / "figG_stress_conditions")
    (pdir / "figure_index.json").write_text(json.dumps(sorted(written), indent=1), encoding="utf-8")
    return written
