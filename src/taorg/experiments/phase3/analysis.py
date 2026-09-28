"""Phase 3 analysis: descriptive tables, paired tests, frontiers, H1 decision.

Reads only saved artifacts (``raw/evaluation_runs.csv.gz``,
``configs/selected_params.json``, ``configs/pareto_sets.json``) and writes
``aggregated/`` and ``tables/``.  All numbers in the paper's Phase 3 tables
must come from these outputs.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taorg.experiments.phase3.conditions import derive_seed
from taorg.experiments.phase3.h1 import classify_cell, decide_h1, frontier_advantage
from taorg.experiments.phase3.pipeline import (
    condition_specs, finish_stage_metadata, new_stage_metadata,
)
from taorg.experiments.phase3.stats import (
    descriptive, holm_adjust, hypervolume_2d, paired_comparison,
)

logger = logging.getLogger(__name__)

PRIMARY_METRICS = ("persistence_f1", "term_delay_mean", "false_pers_mean", "churn_mean")
REPORT_METRICS = PRIMARY_METRICS + (
    "persistence_recall", "persistence_precision", "term_num_censored",
    "diag_clean_alive_recall", "diag_spurious_rate", "diag_ghost_edge_frames",
    "diag_recovery_lag", "diag_swap_recall", "method_runtime_s",
)
METHOD_ORDER = ("no_memory", "fixed_window", "ema", "gated_ema", "hmm")


def load_eval(out_dir: Path) -> pd.DataFrame:
    """Load evaluation rows as a DataFrame with numeric metric columns."""
    df = pd.read_csv(out_dir / "raw" / "evaluation_runs.csv.gz")
    for c in REPORT_METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class EvalIndex:
    """Fast lookup of per-seed metric vectors keyed by (condition, gap, config)."""

    def __init__(self, df: pd.DataFrame, seeds: list[int]) -> None:
        self.seeds = seeds
        self._groups = {
            k: g.set_index("seed").reindex(seeds)
            for k, g in df.groupby(["condition", "gap", "config_id"], sort=False)
        }

    def values(self, cond: str, gap: int, config_id: str, metric: str) -> np.ndarray:
        """Per-seed values (NaN where missing), aligned with ``self.seeds``."""
        g = self._groups.get((cond, gap, config_id))
        if g is None:
            raise KeyError(f"No evaluation rows for ({cond}, gap={gap}, {config_id})")
        return g[metric].to_numpy(dtype=float)


def selection_table(cfg: dict[str, Any], selected: dict[str, Any]) -> pd.DataFrame:
    """Flat table of selected operating points per (condition, method, budget)."""
    rows = []
    for cond, by_m in selected.items():
        for method, by_b in by_m.items():
            for b, sel in by_b.items():
                rows.append({"tuning_condition": cond, "method": method, "budget": int(b),
                             "config_id": sel["config_id"], "feasible": sel["feasible"],
                             "n_feasible": sel["n_feasible"],
                             "calib_persistence_f1": sel["ranking"][0]["persistence_f1"],
                             "calib_term_delay_mean": sel["ranking"][0]["term_delay_mean"],
                             "calib_churn_mean": sel["ranking"][0]["churn_mean"],
                             "calib_false_pers_mean": sel["ranking"][0]["false_pers_mean"],
                             "params_json": json.dumps(sel["params"], sort_keys=True)})
    return pd.DataFrame(rows)


def run_analysis(cfg: dict[str, Any], out_dir: Path, smoke: bool = False) -> dict[str, Any]:
    """Run the full pre-registered analysis and write aggregated artifacts."""
    meta = new_stage_metadata("analyze", cfg, smoke)
    agg_dir = out_dir / "aggregated"
    tab_dir = out_dir / "tables"
    agg_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval(out_dir)
    selected = json.loads((out_dir / "configs" / "selected_params.json").read_text(encoding="utf-8"))
    pareto = json.loads((out_dir / "configs" / "pareto_sets.json").read_text(encoding="utf-8"))
    specs = condition_specs(cfg)
    st = cfg["statistics"]
    crit = cfg["h1_criteria"]
    budgets = [str(b) for b in cfg["budgets_frames"]]
    seeds = sorted(df.seed.unique().tolist())
    ix = EvalIndex(df, seeds)
    primary = next(s.name for s in specs.values() if s.role == "h1_primary")
    robust = [s.name for s in specs.values() if s.role == "h1_robustness"]
    h1_conds = [primary, *robust]
    treat = crit["treatment"]

    sel_df = selection_table(cfg, selected)
    sel_df.to_csv(agg_dir / "selected_operating_points.csv", index=False)

    def rng_for(*key: Any) -> np.random.Generator:
        return np.random.default_rng(derive_seed(cfg["seeds"]["bootstrap"], *key))

    # ---------------- descriptive statistics of selected operating points
    desc_rows = []
    for spec in specs.values():
        sources = [(spec.name if spec.calibrate else spec.tuned_on, "matched" if spec.calibrate else "tuned_on")]
        if spec.role == "h1_robustness":
            sources.append((primary, "mismatched_C0"))
        for src, src_kind in sources:
            for method in METHOD_ORDER:
                if method not in selected[src]:
                    continue
                for b in budgets:
                    cid = selected[src][method][b]["config_id"]
                    for gap in spec.gaps:
                        row = {"condition": spec.name, "role": spec.role, "selection_source": src,
                               "selection_kind": src_kind, "method": method, "budget": int(b),
                               "gap": gap, "config_id": cid}
                        for m in REPORT_METRICS:
                            d = descriptive(ix.values(spec.name, gap, cid, m).tolist(),
                                            st["ci_level"])
                            for k, v in d.items():
                                row[f"{m}__{k}"] = v
                        desc_rows.append(row)
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(agg_dir / "descriptive_selected.csv", index=False)

    # ---------------- paired comparisons: treatment vs every other method
    pair_rows = []
    for spec in specs.values():
        src = spec.name if spec.calibrate else spec.tuned_on
        family = "h1" if spec.name in h1_conds else "exploratory"
        for b in budgets:
            g_cid = selected[src][treat][b]["config_id"]
            for comp in METHOD_ORDER:
                if comp == treat:
                    continue
                c_cid = selected[src][comp][b]["config_id"]
                for gap in spec.gaps:
                    for m in REPORT_METRICS:
                        a = ix.values(spec.name, gap, g_cid, m)
                        bb = ix.values(spec.name, gap, c_cid, m)
                        pc = paired_comparison(a, bb, st["bootstrap_resamples"], st["ci_level"],
                                               rng_for("pair", spec.name, b, comp, gap, m))
                        pair_rows.append({"condition": spec.name, "role": spec.role,
                                          "family": family, "budget": int(b), "gap": gap,
                                          "treatment": treat, "comparator": comp,
                                          "treatment_config": g_cid, "comparator_config": c_cid,
                                          "metric": m, **pc})
    pairs = pd.DataFrame(pair_rows)
    # Holm over the registered family: F1 tests, H1 conditions, unaware comparators.
    fam = ((pairs.family == "h1") & (pairs.metric == "persistence_f1")
           & (pairs.comparator.isin(crit["comparators"])))
    pairs["p_holm"] = np.nan
    pairs.loc[fam, "p_holm"] = holm_adjust(pairs.loc[fam, "p_wilcoxon"].tolist())
    pairs["holm_family_size"] = np.where(fam, int(fam.sum()), 0)
    pairs.to_csv(agg_dir / "paired_tests.csv", index=False)

    # ---------------- H1 cells
    cell_rows = []
    cell_outcomes: dict[str, list[str]] = {c: [] for c in h1_conds}
    for cond in h1_conds:
        for b in budgets:
            for gap in specs[cond].gaps:
                per_comp = {}
                for comp in crit["comparators"]:
                    sub = pairs[(pairs.condition == cond) & (pairs.budget == int(b))
                                & (pairs.gap == gap) & (pairs.comparator == comp)]
                    per_comp[comp] = {r.metric: r._asdict() for r in sub.itertuples(index=False)}
                res = classify_cell(per_comp, crit)
                cell_outcomes[cond].append(res["outcome"])
                row = {"condition": cond, "budget": int(b), "gap": gap, "outcome": res["outcome"]}
                for comp, d in res["detail"].items():
                    f1 = per_comp[comp]["persistence_f1"]
                    row[f"{comp}__win"] = d["win"]
                    row[f"{comp}__loss_type"] = d["loss_type"]
                    row[f"{comp}__failed_checks"] = ",".join(k for k, v in d["checks"].items() if not v)
                    row[f"{comp}__dF1"] = f1["mean_diff"]
                    row[f"{comp}__dF1_ci"] = f"[{f1['ci_lo']:.3f}, {f1['ci_hi']:.3f}]"
                    row[f"{comp}__dDelay"] = per_comp[comp]["term_delay_mean"]["mean_diff"]
                    row[f"{comp}__dFPR"] = per_comp[comp]["false_pers_mean"]["mean_diff"]
                    row[f"{comp}__dChurn"] = per_comp[comp]["churn_mean"]["mean_diff"]
                cell_rows.append(row)
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(agg_dir / "h1_cells.csv", index=False)

    # ---------------- frontiers (calibration Pareto sets evaluated on held-out seeds)
    hv_rows, front_rows = [], []
    hv_gaps_won: dict[str, int] = {c: 0 for c in h1_conds}
    R = st["hv_bootstrap_resamples"]
    for cond in h1_conds:
        for gap in specs[cond].gaps:
            mats = {}
            for method in METHOD_ORDER:
                cids = [it["config_id"] for it in pareto[cond][method]]
                F = np.vstack([ix.values(cond, gap, c, "persistence_f1") for c in cids])
                D = np.vstack([ix.values(cond, gap, c, "term_delay_mean") for c in cids])
                P = np.vstack([ix.values(cond, gap, c, "false_pers_mean") for c in cids])
                mats[method] = (cids, F, D, P)
                for i, c in enumerate(cids):
                    front_rows.append({"condition": cond, "gap": gap, "method": method,
                                       "config_id": c,
                                       "persistence_f1": np.nanmean(F[i]),
                                       "term_delay_mean": np.nanmean(D[i]),
                                       "false_pers_mean": np.nanmean(P[i])})
            rng = rng_for("hv", cond, gap)
            n = len(seeds)
            idx = rng.integers(0, n, size=(R, n))

            def hv_of(F: np.ndarray, D: np.ndarray, cols: np.ndarray | None,
                      ref: float, lo: float) -> float:
                f = np.nanmean(F if cols is None else F[:, cols], axis=1)
                d = np.nanmean(D if cols is None else D[:, cols], axis=1)
                return hypervolume_2d(list(zip(f, d)), lo, ref)

            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    point = {m: hv_of(v[1], v[2], None, st["hv_delay_ref"], st["hv_delay_min"])
                             for m, v in mats.items()}
                    point_fpr = {m: hv_of(v[1], v[3], None, st["hv_fpr_ref"], 0.0)
                                 for m, v in mats.items()}
                    boot = {m: np.array([hv_of(v[1], v[2], idx[r], st["hv_delay_ref"],
                                               st["hv_delay_min"]) for r in range(R)])
                            for m, v in mats.items()}
            lo_q, hi_q = 0.5 - st["ci_level"] / 2, 0.5 + st["ci_level"] / 2
            diff_vs = {}
            for m in METHOD_ORDER:
                row = {"condition": cond, "gap": gap, "method": m, "hv_delay": point[m],
                       "hv_delay_ci_lo": float(np.quantile(boot[m], lo_q)),
                       "hv_delay_ci_hi": float(np.quantile(boot[m], hi_q)),
                       "hv_fpr": point_fpr[m], "n_frontier_configs": len(mats[m][0])}
                if m != treat:
                    d = boot[treat] - boot[m]
                    diff_vs[m] = {"mean_diff": point[treat] - point[m],
                                  "ci_lo": float(np.quantile(d, lo_q)),
                                  "ci_hi": float(np.quantile(d, hi_q))}
                    row.update({"diff_treat_minus_this": diff_vs[m]["mean_diff"],
                                "diff_ci_lo": diff_vs[m]["ci_lo"], "diff_ci_hi": diff_vs[m]["ci_hi"]})
                hv_rows.append(row)
            won = frontier_advantage({"diff_vs": diff_vs}, crit)
            hv_gaps_won[cond] += int(won)
            hv_rows.append({"condition": cond, "gap": gap, "method": "__frontier_criterion__",
                            "hv_delay": float(won)})
    pd.DataFrame(hv_rows).to_csv(agg_dir / "hypervolume.csv", index=False)
    pd.DataFrame(front_rows).to_csv(agg_dir / "frontier_points.csv", index=False)

    decision = decide_h1(cell_outcomes, hv_gaps_won, primary, robust, crit)
    decision["criteria"] = crit
    decision["primary_condition"] = primary
    decision["robustness_conditions"] = robust
    decision["n_eval_seeds"] = len(seeds)
    decision["eval_seeds"] = seeds
    (agg_dir / "h1_decision.json").write_text(json.dumps(decision, indent=2, default=str),
                                              encoding="utf-8")
    write_tables(cfg, out_dir, desc, cells, pd.DataFrame(hv_rows), sel_df, decision)
    finish_stage_metadata(meta, out_dir, {"decision": decision["decision"]})
    return decision


# ---------------------------------------------------------------------------
# Markdown tables (generated, never hand-edited)
# ---------------------------------------------------------------------------

def _fmt(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "n/a"
    if math.isnan(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_tables(cfg: dict[str, Any], out_dir: Path, desc: pd.DataFrame, cells: pd.DataFrame,
                 hv: pd.DataFrame, sel: pd.DataFrame, decision: dict[str, Any]) -> None:
    """Write generated Markdown tables to ``tables/``."""
    tab = out_dir / "tables"
    labels = cfg["method_labels"]
    header = ("<!-- GENERATED by taorg.experiments.phase3.analysis from "
              "raw/evaluation_runs.csv.gz — do not edit by hand. -->\n\n")

    # Selected parameters
    lines = [header, "# Selected operating points (calibration seeds only)\n",
             "| Tuning condition | Method | Budget (frames) | Config | Feasible | Calib F1 | Calib delay | Params |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sel.sort_values(["tuning_condition", "method", "budget"]).itertuples():
        lines.append(f"| {r.tuning_condition} | {labels[r.method]} | {r.budget} | {r.config_id} | "
                     f"{r.feasible} | {r.calib_persistence_f1:.3f} | {r.calib_term_delay_mean:.2f} | "
                     f"`{r.params_json}` |")
    (tab / "selected_parameters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per condition x budget: metric by gap for each method (matched/tuned_on selection)
    metrics = [("persistence_f1", "Persistence F1", 3), ("term_delay_mean", "Termination delay (frames)", 2),
               ("false_pers_mean", "False persistence rate", 3), ("churn_mean", "Churn rate", 4)]
    main = desc[desc.selection_kind != "mismatched_C0"]
    for cond in main.condition.unique():
        sub_c = main[main.condition == cond]
        lines = [header, f"# {cond} — selected operating points, mean [95% CI], n = seeds\n"]
        for b in sorted(sub_c.budget.unique()):
            lines.append(f"\n## Budget {b} frames\n")
            gaps = sorted(sub_c.gap.unique())
            for key, title, dg in metrics:
                lines.append(f"\n### {title}\n")
                lines.append("| Method | " + " | ".join(f"gap {g}" for g in gaps) + " | n |")
                lines.append("|---|" + "---|" * (len(gaps) + 1))
                for m in METHOD_ORDER:
                    rr = sub_c[(sub_c.budget == b) & (sub_c.method == m)].set_index("gap")
                    if rr.empty:
                        continue
                    cells_txt = [_fmt(rr.loc[g, f"{key}__mean"], rr.loc[g, f"{key}__ci_lo"],
                                      rr.loc[g, f"{key}__ci_hi"], dg) for g in gaps]
                    n = int(rr[f"{key}__n"].min())
                    lines.append(f"| {labels[m]} | " + " | ".join(cells_txt) + f" | {n} |")
        (tab / f"results_{cond}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # H1 cells
    lines = [header, f"# H1 decision: **{decision['decision']}**\n",
             "| Condition | Wins | Losses | Cells | Frontier gaps won |", "|---|---|---|---|---|"]
    for c in decision["wins"]:
        lines.append(f"| {c} | {decision['wins'][c]} | {decision['losses'][c]} | "
                     f"{decision['n_cells'][c]} | {decision['hv_gaps_won'][c]} / 6 |")
    lines += ["", "## Criteria checks", "", "| Check | Result |", "|---|---|"]
    for k, v in {**{f"PASS: {k}": v for k, v in decision["pass_checks"].items()},
                 **{f"FAIL: {k}": v for k, v in decision["fail_checks"].items()}}.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Cell outcomes (dF1 = Vis-Gated EMA minus comparator)", "",
              "| Condition | Budget | Gap | Outcome | dF1 vs EMA | dF1 vs Fixed Window | dF1 vs HMM | dDelay vs EMA | failed checks vs EMA |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in cells.itertuples():
        d = r._asdict()
        lines.append(f"| {r.condition} | {r.budget} | {r.gap} | {r.outcome} | "
                     f"{d['ema__dF1']:.3f} {d['ema__dF1_ci']} | "
                     f"{d['fixed_window__dF1']:.3f} {d['fixed_window__dF1_ci']} | "
                     f"{d['hmm__dF1']:.3f} {d['hmm__dF1_ci']} | {d['ema__dDelay']:.2f} | "
                     f"{d['ema__failed_checks']} |")
    (tab / "h1_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Hypervolume
    hvm = hv[hv.method != "__frontier_criterion__"]
    lines = [header, "# Frontier hypervolume (F1 vs termination delay), eval seeds, 95% bootstrap CI\n"]
    for cond in hvm.condition.unique():
        sub = hvm[hvm.condition == cond]
        gaps = sorted(sub.gap.unique())
        lines += [f"\n## {cond}\n", "| Method | " + " | ".join(f"gap {g}" for g in gaps) + " |",
                  "|---|" + "---|" * len(gaps)]
        for m in METHOD_ORDER:
            rr = sub[sub.method == m].set_index("gap")
            lines.append(f"| {labels[m]} | " + " | ".join(
                _fmt(rr.loc[g, "hv_delay"], rr.loc[g, "hv_delay_ci_lo"], rr.loc[g, "hv_delay_ci_hi"])
                for g in gaps) + " |")
    (tab / "hypervolume.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
