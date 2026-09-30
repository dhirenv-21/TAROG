"""Phase 4 EXPLORATORY, POST-HOC diagnostics (defined after seeing results; never decisive).

Motivation (recorded in research/phase4_research_log.md): the joint (Z, V)
filter can dip below threshold at occlusion onset and then climb back
("dip-and-return"). The frozen termination-delay metric records only the
FIRST post-termination drop, so the registered default frontier can reward
such behaviour; false persistence does not.

Outputs (results/phase4/aggregated/exploratory_*.csv, tables/exploratory_diagnostics.md):
  1. relapse diagnostic at the budget-5 operating points (all calibrated conditions, gap 10):
     fraction of terminating edges that are output again after their first
     post-termination drop, and mean relapse frames per terminating edge;
  2. an F1-vs-false-persistence frontier (box FPR in [0, 1], as the Phase 3
     secondary frontier) for the H2 conditions: dHV(M5 - M0) and dHV(M5 - M1)
     with paired bootstrap CIs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from taorg.experiments.phase3.conditions import derive_seed
from taorg.experiments.phase3.evaluation import build_model
from taorg.experiments.phase3.pipeline import read_csv_gz
from taorg.experiments.phase3.stats import hypervolume_2d
from taorg.experiments.phase4 import pipeline as P

logging.disable(logging.WARNING)
OUT = Path("results/phase4")
METHODS = ("gated_ema", "va_hmm", "vhped_m2", "vhped_m3", "vhped_m5")


def relapse(cfg, sp, sel, ctx, cond: str, method: str, gap: int = 10) -> dict[str, float]:
    B = str(cfg["primary_budget_frames"])
    params = P.model_params(method, sel[cond][method][B]["params"], ctx[cond])
    n_term = n_rel = rel_frames = 0
    for seed in cfg["seeds"]["evaluation"]:
        inst = P._build(P._slim_cfg(cfg), sp[cond], gap, seed)
        m = build_model(method, params)
        m.reset_video("v")
        outs = [{e.edge_key for e in m.step(f).edge_observations} for f in inst.input_frames]
        for lt in inst.lifetimes:
            if not lt.terminates:
                continue
            n_term += 1
            post = [lt.edge_key in outs[t] for t in range(lt.end_frame + 1, len(outs))]
            if False not in post:
                continue
            first = post.index(False)
            extra = sum(post[first:])
            rel_frames += extra
            n_rel += extra > 0
    return {"condition": cond, "method": method, "gap": gap, "n_terminating_edges": n_term,
            "relapse_fraction": n_rel / n_term if n_term else float("nan"),
            "relapse_frames_per_terminating_edge": rel_frames / n_term if n_term else float("nan")}


def main() -> None:
    cfg = P.load_config(Path("configs/experiments/phase4.yaml"))
    sp = P.specs(cfg)
    ctx = P.load_context(OUT)
    sel, par = {}, {}
    for g in ("baseline", "proposed"):
        for c, v in json.loads((OUT / "configs" / f"selected_{g}.json").read_text(encoding="utf-8")).items():
            sel.setdefault(c, {}).update(v)
        for c, v in json.loads((OUT / "configs" / f"pareto_{g}.json").read_text(encoding="utf-8")).items():
            for fk, by in v.items():
                par.setdefault(c, {}).setdefault(fk, {}).update(by)
    rel = [relapse(cfg, sp, sel, ctx, c, m) for c, s in sp.items() if s.calibrate for m in METHODS]
    R = pd.DataFrame(rel)
    R.to_csv(OUT / "aggregated" / "exploratory_relapse.csv", index=False)

    ev = pd.concat([pd.DataFrame(read_csv_gz(OUT / "raw" / f"evaluation_{g}.csv.gz")) for g in ("baseline", "proposed")])
    for c in ("gap", "seed", "persistence_f1", "false_pers_mean"):
        ev[c] = ev[c].astype(float)
    rows = []
    nb = int(cfg["statistics"]["hv_bootstrap_resamples"])
    h2 = [c for c, s in sp.items() if s.role in ("h2_failure", "h2_noninferiority")]
    for cond in h2:
        for gap in sp[cond].gaps:
            e = ev[(ev.condition == cond) & (ev.gap == gap)]
            piv_q = e.pivot(index="seed", columns="config_id", values="persistence_f1")
            piv_c = e.pivot(index="seed", columns="config_id", values="false_pers_mean")
            rng = np.random.default_rng(derive_seed(int(cfg["seeds"]["bootstrap"]), cond, gap, "fpr_frontier"))
            idx = rng.integers(0, len(piv_q), size=(nb, len(piv_q)))
            hv = {}
            for m in ("gated_ema", "va_hmm", "vhped_m5"):
                ids = par[cond]["default"][m]
                q, c = piv_q[ids].to_numpy(), piv_c[ids].to_numpy()
                point = hypervolume_2d(list(zip(np.nanmean(q, 0), np.nanmean(c, 0))), 0.0, 1.0)
                qb, cb = np.nanmean(q[idx], 1), np.nanmean(c[idx], 1)
                hv[m] = (point, np.array([hypervolume_2d(list(zip(qb[r], cb[r])), 0.0, 1.0) for r in range(nb)]))
            for comp in ("gated_ema", "va_hmm"):
                d = hv["vhped_m5"][1] - hv[comp][1]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append({"condition": cond, "gap": gap, "comparator": comp, "hv_m5": hv["vhped_m5"][0],
                             "hv_comp": hv[comp][0], "mean_diff": hv["vhped_m5"][0] - hv[comp][0],
                             "ci_lo": lo, "ci_hi": hi})
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "aggregated" / "exploratory_fpr_frontier.csv", index=False)

    lab = cfg["method_labels"]
    lines = ["<!-- GENERATED by scripts/phase4/exploratory_diagnostics.py — EXPLORATORY, POST-HOC -->\n",
             "# Exploratory post-hoc diagnostics (defined after seeing Phase 4 results; never decisive)\n",
             "## 1. Relapse of terminated edges (budget-5 operating points, gap 10, 30 videos)\n",
             "Relapse = a terminating edge is output again after its first post-termination drop "
             "(invisible to the frozen termination-delay metric).\n",
             "| Condition | " + " | ".join(lab[m] for m in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for cond, d in R.groupby("condition", sort=False):
        cells = []
        for m in METHODS:
            r = d[d.method == m].iloc[0]
            cells.append(f"{r.relapse_fraction:.2f} ({r.relapse_frames_per_terminating_edge:.1f} fr)")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["\nCells: fraction of terminating edges that relapse (mean relapse frames per terminating edge).\n",
              "## 2. F1 vs false-persistence frontier: ΔHV(M5 − comparator) [95% paired bootstrap CI]\n",
              "| Condition | Gap | vs GatedEMA | vs VA-HMM |", "|---|---|---|---|"]
    for (cond, gap), d in F.groupby(["condition", "gap"], sort=False):
        g_ = d[d.comparator == "gated_ema"].iloc[0]
        v_ = d[d.comparator == "va_hmm"].iloc[0]
        lines.append(f"| {cond} | {gap} | {g_.mean_diff:+.3f} [{g_.ci_lo:+.3f}, {g_.ci_hi:+.3f}] | "
                     f"{v_.mean_diff:+.3f} [{v_.ci_lo:+.3f}, {v_.ci_hi:+.3f}] |")
    (OUT / "tables" / "exploratory_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
