"""Phase 4 analysis: descriptive statistics, frontiers, paired tests, decisions.

Reads only saved artifacts (evaluation rows, selections, Pareto sets,
latency).  Decisions are delegated to the pre-registered
``taorg.experiments.phase4.decision`` functions.

If only the baseline group has been evaluated, a baseline-only report is
written (descriptive, frontiers, M1 vs M0); no H2/H3/H4 decision is made.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taorg.experiments.phase3.conditions import derive_seed
from taorg.experiments.phase3.pipeline import read_csv_gz
from taorg.experiments.phase3.stats import descriptive, holm_adjust, hypervolume_2d, paired_comparison
from taorg.experiments.phase4 import decision as D
from taorg.experiments.phase4.pipeline import frontier_keys, specs

DESC_METRICS = ("persistence_f1", "term_delay_mean", "delay_s", "false_pers_mean", "churn_mean",
                "diag_swap_recall", "diag_ghost_edge_frames", "diag_spurious_rate",
                "diag_clean_alive_recall", "persistence_recall")
ABLATION = (("A0", "va_hmm", "gated_ema"), ("A1", "vhped_m2", "va_hmm"), ("A2", "vhped_m3", "vhped_m2"),
            ("A3", "vhped_m4", "vhped_m3"), ("A4", "vhped_m5", "vhped_m4"), ("A5", "vhped_m6", "vhped_m5"))


def _load(out: Path) -> tuple[pd.DataFrame, dict, dict, list[str]]:
    frames, selected, pareto, groups = [], {}, {}, []
    for g in ("baseline", "proposed"):
        p = out / "raw" / f"evaluation_{g}.csv.gz"
        if not p.exists():
            continue
        groups.append(g)
        frames.append(pd.DataFrame(read_csv_gz(p)))
        for src, dst in ((f"selected_{g}.json", selected), (f"pareto_{g}.json", pareto)):
            d = json.loads((out / "configs" / src).read_text(encoding="utf-8"))
            for c, v in d.items():
                if src.startswith("pareto"):
                    for fk, by_m in v.items():
                        dst.setdefault(c, {}).setdefault(fk, {}).update(by_m)
                else:
                    dst.setdefault(c, {}).update(v)
    ev = pd.concat(frames, ignore_index=True)
    for c in ev.columns:
        if c not in ("run_id", "condition", "role", "instance_id", "manifest_sha256", "config_id",
                     "method", "params_json"):
            ev[c] = ev[c].astype(float)
    ev["delay_s"] = ev["term_delay_mean"] / (30.0 * ev["fps_scale"])
    return ev, selected, pareto, groups


class Cube:
    """Per (condition, gap): seed x config matrices of each metric."""

    def __init__(self, ev: pd.DataFrame, seeds: list[int]):
        self.seeds = seeds
        self._m: dict[tuple[str, int, str], pd.DataFrame] = {}
        for (c, g), df in ev.groupby(["condition", "gap"]):
            for metric in DESC_METRICS:
                self._m[(c, int(g), metric)] = df.pivot(index="seed", columns="config_id",
                                                        values=metric).reindex(seeds)

    def values(self, c: str, g: int, metric: str, cid: str) -> np.ndarray:
        return self._m[(c, g, metric)][cid].to_numpy(dtype=float)

    def matrix(self, c: str, g: int, metric: str, cids: list[str]) -> np.ndarray:
        return self._m[(c, g, metric)][cids].to_numpy(dtype=float)


def _dz_fix(pc: dict[str, float], min_sd: float) -> dict[str, float]:
    pc = dict(pc)
    pc["dz_undefined"] = bool(not (pc["sd_diff"] >= min_sd))
    if pc["dz_undefined"]:
        pc["dz"] = math.nan
    return pc


def run_analysis(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Compute all Phase 4 statistics and decisions; write aggregated/ and tables/."""
    ev, selected, pareto, groups = _load(out)
    sp = specs(cfg)
    seeds = sorted(int(s) for s in ev["seed"].unique())
    cube = Cube(ev, seeds)
    st = cfg["statistics"]
    crit = cfg["criteria"]
    B = str(cfg["primary_budget_frames"])
    agg = out / "aggregated"
    agg.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    present = set(ev["method"].unique())

    def sel_id(cond: str, name: str, b: str = B) -> str | None:
        src = sp[cond].tuning_source
        return selected.get(src, {}).get(name, {}).get(b, {}).get("config_id")

    # ---------------- descriptive (selected operating points) ----------------
    desc_rows = []
    for cond, spec in sp.items():
        for name in sorted(selected.get(spec.tuning_source, {})):
            for b in map(str, cfg["budgets_frames"]):
                cid = sel_id(cond, name, b)
                if cid is None:
                    continue
                for g in spec.gaps:
                    row = {"condition": cond, "role": spec.role, "gap": g, "method": name, "budget": b,
                           "config_id": cid}
                    for m in DESC_METRICS:
                        d = descriptive(cube.values(cond, g, m, cid))
                        for k, v in d.items():
                            row[f"{m}__{k}"] = v
                    desc_rows.append(row)
    desc = pd.DataFrame(desc_rows)
    desc.to_csv(agg / "descriptive.csv", index=False)

    # ---------------- frontiers: HV with paired bootstrap ----------------
    R = int(st["hv_bootstrap_resamples"])
    hv_rows, hv_boot = [], {}
    for cond, spec in sp.items():
        if not spec.calibrate:
            continue
        for fk in frontier_keys(cfg, cond):
            f = cfg["frontiers"][fk]
            for g in spec.gaps:
                rng = np.random.default_rng(derive_seed(int(cfg["seeds"]["bootstrap"]), cond, g, fk, "hv"))
                idx = rng.integers(0, len(seeds), size=(R, len(seeds)))
                for name, cids in pareto.get(cond, {}).get(fk, {}).items():
                    q = cube.matrix(cond, g, f["quality"], cids)
                    c = cube.matrix(cond, g, f["cost"], cids)
                    point = hypervolume_2d(list(zip(np.nanmean(q, 0), np.nanmean(c, 0))), f["cost_min"], f["cost_ref"])
                    qb, cb = np.nanmean(q[idx], axis=1), np.nanmean(c[idx], axis=1)
                    boots = np.array([hypervolume_2d(list(zip(qb[r], cb[r])), f["cost_min"], f["cost_ref"])
                                      for r in range(R)])
                    hv_boot[(cond, g, fk, name)] = (point, boots)
                    lo, hi = np.quantile(boots, [0.025, 0.975])
                    hv_rows.append({"condition": cond, "gap": g, "frontier": fk, "method": name, "hv": point,
                                    "ci_lo": lo, "ci_hi": hi, "n_frontier_configs": len(cids)})
    pd.DataFrame(hv_rows).to_csv(agg / "hypervolume.csv", index=False)

    def dhv(cond: str, g: int, fk: str, a: str, b: str) -> dict[str, float] | None:
        if (cond, g, fk, a) not in hv_boot or (cond, g, fk, b) not in hv_boot:
            return None
        pa, ba = hv_boot[(cond, g, fk, a)]
        pb, bb = hv_boot[(cond, g, fk, b)]
        lo, hi = np.quantile(ba - bb, [0.025, 0.975])
        return {"mean_diff": pa - pb, "ci_lo": float(lo), "ci_hi": float(hi)}

    def op_pair(cond: str, g: int, a: str, b: str, metric: str, key: tuple) -> dict[str, float] | None:
        ca, cb = sel_id(cond, a), sel_id(cond, b)
        if ca is None or cb is None:
            return None
        rng = np.random.default_rng(derive_seed(int(cfg["seeds"]["bootstrap"]), *key))
        pc = paired_comparison(cube.values(cond, g, metric, ca), cube.values(cond, g, metric, cb),
                               int(st["bootstrap_resamples"]), float(st["ci_level"]), rng)
        return _dz_fix(pc, float(st["dz_min_sd"]))

    diff_rows: list[dict[str, Any]] = []

    def record(tag: str, cond: str, g: int, fk: str, a: str, b: str, d: dict | None) -> None:
        if d is not None:
            diff_rows.append({"comparison": tag, "condition": cond, "gap": g, "frontier": fk,
                              "treatment": a, "comparator": b, **d})

    frc = {"min_hv_gain": crit["min_hv_gain"], "churn_margin_abs": crit["churn_margin_abs"],
           "churn_margin_rel": crit["churn_margin_rel"]}

    def frontier_cell(cond: str, g: int, treat: str, comps: list[str], tag: str) -> tuple[str, dict]:
        fk = "identity" if cond in cfg["identity_conditions"] else "default"
        per = {}
        for m in comps:
            d = dhv(cond, g, fk, treat, m)
            record(tag, cond, g, fk, treat, m, d)
            ch = op_pair(cond, g, treat, m, "churn_mean", (tag, cond, g, treat, m, "churn"))
            comp_churn = ch["mean_b"] if ch else math.nan
            dd = None
            if fk == "identity":
                dd = dhv(cond, g, "default", treat, m)
                record(tag + "_defaultNI", cond, g, "default", treat, m, dd)
            if d is None:
                per[m] = {"win": False, "loss": False, "failed_checks": ["missing"]}
                continue
            r = D.frontier_vs(d, ch, comp_churn, frc, dd, crit["h2"]["identity_default_frontier_ni"]
                              if dd is not None else None)
            r.update({"d_hv": d, "d_churn": ch})
            per[m] = r
        return D.cell_outcome(per), per

    result: dict[str, Any] = {"groups": groups, "decisions": {}}
    roles = {r: [c for c, s in sp.items() if s.role == r] for r in
             ("h2_failure", "h2_noninferiority", "h3_benefit", "h3_noninferiority", "h4_shift")}

    # ---------------- ablation / baseline comparisons ----------------
    abl = {}
    for tag, a, b in ABLATION:
        if a not in present or b not in present:
            continue
        rows_ab = {}
        for cond, spec in sp.items():
            if not spec.calibrate:
                continue
            fk = "identity" if cond in cfg["identity_conditions"] else "default"
            for g in spec.gaps:
                d = dhv(cond, g, fk, a, b)
                record(tag, cond, g, fk, a, b, d)
                if d is not None:
                    rows_ab[(cond, g)] = d
        abl[tag] = rows_ab
    retention = {}
    for tag, a, b in ABLATION[2:5]:
        if tag not in abl:
            continue
        tg = crit["ablation"]["targets"][a]
        tgt = [v for (c, g), v in abl[tag].items() if c in tg]
        ni = [v for (c, g), v in abl[tag].items() if c in roles["h2_noninferiority"]]
        retention[a] = D.rung_retained(tgt, ni, crit["ablation"])
    result["ablation_retention"] = retention

    # ---------------- H2 / tuning control / H3 ----------------
    cells_out: dict[str, Any] = {}
    h2 = crit["h2"]
    if h2["treatment"] in present:
        for variant, treat in (("H2", h2["treatment"]), ("H2_tuning_control", h2["treatment"] + "__half")):
            fc, nc, detail = {}, {}, {}
            for cond in roles["h2_failure"] + roles["h2_noninferiority"]:
                for g in sp[cond].gaps:
                    oc, per = frontier_cell(cond, g, treat, h2["comparators"], variant)
                    (fc if cond in roles["h2_failure"] else nc)[(cond, g)] = oc
                    detail[f"{cond}|{g}"] = {"outcome": oc, "per_comparator": per}
            result["decisions"][variant] = D.decide_h2(fc, nc, h2)
            cells_out[variant] = detail
    h3 = crit["h3"]
    if h3["treatment"] in present:
        bc, nc, detail = {}, {}, {}
        for cond in h3["benefit_conditions"] + h3["noninferiority_conditions"]:
            for g in sp[cond].gaps:
                oc, per = frontier_cell(cond, g, h3["treatment"], [h3["comparator"]], "H3")
                (bc if cond in h3["benefit_conditions"] else nc)[(cond, g)] = oc
                detail[f"{cond}|{g}"] = {"outcome": oc, "per_comparator": per}
        result["decisions"]["H3"] = D.decide_h3(bc, nc, h3)
        cells_out["H3"] = detail

    # ---------------- H4: operating points under shift ----------------
    h4 = crit["h4"]
    op_rows = []
    if h4["treatment"] in present:
        tests = {}
        for cond in roles["h4_shift"]:
            for g in sp[cond].gaps:
                for m in h4["comparators"]:
                    for metric in ("persistence_f1", "delay_s", "false_pers_mean", "churn_mean"):
                        pc = op_pair(cond, g, h4["treatment"], m, metric, ("H4", cond, g, m, metric))
                        tests[(cond, g, m, metric)] = pc
        keys = [k for k in tests if k[3] == "persistence_f1"]
        adj = holm_adjust([tests[k]["p_wilcoxon"] for k in keys])
        for k, p in zip(keys, adj):
            tests[k]["p_holm"] = p
        cells, detail = {}, {}
        for cond in roles["h4_shift"]:
            for g in sp[cond].gaps:
                per = {}
                for m in h4["comparators"]:
                    t = {mt: tests[(cond, g, m, mt)] for mt in ("persistence_f1", "delay_s", "false_pers_mean",
                                                                   "churn_mean")}
                    r = D.shift_vs(t["persistence_f1"], t["delay_s"], t["false_pers_mean"], t["churn_mean"],
                                   t["churn_mean"]["mean_b"], t["churn_mean"]["mean_a"], h4,
                                   crit["churn_margin_abs"], crit["churn_margin_rel"])
                    per[m] = r
                oc = D.cell_outcome(per)
                cells[(cond, g)] = oc
                detail[f"{cond}|{g}"] = {"outcome": oc, "per_comparator": per}
        result["decisions"]["H4"] = D.decide_h4(cells, h4)
        cells_out["H4"] = detail
        for (cond, g, m, metric), pc in tests.items():
            op_rows.append({"family": "H4", "condition": cond, "gap": g, "treatment": h4["treatment"],
                            "comparator": m, "metric": metric, **pc})
    # Exploratory operating-point comparisons (budget 5) on calibrated conditions
    for cond, spec in sp.items():
        if not spec.calibrate:
            continue
        for g in spec.gaps:
            for a, b in (("vhped_m5", "gated_ema"), ("vhped_m5", "va_hmm"), ("va_hmm", "gated_ema")):
                if a not in present or b not in present:
                    continue
                for metric in ("persistence_f1", "term_delay_mean", "false_pers_mean", "churn_mean",
                               "diag_ghost_edge_frames", "diag_swap_recall"):
                    pc = op_pair(cond, g, a, b, metric, ("explore", cond, g, a, b, metric))
                    if pc:
                        op_rows.append({"family": "exploratory", "condition": cond, "gap": g, "treatment": a,
                                        "comparator": b, "metric": metric, **pc})
    pd.DataFrame(op_rows).to_csv(agg / "paired_operating_points.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(agg / "hv_differences.csv", index=False)

    # ---------------- latency ----------------
    lat = []
    for g in groups:
        p = out / "raw" / f"latency_{g}.csv.gz"
        if p.exists():
            lat.append(pd.DataFrame(read_csv_gz(p)))
    if lat:
        L = pd.concat(lat)
        for c in ("latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms"):
            L[c] = pd.to_numeric(L[c])
        L.groupby("method")[["latency_mean_ms", "latency_median_ms", "latency_p95_ms", "latency_p99_ms"]] \
            .mean().to_csv(agg / "latency.csv")

    def _ser(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k) if not isinstance(k, str) else k: _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (agg / "decisions.json").write_text(json.dumps(_ser({**result, "cells": cells_out}), indent=1),
                                        encoding="utf-8")
    from taorg.experiments.phase4.report import write_tables
    write_tables(cfg, out, desc, pd.DataFrame(hv_rows), pd.DataFrame(diff_rows), pd.DataFrame(op_rows),
                 result, cells_out)
    return result
