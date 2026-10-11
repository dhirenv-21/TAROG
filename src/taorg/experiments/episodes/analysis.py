"""Episode-study registered analysis (prereg §7-§9).

Reads only saved artifacts (frozen parameters, evaluation rows, latency rows,
calibration bookkeeping) and writes:
    aggregated/completeness.json, aggregated/analysis.json, aggregated/descriptive.csv,
    aggregated/paired.csv, tables/*.md
Decisions are made exclusively by ``taorg.experiments.episodes.decision``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from taorg.experiments.episodes import decision as D
from taorg.experiments.episodes.evaluation import METRIC_FIELDS
from taorg.experiments.episodes.pipeline import read_csv_gz, seeds, conditions as cond_specs
from taorg.experiments.phase3.conditions import derive_seed
from taorg.experiments.phase3.stats import descriptive, holm_adjust, paired_comparison

PAIRED_METRICS = ("J", "frame_f1", "episode_f1", "frr", "fpt_s", "ghost_time_s", "std_mean_s", "rdd_mean_s",
                  "onset_mean_s", "gap_recall", "persistence_f1", "churn", "missed_rate", "fragmentation",
                  "swap_recall", "bridge_rate")
REPORT_METRICS = PAIRED_METRICS + ("frame_precision", "frame_recall", "reactivation_frames",
                                   "term_delay_frozen", "false_pers_frozen", "std_censored", "rdd_censored")
RUNGS = ["episode", "episode_id", "episode_ctx", "episode_hist"]


class Data:
    """Evaluation rows indexed by (condition, variant) -> metric -> per-seed array (seed order)."""

    def __init__(self, rows: list[dict[str, str]], seed_list: list[int]):
        self.seeds = seed_list
        pos = {s: i for i, s in enumerate(seed_list)}
        self.v: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(
            lambda: {f: np.full(len(seed_list), np.nan) for f in METRIC_FIELDS})
        for r in rows:
            arr = self.v[(r["condition"], r["variant"])]
            i = pos[int(r["seed"])]
            for f in METRIC_FIELDS:
                try:
                    arr[f][i] = float(r[f])
                except (TypeError, ValueError):
                    arr[f][i] = np.nan

    def get(self, cond: str, variant: str, metric: str) -> np.ndarray:
        return self.v[(cond, variant)][metric]


def _rng(cfg: dict[str, Any], *labels: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(int(cfg["seeds"]["bootstrap"]), *labels))


def paired(cfg: dict[str, Any], a: np.ndarray, b: np.ndarray, *labels: Any) -> dict[str, float]:
    """Paired comparison a - b with the registered bootstrap; d_z undefined if SD < dz_min_sd."""
    st = cfg["statistics"]
    r = paired_comparison(list(a), list(b), int(st["bootstrap_resamples"]), float(st["ci_level"]), _rng(cfg, *labels))
    if not (isinstance(r["sd_diff"], float) and r["sd_diff"] >= float(st["dz_min_sd"])):
        r["dz"] = math.nan
        r["dz_undefined"] = True
    return r


def stats_set(cfg: dict[str, Any], data: Data, cond: str, x: str, y: str, metrics=PAIRED_METRICS) -> dict[str, dict]:
    """Paired stats of variant x vs variant y on the given metrics in one condition."""
    return {m: paired(cfg, data.get(cond, x, m), data.get(cond, y, m), "paired", cond, x, y, m) for m in metrics}


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------

def completeness(cfg: dict[str, Any], out: Path, rows: list[dict[str, str]], frozen: dict[str, Any]) -> dict[str, Any]:
    """Expected / actual / missing / duplicate / failed runs (checked before analysis)."""
    conds = list(cfg["conditions"])
    ev_seeds = seeds(cfg, "evaluation", False)
    variants = sorted(frozen["selected"])
    expected = {(c, s, v) for c in conds for s in ev_seeds for v in variants}
    seen: dict[tuple, int] = defaultdict(int)
    failed = []
    sha_by_inst: dict[str, set] = defaultdict(set)
    for r in rows:
        key = (r["condition"], int(r["seed"]), r["variant"])
        seen[key] += 1
        sha_by_inst[r["instance_id"]].add(r["manifest_sha256"])
        if r.get("error") or not np.isfinite(float(r["J"])):
            failed.append({"key": list(key), "error": r.get("error", "")})
    actual = set(seen)
    calib = json.loads((out / "raw" / "calibration_instances.json").read_text(encoding="utf-8"))
    n_cal_expected = sum(1 for c in cond_specs(cfg).values() if c.calibrate) * len(seeds(cfg, "calibration", False))
    stage_cal = json.loads((out / "raw" / "stage_calibrate.json").read_text(encoding="utf-8"))
    res = {
        "evaluation": {"expected": len(expected), "actual_unique": len(actual), "rows": len(rows),
                       "missing": sorted([list(k) for k in expected - actual])[:50],
                       "n_missing": len(expected - actual),
                       "unexpected": sorted([list(k) for k in actual - expected])[:50],
                       "n_duplicates": sum(1 for v in seen.values() if v > 1), "n_failed": len(failed),
                       "failed": failed[:50],
                       "instances_with_inconsistent_manifest": [i for i, s in sha_by_inst.items() if len(s) != 1]},
        "calibration": {"expected_instances": n_cal_expected, "actual_instances": len(calib),
                        "n_configs": stage_cal.get("n_configs"), "max_samples": stage_cal.get("max_samples"),
                        "smoke": stage_cal.get("smoke"),
                        "calibration_seed_leak": sorted({int(c["instance_id"].split("|s")[1]) for c in calib}
                                                        & set(ev_seeds))},
    }
    res["complete"] = (res["evaluation"]["n_missing"] == 0 and res["evaluation"]["n_duplicates"] == 0
                       and res["evaluation"]["n_failed"] == 0
                       and not res["evaluation"]["instances_with_inconsistent_manifest"]
                       and n_cal_expected == len(calib) and not res["calibration"]["calibration_seed_leak"]
                       and not stage_cal.get("smoke") and stage_cal.get("max_samples") is None)
    return res


# ---------------------------------------------------------------------------
# decision procedure (also re-run for the half-sample control)
# ---------------------------------------------------------------------------

def decide(cfg: dict[str, Any], data: Data, latency: dict[str, float], suffix: str = "") -> dict[str, Any]:
    """Full registered decision for one variant family ('' = primary, '__half' = control)."""
    crit = cfg["criteria"]
    conds = list(cfg["conditions"])
    cal = crit["benefit_conditions"] + crit["ni_conditions"]
    v = lambda m: m + suffix  # noqa: E731
    minr = int(cfg["statistics"]["min_paired_rdd"])

    # ladder
    lad: dict[str, Any] = {}
    pc = {c: stats_set(cfg, data, c, v("episode_id"), v("episode"), ("J", "ghost_time_s", "std_mean_s")) for c in cal}
    lad["identity"] = {**D.retain_identity(pc, crit, cal), "stats": pc}
    pc = {c: stats_set(cfg, data, c, v("episode_ctx"), v("episode_id"), ("J", "std_mean_s")) for c in cal}
    lad["context"] = {**D.retain_context(pc, crit, cal), "stats": pc}
    pc = {c: stats_set(cfg, data, c, v("episode_hist"), v("episode_ctx"), ("J", "std_mean_s")) for c in cal}
    a = np.nanmean([data.get(c, v("episode_hist"), "J") for c in cal], axis=0)
    b = np.nanmean([data.get(c, v("episode_ctx"), "J") for c in cal], axis=0)
    pooled = paired(cfg, a, b, "pooled_history", suffix)
    lad["history"] = {**D.retain_history(pc, pooled, crit, cal), "pooled_J": pooled, "stats": pc}
    T = D.select_treatment(lad["identity"]["retained"], lad["context"]["retained"], lad["history"]["retained"], RUNGS)

    # T vs comparators
    pair: dict[str, dict[str, Any]] = {}
    cells: dict[str, str] = {}
    for c in conds:
        pair[c] = {}
        for y in D.comparators_for(c, crit):
            st = stats_set(cfg, data, c, v(T), v(y))
            pair[c][y] = {"result": D.pairwise(st, c, crit, minr), "stats": st}
        cells[c] = D.cell_outcome({y: r["result"] for y, r in pair[c].items()})
    vs_simple = {c: pair[c][crit["simple_baseline"]]["result"] for c in crit["relapse_conditions"]}
    lat_T = latency.get(T, math.nan)
    lat_ratio = lat_T / latency["gated_ema"] if latency.get("gated_ema") else math.nan

    # registered answers (P1-based Q1/Q2, P2-based Q3)
    p1_vs = {c: {y: D.pairwise(stats_set(cfg, data, c, v("episode"), v(y)), c, crit, minr) for y in crit["comparators"]}
             for c in ("C3_misleading_post", "C5_boundary_flicker")}
    p1_rdd = {c: {y: paired(cfg, data.get(c, v("episode"), "rdd_mean_s"), data.get(c, v(y), "rdd_mean_s"),
                            "q2", c, y, suffix) for y in crit["comparators"]}
              for c in ("C4_recurrence", "C5_boundary_flicker")}
    c6 = "C6_id_switch"
    p2_vs_b5 = D.pairwise(stats_set(cfg, data, c6, v("episode_id"), v("gated_ema_id")), c6, crit, minr)
    inter = paired(cfg, data.get(c6, v("episode_id"), "ghost_time_s") - data.get(c6, v("episode"), "ghost_time_s"),
                   data.get(c6, v("gated_ema_id"), "ghost_time_s") - data.get(c6, v("gated_ema"), "ghost_time_s"),
                   "q3_interaction", suffix)
    return {"T": T, "ladder": lad, "pairwise": pair, "cells": cells, "vs_simple": vs_simple,
            "latency_T_ms": lat_T, "latency_ratio": lat_ratio,
            "q1": {**D.answer_q1(p1_vs, crit), "detail": p1_vs},
            "q2": {**D.answer_q2(p1_rdd, crit, minr), "stats": p1_rdd},
            "q3": {**D.answer_q3(p2_vs_b5, inter), "p2_vs_b5": p2_vs_b5, "interaction": inter},
            "q4": {"answer": "YES" if lad["history"]["retained"] else "NO", **{k: lad["history"][k] for k in
                                                                              ("pooled_ok", "target_hits", "J_losses")}},
            "q5": D.answer_q5(cells, crit)}


def _fmt(x: float, d: int = 3) -> str:
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{d}f}"


def run_analysis(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Completeness -> statistics -> registered decisions -> tables."""
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    rows = read_csv_gz(out / "raw" / "evaluation.csv.gz")
    comp = completeness(cfg, out, rows, frozen)
    (out / "aggregated").mkdir(parents=True, exist_ok=True)
    (out / "aggregated" / "completeness.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
    if not comp["complete"]:
        raise RuntimeError("evaluation/calibration INCOMPLETE - see aggregated/completeness.json; no analysis run")
    data = Data(rows, seeds(cfg, "evaluation", False))
    lat_rows = read_csv_gz(out / "raw" / "latency.csv.gz")
    latency = {m: float(np.mean([float(r["latency_mean_ms"]) for r in lat_rows if r["method"] == m]))
               for m in cfg["methods_order"]}

    # descriptive
    desc = []
    for c in cfg["conditions"]:
        for var in sorted(frozen["selected"]):
            d = {"condition": c, "variant": var}
            for m in REPORT_METRICS:
                s = descriptive(list(data.get(c, var, m)))
                d[f"{m}_mean"], d[f"{m}_lo"], d[f"{m}_hi"], d[f"{m}_n"] = s["mean"], s["ci_lo"], s["ci_hi"], s["n"]
            desc.append(d)
    import csv
    with open(out / "aggregated" / "descriptive.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(desc[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(desc)

    primary = decide(cfg, data, latency, "")
    half = decide(cfg, data, latency, "__half")
    crit = cfg["criteria"]
    half_dec = D.study_decision(half["cells"], half["vs_simple"], half["latency_T_ms"], half["latency_ratio"], None, crit)
    study = D.study_decision(primary["cells"], primary["vs_simple"], primary["latency_T_ms"],
                             primary["latency_ratio"], half_dec["decision"], crit)

    # Holm families (reported, not decisive)
    fam, keys = [], []
    for c, by in primary["pairwise"].items():
        for y, r in by.items():
            fam.append(r["stats"]["J"]["p_wilcoxon"])
            keys.append((c, y))
    holm = dict(zip([f"{c}|{y}" for c, y in keys], holm_adjust(fam)))
    lfam, lkeys = [], []
    for rung in ("identity", "context", "history"):
        for c, st in primary["ladder"][rung]["stats"].items():
            lfam.append(st["J"]["p_wilcoxon"])
            lkeys.append(f"{rung}|{c}")
    holm_ladder = dict(zip(lkeys, holm_adjust(lfam)))

    # paired table (flat)
    flat = []
    for c, by in primary["pairwise"].items():
        for y, r in by.items():
            for m, st in r["stats"].items():
                flat.append({"condition": c, "treatment": primary["T"], "comparator": y, "metric": m, **st})
    for rung, (x, y) in {"identity": ("episode_id", "episode"), "context": ("episode_ctx", "episode_id"),
                         "history": ("episode_hist", "episode_ctx")}.items():
        for c, sts in primary["ladder"][rung]["stats"].items():
            for m, st in sts.items():
                flat.append({"condition": c, "treatment": x, "comparator": y, "metric": m, **st})
    with open(out / "aggregated" / "paired.csv", "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in flat for k in r}, key=lambda k: (k not in ("condition", "treatment", "comparator", "metric"), k))
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(flat)

    q6 = {"answer": "YES" if primary["T"] == "episode_hist" else "NO", "T": primary["T"],
          "retained": {k: primary["ladder"][k]["retained"] for k in ("identity", "context", "history")}}
    result = {"study_decision": study, "half_control_decision": half_dec, "T": primary["T"], "T_half": half["T"],
              "cells": primary["cells"], "cells_half": half["cells"], "latency_ms": latency,
              "latency_T_ms": primary["latency_T_ms"], "latency_ratio": primary["latency_ratio"],
              "ladder": {k: {kk: vv for kk, vv in primary["ladder"][k].items() if kk != "stats"}
                         for k in primary["ladder"]},
              "pairwise_results": {c: {y: r["result"] for y, r in by.items()} for c, by in primary["pairwise"].items()},
              "vs_simple": primary["vs_simple"],
              "answers": {"Q1": {k: v for k, v in primary["q1"].items() if k != "detail"},
                          "Q1_detail": {c: {y: r for y, r in by.items()} for c, by in primary["q1"]["detail"].items()},
                          "Q2": {"answer": primary["q2"]["answer"], "cells": primary["q2"]["cells"]},
                          "Q3": {k: primary["q3"][k] for k in ("answer", "p2_wins_vs_b5", "interaction_negative")},
                          "Q3_interaction": primary["q3"]["interaction"],
                          "Q4": primary["q4"], "Q5": primary["q5"], "Q6": q6},
              "holm_J_T_vs_comparators": holm, "holm_J_ladder": holm_ladder,
              "completeness": {"complete": comp["complete"], "rows": comp["evaluation"]["rows"]}}
    (out / "aggregated" / "analysis.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    write_tables(cfg, out, data, frozen, primary, result)
    return result


def write_tables(cfg: dict[str, Any], out: Path, data: Data, frozen: dict[str, Any], primary: dict[str, Any],
                 result: dict[str, Any]) -> None:
    """Markdown tables generated from the artifacts (never hand-copied)."""
    t = out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    lab = cfg["method_labels"]
    ms = cfg["methods_order"]
    lines = ["# Descriptive means (evaluation, n = 50 seeds per condition; primary selection)", ""]
    for metric, d in (("J", 3), ("frame_f1", 3), ("episode_f1", 3), ("frr", 3), ("fpt_s", 2), ("ghost_time_s", 2),
                      ("std_mean_s", 3), ("rdd_mean_s", 3), ("onset_mean_s", 3), ("gap_recall", 3),
                      ("churn", 4), ("missed_rate", 3), ("fragmentation", 3), ("bridge_rate", 3),
                      ("swap_recall", 3), ("persistence_f1", 3), ("term_delay_frozen", 1), ("false_pers_frozen", 3)):
        lines += [f"## {metric}", "", "| Condition | " + " | ".join(lab[m] for m in ms) + " |",
                  "|---|" + "---|" * len(ms)]
        for c in cfg["conditions"]:
            vals = [_fmt(float(np.nanmean(data.get(c, m, metric))) if np.isfinite(data.get(c, m, metric)).any()
                         else math.nan, d) for m in ms]
            lines.append(f"| {c} | " + " | ".join(vals) + " |")
        lines.append("")
    (t / "descriptive.md").write_text("\n".join(lines), encoding="utf-8")

    T = primary["T"]
    lines = [f"# Decision matrix: T = {lab[T]} vs registered comparators", "",
             "Cell entries: W (win), L (loss), · (neutral); route J/harm; failed guards in brackets.", "",
             "| Condition | role | " + " | ".join(lab[y] for y in cfg["criteria"]["comparators"] + ["gated_ema_id"])
             + " | cell |", "|---|---|" + "---|" * (len(cfg["criteria"]["comparators"]) + 2)]
    roles = {c: d["role"] for c, d in cfg["conditions"].items()}
    for c, by in primary["pairwise"].items():
        cols = []
        for y in cfg["criteria"]["comparators"] + ["gated_ema_id"]:
            if y not in by:
                cols.append("—")
                continue
            r = by[y]["result"]
            s = by[y]["stats"]["J"]
            mark = "W" if r["win"] else ("L" if r["loss"] else "·")
            extra = (f" {r['route']}" if r["win"] else "") + (f" [{','.join(r['failed_guards'])}]" if r["failed_guards"] else "") \
                + (f" ({','.join(r['loss_reasons'])})" if r["loss"] else "")
            cols.append(f"{mark}{extra} ΔJ {_fmt(s['mean_diff'])} [{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]")
        lines.append(f"| {c} | {roles[c]} | " + " | ".join(cols) + f" | **{primary['cells'][c]}** |")
    sd = result["study_decision"]
    lines += ["", f"**Study decision: {sd['decision']}**", "", "GO checks: " + json.dumps(sd["go_checks"]),
              "", "STOP checks: " + json.dumps(sd["stop_checks"]), "",
              f"Half-sample control decision: {result['half_control_decision']['decision']} (T = {result['T_half']})"]
    (t / "decision.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Ladder (each rung vs the rung below), evaluation", "",
             "| Rung | Condition | ΔJ [95% CI] | ΔSTD s [CI] | Δghost s [CI] | Holm p (ΔJ) |", "|---|---|---|---|---|---|"]
    for rung in ("identity", "context", "history"):
        for c, st in primary["ladder"][rung]["stats"].items():
            def f(m):
                s = st.get(m)
                return "—" if s is None else f"{_fmt(s['mean_diff'])} [{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]"
            lines.append(f"| {rung} | {c} | {f('J')} | {f('std_mean_s')} | {f('ghost_time_s')} | "
                         f"{_fmt(result['holm_J_ladder'].get(f'{rung}|{c}', math.nan))} |")
    pj = primary["ladder"]["history"]["pooled_J"]
    lines += ["", f"Pooled history ΔJ (P4 − P3) over C1–C7: {_fmt(pj['mean_diff'], 4)} "
                  f"[{_fmt(pj['ci_lo'], 4)}, {_fmt(pj['ci_hi'], 4)}]", "",
              "Retained: " + json.dumps(result["ladder"] and {k: v["retained"] for k, v in result["ladder"].items()}),
              f"\nSelected T = {T}"]
    (t / "ladder.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Frozen operating points (pooled calibration J)", "", "| Method | config | cal. J | cal. churn | params |",
             "|---|---|---|---|---|"]
    for m in ms:
        s = frozen["selected"][m]
        lines.append(f"| {lab[m]} | {s['config_id']} | {_fmt(s['J'])} | {_fmt(s['churn'], 4)} | "
                     f"{json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s['params'].items()})} |")
    (t / "frozen_parameters.md").write_text("\n".join(lines), encoding="utf-8")

    lat = result["latency_ms"]
    lines = ["# Latency (mean ms/frame, C1 evaluation instances, one machine)", "", "| Method | ms/frame | × GatedEMA |",
             "|---|---|---|"] + [f"| {lab[m]} | {lat[m]:.4f} | {lat[m] / lat['gated_ema']:.2f} |" for m in ms]
    (t / "latency.md").write_text("\n".join(lines), encoding="utf-8")
