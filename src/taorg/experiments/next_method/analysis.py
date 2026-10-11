"""Next-method registered analysis (prereg §6-§9): completeness -> paired stats -> gate -> tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from taorg.experiments.episodes.analysis import Data, paired
from taorg.experiments.episodes.evaluation import METRIC_FIELDS
from taorg.experiments.episodes.pipeline import conditions, read_csv_gz, seeds
from taorg.experiments.next_method import decision as D
from taorg.experiments.phase3.stats import descriptive, holm_adjust

GATE_METRICS = ("J", "churn", "fpt_s", "frr", "std_mean_s", "ghost_time_s", "rdd_mean_s")
REPORT = ("J", "frame_f1", "episode_f1", "gap_recall", "fpt_s", "frr", "std_mean_s", "rdd_mean_s", "onset_mean_s",
          "ghost_time_s", "swap_recall", "churn", "bridge_rate", "fragmentation", "missed_rate", "persistence_f1",
          "term_delay_frozen", "false_pers_frozen")


def completeness(cfg: dict[str, Any], out: Path, rows: list[dict[str, str]], frozen: dict[str, Any]) -> dict[str, Any]:
    """Expected / actual / missing / duplicate / failed (checked before any analysis)."""
    variants = sorted(list(frozen["selected"]) + list(frozen["references"]))
    ev = seeds(cfg, "evaluation", False)
    expected = {(c, s, v) for c in cfg["conditions"] for s in ev for v in variants}
    seen: dict[tuple, int] = defaultdict(int)
    failed, shas = [], defaultdict(set)
    for r in rows:
        k = (r["condition"], int(r["seed"]), r["variant"])
        seen[k] += 1
        shas[r["instance_id"]].add(r["manifest_sha256"])
        if r.get("error") or not np.isfinite(float(r["J"])):
            failed.append({"key": list(k), "error": r.get("error", "")})
    calib = json.loads((out / "raw" / "calibration_instances.json").read_text(encoding="utf-8"))
    stage = json.loads((out / "raw" / "stage_calibrate.json").read_text(encoding="utf-8"))
    n_cal = sum(1 for c in conditions(cfg).values() if c.calibrate) * len(seeds(cfg, "calibration", False))
    leak = sorted({int(c["instance_id"].split("|s")[1]) for c in calib} & (set(ev) | set(cfg["seeds"]["development_excluded"])))
    res = {"evaluation": {"expected": len(expected), "actual_unique": len(seen), "rows": len(rows),
                          "n_missing": len(expected - set(seen)), "missing": sorted(map(list, expected - set(seen)))[:50],
                          "n_unexpected": len(set(seen) - expected), "n_duplicates": sum(v > 1 for v in seen.values()),
                          "n_failed": len(failed), "failed": failed[:50],
                          "inconsistent_manifests": [i for i, s in shas.items() if len(s) != 1]},
           "calibration": {"expected_instances": n_cal, "actual_instances": len(calib), "n_configs": stage.get("n_configs"),
                           "smoke": stage.get("smoke"), "max_samples": stage.get("max_samples"), "seed_leak": leak}}
    e = res["evaluation"]
    res["complete"] = (e["n_missing"] == 0 and e["n_unexpected"] == 0 and e["n_duplicates"] == 0 and e["n_failed"] == 0
                       and not e["inconsistent_manifests"] and len(calib) == n_cal and not leak
                       and not stage.get("smoke") and stage.get("max_samples") is None)
    return res


def pairset(cfg, data: Data, x: str, y: str, metrics=GATE_METRICS) -> dict[str, dict[str, dict]]:
    """condition -> metric -> paired stats of x - y."""
    return {c: {m: paired(cfg, data.get(c, x, m), data.get(c, y, m), "nm", c, x, y, m) for m in metrics}
            for c in cfg["conditions"]}


def run_analysis(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Registered analysis; refuses to run on incomplete data."""
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    rows = read_csv_gz(out / "raw" / "evaluation.csv.gz")
    comp = completeness(cfg, out, rows, frozen)
    (out / "aggregated").mkdir(parents=True, exist_ok=True)
    (out / "aggregated" / "completeness.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")
    if not comp["complete"]:
        raise RuntimeError("INCOMPLETE evaluation/calibration - see aggregated/completeness.json")
    data = Data(rows, seeds(cfg, "evaluation", False))
    cr = cfg["criteria"]
    T = cr["treatment"]
    comps = {"B2": cr["baseline"], "BUF": cr["buffer"], "GEO": cr["geometric"], "STATIC": cr["static"],
             "POOLED": cr["pooled"]}
    stats = {k: pairset(cfg, data, T, y) for k, y in comps.items()}
    stats["BUF_vs_B2"] = pairset(cfg, data, cr["buffer"], cr["baseline"])
    stats["half_B2"] = pairset(cfg, data, T + "__half", cr["baseline"] + "__half")
    stats["half_BUF"] = pairset(cfg, data, T + "__half", cr["buffer"] + "__half")
    res = {k: {c: D.pairwise(v[c], cr) for c in v} for k, v in stats.items()}
    lat_rows = read_csv_gz(out / "raw" / "latency.csv.gz")
    lat = {m: float(np.mean([float(r["latency_mean_ms"]) for r in lat_rows if r["method"] == m]))
           for m in cfg["methods_order"]}
    ratio = lat[T] / lat[cr["baseline"]]
    g = D.gate(res["B2"], res["BUF"], res["STATIC"], res["GEO"], res["half_B2"], res["half_BUF"], ratio, cr)
    fam, keys = [], []
    for k in ("B2", "BUF", "GEO", "STATIC"):
        for c in cfg["conditions"]:
            fam.append(stats[k][c]["J"]["p_wilcoxon"])
            keys.append(f"{k}|{c}")
    holm = dict(zip(keys, holm_adjust(fam)))
    desc = []
    variants = sorted(list(frozen["selected"]) + list(frozen["references"]))
    for c in cfg["conditions"]:
        for v in variants:
            d = {"condition": c, "variant": v}
            for m in REPORT:
                s = descriptive(list(data.get(c, v, m)))
                d.update({f"{m}_mean": s["mean"], f"{m}_lo": s["ci_lo"], f"{m}_hi": s["ci_hi"], f"{m}_n": s["n"]})
            desc.append(d)
    with open(out / "aggregated" / "descriptive.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(desc[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(desc)
    flat = [{"contrast": k, "condition": c, "metric": m, **s} for k, v in stats.items() for c, ms in v.items()
            for m, s in ms.items()]
    with open(out / "aggregated" / "paired.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["contrast", "condition", "metric"] + sorted({x for r in flat for x in r} - {"contrast", "condition", "metric"})
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(flat)
    result = {"gate": g, "pairwise": res, "latency_ms": lat, "latency_ratio_K1_B2": ratio, "holm_J": holm,
              "completeness": {"complete": True, "rows": len(rows)}}
    (out / "aggregated" / "analysis.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    write_tables(cfg, out, data, frozen, stats, res, result)
    return result


def _f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_tables(cfg, out: Path, data: Data, frozen, stats, res, result) -> None:
    """Markdown tables generated from artifacts."""
    t = out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    lab = cfg["method_labels"]
    vs = cfg["methods_order"] + [f"ref_{m}" for m in cfg["reference_methods"]["methods"]]
    labv = lambda v: lab.get(v, lab.get(v.replace("ref_", ""), v))  # noqa: E731
    lines = ["# Descriptive means (evaluation seeds 8000-8049, n = 50 per condition)", ""]
    for m, d in (("J", 3), ("gap_recall", 3), ("fpt_s", 2), ("frr", 3), ("std_mean_s", 3), ("rdd_mean_s", 3),
                 ("ghost_time_s", 2), ("churn", 4), ("bridge_rate", 3), ("onset_mean_s", 3), ("episode_f1", 3),
                 ("frame_f1", 3)):
        lines += [f"## {m}", "", "| Condition | " + " | ".join(labv(v) for v in vs) + " |", "|---|" + "---|" * len(vs)]
        for c in cfg["conditions"]:
            lines.append(f"| {c} | " + " | ".join(_f(float(np.nanmean(data.get(c, v, m))) if np.isfinite(
                data.get(c, v, m)).any() else math.nan, d) for v in vs) + " |")
        lines.append("")
    (t / "descriptive.md").write_text("\n".join(lines), encoding="utf-8")
    lines = ["# K1 (B2 + online KM hold) vs comparators: ΔJ [95% CI], outcome", "",
             "| Condition | vs B2 | vs BUF | vs GEO | vs K1-static | vs K1-pooled |", "|---|---|---|---|---|---|"]
    for c in cfg["conditions"]:
        cells = []
        for k in ("B2", "BUF", "GEO", "STATIC", "POOLED"):
            s, r = stats[k][c]["J"], res[k][c]
            mark = "W" if r["win"] else ("L" if r["loss"] else "·")
            extra = f" ({','.join(r['loss_reasons'])})" if r["loss"] else ("" if r["churn_guard"] else " [churn]")
            cells.append(f"{mark}{extra} {_f(s['mean_diff'])} [{_f(s['ci_lo'])}, {_f(s['ci_hi'])}]")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "BUF vs B2 (does the established fixed rule already fix B2?):", "",
              "| Condition | ΔJ [CI] | outcome |", "|---|---|---|"]
    for c in cfg["conditions"]:
        s, r = stats["BUF_vs_B2"][c]["J"], res["BUF_vs_B2"][c]
        lines.append(f"| {c} | {_f(s['mean_diff'])} [{_f(s['ci_lo'])}, {_f(s['ci_hi'])}] | "
                     f"{'W' if r['win'] else ('L' if r['loss'] else '·')} |")
    g = result["gate"]
    lines += ["", f"**Decision: {g['decision']}** — algorithm contribution: **{g['algorithm_contribution']}**", "",
              "Gates: " + json.dumps(g["gates"]), "", f"Latency ratio K1/B2: {result['latency_ratio_K1_B2']:.2f}"]
    (t / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    lines = ["# Frozen operating points (pooled calibration J; committed before evaluation)", "",
             "| Method | config | cal. J | cal. churn | half-sample pick | params |", "|---|---|---|---|---|---|"]
    for m in cfg["methods_order"]:
        s, h = frozen["selected"][m], frozen["selected"][m + "__half"]
        p = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s["params"].items()}
        lines.append(f"| {lab[m]} | {s['config_id']} | {_f(s['J'], 4)} | {_f(s['churn'], 4)} | {h['config_id']} | `{json.dumps(p)}` |")
    (t / "frozen_parameters.md").write_text("\n".join(lines), encoding="utf-8")
    lat = result["latency_ms"]
    lines = ["# Latency (mean ms/frame, C1, one machine)", "", "| Method | ms/frame | × B2 |", "|---|---|---|"] + \
        [f"| {lab[m]} | {lat[m]:.4f} | {lat[m] / lat[cfg['criteria']['baseline']]:.2f} |" for m in cfg["methods_order"]]
    (t / "latency.md").write_text("\n".join(lines), encoding="utf-8")
