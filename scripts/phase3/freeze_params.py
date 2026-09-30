"""Phase 3 — verify calibration and write the frozen-parameter artifact.

Run AFTER ``--stage calibrate`` and ``--stage select`` and BEFORE ``--stage
evaluate``.  Reads only calibration artifacts (never evaluation data) and writes
``configs/frozen_parameters.json``: every selected operating point with the
reason it was chosen under the pre-registered rule (§5.3), SHA-256 hashes of
the calibration/selection artifacts, and the calibration completeness check.

Exit status is non-zero if any calibration check fails.

Usage:
    python scripts/phase3/freeze_params.py [--out results/phase3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from taorg.experiments.phase3.grids import expand_all
from taorg.experiments.phase3.pipeline import (
    _float_row, condition_specs, evaluation_plan, load_phase3_config, read_csv_gz,
)
from taorg.experiments.phase3.selection import TIE_TOL

RULE = ("Among configurations whose mean termination delay over all calibration instances "
        "(20 seeds x 6 gaps) is <= B, choose maximal mean Persistence F1; ties (|dF1| <= 1e-9) "
        "broken by lower churn, lower delay, lower false persistence, then grid order. If nothing "
        "is feasible, the minimum-delay configuration is used and flagged (prereg §5.3).")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_calibration(cfg: dict, out: Path, rows: list[dict]) -> dict:
    """Completeness / leakage checks on the calibration artifacts."""
    specs = condition_specs(cfg)
    calib_seeds = set(cfg["seeds"]["calibration"])
    eval_seeds = set(cfg["seeds"]["evaluation"])
    meta = json.loads((out / "raw" / "stage_calibrate.json").read_text(encoding="utf-8"))
    inst = json.loads((out / "raw" / "calibration_instances.json").read_text(encoding="utf-8"))
    n_configs = len(expand_all(cfg["grids"]))
    calibrated = [s for s in specs.values() if s.calibrate]
    expected_inst = {f"{s.name}|g{g}|s{seed}" for s in calibrated for g in s.gaps for seed in calib_seeds}
    actual_ids = [r["instance_id"] for r in inst]
    seen_seeds = {r["seed"] for r in inst}
    per_cond = {}
    for s in calibrated:
        rs = [r for r in rows if r["condition"] == s.name]
        per_cond[s.name] = {
            "n_config_rows": len(rs),
            "n_instances_min": min(int(r["n_instances"]) for r in rs) if rs else 0,
            "n_instances_max": max(int(r["n_instances"]) for r in rs) if rs else 0,
            "n_nan_f1": sum(math.isnan(r["persistence_f1"]) for r in rs),
            "n_nan_delay": sum(math.isnan(r["term_delay_mean"]) for r in rs),
        }
    expected_per_cond = len(calib_seeds) * 6
    checks = {
        "stage_not_smoke": meta.get("smoke") is False,
        "grid_stride_1": meta.get("grid_stride") == 1,
        "config_sha_matches": meta.get("config_sha256") == cfg["_sha256"],
        "n_configs_full_grid": meta.get("n_configs") == n_configs,
        "instances_expected_count": len(actual_ids) == len(expected_inst),
        "instances_no_duplicates": len(set(actual_ids)) == len(actual_ids),
        "instances_exact_match": set(actual_ids) == expected_inst,
        "only_calibration_seeds": seen_seeds <= calib_seeds,
        "no_evaluation_seeds": not (seen_seeds & eval_seeds),
        "all_config_rows_present": all(v["n_config_rows"] == n_configs for v in per_cond.values()),
        "every_config_saw_every_instance": all(
            v["n_instances_min"] == v["n_instances_max"] == expected_per_cond
            for v in per_cond.values()),
        "no_nan_f1_or_delay": all(v["n_nan_f1"] == 0 and v["n_nan_delay"] == 0
                                  for v in per_cond.values()),
    }
    return {"checks": checks, "all_passed": all(checks.values()),
            "expected_instances": len(expected_inst), "actual_instances": len(actual_ids),
            "missing_instances": sorted(expected_inst - set(actual_ids)),
            "calibration_seeds": sorted(calib_seeds), "seeds_seen": sorted(seen_seeds),
            "per_condition": per_cond, "calibration_run_id": meta.get("run_id"),
            "calibration_git_commit": meta.get("git_commit"),
            "calibration_git_dirty": meta.get("git_dirty"),
            "calibration_elapsed_s": meta.get("elapsed_s")}


def rationale(cond: str, method: str, b: str, sel: dict, summ: list[dict]) -> dict:
    """Explain one selection under the registered rule, recomputed from calibration rows."""
    budget = float(b)
    feas = [s for s in summ if not math.isnan(s["term_delay_mean"]) and s["term_delay_mean"] <= budget]
    chosen = next(s for s in summ if s["config_id"] == sel["config_id"])
    info = {"n_configs_searched": len(summ), "n_feasible": len(feas), "feasible": sel["feasible"],
            "calib_persistence_f1": chosen["persistence_f1"],
            "calib_term_delay_mean": chosen["term_delay_mean"],
            "calib_churn_mean": chosen["churn_mean"],
            "calib_false_pers_mean": chosen["false_pers_mean"]}
    if not feas:
        info["reason"] = (f"No configuration met mean delay <= {b}; minimum-delay configuration "
                          f"chosen (delay {chosen['term_delay_mean']:.3f}) and FLAGGED infeasible.")
        return info
    best = max(s["persistence_f1"] for s in feas)
    tied = [s for s in feas if s["persistence_f1"] >= best - TIE_TOL]
    below = sorted({round(s["persistence_f1"], 12) for s in feas if s["persistence_f1"] < best - TIE_TOL},
                   reverse=True)
    info.update({"best_feasible_f1": best, "n_tied_at_best_f1": len(tied),
                 "tie_break_decisive": len(tied) > 1,
                 "next_best_distinct_f1": below[0] if below else None,
                 "margin_to_next_f1": (best - below[0]) if below else None})
    if len(tied) > 1:
        info["reason"] = (f"{len(tied)} feasible configurations tied at max calibration F1 "
                          f"{best:.4f} (of {len(feas)} feasible, budget {b}); tie broken by the "
                          "registered order churn -> delay -> false persistence -> grid index.")
    else:
        info["reason"] = (f"Unique max calibration F1 {best:.4f} among {len(feas)} feasible "
                          f"configurations (mean delay <= {b}); next best F1 "
                          f"{below[0] if below else float('nan'):.4f}.")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/phase3_pilot.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3"))
    args = ap.parse_args(argv)
    cfg = load_phase3_config(args.config)
    out = args.out
    if (out / "raw" / "evaluation_runs.csv.gz").exists():
        print("ERROR: evaluation artifacts already exist; freezing must precede evaluation.")
        return 2
    summ_path = out / "raw" / "calibration_summary.csv.gz"
    rows = [_float_row(r) for r in read_csv_gz(summ_path)]
    verification = verify_calibration(cfg, out, rows)

    sel_path, par_path = out / "configs" / "selected_params.json", out / "configs" / "pareto_sets.json"
    selected = json.loads(sel_path.read_text(encoding="utf-8"))
    pareto = json.loads(par_path.read_text(encoding="utf-8"))
    by_cm: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cm.setdefault((r["condition"], r["method"]), []).append(r)

    frozen = {}
    for cond, by_m in selected.items():
        for method, by_b in by_m.items():
            for b, sel in by_b.items():
                entry = {"config_id": sel["config_id"], "params": sel["params"],
                         **rationale(cond, method, b, sel, by_cm[(cond, method)])}
                frozen.setdefault(cond, {}).setdefault(method, {})[b] = entry
    plan = evaluation_plan(cfg, selected, pareto)
    artifact = {
        "description": "Phase 3 frozen parameters (selected on calibration seeds only).",
        "registration_commit": "d8728b3",
        "config_path": cfg["_source_path"], "config_sha256": cfg["_sha256"],
        "selection_rule": RULE, "tie_tolerance": TIE_TOL,
        "budgets_frames": cfg["budgets_frames"],
        "artifact_sha256": {
            "raw/calibration_summary.csv.gz": sha256(summ_path),
            "raw/calibration_instances.json": sha256(out / "raw" / "calibration_instances.json"),
            "configs/selected_params.json": sha256(sel_path),
            "configs/pareto_sets.json": sha256(par_path),
        },
        "calibration_verification": verification,
        "pareto_set_sizes": {c: {m: len(v) for m, v in d.items()} for c, d in pareto.items()},
        "evaluation_plan_config_counts": {c: len(v) for c, v in plan.items()},
        "stress_conditions_use": {s.name: s.tuned_on for s in condition_specs(cfg).values()
                                  if not s.calibrate},
        "frozen": frozen,
    }
    path = out / "configs" / "frozen_parameters.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # Human-readable table.
    labels = cfg["method_labels"]
    lines = ["<!-- GENERATED by scripts/phase3/freeze_params.py — do not edit by hand. -->\n",
             "# Phase 3 frozen parameters (calibration seeds 1000–1019 only)\n",
             f"Selection rule: {RULE}\n",
             "| Tuning condition | Method | Budget | Params | Calib F1 | Calib delay | Feasible (n) | Ties | Reason |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cond, by_m in frozen.items():
        for method in ("no_memory", "fixed_window", "ema", "gated_ema", "hmm"):
            for b, e in by_m.get(method, {}).items():
                lines.append(
                    f"| {cond} | {labels[method]} | {b} | `{json.dumps(e['params'], sort_keys=True)}` | "
                    f"{e['calib_persistence_f1']:.4f} | {e['calib_term_delay_mean']:.3f} | "
                    f"{e['feasible']} ({e['n_feasible']}) | {e.get('n_tied_at_best_f1', '—')} | "
                    f"{e['reason']} |")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "frozen_parameters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(verification["checks"], indent=1))
    print("calibration verification:", "PASS" if verification["all_passed"] else "FAIL")
    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
