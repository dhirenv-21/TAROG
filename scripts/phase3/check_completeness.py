"""Phase 3 — explicit completeness check of the evaluation experiment matrix.

Run AFTER ``--stage evaluate`` / ``--stage latency`` and BEFORE ``--stage
analyze``.  Compares the expected matrix (frozen evaluation plan x gaps x
evaluation seeds) with the saved rows and writes
``aggregated/completeness.json`` + ``tables/completeness.md``.

Reported per condition: expected runs, actual runs, missing, duplicate, failed
(metric that must always be defined is NaN), undefined-by-design (e.g. no
occlusion interval at gap 0, or no terminating edge in a video) — the latter
are not failures but are counted so nothing is silently ignored.

Exit status is non-zero if anything is missing, duplicated or failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from taorg.experiments.phase3.pipeline import condition_specs, load_phase3_config, read_csv_gz


def _nan(x: str) -> bool:
    try:
        return math.isnan(float(x))
    except ValueError:
        return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/phase3_pilot.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3"))
    args = ap.parse_args(argv)
    cfg = load_phase3_config(args.config)
    out = args.out
    specs = condition_specs(cfg)
    seeds = list(cfg["seeds"]["evaluation"])
    plan = json.loads((out / "configs" / "evaluation_plan.json").read_text(encoding="utf-8"))
    rows = read_csv_gz(out / "raw" / "evaluation_runs.csv.gz")
    meta = json.loads((out / "raw" / "stage_evaluate.json").read_text(encoding="utf-8"))

    keys = Counter((r["condition"], int(r["gap"]), int(r["seed"]), r["config_id"]) for r in rows)
    report: dict[str, dict] = {}
    totals = Counter()
    for spec in specs.values():
        expected = {(spec.name, g, s, c) for g in spec.gaps for s in seeds for c in plan[spec.name]}
        actual = {k for k in keys if k[0] == spec.name}
        rs = [r for r in rows if r["condition"] == spec.name]
        failed = [r for r in rs if _nan(r["churn_mean"]) or _nan(r["method_runtime_s"])
                  or (int(r["gap"]) > 0 and _nan(r["persistence_f1"]))]
        undefined_f1 = sum(int(r["gap"]) == 0 and _nan(r["persistence_f1"]) for r in rs)
        undefined_delay = sum(_nan(r["term_delay_mean"]) for r in rs)
        rep = {
            "gaps": list(spec.gaps), "n_configs": len(plan[spec.name]), "n_seeds": len(seeds),
            "expected": len(expected), "actual": len(rs),
            "missing": len(expected - actual), "unexpected": len(actual - expected),
            "duplicates": sum(v - 1 for k, v in keys.items() if k[0] == spec.name and v > 1),
            "failed": len(failed),
            "undefined_f1_by_design_gap0": undefined_f1,
            "undefined_delay_no_termination": undefined_delay,
            "missing_examples": [list(k) for k in sorted(expected - actual)[:5]],
        }
        report[spec.name] = rep
        for k in ("expected", "actual", "missing", "unexpected", "duplicates", "failed"):
            totals[k] += rep[k]

    seeds_seen = sorted({int(r["seed"]) for r in rows})
    calib_leak = sorted(set(seeds_seen) & set(cfg["seeds"]["calibration"]))
    run_ids = sorted({r["run_id"] for r in rows})
    manifests_n = 0
    import gzip
    with gzip.open(out / "raw" / "manifests_evaluation.jsonl.gz", "rt", encoding="utf-8") as f:
        manifest_ids = [json.loads(line)["instance_id"] for line in f]
    manifests_n = len(manifest_ids)
    expected_instances = sum(len(s.gaps) * len(seeds) for s in specs.values())

    lat_rows = []
    lat_path = out / "raw" / "latency.csv.gz"
    if lat_path.exists():
        lat_rows = read_csv_gz(lat_path)
    primary = next(s for s in specs.values() if s.role == "h1_primary")
    lat_expected = len(primary.gaps) * len(seeds) * len(cfg["grids"])

    ok = (totals["missing"] == 0 and totals["duplicates"] == 0 and totals["failed"] == 0
          and totals["unexpected"] == 0 and not calib_leak and seeds_seen == sorted(seeds)
          and manifests_n == expected_instances == len(set(manifest_ids))
          and len(run_ids) == 1 and meta.get("smoke") is False and len(lat_rows) == lat_expected)
    result = {"all_complete": ok, "totals": dict(totals), "per_condition": report,
              "eval_seeds_expected": seeds, "eval_seeds_seen": seeds_seen,
              "calibration_seed_leak": calib_leak, "evaluation_run_ids": run_ids,
              "evaluation_git_commit": meta.get("git_commit"),
              "evaluation_git_dirty": meta.get("git_dirty"),
              "manifests": {"expected_instances": expected_instances, "actual": manifests_n,
                            "unique": len(set(manifest_ids))},
              "latency_rows": {"expected": lat_expected, "actual": len(lat_rows)}}
    (out / "aggregated").mkdir(parents=True, exist_ok=True)
    (out / "aggregated" / "completeness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["<!-- GENERATED by scripts/phase3/check_completeness.py — do not edit by hand. -->\n",
             "# Phase 3 evaluation completeness\n",
             f"Overall: **{'COMPLETE' if ok else 'INCOMPLETE'}** — evaluation run(s) {', '.join(run_ids)}; "
             f"seeds {seeds[0]}–{seeds[-1]} (n={len(seeds)}); calibration-seed leak: {calib_leak or 'none'}; "
             f"manifests {manifests_n}/{expected_instances}; latency rows "
             f"{len(lat_rows)}/{lat_expected}.\n",
             "| Condition | Gaps | Configs | Expected runs | Actual | Missing | Duplicate | Failed | "
             "F1 undefined (gap 0, by design) | Delay undefined (no termination in video) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        lines.append(f"| {name} | {','.join(map(str, r['gaps']))} | {r['n_configs']} | {r['expected']} | "
                     f"{r['actual']} | {r['missing']} | {r['duplicates']} | {r['failed']} | "
                     f"{r['undefined_f1_by_design_gap0']} | {r['undefined_delay_no_termination']} |")
    lines.append(f"| **Total** | | | {totals['expected']} | {totals['actual']} | {totals['missing']} | "
                 f"{totals['duplicates']} | {totals['failed']} | | |")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "completeness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
