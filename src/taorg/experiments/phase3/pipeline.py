"""Phase 3 pipeline stages: calibrate -> select -> evaluate -> latency.

Analysis, plotting and failure extraction live in ``analysis.py``,
``plots.py`` and ``failures.py``; all of them read only the artifacts written
here, so every table and figure is reproducible from saved files.

Artifact layout (``out_dir``):
    configs/   resolved config, selected parameters, Pareto sets
    raw/       calibration summaries, per-(config, instance) evaluation rows,
               manifests (gzipped JSONL), latency rows, stage metadata
    aggregated/, tables/, plots/, failures/   (written by later stages)
"""

from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import io
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from taorg.experiments.phase3.conditions import ConditionSpec, Instance, build_instance
from taorg.experiments.phase3.evaluation import METRIC_FIELDS, evaluate_config, evaluate_latency
from taorg.experiments.phase3.grids import MethodConfig, expand_all
from taorg.experiments.phase3.selection import pareto_front, select_under_budget
from taorg.utils.git import get_git_branch, get_git_commit, get_git_dirty

logger = logging.getLogger(__name__)

SUMMARY_METRICS = (
    "persistence_f1", "persistence_recall", "persistence_precision",
    "term_delay_mean", "false_pers_mean", "churn_mean",
    "diag_clean_alive_recall", "diag_spurious_rate", "diag_recovery_lag",
)
PER_GAP_METRICS = ("persistence_f1", "term_delay_mean", "false_pers_mean", "churn_mean")


# ---------------------------------------------------------------------------
# Config / provenance helpers
# ---------------------------------------------------------------------------

def load_phase3_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the Phase 3 experiment config."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("world", "seeds", "occlusion", "gap_lengths", "budgets_frames",
                "conditions", "grids", "statistics", "h1_criteria"):
        if key not in cfg:
            raise KeyError(f"Phase 3 config {path} is missing required key '{key}'")
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def condition_specs(cfg: dict[str, Any]) -> dict[str, ConditionSpec]:
    """Parse all condition specs, validating tuned_on references."""
    specs = {name: ConditionSpec.from_dict(name, d) for name, d in cfg["conditions"].items()}
    for s in specs.values():
        if not s.calibrate and s.tuned_on not in specs:
            raise KeyError(f"Condition {s.name}: tuned_on '{s.tuned_on}' is not defined")
    return specs


def environment_info() -> dict[str, str]:
    """Collect environment information for provenance."""
    import matplotlib
    import scipy
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "cpu_count": str(os.cpu_count()),
        "machine": platform.machine(),
    }


def new_stage_metadata(stage: str, cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """Create a provenance record for one pipeline stage."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "run_id": f"phase3_{stage}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "stage": stage, "smoke": smoke,
        "start_time_utc": now.isoformat(),
        "git_commit": get_git_commit(), "git_dirty": get_git_dirty(),
        "git_branch": get_git_branch(),
        "command": " ".join(sys.argv),
        "config_path": cfg.get("_source_path"), "config_sha256": cfg.get("_sha256"),
        "environment": environment_info(),
    }


def finish_stage_metadata(meta: dict[str, Any], out_dir: Path, extra: dict | None = None) -> None:
    """Complete and save a stage provenance record."""
    meta["end_time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if extra:
        meta.update(extra)
    path = out_dir / "raw" / f"stage_{meta['stage']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _write_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    """Read a gzipped CSV written by this pipeline."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _append_jsonl_gz(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    for r in records:
        buf.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(buf.getvalue())


def _seeds(cfg: dict[str, Any], split: str, smoke: bool) -> list[int]:
    return list(cfg["seeds"]["smoke"] if smoke else cfg["seeds"][split])


def _quiet_worker() -> None:
    """Silence per-call INFO logs and the intended GatedEMA 'gating disabled' warning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("taorg").setLevel(logging.WARNING)
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)


def _pool(workers: int):
    return mp.get_context("spawn").Pool(workers, initializer=_quiet_worker)


# ---------------------------------------------------------------------------
# Stage 1: calibration
# ---------------------------------------------------------------------------

def _calibration_task(args: tuple) -> tuple[str, int, int, str, str, np.ndarray]:
    """Evaluate every configuration on ONE calibration instance."""
    spec, gap, seed, world, occ, configs = args
    inst = build_instance(spec, gap, seed, world, occ)
    mat = np.empty((len(configs), len(METRIC_FIELDS)), dtype=np.float64)
    for i, (method, params_json) in enumerate(configs):
        metrics, _ = evaluate_config(inst, method, json.loads(params_json))
        mat[i] = [metrics[k] for k in METRIC_FIELDS]
    return spec.name, gap, seed, inst.instance_id, inst.manifest_sha256, mat


def run_calibration(
    cfg: dict[str, Any], out_dir: Path, workers: int, smoke: bool = False,
    grid_stride: int = 1,
) -> Path:
    """Evaluate every grid configuration on calibration instances.

    Writes ``raw/calibration_summary.csv.gz``: one row per (condition, config)
    with means over all calibration instances (the selection objective) and
    per-gap means of the primary metrics.
    """
    meta = new_stage_metadata("calibrate", cfg, smoke)
    specs = condition_specs(cfg)
    configs = expand_all(cfg["grids"])
    if grid_stride > 1:
        configs = [c for c in configs if c.grid_index % grid_stride == 0]
    cfg_tuples = [(c.method, c.params_json) for c in configs]
    seeds = _seeds(cfg, "calibration", smoke)
    fields_idx = {k: i for i, k in enumerate(METRIC_FIELDS)}

    tasks = []
    for spec in specs.values():
        if not spec.calibrate:
            continue
        for gap in spec.gaps:
            for seed in seeds:
                tasks.append((spec, gap, seed, cfg["world"], cfg["occlusion"], cfg_tuples))
    logger.info("Calibration: %d instances x %d configs", len(tasks), len(configs))

    # Accumulators: per condition, overall and per gap (nansum + count).
    # pool.imap (ordered) keeps the floating-point summation order independent
    # of worker scheduling, so reruns are bit-identical.
    acc: dict[tuple[str, Any], list[np.ndarray]] = {}
    manifest_records = []
    t0 = time.time()
    with _pool(workers) as pool:
        for k, (cond, gap, seed, iid, sha, mat) in enumerate(
                pool.imap(_calibration_task, tasks, chunksize=1)):
            for key in ((cond, "all"), (cond, gap)):
                if key not in acc:
                    acc[key] = [np.zeros_like(mat), np.zeros_like(mat)]
                s, c = acc[key]
                ok = ~np.isnan(mat)
                s += np.where(ok, mat, 0.0)
                c += ok
            manifest_records.append({"instance_id": iid, "split": "calibration",
                                     "condition": cond, "gap": gap, "seed": seed,
                                     "manifest_sha256": sha})
            if (k + 1) % 50 == 0:
                logger.info("  %d/%d calibration instances (%.0fs)", k + 1, len(tasks), time.time() - t0)

    rows = []
    for spec in specs.values():
        if not spec.calibrate:
            continue
        s, c = acc[(spec.name, "all")]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_all = s / c
            per_gap = {g: acc[(spec.name, g)][0] / acc[(spec.name, g)][1] for g in spec.gaps}
        for i, mc in enumerate(configs):
            row = {"condition": spec.name, "config_id": mc.config_id, "method": mc.method,
                   "grid_index": mc.grid_index, "params_json": mc.params_json,
                   "n_instances": int(c[i, fields_idx["churn_mean"]])}
            for m in SUMMARY_METRICS:
                row[m] = float(mean_all[i, fields_idx[m]])
            for g in spec.gaps:
                for m in PER_GAP_METRICS:
                    row[f"g{g}_{m}"] = float(per_gap[g][i, fields_idx[m]])
            rows.append(row)
    all_gaps = sorted({g for s in specs.values() if s.calibrate for g in s.gaps})
    fields = (["condition", "config_id", "method", "grid_index", "params_json", "n_instances"]
              + list(SUMMARY_METRICS)
              + [f"g{g}_{m}" for g in all_gaps for m in PER_GAP_METRICS])
    for r in rows:
        for f in fields:
            r.setdefault(f, math.nan)
    path = out_dir / "raw" / "calibration_summary.csv.gz"
    _write_csv_gz(path, rows, fields)
    manifest_records.sort(key=lambda r: r["instance_id"])
    (out_dir / "raw" / "calibration_instances.json").write_text(
        json.dumps(manifest_records, indent=1), encoding="utf-8")
    finish_stage_metadata(meta, out_dir, {
        "n_instances": len(tasks), "n_configs": len(configs), "seeds": seeds,
        "grid_stride": grid_stride, "elapsed_s": time.time() - t0,
    })
    return path


# ---------------------------------------------------------------------------
# Stage 2: selection
# ---------------------------------------------------------------------------

def _float_row(r: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(r)
    for k in SUMMARY_METRICS:
        out[k] = float(r[k])
    out["grid_index"] = int(r["grid_index"])
    return out


def run_selection(cfg: dict[str, Any], out_dir: Path, smoke: bool = False) -> Path:
    """Select operating points per (condition, method, budget) + Pareto sets.

    Writes ``configs/selected_params.json`` and ``configs/pareto_sets.json``.
    """
    meta = new_stage_metadata("select", cfg, smoke)
    rows = [_float_row(r) for r in read_csv_gz(out_dir / "raw" / "calibration_summary.csv.gz")]
    by_cm: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cm.setdefault((r["condition"], r["method"]), []).append(r)
    params_of = {(r["condition"], r["config_id"]): r["params_json"] for r in rows}

    selected: dict[str, Any] = {}
    pareto: dict[str, Any] = {}
    for (cond, method), summ in sorted(by_cm.items()):
        for b in cfg["budgets_frames"]:
            sel = select_under_budget(summ, float(b))
            sel["params"] = json.loads(params_of[(cond, sel["config_id"])])
            sel["method"] = method
            selected.setdefault(cond, {}).setdefault(method, {})[str(b)] = sel
        front = pareto_front(summ)
        pareto.setdefault(cond, {})[method] = [
            {"config_id": cid, "params": json.loads(params_of[(cond, cid)])} for cid in front
        ]
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    path = out_dir / "configs" / "selected_params.json"
    path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (out_dir / "configs" / "pareto_sets.json").write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    finish_stage_metadata(meta, out_dir)
    return path


# ---------------------------------------------------------------------------
# Stage 3: evaluation
# ---------------------------------------------------------------------------

def evaluation_plan(
    cfg: dict[str, Any], selected: dict[str, Any], pareto: dict[str, Any]
) -> dict[str, dict[str, MethodConfig]]:
    """Configurations to evaluate per condition (deduplicated by config_id).

    - every condition: operating points selected on its tuning source
      (itself if calibrated, else ``tuned_on``) for every method and budget;
    - H1 conditions: additionally their own calibration Pareto sets;
    - H1 robustness conditions: additionally the C0-selected operating points
      (secondary "mismatched tuning" analysis).
    """
    specs = condition_specs(cfg)
    primary = [s.name for s in specs.values() if s.role == "h1_primary"]
    grid_index = {c.config_id: c for c in expand_all(cfg["grids"])}
    plan: dict[str, dict[str, MethodConfig]] = {}
    for spec in specs.values():
        chosen: dict[str, MethodConfig] = {}
        sources = [spec.name if spec.calibrate else spec.tuned_on]
        if spec.role == "h1_robustness":
            sources += primary
        for src in sources:
            for method, by_budget in selected[src].items():
                for sel in by_budget.values():
                    chosen[sel["config_id"]] = grid_index[sel["config_id"]]
        if spec.role in ("h1_primary", "h1_robustness"):
            for method, items in pareto[spec.name].items():
                for it in items:
                    chosen[it["config_id"]] = grid_index[it["config_id"]]
        plan[spec.name] = dict(sorted(chosen.items()))
    return plan


def _evaluation_task(args: tuple) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec, gap, seed, world, occ, configs = args
    inst = build_instance(spec, gap, seed, world, occ)
    rows = []
    for mc in configs:
        metrics, _ = evaluate_config(inst, mc.method, mc.params)
        row = {"condition": spec.name, "role": spec.role, "gap": gap, "seed": seed,
               "instance_id": inst.instance_id, "manifest_sha256": inst.manifest_sha256,
               "config_id": mc.config_id, "method": mc.method, "params_json": mc.params_json}
        row.update(metrics)
        rows.append(row)
    manifest = {"instance_id": inst.instance_id, "split": "evaluation", "condition": spec.name,
                "gap": gap, "seed": seed, "manifest_sha256": inst.manifest_sha256,
                "lifetimes": [
                    {"edge_key": list(lt.edge_key), "start": lt.start_frame, "end": lt.end_frame,
                     "terminates": lt.terminates, "score_mean": lt.score_mean,
                     "post_termination_occluded_frames": lt.post_termination_occluded_frames}
                    for lt in inst.lifetimes],
                "manifests": inst.manifests}
    return rows, manifest


EVAL_ID_FIELDS = ["run_id", "condition", "role", "gap", "seed", "instance_id", "manifest_sha256",
                  "config_id", "method", "params_json"]


def run_evaluation(cfg: dict[str, Any], out_dir: Path, workers: int, smoke: bool = False) -> Path:
    """Evaluate the planned configurations on held-out evaluation instances.

    Writes ``raw/evaluation_runs.csv.gz`` (one row per config x instance) and
    ``raw/manifests_evaluation.jsonl.gz`` (every corruption manifest + GT lifetimes).
    """
    meta = new_stage_metadata("evaluate", cfg, smoke)
    selected = json.loads((out_dir / "configs" / "selected_params.json").read_text(encoding="utf-8"))
    pareto = json.loads((out_dir / "configs" / "pareto_sets.json").read_text(encoding="utf-8"))
    plan = evaluation_plan(cfg, selected, pareto)
    (out_dir / "configs" / "evaluation_plan.json").write_text(
        json.dumps({c: list(v) for c, v in plan.items()}, indent=1), encoding="utf-8")
    specs = condition_specs(cfg)
    seeds = _seeds(cfg, "evaluation", smoke)
    tasks = [(spec, gap, seed, cfg["world"], cfg["occlusion"], list(plan[spec.name].values()))
             for spec in specs.values() for gap in spec.gaps for seed in seeds]
    logger.info("Evaluation: %d instances, %d config-instance runs", len(tasks),
                sum(len(t[-1]) for t in tasks))
    manifest_path = out_dir / "raw" / "manifests_evaluation.jsonl.gz"
    if manifest_path.exists():
        manifest_path.unlink()
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    t0 = time.time()
    with _pool(workers) as pool:
        for k, (r, m) in enumerate(pool.imap(_evaluation_task, tasks, chunksize=1)):
            rows.extend(r)
            manifests.append(m)
            if (k + 1) % 200 == 0:
                logger.info("  %d/%d evaluation instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    for r in rows:
        r["run_id"] = meta["run_id"]
    rows.sort(key=lambda r: (r["condition"], r["gap"], r["seed"], r["config_id"]))
    manifests.sort(key=lambda m: m["instance_id"])
    _append_jsonl_gz(manifest_path, manifests)
    path = out_dir / "raw" / "evaluation_runs.csv.gz"
    _write_csv_gz(path, rows, EVAL_ID_FIELDS + list(METRIC_FIELDS))
    finish_stage_metadata(meta, out_dir, {
        "n_instances": len(tasks), "n_rows": len(rows), "seeds": seeds,
        "elapsed_s": time.time() - t0,
    })
    return path


# ---------------------------------------------------------------------------
# Stage 4: latency (serial, to avoid contention between workers)
# ---------------------------------------------------------------------------

def run_latency(cfg: dict[str, Any], out_dir: Path, smoke: bool = False) -> Path:
    """Per-frame latency of the primary-budget operating points on C0 instances."""
    _quiet_worker()
    meta = new_stage_metadata("latency", cfg, smoke)
    selected = json.loads((out_dir / "configs" / "selected_params.json").read_text(encoding="utf-8"))
    specs = condition_specs(cfg)
    primary = next(s for s in specs.values() if s.role == "h1_primary")
    b = str(cfg["primary_budget_frames"])
    seeds = _seeds(cfg, "evaluation", smoke)
    rows = []
    for gap in primary.gaps:
        for seed in seeds:
            inst: Instance = build_instance(primary, gap, seed, cfg["world"], cfg["occlusion"])
            for method, by_budget in selected[primary.name].items():
                sel = by_budget[b]
                lat = evaluate_latency(inst, method, sel["params"])
                rows.append({"run_id": meta["run_id"], "condition": primary.name, "gap": gap,
                             "seed": seed, "method": method, "config_id": sel["config_id"],
                             "budget": b, **lat})
    path = out_dir / "raw" / "latency.csv.gz"
    _write_csv_gz(path, rows, list(rows[0].keys()))
    finish_stage_metadata(meta, out_dir, {"n_rows": len(rows)})
    return path


def save_resolved_config(cfg: dict[str, Any], out_dir: Path) -> None:
    """Copy the resolved config into the artifact tree."""
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out_dir / "configs" / "phase3_pilot.resolved.yaml").write_text(
        yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    (out_dir / "configs" / "config_sha256.txt").write_text(
        f"{cfg['_sha256']}  {cfg['_source_path']}\n", encoding="utf-8")
