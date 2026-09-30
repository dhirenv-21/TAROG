"""Phase 4 pipeline stages: calibrate -> select -> evaluate -> latency.

Methods are processed in two registered GROUPS (``method_groups`` in the
config): ``baseline`` (No Memory ... VA-HMM) is calibrated, frozen and
evaluated BEFORE ``proposed`` (V-HPED rungs).  Selection is per method, so the
split cannot change any selection.  Artifacts are suffixed by group.

Metric computation reuses ``taorg.experiments.phase3.evaluation`` unchanged
(frozen metrics + diagnostics).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from taorg.experiments.phase3.evaluation import METRIC_FIELDS, evaluate_config, evaluate_latency
from taorg.experiments.phase3.grids import MethodConfig
from taorg.experiments.phase3.pipeline import (
    _append_jsonl_gz, _write_csv_gz, finish_stage_metadata, new_stage_metadata, read_csv_gz,
)
from taorg.experiments.phase3.selection import pareto_front, select_under_budget
from taorg.experiments.phase4.conditions import (
    P4ConditionSpec, build_instance, exclusivity_matrix, world_config,
)
from taorg.data.synthetic_world import sample_lifetimes
from taorg.experiments.phase4.search import method_configs
from taorg.experiments.phase3.conditions import derive_seed

logger = logging.getLogger(__name__)

SUMMARY_FIELDS = tuple(METRIC_FIELDS)
PER_GAP = ("persistence_f1", "term_delay_mean", "diag_swap_recall", "diag_ghost_edge_frames")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load the Phase 4 config and record its source and SHA-256."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("world", "seeds", "occlusion", "gap_lengths", "conditions", "search",
                "method_groups", "criteria", "statistics", "frontiers"):
        if key not in cfg:
            raise KeyError(f"Phase 4 config is missing '{key}'")
    cfg["predicate_partners"] = {int(k): int(v) for k, v in cfg["predicate_partners"].items()}
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def specs(cfg: dict[str, Any]) -> dict[str, P4ConditionSpec]:
    """Parse and validate all condition specs."""
    out = {n: P4ConditionSpec.from_dict(n, d, cfg["gap_lengths"]) for n, d in cfg["conditions"].items()}
    for s in out.values():
        if not s.calibrate and s.tuned_on not in out:
            raise KeyError(f"{s.name}: tuned_on '{s.tuned_on}' undefined")
    return out


def methods_for(cfg: dict[str, Any], group: str, condition: str) -> list[str]:
    """Methods of a group that run on a condition (M6 only on its registered conditions)."""
    ms = list(cfg["method_groups"][group])
    if condition not in cfg["m6_conditions"]:
        ms = [m for m in ms if m != "vhped_m6"]
    return ms


def frontier_keys(cfg: dict[str, Any], condition: str) -> list[str]:
    """Registered frontier types of a condition."""
    return ["default", "identity"] if condition in cfg["identity_conditions"] else ["default"]


def model_params(method: str, params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Constructor kwargs: sampled params + condition context (M6 exclusivity)."""
    if method == "vhped_m6":
        return {**params, "exclusivity": context["exclusivity"]}
    return params


def _seeds(cfg: dict[str, Any], split: str, smoke: bool) -> list[int]:
    return list(cfg["seeds"]["smoke"] if smoke else cfg["seeds"][split])


def _quiet_worker() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("taorg").setLevel(logging.WARNING)
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)


def _pool(workers: int):
    return mp.get_context("spawn").Pool(workers, initializer=_quiet_worker)


def _build(cfg: dict[str, Any], spec: P4ConditionSpec, gap: int, seed: int):
    return build_instance(spec, gap, seed, cfg["world"], cfg["occlusion"],
                          float(cfg["bbox_jitter_std"]), cfg["predicate_partners"])


# ---------------------------------------------------------------------------
# Context: exclusivity statistic from calibration ground truth
# ---------------------------------------------------------------------------

def compute_context(cfg: dict[str, Any], out_dir: Path, smoke: bool = False) -> dict[str, Any]:
    """Exclusivity matrix E per calibrated condition from CALIBRATION GT lifetimes.

    Succession is re-derived exactly as in ``build_instance`` (no corruption
    is needed to obtain ground truth).
    """
    from taorg.experiments.phase4.conditions import add_successors
    ctx: dict[str, Any] = {}
    npred = int(cfg["world"]["num_predicates"])
    for spec in specs(cfg).values():
        if not spec.calibrate:
            continue
        wcfg = world_config(cfg["world"], spec)
        sets = []
        for seed in _seeds(cfg, "calibration", smoke):
            lts = sample_lifetimes(seed, wcfg)
            if spec.successor_prob > 0:
                lts, _ = add_successors(lts, seed, wcfg, spec.successor_prob, cfg["predicate_partners"])
            sets.append(lts)
        ctx[spec.name] = {"exclusivity": exclusivity_matrix(sets, npred)}
    for spec in specs(cfg).values():
        if not spec.calibrate:
            ctx[spec.name] = ctx[spec.tuning_source]
    path = out_dir / "configs" / "context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx, indent=1), encoding="utf-8")
    return ctx


def load_context(out_dir: Path) -> dict[str, Any]:
    """Load the saved condition context."""
    return json.loads((out_dir / "configs" / "context.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 1: calibration
# ---------------------------------------------------------------------------

def _calib_task(args: tuple) -> tuple[str, int, int, str, str, np.ndarray]:
    cfg, spec, gap, seed, configs, context = args
    inst = _build(cfg, spec, gap, seed)
    mat = np.empty((len(configs), len(SUMMARY_FIELDS)))
    for i, (method, pj) in enumerate(configs):
        m, _ = evaluate_config(inst, method, model_params(method, json.loads(pj), context))
        mat[i] = [m[k] for k in SUMMARY_FIELDS]
    return spec.name, gap, seed, inst.instance_id, inst.manifest_sha256, mat


def _slim_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in ("world", "occlusion", "bbox_jitter_std", "predicate_partners")}


def run_calibration(cfg: dict[str, Any], out_dir: Path, group: str, workers: int,
                    smoke: bool = False, max_samples: int | None = None) -> Path:
    """Evaluate every configuration of a method group on calibration instances."""
    meta = new_stage_metadata(f"calibrate_{group}", cfg, smoke)
    meta["config_sha256"], meta["config_path"] = cfg["_sha256"], cfg["_source_path"]
    ctx = compute_context(cfg, out_dir, smoke) if group == "baseline" or not (
        out_dir / "configs" / "context.json").exists() else load_context(out_dir)
    sp = specs(cfg)
    seeds = _seeds(cfg, "calibration", smoke)
    cond_cfgs: dict[str, list[MethodConfig]] = {}
    tasks = []
    slim = _slim_cfg(cfg)
    for spec in sp.values():
        if not spec.calibrate:
            continue
        mcs = method_configs(cfg, methods_for(cfg, group, spec.name))
        if max_samples is not None:
            mcs = [c for c in mcs if c.grid_index < max_samples]
        cond_cfgs[spec.name] = mcs
        tup = [(c.method, c.params_json) for c in mcs]
        for gap in spec.gaps:
            for seed in seeds:
                tasks.append((slim, spec, gap, seed, tup, ctx[spec.name]))
    logger.info("Calibration[%s]: %d instances, %d config-instance runs", group, len(tasks),
                sum(len(t[4]) for t in tasks))
    acc: dict[tuple[str, Any], list[np.ndarray]] = {}
    records = []
    t0 = time.time()
    with _pool(workers) as pool:
        for k, (cond, gap, seed, iid, sha, mat) in enumerate(pool.imap(_calib_task, tasks, chunksize=1)):
            for key in ((cond, "all"), (cond, gap)):
                if key not in acc:
                    acc[key] = [np.zeros_like(mat), np.zeros_like(mat)]
                ok = ~np.isnan(mat)
                acc[key][0] += np.where(ok, mat, 0.0)
                acc[key][1] += ok
            records.append({"instance_id": iid, "split": "calibration", "condition": cond,
                            "gap": gap, "seed": seed, "manifest_sha256": sha})
            if (k + 1) % 20 == 0:
                logger.info("  %d/%d calibration instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    rows = []
    idx = {f: i for i, f in enumerate(SUMMARY_FIELDS)}
    for cond, mcs in cond_cfgs.items():
        s, c = acc[(cond, "all")]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / c
            pg = {g: acc[(cond, g)][0] / acc[(cond, g)][1] for g in sp[cond].gaps}
        for i, mc in enumerate(mcs):
            r = {"condition": cond, "config_id": mc.config_id, "method": mc.method,
                 "grid_index": mc.grid_index, "params_json": mc.params_json,
                 "n_instances": int(c[i, idx["churn_mean"]])}
            for f in SUMMARY_FIELDS:
                r[f] = float(mean[i, idx[f]])
            for g in sp[cond].gaps:
                for f in PER_GAP:
                    r[f"g{g}_{f}"] = float(pg[g][i, idx[f]])
            rows.append(r)
    gaps_all = sorted({g for s_ in sp.values() if s_.calibrate for g in s_.gaps})
    fields = (["condition", "config_id", "method", "grid_index", "params_json", "n_instances"]
              + list(SUMMARY_FIELDS) + [f"g{g}_{f}" for g in gaps_all for f in PER_GAP])
    for r in rows:
        for f in fields:
            r.setdefault(f, math.nan)
    path = out_dir / "raw" / f"calibration_{group}.csv.gz"
    _write_csv_gz(path, rows, fields)
    records.sort(key=lambda r: r["instance_id"])
    (out_dir / "raw" / f"calibration_instances_{group}.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8")
    finish_stage_metadata(meta, out_dir, {"group": group, "n_instances": len(tasks),
                                          "n_config_rows": len(rows), "seeds": seeds,
                                          "max_samples": max_samples, "elapsed_s": time.time() - t0})
    return path


# ---------------------------------------------------------------------------
# Stage 2: selection
# ---------------------------------------------------------------------------

def _float_row(r: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(r)
    for k in SUMMARY_FIELDS:
        out[k] = float(r[k])
    out["grid_index"] = int(r["grid_index"])
    return out


def pareto_generic(summ: list[dict[str, Any]], q: str, c: str) -> list[str]:
    """Calibration Pareto set for quality ``q`` (max) vs cost ``c`` (min)."""
    remap = [{**s, "persistence_f1": s[q], "term_delay_mean": s[c]} for s in summ]
    return pareto_front(remap)


def select_group(cfg: dict[str, Any], rows: list[dict[str, Any]], half_for: tuple[str, ...] = ()
                 ) -> tuple[dict, dict]:
    """Operating points and Pareto sets per (condition, method).

    ``half_for``: methods for which the pre-registered tuning-capacity control
    (selection from a seeded random subset of samples) is also computed and
    stored under the key ``<method>__half``.
    """
    by_cm: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_cm.setdefault((r["condition"], r["method"]), []).append(r)
    selected: dict[str, Any] = {}
    pareto: dict[str, Any] = {}
    frac = float(cfg["criteria"]["h2"]["tuning_control_fraction"])
    for (cond, method), summ in sorted(by_cm.items()):
        variants = [(method, summ)]
        if method in half_for:
            ids = sorted(s["grid_index"] for s in summ)
            rng = random.Random(derive_seed(int(cfg["seeds"]["sampler"]), "tuning_control", method))
            keep = set(rng.sample(ids, int(round(frac * len(ids)))))
            variants.append((f"{method}__half", [s for s in summ if s["grid_index"] in keep]))
        for name, ss in variants:
            for b in cfg["budgets_frames"]:
                sel = select_under_budget(ss, float(b))
                row = next(s for s in ss if s["config_id"] == sel["config_id"])
                sel.update(params=json.loads(row["params_json"]), method=method)
                selected.setdefault(cond, {}).setdefault(name, {})[str(b)] = sel
            for fk in frontier_keys(cfg, cond):
                f = cfg["frontiers"][fk]
                ids = pareto_generic(ss, f["quality"], f["cost"])
                pareto.setdefault(cond, {}).setdefault(fk, {})[name] = ids
    return selected, pareto


def run_selection(cfg: dict[str, Any], out_dir: Path, group: str, smoke: bool = False) -> None:
    """Select operating points / Pareto sets from calibration artifacts only."""
    meta = new_stage_metadata(f"select_{group}", cfg, smoke)
    meta["config_sha256"] = cfg["_sha256"]
    rows = [_float_row(r) for r in read_csv_gz(out_dir / "raw" / f"calibration_{group}.csv.gz")]
    half = (cfg["criteria"]["h2"]["treatment"],) if group == "proposed" else ()
    selected, pareto = select_group(cfg, rows, half)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    (out_dir / "configs" / f"selected_{group}.json").write_text(json.dumps(selected, indent=1), encoding="utf-8")
    (out_dir / "configs" / f"pareto_{group}.json").write_text(json.dumps(pareto, indent=1), encoding="utf-8")
    finish_stage_metadata(meta, out_dir, {"group": group})


# ---------------------------------------------------------------------------
# Stage 3: evaluation
# ---------------------------------------------------------------------------

def evaluation_plan(cfg: dict[str, Any], group: str, selected: dict, pareto: dict,
                    calib_rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, str, str]]]:
    """(config_id, method, params_json) per condition, deduplicated.

    Every condition gets the operating points of its tuning source (all
    budgets, incl. tuning-control variants); calibrated conditions also get
    their Pareto sets (all registered frontiers).
    """
    params = {(r["condition"], r["config_id"]): (r["method"], r["params_json"]) for r in calib_rows}
    plan: dict[str, list[tuple[str, str, str]]] = {}
    for spec in specs(cfg).values():
        src = spec.tuning_source
        chosen: dict[str, tuple[str, str]] = {}
        for name, by_b in selected.get(src, {}).items():
            base = name.split("__")[0]
            if base not in methods_for(cfg, group, spec.name):
                continue
            for sel in by_b.values():
                chosen[sel["config_id"]] = params[(src, sel["config_id"])]
        if spec.calibrate:
            for fk, by_m in pareto.get(spec.name, {}).items():
                for name, ids in by_m.items():
                    for cid in ids:
                        chosen[cid] = params[(spec.name, cid)]
        plan[spec.name] = [(cid, m, pj) for cid, (m, pj) in sorted(chosen.items())]
    return plan


def _eval_task(args: tuple) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, spec, gap, seed, configs, context = args
    inst = _build(cfg, spec, gap, seed)
    rows = []
    for cid, method, pj in configs:
        m, _ = evaluate_config(inst, method, model_params(method, json.loads(pj), context))
        row = {"condition": spec.name, "role": spec.role, "gap": gap, "seed": seed,
               "fps_scale": spec.fps_scale, "instance_id": inst.instance_id,
               "manifest_sha256": inst.manifest_sha256, "config_id": cid, "method": method,
               "params_json": pj}
        row.update(m)
        rows.append(row)
    man = {"instance_id": inst.instance_id, "condition": spec.name, "gap": gap, "seed": seed,
           "manifest_sha256": inst.manifest_sha256,
           "lifetimes": [{"edge_key": list(lt.edge_key), "start": lt.start_frame, "end": lt.end_frame,
                          "terminates": lt.terminates,
                          "post_termination_occluded_frames": lt.post_termination_occluded_frames}
                         for lt in inst.lifetimes],
           "manifests": inst.manifests}
    return rows, man


EVAL_ID_FIELDS = ["run_id", "condition", "role", "gap", "seed", "fps_scale", "instance_id",
                  "manifest_sha256", "config_id", "method", "params_json"]


def run_evaluation(cfg: dict[str, Any], out_dir: Path, group: str, workers: int,
                   smoke: bool = False) -> Path:
    """Evaluate a group's planned configurations on evaluation seeds."""
    meta = new_stage_metadata(f"evaluate_{group}", cfg, smoke)
    meta["config_sha256"] = cfg["_sha256"]
    selected = json.loads((out_dir / "configs" / f"selected_{group}.json").read_text(encoding="utf-8"))
    pareto = json.loads((out_dir / "configs" / f"pareto_{group}.json").read_text(encoding="utf-8"))
    calib = read_csv_gz(out_dir / "raw" / f"calibration_{group}.csv.gz")
    plan = evaluation_plan(cfg, group, selected, pareto, calib)
    (out_dir / "configs" / f"evaluation_plan_{group}.json").write_text(
        json.dumps({c: [p[0] for p in v] for c, v in plan.items()}, indent=1), encoding="utf-8")
    ctx = load_context(out_dir)
    slim = _slim_cfg(cfg)
    seeds = _seeds(cfg, "evaluation", smoke)
    tasks = [(slim, spec, gap, seed, plan[spec.name], ctx[spec.name])
             for spec in specs(cfg).values() for gap in spec.gaps for seed in seeds]
    logger.info("Evaluation[%s]: %d instances, %d runs", group, len(tasks), sum(len(t[4]) for t in tasks))
    rows: list[dict[str, Any]] = []
    mans: list[dict[str, Any]] = []
    t0 = time.time()
    with _pool(workers) as pool:
        for k, (r, m) in enumerate(pool.imap(_eval_task, tasks, chunksize=1)):
            rows.extend(r)
            mans.append(m)
            if (k + 1) % 200 == 0:
                logger.info("  %d/%d evaluation instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    for r in rows:
        r["run_id"] = meta["run_id"]
    rows.sort(key=lambda r: (r["condition"], r["gap"], r["seed"], r["config_id"]))
    mans.sort(key=lambda m: m["instance_id"])
    mpath = out_dir / "raw" / f"manifests_evaluation_{group}.jsonl.gz"
    if mpath.exists():
        mpath.unlink()
    _append_jsonl_gz(mpath, mans)
    path = out_dir / "raw" / f"evaluation_{group}.csv.gz"
    _write_csv_gz(path, rows, EVAL_ID_FIELDS + list(METRIC_FIELDS))
    finish_stage_metadata(meta, out_dir, {"group": group, "n_instances": len(tasks),
                                          "n_rows": len(rows), "seeds": seeds,
                                          "elapsed_s": time.time() - t0})
    return path


# ---------------------------------------------------------------------------
# Stage 4: latency (serial)
# ---------------------------------------------------------------------------

def run_latency(cfg: dict[str, Any], out_dir: Path, group: str, smoke: bool = False) -> Path:
    """Per-frame latency of primary-budget operating points on N0 instances."""
    _quiet_worker()
    meta = new_stage_metadata(f"latency_{group}", cfg, smoke)
    selected = json.loads((out_dir / "configs" / f"selected_{group}.json").read_text(encoding="utf-8"))
    ctx = load_context(out_dir)
    cond = "N0_clean"
    spec = specs(cfg)[cond]
    b = str(cfg["primary_budget_frames"])
    rows = []
    for gap in spec.gaps:
        for seed in _seeds(cfg, "evaluation", smoke):
            inst = _build(_slim_cfg(cfg), spec, gap, seed)
            for name, by_b in selected[cond].items():
                if "__" in name:
                    continue
                sel = by_b[b]
                lat = evaluate_latency(inst, name, model_params(name, sel["params"], ctx[cond]))
                rows.append({"run_id": meta["run_id"], "condition": cond, "gap": gap, "seed": seed,
                             "method": name, "config_id": sel["config_id"], "budget": b, **lat})
    path = out_dir / "raw" / f"latency_{group}.csv.gz"
    _write_csv_gz(path, rows, list(rows[0].keys()))
    finish_stage_metadata(meta, out_dir, {"group": group, "n_rows": len(rows)})
    return path


def save_resolved_config(cfg: dict[str, Any], out_dir: Path) -> None:
    """Copy the resolved config and its hash into the artifact tree."""
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out_dir / "configs" / "phase4.resolved.yaml").write_text(yaml.safe_dump(clean, sort_keys=False),
                                                              encoding="utf-8")
    (out_dir / "configs" / "config_sha256.txt").write_text(f"{cfg['_sha256']}  {cfg['_source_path']}\n",
                                                           encoding="utf-8")
