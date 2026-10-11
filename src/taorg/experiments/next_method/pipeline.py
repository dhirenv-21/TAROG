"""Next-method study pipeline: fit (prior) -> calibrate -> select -> evaluate -> latency.

Reuses the episode-study instance builder, metrics, sampler and checkpoint helpers
unchanged.  The only model-side input beyond each method's sampled parameters is the PRIOR,
fitted on CALIBRATION INPUT streams (observable data only) and passed to the prior methods.
Checkpointed per instance (atomic write + SHA-256 sidecar), resumable.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance
from taorg.experiments.episodes.evaluation import METRIC_FIELDS, evaluate_config, evaluate_latency
from taorg.experiments.episodes.pipeline import (
    _atomic_write, _pool, _valid, conditions, finish_meta, read_csv_gz, seeds, select_one, sha256_file,
    stage_meta, write_csv_gz,
)
from taorg.experiments.episodes.search import half_indices, method_configs
from taorg.temporal.cure import fit_prior

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load the study config; record path and SHA-256."""
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    for key in ("world", "base_corruption", "seeds", "conditions", "search", "methods_order", "criteria",
                "metrics", "statistics", "prior", "prior_methods", "reference_methods"):
        if key not in cfg:
            raise KeyError(f"next-method config {path} is missing '{key}'")
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def _slim(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in ("world", "base_corruption", "metrics")}


def with_prior(method: str, params: dict[str, Any], cfg: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
    """Constructor kwargs (+ the calibration-input prior for the prior methods)."""
    if method in cfg["prior_methods"]:
        if prior is None:
            raise ValueError(f"{method} needs the fitted prior")
        return {**params, "prior": prior}
    return dict(params)


# --------------------------------------------------------------------------- prior
def run_fit(cfg: dict[str, Any], out: Path, smoke: bool = False) -> dict[str, Any]:
    """Prior from the INPUT streams of calibration seeds of the calibrated conditions."""
    meta = stage_meta("fit", cfg, smoke)
    streams, used = [], []
    for c in conditions(cfg).values():
        if not c.calibrate:
            continue
        for s in seeds(cfg, "calibration", smoke):
            inst = build_instance(c, s, cfg["world"], cfg["base_corruption"])
            streams.append(inst.input_frames)                  # observable input only
            used.append(f"{c.name}|s{s}")
    prior = fit_prior(streams, float(cfg["prior"]["horizon_s"]))
    prior["fitted_on"] = used
    p = out / "configs" / "prior.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prior), encoding="utf-8")
    finish_meta(meta, out, {"n_streams": len(used)})
    return prior


def load_prior(out: Path) -> dict[str, Any]:
    """The saved prior (without the provenance list)."""
    pr = json.loads((out / "configs" / "prior.json").read_text(encoding="utf-8"))
    pr.pop("fitted_on", None)
    return pr


# --------------------------------------------------------------------------- calibration
def _calib_task(args: tuple) -> str:
    slim, cdict, cname, seed, confs, prior, part = args
    part = Path(part)
    if _valid(part):
        return "skipped"
    inst = build_instance(EpisodeCondition.from_dict(cname, cdict), seed, slim["world"], slim["base_corruption"])
    mat = np.empty((len(confs), len(METRIC_FIELDS)))
    for i, (method, pj, needs) in enumerate(confs):
        params = json.loads(pj)
        if needs:
            params["prior"] = prior
        m, _ = evaluate_config(inst, method, params, slim["metrics"])
        mat[i] = [m[f] for f in METRIC_FIELDS]
    buf = io.BytesIO()
    np.savez_compressed(buf, metrics=mat, config_ids=np.array([c[0] + "|" + c[1] for c in confs]),
                        manifest_sha256=np.array(inst.manifest_sha256), instance_id=np.array(inst.instance_id))
    _atomic_write(part, buf.getvalue())
    return "done"


def run_calibration(cfg: dict[str, Any], out: Path, run_dir: Path, workers: int, smoke: bool = False,
                    max_samples: int | None = None) -> Path:
    """All configurations x calibration instances (C1-C7), checkpointed."""
    meta = stage_meta("calibrate", cfg, smoke)
    prior = load_prior(out)
    mcs = method_configs(cfg, cfg["methods_order"])
    if max_samples is not None:
        mcs = [c for c in mcs if c.grid_index < max_samples]
    confs = [(c.method, c.params_json, c.method in cfg["prior_methods"]) for c in mcs]
    tasks = []
    for c in conditions(cfg).values():
        if c.calibrate:
            for s in seeds(cfg, "calibration", smoke):
                tasks.append((_slim(cfg), cfg["conditions"][c.name], c.name, s, confs, prior,
                              str(run_dir / "calibration" / f"{c.name}__s{s}.npz")))
    t0, status = time.time(), {}
    with _pool(workers) as pool:
        for k, st in enumerate(pool.imap_unordered(_calib_task, tasks, chunksize=1)):
            status[st] = status.get(st, 0) + 1
            if (k + 1) % 10 == 0:
                logger.info("  %d/%d calibration instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    S = np.zeros((len(confs), len(METRIC_FIELDS)))
    Cn = np.zeros_like(S)
    per_cond: dict[str, list] = {}
    inst_rec, sums = [], []
    for t in tasks:
        part = Path(t[6])
        if not _valid(part):
            raise RuntimeError(f"calibration part missing or corrupt: {part}")
        z = np.load(part, allow_pickle=False)
        if list(z["config_ids"]) != [c[0] + "|" + c[1] for c in confs]:
            raise RuntimeError(f"config list mismatch in {part}")
        mat = z["metrics"]
        ok = ~np.isnan(mat)
        S += np.where(ok, mat, 0.0)
        Cn += ok
        per_cond.setdefault(t[2], []).append(mat[:, METRIC_FIELDS.index("J")])
        inst_rec.append({"instance_id": str(z["instance_id"]), "manifest_sha256": str(z["manifest_sha256"])})
        sums.append({"file": str(part.relative_to(run_dir)), "sha256": sha256_file(part)})
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / Cn
    rows = []
    for i, mc in enumerate(mcs):
        r = {"config_id": mc.config_id, "method": mc.method, "grid_index": mc.grid_index,
             "params_json": mc.params_json, "n_instances": int(Cn[i, METRIC_FIELDS.index("J")])}
        r.update({f: float(mean[i, j]) for j, f in enumerate(METRIC_FIELDS)})
        r.update({f"J__{cn}": float(np.mean([v[i] for v in lst])) for cn, lst in per_cond.items()})
        rows.append(r)
    fields = ["config_id", "method", "grid_index", "params_json", "n_instances"] + list(METRIC_FIELDS) + \
        [f"J__{cn}" for cn in per_cond]
    path = out / "raw" / "calibration_summary.csv.gz"
    write_csv_gz(path, rows, fields)
    (out / "raw" / "calibration_instances.json").write_text(
        json.dumps(sorted(inst_rec, key=lambda r: r["instance_id"]), indent=1), encoding="utf-8")
    (out / "raw" / "calibration_parts_sha256.json").write_text(
        json.dumps({"run_dir": str(run_dir), "parts": sums}, indent=1), encoding="utf-8")
    finish_meta(meta, out, {"n_instances": len(tasks), "n_configs": len(confs), "status": status,
                            "max_samples": max_samples, "elapsed_s": time.time() - t0, "run_dir": str(run_dir)})
    return path


# --------------------------------------------------------------------------- selection
def _fl(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def run_selection(cfg: dict[str, Any], out: Path, smoke: bool = False) -> dict[str, Any]:
    """Pooled-J operating points + the half-sample control (calibration artifacts only)."""
    meta = stage_meta("select", cfg, smoke)
    rows = read_csv_gz(out / "raw" / "calibration_summary.csv.gz")
    for r in rows:
        r["grid_index"] = int(r["grid_index"])
        for f in METRIC_FIELDS:
            r[f] = _fl(r[f])
    half = half_indices(cfg)
    sel = {}
    for m in cfg["methods_order"]:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            raise RuntimeError(f"no calibration rows for {m}")
        sel[m] = select_one(rs)
        sel[f"{m}__half"] = {**select_one([r for r in rs if r["grid_index"] in half] or rs), "half_of": m}
    ref = json.loads(Path(cfg["reference_methods"]["from"]).read_text(encoding="utf-8"))["selected"]
    refs = {f"ref_{m}": {"config_id": ref[m]["config_id"], "params": ref[m]["params"], "ref_of": m,
                         "note": "episode-study frozen point; descriptive only"} for m in cfg["reference_methods"]["methods"]}
    frozen = {"selection_rule": "max pooled mean J over C1-C7 calibration instances; ties: churn, index",
              "config_sha256": cfg["_sha256"], "half_indices": sorted(half), "selected": sel, "references": refs}
    p = out / "configs" / "frozen_parameters.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    finish_meta(meta, out, {})
    return frozen


# --------------------------------------------------------------------------- evaluation
EVAL_ID = ["condition", "role", "seed", "instance_id", "manifest_sha256", "variant", "method", "config_id"]


def evaluation_plan(cfg: dict[str, Any], frozen: dict[str, Any]) -> list[tuple]:
    """(variant, method, config_id, params, needs_prior) for every frozen point and reference."""
    plan = []
    for name, s in frozen["selected"].items():
        m = s.get("half_of", name)
        plan.append((name, m, s["config_id"], s["params"], m in cfg["prior_methods"]))
    for name, s in frozen["references"].items():
        plan.append((name, s["ref_of"], s["config_id"], s["params"], False))
    return plan


def _eval_task(args: tuple) -> str:
    slim, cdict, cname, role, seed, plan, prior, part = args
    part = Path(part)
    if _valid(part):
        return "skipped"
    inst = build_instance(EpisodeCondition.from_dict(cname, cdict), seed, slim["world"], slim["base_corruption"])
    rows = []
    for variant, method, cid, params, needs in plan:
        p = {**params, "prior": prior} if needs else dict(params)
        try:
            m, _ = evaluate_config(inst, method, p, slim["metrics"])
            err = ""
        except Exception as e:  # noqa: BLE001 - recorded as a FAILED run, never dropped silently
            m, err = {f: math.nan for f in METRIC_FIELDS}, repr(e)
        rows.append({"condition": cname, "role": role, "seed": seed, "instance_id": inst.instance_id,
                     "manifest_sha256": inst.manifest_sha256, "variant": variant, "method": method,
                     "config_id": cid, "error": err, **m})
    _atomic_write(part, json.dumps(rows, sort_keys=True).encode("utf-8"))
    return "done"


def run_evaluation(cfg: dict[str, Any], out: Path, run_dir: Path, workers: int, smoke: bool = False) -> Path:
    """Frozen operating points (+ references) x evaluation seeds of all conditions."""
    meta = stage_meta("evaluate", cfg, smoke)
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    if frozen["config_sha256"] != cfg["_sha256"]:
        raise RuntimeError("config changed since the parameter freeze")
    prior = load_prior(out)
    plan = evaluation_plan(cfg, frozen)
    tasks = [(_slim(cfg), cfg["conditions"][c.name], c.name, c.role, s, plan, prior,
              str(run_dir / "evaluation" / f"{c.name}__s{s}.json"))
             for c in conditions(cfg).values() for s in seeds(cfg, "evaluation", smoke)]
    t0, status = time.time(), {}
    with _pool(workers) as pool:
        for k, st in enumerate(pool.imap_unordered(_eval_task, tasks, chunksize=1)):
            status[st] = status.get(st, 0) + 1
            if (k + 1) % 50 == 0:
                logger.info("  %d/%d evaluation instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    rows, sums = [], []
    for t in tasks:
        part = Path(t[7])
        if not _valid(part):
            raise RuntimeError(f"evaluation part missing or corrupt: {part}")
        rows.extend(json.loads(part.read_text(encoding="utf-8")))
        sums.append({"file": str(part.relative_to(run_dir)), "sha256": sha256_file(part)})
    rows.sort(key=lambda r: (r["condition"], r["seed"], r["variant"]))
    for r in rows:
        r["run_id"] = meta["run_id"]
    path = out / "raw" / "evaluation.csv.gz"
    write_csv_gz(path, rows, ["run_id"] + EVAL_ID + ["error"] + list(METRIC_FIELDS))
    (out / "raw" / "evaluation_parts_sha256.json").write_text(
        json.dumps({"run_dir": str(run_dir), "parts": sums}, indent=1), encoding="utf-8")
    finish_meta(meta, out, {"n_instances": len(tasks), "n_rows": len(rows), "status": status,
                            "elapsed_s": time.time() - t0, "run_dir": str(run_dir)})
    return path


def run_latency(cfg: dict[str, Any], out: Path, smoke: bool = False) -> Path:
    """Per-frame latency of the frozen points on C1 evaluation instances (serial)."""
    logging.getLogger("taorg").setLevel(logging.ERROR)
    meta = stage_meta("latency", cfg, smoke)
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    prior = load_prior(out)
    cname = cfg["metrics"]["latency_condition"]
    cond = conditions(cfg)[cname]
    rows = []
    for s in seeds(cfg, "evaluation", smoke)[: int(cfg["metrics"]["latency_instances"])]:
        inst = build_instance(cond, s, cfg["world"], cfg["base_corruption"])
        for m in cfg["methods_order"]:
            sel = frozen["selected"][m]
            lat = evaluate_latency(inst, m, with_prior(m, sel["params"], cfg, prior))
            rows.append({"run_id": meta["run_id"], "condition": cname, "seed": s, "method": m,
                         "config_id": sel["config_id"], **lat})
    path = out / "raw" / "latency.csv.gz"
    write_csv_gz(path, rows, list(rows[0].keys()))
    finish_meta(meta, out, {"n_rows": len(rows)})
    return path


def save_resolved_config(cfg: dict[str, Any], out: Path) -> None:
    """Resolved config + hash in the results tree."""
    (out / "configs").mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "configs" / "next_method.resolved.yaml").write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    (out / "configs" / "config_sha256.txt").write_text(f"{cfg['_sha256']}  {cfg['_source_path']}\n", encoding="utf-8")
