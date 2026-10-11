"""Episode-study pipeline: fit -> calibrate -> select -> evaluate -> latency.

Checkpointing / safe resume: every (condition, seed) instance of the calibration
and evaluation stages is written atomically (temp file + rename) together with a
SHA-256 sidecar into the RUN directory (heavy, outside git); on restart, complete
and checksum-valid parts are skipped.  Compact aggregates are written to the
RESULTS directory (in git).

Selection reads ONLY calibration artifacts.  Evaluation reads ONLY the frozen
selection.  The transition statistic E is fitted from calibration ground truth.
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
from typing import Any

import numpy as np
import yaml

from taorg.data.episode_world import sample_world
from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance, fit_transition, world_config
from taorg.experiments.episodes.evaluation import METRIC_FIELDS, evaluate_config, evaluate_latency
from taorg.experiments.episodes.search import half_indices, method_configs
from taorg.utils.git import get_git_branch, get_git_commit, get_git_dirty

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config / provenance
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load the study config; record its path and SHA-256."""
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    for key in ("world", "base_corruption", "seeds", "conditions", "search", "methods_order", "criteria",
                "metrics", "statistics", "context_fit"):
        if key not in cfg:
            raise KeyError(f"episode config {path} is missing '{key}'")
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def conditions(cfg: dict[str, Any]) -> dict[str, EpisodeCondition]:
    """Parsed conditions in config order."""
    return {n: EpisodeCondition.from_dict(n, d) for n, d in cfg["conditions"].items()}


def seeds(cfg: dict[str, Any], split: str, smoke: bool) -> list[int]:
    """Seed list of a split (smoke seeds replace every split in smoke mode)."""
    return list(cfg["seeds"]["smoke"] if smoke else cfg["seeds"][split])


def stage_meta(stage: str, cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """Provenance record of one stage."""
    import scipy
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"run_id": f"episodes_{stage}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            "stage": stage, "smoke": smoke, "start_time_utc": now.isoformat(),
            "git_commit": get_git_commit(), "git_dirty": get_git_dirty(), "git_branch": get_git_branch(),
            "command": " ".join(sys.argv), "config_path": cfg["_source_path"], "config_sha256": cfg["_sha256"],
            "environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__, "cpu_count": os.cpu_count()}}


def finish_meta(meta: dict[str, Any], out: Path, extra: dict[str, Any]) -> None:
    """Write the stage provenance record."""
    meta["end_time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta.update(extra)
    p = out / "raw" / f"stage_{meta['stage']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    (path.parent / (path.name + ".sha256")).write_text(hashlib.sha256(data).hexdigest(), encoding="utf-8")


def _valid(path: Path) -> bool:
    side = path.parent / (path.name + ".sha256")
    return path.exists() and side.exists() and sha256_file(path) == side.read_text(encoding="utf-8").strip()


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Gzipped CSV (deterministic member mtime)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    """Read a gzipped CSV."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _quiet() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("taorg").setLevel(logging.ERROR)


def _pool(workers: int):
    return mp.get_context("spawn").Pool(workers, initializer=_quiet)


def _slim(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: cfg[k] for k in ("world", "base_corruption", "metrics")}


def model_params(method: str, params: dict[str, Any], cfg: dict[str, Any], transition: list[list[float]] | None
                 ) -> dict[str, Any]:
    """Constructor kwargs: sampled params (+ the fitted transition statistic for P3/P4)."""
    if method in cfg["context_methods"]:
        if transition is None:
            raise ValueError(f"{method} needs the fitted transition statistic")
        return {**params, "transition": transition}
    return dict(params)


# ---------------------------------------------------------------------------
# stage 0: fit the transition statistic on calibration ground truth
# ---------------------------------------------------------------------------

def run_fit(cfg: dict[str, Any], out: Path, smoke: bool = False) -> list[list[float]]:
    """E from the GT episodes of calibration seeds of calibrated conditions only."""
    meta = stage_meta("fit", cfg, smoke)
    gts, used = [], []
    for c in conditions(cfg).values():
        if not c.calibrate:
            continue
        wc = world_config(cfg["world"], c)
        for s in seeds(cfg, "calibration", smoke):
            w = sample_world(s, wc)
            gts.append({ke.edge_key: list(ke.episodes) for ke in w.keys})
            used.append(f"{c.name}|s{s}")
    E = fit_transition(gts, int(cfg["world"]["num_frames"]), int(cfg["world"]["num_predicates"]),
                       int(cfg["context_fit"]["window_before"]), int(cfg["context_fit"]["window_after"]))
    p = out / "configs" / "transition.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"E": E, "fitted_on": used, "split": "calibration"}, indent=1), encoding="utf-8")
    finish_meta(meta, out, {"n_instances": len(used)})
    return E


def load_transition(out: Path) -> list[list[float]]:
    """The saved transition statistic."""
    return json.loads((out / "configs" / "transition.json").read_text(encoding="utf-8"))["E"]


# ---------------------------------------------------------------------------
# stage 1: calibration (checkpointed per instance)
# ---------------------------------------------------------------------------

def _calib_task(args: tuple) -> str:
    slim, cdict, cname, seed, confs, E, part = args
    part = Path(part)
    if _valid(part):
        return "skipped"
    cond = EpisodeCondition.from_dict(cname, cdict)
    inst = build_instance(cond, seed, slim["world"], slim["base_corruption"])
    mat = np.empty((len(confs), len(METRIC_FIELDS)), dtype=np.float64)
    for i, (method, pj, ctx) in enumerate(confs):
        params = json.loads(pj)
        if ctx:
            params["transition"] = E
        m, _ = evaluate_config(inst, method, params, slim["metrics"])
        mat[i] = [m[f] for f in METRIC_FIELDS]
    buf = io.BytesIO()
    np.savez_compressed(buf, metrics=mat, config_ids=np.array([c[0] + "|" + c[1] for c in confs]),
                        manifest_sha256=np.array(inst.manifest_sha256), instance_id=np.array(inst.instance_id))
    _atomic_write(part, buf.getvalue())
    return "done"


def run_calibration(cfg: dict[str, Any], out: Path, run_dir: Path, workers: int, smoke: bool = False,
                    max_samples: int | None = None) -> Path:
    """Evaluate every configuration on every calibration instance (pooled C conditions)."""
    meta = stage_meta("calibrate", cfg, smoke)
    E = load_transition(out)
    mcs = method_configs(cfg, cfg["methods_order"])
    if max_samples is not None:
        mcs = [c for c in mcs if c.grid_index < max_samples]
    confs = [(c.method, c.params_json, c.method in cfg["context_methods"]) for c in mcs]
    ids = [c.config_id for c in mcs]
    tasks = []
    for c in conditions(cfg).values():
        if not c.calibrate:
            continue
        for s in seeds(cfg, "calibration", smoke):
            part = run_dir / "calibration" / f"{c.name}__s{s}.npz"
            tasks.append((_slim(cfg), cfg["conditions"][c.name], c.name, s, confs, E, str(part)))
    logger.info("calibration: %d instances x %d configs", len(tasks), len(confs))
    t0 = time.time()
    status: dict[str, int] = {}
    with _pool(workers) as pool:
        for k, st in enumerate(pool.imap_unordered(_calib_task, tasks, chunksize=1)):
            status[st] = status.get(st, 0) + 1
            if (k + 1) % 10 == 0:
                logger.info("  %d/%d instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    # aggregate (sequential, verifies every part)
    S = np.zeros((len(confs), len(METRIC_FIELDS)))
    Cn = np.zeros_like(S)
    per_cond: dict[str, list[np.ndarray]] = {}
    inst_rec, checksums = [], []
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
        checksums.append({"file": str(part.relative_to(run_dir)), "sha256": sha256_file(part)})
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / Cn
    rows = []
    for i, mc in enumerate(mcs):
        r = {"config_id": ids[i], "method": mc.method, "grid_index": mc.grid_index, "params_json": mc.params_json,
             "n_instances": int(Cn[i, METRIC_FIELDS.index("J")])}
        for j, f in enumerate(METRIC_FIELDS):
            r[f] = float(mean[i, j])
        for cn, lst in per_cond.items():
            r[f"J__{cn}"] = float(np.mean([v[i] for v in lst]))
        rows.append(r)
    fields = ["config_id", "method", "grid_index", "params_json", "n_instances"] + list(METRIC_FIELDS) + \
        [f"J__{cn}" for cn in per_cond]
    path = out / "raw" / "calibration_summary.csv.gz"
    write_csv_gz(path, rows, fields)
    (out / "raw" / "calibration_instances.json").write_text(json.dumps(sorted(inst_rec, key=lambda r: r["instance_id"]),
                                                                       indent=1), encoding="utf-8")
    (out / "raw" / "calibration_parts_sha256.json").write_text(json.dumps({"run_dir": str(run_dir),
                                                                          "parts": checksums}, indent=1),
                                                              encoding="utf-8")
    finish_meta(meta, out, {"n_instances": len(tasks), "n_configs": len(confs), "status": status,
                            "max_samples": max_samples, "elapsed_s": time.time() - t0, "run_dir": str(run_dir)})
    return path


# ---------------------------------------------------------------------------
# stage 2: selection (calibration artifacts only)
# ---------------------------------------------------------------------------

def _fl(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def select_one(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Max mean J; ties (|dJ| < 1e-12): lower churn, then lower index (prereg §5)."""
    def key(r):
        j = r["J"] if not math.isnan(r["J"]) else -math.inf
        c = r["churn"] if not math.isnan(r["churn"]) else math.inf
        return (-round(j, 12), round(c, 12), r["grid_index"])
    ranked = sorted(rows, key=key)
    best = ranked[0]
    return {"config_id": best["config_id"], "params": json.loads(best["params_json"]), "J": best["J"],
            "churn": best["churn"], "n_candidates": len(rows),
            "ranking": [{"config_id": r["config_id"], "J": r["J"], "churn": r["churn"]} for r in ranked[:5]]}


def run_selection(cfg: dict[str, Any], out: Path, smoke: bool = False) -> dict[str, Any]:
    """Frozen operating points (pooled J) + the half-sample control."""
    meta = stage_meta("select", cfg, smoke)
    rows = read_csv_gz(out / "raw" / "calibration_summary.csv.gz")
    for r in rows:
        r["grid_index"] = int(r["grid_index"])
        for f in METRIC_FIELDS:
            r[f] = _fl(r[f])
    half = half_indices(cfg)
    sel: dict[str, Any] = {}
    for m in cfg["methods_order"]:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            raise RuntimeError(f"no calibration rows for {m}")
        sel[m] = select_one(rs)
        hs = [r for r in rs if r["grid_index"] in half] or rs        # enumerated (1 config): no half
        sel[f"{m}__half"] = select_one(hs)
        sel[f"{m}__half"]["half_of"] = m
    frozen = {"selection_rule": "max pooled mean J over C1-C7 calibration instances; ties: churn, index",
              "config_sha256": cfg["_sha256"], "half_indices": sorted(half), "selected": sel}
    p = out / "configs" / "frozen_parameters.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    finish_meta(meta, out, {"n_methods": len(cfg["methods_order"])})
    return frozen


# ---------------------------------------------------------------------------
# stage 3: evaluation (checkpointed per instance)
# ---------------------------------------------------------------------------

EVAL_ID = ["condition", "role", "seed", "instance_id", "manifest_sha256", "variant", "method", "config_id"]


def _eval_task(args: tuple) -> str:
    slim, cdict, cname, role, seed, plan, E, part = args
    part = Path(part)
    if _valid(part):
        return "skipped"
    cond = EpisodeCondition.from_dict(cname, cdict)
    inst = build_instance(cond, seed, slim["world"], slim["base_corruption"])
    rows = []
    for variant, method, cid, params, ctx in plan:
        p = dict(params)
        if ctx:
            p["transition"] = E
        try:
            m, _ = evaluate_config(inst, method, p, slim["metrics"])
            err = ""
        except Exception as e:  # noqa: BLE001 - recorded as a FAILED run, never silently dropped
            m, err = {f: math.nan for f in METRIC_FIELDS}, repr(e)
        rows.append({"condition": cname, "role": role, "seed": seed, "instance_id": inst.instance_id,
                     "manifest_sha256": inst.manifest_sha256, "variant": variant, "method": method,
                     "config_id": cid, "error": err, **m})
    _atomic_write(part, json.dumps(rows, sort_keys=True).encode("utf-8"))
    return "done"


def evaluation_plan(cfg: dict[str, Any], frozen: dict[str, Any]) -> list[tuple]:
    """(variant, method, config_id, params, needs_context) for every selected operating point."""
    plan = []
    for name, s in frozen["selected"].items():
        m = s.get("half_of", name)
        plan.append((name, m, s["config_id"], s["params"], m in cfg["context_methods"]))
    return plan


def run_evaluation(cfg: dict[str, Any], out: Path, run_dir: Path, workers: int, smoke: bool = False) -> Path:
    """Evaluate the frozen operating points on evaluation seeds of all conditions."""
    meta = stage_meta("evaluate", cfg, smoke)
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    if frozen["config_sha256"] != cfg["_sha256"]:
        raise RuntimeError("config changed since the parameter freeze")
    E = load_transition(out)
    plan = evaluation_plan(cfg, frozen)
    tasks = []
    for c in conditions(cfg).values():
        for s in seeds(cfg, "evaluation", smoke):
            part = run_dir / "evaluation" / f"{c.name}__s{s}.json"
            tasks.append((_slim(cfg), cfg["conditions"][c.name], c.name, c.role, s, plan, E, str(part)))
    t0 = time.time()
    status: dict[str, int] = {}
    with _pool(workers) as pool:
        for k, st in enumerate(pool.imap_unordered(_eval_task, tasks, chunksize=1)):
            status[st] = status.get(st, 0) + 1
            if (k + 1) % 50 == 0:
                logger.info("  %d/%d evaluation instances (%.0fs)", k + 1, len(tasks), time.time() - t0)
    rows, checksums = [], []
    for t in tasks:
        part = Path(t[7])
        if not _valid(part):
            raise RuntimeError(f"evaluation part missing or corrupt: {part}")
        rows.extend(json.loads(part.read_text(encoding="utf-8")))
        checksums.append({"file": str(part.relative_to(run_dir)), "sha256": sha256_file(part)})
    rows.sort(key=lambda r: (r["condition"], r["seed"], r["variant"]))
    for r in rows:
        r["run_id"] = meta["run_id"]
    path = out / "raw" / "evaluation.csv.gz"
    write_csv_gz(path, rows, ["run_id"] + EVAL_ID + ["error"] + list(METRIC_FIELDS))
    (out / "raw" / "evaluation_parts_sha256.json").write_text(json.dumps({"run_dir": str(run_dir), "parts": checksums},
                                                                         indent=1), encoding="utf-8")
    finish_meta(meta, out, {"n_instances": len(tasks), "n_rows": len(rows), "status": status,
                            "elapsed_s": time.time() - t0, "run_dir": str(run_dir)})
    return path


# ---------------------------------------------------------------------------
# stage 4: latency (serial)
# ---------------------------------------------------------------------------

def run_latency(cfg: dict[str, Any], out: Path, smoke: bool = False) -> Path:
    """Per-frame step() latency of the selected operating points on C1 evaluation instances."""
    _quiet()
    meta = stage_meta("latency", cfg, smoke)
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))
    E = load_transition(out)
    cname = cfg["metrics"]["latency_condition"]
    cond = conditions(cfg)[cname]
    rows = []
    for s in seeds(cfg, "evaluation", smoke)[: int(cfg["metrics"]["latency_instances"])]:
        inst = build_instance(cond, s, cfg["world"], cfg["base_corruption"])
        for m in cfg["methods_order"]:
            sel = frozen["selected"][m]
            lat = evaluate_latency(inst, m, model_params(m, sel["params"], cfg, E))
            rows.append({"run_id": meta["run_id"], "condition": cname, "seed": s, "method": m,
                         "config_id": sel["config_id"], **lat})
    path = out / "raw" / "latency.csv.gz"
    write_csv_gz(path, rows, list(rows[0].keys()))
    finish_meta(meta, out, {"n_rows": len(rows)})
    return path


def save_resolved_config(cfg: dict[str, Any], out: Path) -> None:
    """Copy the resolved config and its hash into the results tree."""
    (out / "configs").mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "configs" / "episodes.resolved.yaml").write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
    (out / "configs" / "config_sha256.txt").write_text(f"{cfg['_sha256']}  {cfg['_source_path']}\n", encoding="utf-8")
