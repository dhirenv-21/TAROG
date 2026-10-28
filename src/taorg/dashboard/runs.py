"""Read-only access to saved experiment runs under ``outputs/runs``.

Nothing here runs an experiment: every function reads files that earlier runs wrote.
Run ids are only accepted if they appear in :func:`list_runs`, so a request can never
name an arbitrary path.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CURVE_METRICS = ("frame_f1", "persistence_f1", "termination_delay_mean",
                 "false_persistence_rate", "churn")
MARKERS = {"INTERRUPTED.txt": "interrupted", "STOPPED.txt": "stopped"}


class RunNotFound(KeyError):
    pass


@lru_cache(maxsize=32)
def _read_json_cached(path: str, mtime: float) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_json(path: Path) -> Any:
    return _read_json_cached(str(path), path.stat().st_mtime)


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _matrix_entry(d: Path) -> dict[str, Any] | None:
    meta_p, metrics_p, cfg_p = d / "metadata.json", d / "metrics.json", d / "config.resolved.yaml"
    if not (meta_p.exists() or cfg_p.exists()):
        return None
    meta = read_json(meta_p) if meta_p.exists() else {}
    entry: dict[str, Any] = {
        "id": d.name, "kind": "matrix", "label": "Ablation matrix",
        "command": meta.get("command") or (_read_yaml(cfg_p).get("command") if cfg_p.exists()
                                           else None),
        "start_time": meta.get("start_time"), "end_time": meta.get("end_time"),
        "duration_s": (meta.get("timing") or {}).get("total_seconds"),
        "seeds": meta.get("seeds"), "status": meta.get("status", "unknown"), "note": None,
        "files": sorted(p.name for p in d.iterdir() if p.is_file()),
    }
    if metrics_p.exists():
        entry["label"] = read_json(metrics_p).get("kind", entry["label"])
    else:
        entry["status"] = "incomplete"
        for marker, status in MARKERS.items():
            if (d / marker).exists():
                entry["status"] = status
                entry["note"] = (d / marker).read_text(encoding="utf-8").strip()
    return entry


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Every saved run under ``root``, newest first."""
    runs: list[dict[str, Any]] = []
    if not root.is_dir():
        return runs
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in ("complexity", "param_recovery"):
            for sub in sorted(d.iterdir()):
                res = sub / "results.json"
                if res.exists():
                    data = read_json(res)
                    runs.append({
                        "id": f"{d.name}/{sub.name}", "kind": d.name,
                        "label": ("Lazy vs eager complexity benchmark" if d.name == "complexity"
                                  else "Parameter recovery (synthetic)"),
                        "command": None, "start_time": sub.name, "end_time": None,
                        "duration_s": None, "seeds": None, "status": "completed",
                        "note": None, "git_commit": data.get("git_commit"),
                        "files": sorted(p.name for p in sub.iterdir() if p.is_file()),
                    })
        elif d.name == "vhped_synthetic":
            for f in sorted(d.glob("run_*.json")):
                data = read_json(f)
                runs.append({
                    "id": f"{d.name}/{f.stem}", "kind": "sanity",
                    "label": data.get("kind", "Synthetic sanity benchmark"),
                    "command": None, "start_time": f.stem.removeprefix("run_"),
                    "end_time": None, "duration_s": None, "seeds": data.get("seeds"),
                    "status": "completed", "note": None, "files": [f.name],
                })
        else:
            entry = _matrix_entry(d)
            if entry:
                runs.append(entry)
    runs.sort(key=_sort_key, reverse=True)
    return runs


def _sort_key(run: dict[str, Any]) -> str:
    """YYYYMMDDhhmmss from the start time or, failing that, the run id."""
    stamp = run["start_time"] or run["id"].split("/")[-1]
    return re.sub(r"\D", "", stamp)[:14]


def run_path(root: Path, run_id: str) -> tuple[dict[str, Any], Path]:
    """Look up a run by id; raises RunNotFound for anything not listed."""
    for r in list_runs(root):
        if r["id"] == run_id:
            if r["kind"] == "sanity":
                return r, root / (run_id + ".json")
            return r, root / run_id
    raise RunNotFound(run_id)


def matrix_config(path: Path) -> dict[str, Any]:
    return _read_yaml(path / "config.resolved.yaml")


def mean_curves(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Seed-averaged test curves per ``condition|method`` (threshold on the x axis)."""
    acc: dict[tuple[str, str], list[list[dict[str, float]]]] = {}
    for cell in cells:
        for name, m in cell["methods"].items():
            acc.setdefault((cell["condition"], name), []).append(m["test_curves"])
    out: dict[str, dict[str, Any]] = {}
    for (cond, name), curves in acc.items():
        thr = [c["threshold"] for c in curves[0]]
        out[f"{cond}|{name}"] = {
            "condition": cond, "method": name, "threshold": thr,
            **{k: [float(np.mean([cv[i][k] for cv in curves])) for i in range(len(thr))]
               for k in CURVE_METRICS},
        }
    return out


def matrix_summary(root: Path, run_id: str) -> dict[str, Any]:
    info, path = run_path(root, run_id)
    if info["kind"] != "matrix" or info["status"] != "completed":
        raise RunNotFound(run_id)
    metrics = read_json(path / "metrics.json")
    cfg = matrix_config(path)
    cells = read_json(path / "cells.json")
    return {
        "run": info,
        "kind": metrics.get("kind"),
        "reference_method": cfg.get("reference_method"),
        "conditions": list(cfg.get("conditions", {})),
        "methods": list(cfg.get("methods", {})),
        "method_specs": cfg.get("methods", {}),
        "condition_specs": cfg.get("conditions", {}),
        "splits": cfg.get("splits"),
        "table": metrics["table"],
        "paired": metrics.get("paired_vs_reference", []),
        "curves": mean_curves(cells),
    }


def completed_matrix_runs(root: Path) -> list[dict[str, Any]]:
    return [r for r in list_runs(root) if r["kind"] == "matrix" and r["status"] == "completed"]


def primary_matrix_run(root: Path) -> str | None:
    """The main ablation run: the completed matrix run with the most methods."""
    best, best_n = None, -1
    for r in completed_matrix_runs(root):
        n = len(matrix_config(root / r["id"]).get("methods", {}))
        if n > best_n:
            best, best_n = r["id"], n
    return best


def overview(root: Path) -> dict[str, Any]:
    """Headline numbers, all read from saved runs."""
    runs = list_runs(root)
    out: dict[str, Any] = {
        "runs_root": str(root),
        "counts": {s: sum(r["status"] == s for r in runs)
                   for s in sorted({r["status"] for r in runs})},
        "num_runs": len(runs), "headline": None, "complexity": None, "recovery": None,
    }
    primary = primary_matrix_run(root)
    if primary:
        s = matrix_summary(root, primary)
        ref = s["reference_method"]
        rows = []
        for cond in s["conditions"]:
            by = {t["method"]: t for t in s["table"] if t["condition"] == cond}
            pair = next((p for p in s["paired"] if p["condition"] == cond
                         and p["b"] == "gated_ema"), None)
            rows.append({"condition": cond, "reference": by.get(ref),
                         "gated_ema": by.get("gated_ema"), "paired": pair,
                         "best_frame_f1": max(by.values(), key=lambda t: t["frame_f1"][0])})
        out["headline"] = {"run_id": primary, "reference_method": ref, "rows": rows}
    cx = [r for r in runs if r["kind"] == "complexity"]
    if cx:
        rows = read_json(root / cx[0]["id"] / "results.json")["rows"]
        out["complexity"] = {
            "run_id": cx[0]["id"],
            "max_speedup": max(r["speedup"] for r in rows),
            "min_speedup": min(r["speedup"] for r in rows),
            "mismatches": sum(r["decision_mismatches"] for r in rows),
            "grid_points": len(rows),
        }
    rec = [r for r in runs if r["kind"] == "param_recovery"]
    if rec:
        summ = read_json(root / rec[0]["id"] / "results.json")["summary"]
        out["recovery"] = {"run_id": rec[0]["id"], "summary": summ}
    return out
