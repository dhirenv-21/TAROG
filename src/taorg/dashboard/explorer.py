"""Replay one saved synthetic test video through the temporal filters.

The video, its corruptions, the method specs, the fitted parameters and the decision
threshold all come from a saved ablation-matrix run, so the per-video metrics computed
here can be checked against the ``per_video`` records that run stored.  No model is
fitted and no run directory is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from taorg.dashboard.runs import RunNotFound, matrix_config, read_json, run_path
from taorg.experiments.ablation_matrix import (
    IA_KINDS,
    _run,
    build,
    idswitch_spurious,
    make_videos,
)
from taorg.experiments.vhped_benchmark import ScenarioConfig
from taorg.metrics.threshold_sweep import SweepAccumulator, video_scores
from taorg.schemas.edges import EdgeKey
from taorg.temporal.vhped import VHPEDParams
from taorg.temporal.vhped_ia import IAParams

MAX_METHODS = 6


class ExplorerError(ValueError):
    pass


def _matrix(root: Path, run_id: str) -> tuple[dict[str, Any], Path]:
    info, path = run_path(root, run_id)
    if info["kind"] != "matrix" or info["status"] != "completed":
        raise RunNotFound(run_id)
    return info, path


def options(root: Path, run_id: str) -> dict[str, Any]:
    _, path = _matrix(root, run_id)
    cfg = matrix_config(path)
    return {"run_id": run_id, "conditions": list(cfg["conditions"]),
            "seeds": cfg["seeds"], "num_test_videos": cfg["splits"]["test"],
            "methods": list(cfg["methods"]), "reference_method": cfg.get("reference_method")}


def _fitted(cell: dict[str, Any], cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        spec = cfg["methods"][name]
        if not spec.get("fit"):
            continue
        params = cell["fitted"][name]["params"]
        out[name] = (IAParams.from_dict(params) if spec["method"] in IA_KINDS
                     else VHPEDParams.from_dict(params))
    return out


def _key(k: EdgeKey) -> str:
    return f"{k.subject_track_id}-{k.object_track_id}-{k.predicate_id}"


def _edges(video) -> list[dict[str, Any]]:
    """Per-edge GT / detection / occlusion tracks over the video."""
    t_n = len(video.gt)
    keys: dict[EdgeKey, None] = {}
    for frames in (video.gt, video.noisy):
        for f in frames:
            for o in f.edge_observations:
                if o.observed or frames is video.noisy:
                    keys.setdefault(o.edge_key, None)
    rows = []
    for k in keys:
        gt = np.zeros(t_n, dtype=int)
        obs = np.zeros(t_n, dtype=int)
        occ = np.zeros(t_n, dtype=int)
        for t, f in enumerate(video.gt):
            gt[t] = any(o.edge_key == k and o.observed for o in f.edge_observations)
        for t, f in enumerate(video.noisy):
            for o in f.edge_observations:
                if o.edge_key == k:
                    obs[t] = int(o.observed)
                    occ[t] = int(o.edge_visibility.value != "visible")
        alive_frames = np.flatnonzero(gt)
        end = int(alive_frames[-1]) if alive_frames.size else None
        rows.append({
            "key": _key(k), "subject": k.subject_track_id, "object": k.object_track_id,
            "predicate": k.predicate_id, "gt": gt.tolist(), "observed": obs.tolist(),
            "occluded": occ.tolist(), "in_gt": bool(alive_frames.size),
            "terminates_at": end if end is not None and end < t_n - 1 else None,
            "occluded_frames": int(occ.sum()),
        })
    return rows


def simulate(root: Path, run_id: str, condition: str, seed: int, video_index: int,
             methods: list[str]) -> dict[str, Any]:
    """Run ``methods`` on test video ``video_index`` of cell (condition, seed)."""
    _, path = _matrix(root, run_id)
    cfg = matrix_config(path)
    if condition not in cfg["conditions"]:
        raise ExplorerError(f"unknown condition {condition!r}")
    if seed not in cfg["seeds"]:
        raise ExplorerError(f"seed {seed} is not one of the run's seeds {cfg['seeds']}")
    n_test = cfg["splits"]["test"]
    if not 0 <= video_index < n_test:
        raise ExplorerError(f"video_index must be in [0, {n_test - 1}]")
    if not methods:
        raise ExplorerError("select at least one method")
    if len(methods) > MAX_METHODS:
        raise ExplorerError(f"select at most {MAX_METHODS} methods")
    unknown = [m for m in methods if m not in cfg["methods"]]
    if unknown:
        raise ExplorerError(f"unknown methods: {', '.join(unknown)}")

    cell = next((c for c in read_json(path / "cells.json")
                 if c["condition"] == condition and c["seed"] == seed), None)
    if cell is None:
        raise ExplorerError("this run has no saved cell for that condition and seed")
    cond = cfg["conditions"][condition]
    scenario = ScenarioConfig(**cfg.get("scenario", {}))
    vseed = seed * 10_000 + 2000 + video_index
    video = make_videos("te", range(vseed, vseed + 1), scenario, cond.get("id_switch"), cond)[0]
    fitted = _fitted(cell, cfg, methods)

    scores: dict[str, dict[str, list[float | None]]] = {}
    results = []
    for name in methods:
        spec = cfg["methods"][name]
        saved = cell["methods"][name]
        thr = float(saved["selected_threshold"])
        out, ms = _run(build(spec, fitted, name, 0.0, scenario.fps), video)
        trace: dict[str, list[float | None]] = {}
        for t, f in enumerate(out):
            for o in f.edge_observations:
                trace.setdefault(_key(o.edge_key), [None] * len(out))[t] = float(o.predicate_score)
        scores[name] = trace

        at_thr, _ = _run(build(spec, fitted, name, thr, scenario.fps), video)
        acc = SweepAccumulator(np.array([thr]))
        acc.add(video_scores(video.gt, at_thr, video.manifest))
        r = acc.results()[0]
        spurious, predicted = idswitch_spurious(video, at_thr, window_after=10)
        saved_pv = saved["per_video"][video_index]
        results.append({
            "method": name, "kind": spec["method"], "fitted": bool(spec.get("fit")),
            "threshold": thr, "latency_ms_per_frame": ms,
            "frame_f1": r["frame_f1"], "persistence_f1": r["persistence_f1"],
            "termination_delay_mean": r["termination_delay_mean"],
            "num_terminating_edges": r["num_terminating_edges"],
            "false_persistence_rate": r["false_persistence_rate"], "churn": r["churn"],
            "idswitch_spurious": spurious, "idswitch_predicted": predicted,
            "saved_frame_f1": saved_pv["frame_f1"],
            "matches_saved": abs(saved_pv["frame_f1"] - r["frame_f1"]) < 1e-9
            and saved_pv["video"] == video.gt[0].video_id,
        })

    swaps = [{"start": e.start_frame, "end": e.end_frame, "tracks": list(e.track_ids or [])}
             for e in video.manifest.entries if e.corruption_type == "id_switch"]
    return {
        "run_id": run_id, "condition": condition, "seed": seed, "video_index": video_index,
        "video_id": video.gt[0].video_id, "num_frames": len(video.gt), "fps": scenario.fps,
        "edges": _edges(video), "scores": scores, "results": results, "id_switches": swaps,
    }
