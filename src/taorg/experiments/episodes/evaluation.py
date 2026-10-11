"""Run one method configuration on one episode instance and compute all metrics.

Episode metrics: taorg.metrics.episodes (prereg §6, items 1-11).
Historical metrics: the unchanged frozen implementations (persistence, churn,
termination delay, false persistence) -- reported, never redefined.
The model receives ONLY ``instance.input_frames`` (and its parameters).
"""

from __future__ import annotations

import math
import time
from typing import Any

from taorg.experiments.episodes.conditions import EpisodeInstance
from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.episodes import episode_metrics
from taorg.metrics.latency import measure_latency
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.termination import compute_false_persistence, compute_termination_delay
from taorg.schemas.edges import FrameGraph
from taorg.temporal import METHODS

NAN = float("nan")

METRIC_FIELDS: tuple[str, ...] = (
    "J", "frame_f1", "frame_precision", "frame_recall", "episode_f1", "episode_precision", "episode_recall",
    "missed_rate", "fragmentation", "frr", "n_terminations", "reactivation_frames", "fpt_s", "ghost_time_s",
    "std_mean_s", "std_censored", "rdd_mean_s", "rdd_censored", "n_recurrences", "bridge_rate",
    "onset_mean_s", "onset_censored", "n_gt_episodes", "n_output_runs",
    "persistence_f1", "gap_recall", "churn", "term_delay_frozen", "false_pers_frozen",
    "swap_recall", "ghost_edge_frames", "runtime_s",
)


def build_model(method: str, params: dict[str, Any]):
    """Instantiate a registered method with explicit parameters."""
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'")
    return METHODS[method](**params)


def run_model(model, frames: list[FrameGraph], video_id: str) -> list[FrameGraph]:
    """reset -> step* -> finalize."""
    model.reset_video(video_id)
    out = [model.step(f) for f in frames]
    model.finalize_video()
    return out


def swap_recall(inst: EpisodeInstance, out_sets: list[set]) -> float:
    """Fraction of GT-alive frames of keys touching a swapped track, during swaps, output under the TRUE key."""
    entries = (inst.manifests.get("id_switch") or {}).get("entries", [])
    total = hit = 0
    n = len(out_sets)
    for e in entries:
        tracks = set(e["track_ids"] or ())
        for f in range(e["start_frame"], min(e["end_frame"] + 1, n)):
            for k, eps in inst.gt_episodes.items():
                if (k.subject_track_id in tracks or k.object_track_id in tracks) and any(a <= f <= b for a, b in eps):
                    total += 1
                    hit += k in out_sets[f]
    return hit / total if total else NAN


def compute_all(inst: EpisodeInstance, pred: list[FrameGraph], met: dict[str, Any]) -> dict[str, float]:
    """Episode metrics + frozen metrics + identity diagnostics."""
    out_sets = [{o.edge_key for o in fr.edge_observations} for fr in pred]
    rec = episode_metrics(inst.gt_episodes, out_sets, inst.fps, iou_thr=met["episode_iou"],
                          coverage_thr=met["coverage_detected"], min_dead=met["reactivation_min_dead"],
                          sustained=met["sustained_off_frames"])
    has_gap = any(e.corruption_type == "occlusion" for e in inst.occlusion_manifest.entries)
    pers = compute_persistence_f1(inst.gt_frames, pred, inst.occlusion_manifest)
    rec["persistence_f1"] = pers.f1 if has_gap else NAN
    rec["gap_recall"] = (pers.tp / (pers.tp + pers.fn)) if (has_gap and pers.tp + pers.fn) else NAN
    rec["churn"] = compute_churn_rate(pred, inst.gt_frames).mean_churn
    td = compute_termination_delay(inst.gt_frames, pred)
    rec["term_delay_frozen"] = td.mean_delay if td.num_evaluated else NAN
    fp = compute_false_persistence(inst.gt_frames, pred)
    rec["false_pers_frozen"] = fp.mean_rate if fp.num_evaluated else NAN
    rec["swap_recall"] = swap_recall(inst, out_sets)
    rec["ghost_edge_frames"] = rec["ghost_time_s"] * inst.fps
    return rec


def evaluate_config(inst: EpisodeInstance, method: str, params: dict[str, Any], met: dict[str, Any]
                    ) -> tuple[dict[str, float], list[FrameGraph]]:
    """Run and score one configuration; returns (metrics, predictions)."""
    model = build_model(method, params)
    t0 = time.perf_counter()
    pred = run_model(model, inst.input_frames, inst.gt_frames[0].video_id)
    rt = time.perf_counter() - t0
    rec = compute_all(inst, pred, met)
    rec["runtime_s"] = rt
    return {k: float(rec[k]) for k in METRIC_FIELDS}, pred


def evaluate_latency(inst: EpisodeInstance, method: str, params: dict[str, Any]) -> dict[str, float]:
    """Frozen per-frame latency instrumentation."""
    lat = measure_latency(build_model(method, params), inst.input_frames, video_id=inst.gt_frames[0].video_id)
    return {"latency_mean_ms": lat.mean_ms, "latency_median_ms": lat.median_ms,
            "latency_p95_ms": lat.p95_ms, "latency_frames": float(lat.total_frames)}


def is_nan(x: float) -> bool:
    """True for float NaN."""
    return isinstance(x, float) and math.isnan(x)
