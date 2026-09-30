"""Run one method configuration on one instance and compute metrics.

FROZEN metrics (research/metric_definitions.md) are computed exclusively by the
Phase 2 implementations in ``taorg.metrics`` — nothing here re-implements them.

Reporting conventions (do not change metric definitions):
    - Persistence P/R/F1 is reported as NaN when the instance has no occlusion
      interval (the frozen function returns 0.0 for "undefined").
    - Termination delay / false persistence are NaN when the instance has no
      edge with a GT termination (the frozen functions return 0.0 with
      ``num_evaluated == 0``).

DIAGNOSTICS (prefix ``diag_``) are NOT frozen metrics.  They are exploratory
measurements added in Phase 3 to expose failure modes the frozen metrics do
not see.  They are never used in the pre-registered H1 decision rule.
    - diag_clean_alive_recall: fraction of GT-alive (edge, frame) pairs
      OUTSIDE occlusion intervals that the method outputs (captures drops
      caused by flicker or slow re-acquisition).
    - diag_spurious_rate: fraction of output (edge, frame) pairs whose edge is
      not GT-alive at that frame (post-termination persistence + ghosts).
    - diag_ghost_edge_frames: number of output (edge, frame) pairs whose edge
      key is never alive in GT (e.g. created by an ID switch).
    - diag_recovery_lag: mean over occlusion intervals (whose edge is GT-alive
      right after the gap) of the number of frames after the gap before the
      method outputs the edge again; capped at the remaining alive frames.
    - diag_swap_recall: for ID-switch intervals, fraction of GT-alive frames of
      edges touching a swapped track that are output under their TRUE key.
"""

from __future__ import annotations

import math
import time
from typing import Any

from taorg.experiments.phase3.conditions import Instance
from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.latency import measure_latency
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.termination import compute_false_persistence, compute_termination_delay
from taorg.schemas.edges import EdgeKey, FrameGraph
from taorg.temporal import METHODS

NAN = float("nan")

# Keys of the per-(config, instance) metric record, in output order.
METRIC_FIELDS: tuple[str, ...] = (
    "persistence_f1", "persistence_precision", "persistence_recall",
    "persistence_tp", "persistence_fp", "persistence_fn",
    "churn_mean", "churn_total_flips",
    "term_delay_mean", "term_delay_median", "term_num_censored", "term_num_evaluated",
    "false_pers_mean", "false_pers_num_evaluated",
    "diag_clean_alive_recall", "diag_spurious_rate", "diag_ghost_edge_frames",
    "diag_recovery_lag", "diag_swap_recall",
    "method_runtime_s",
)


def build_model(method: str, params: dict[str, Any]):
    """Instantiate a registered temporal method with explicit parameters."""
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'. Registered: {sorted(METHODS)}")
    return METHODS[method](**params)


def run_model(model, frames: list[FrameGraph], video_id: str) -> list[FrameGraph]:
    """Run a temporal model over a frame sequence (reset -> step* -> finalize)."""
    model.reset_video(video_id)
    out = [model.step(f) for f in frames]
    model.finalize_video()
    return out


def _alive_sets(frames: list[FrameGraph], observed_only: bool) -> list[set[EdgeKey]]:
    if observed_only:
        return [{o.edge_key for o in f.edge_observations if o.observed} for f in frames]
    return [{o.edge_key for o in f.edge_observations} for f in frames]


def compute_diagnostics(inst: Instance, pred_frames: list[FrameGraph]) -> dict[str, float]:
    """Compute non-frozen diagnostics (see module docstring)."""
    gt_alive = _alive_sets(inst.gt_frames, observed_only=True)
    pred = _alive_sets(pred_frames, observed_only=False)
    n = len(gt_alive)
    gt_keys_ever = set().union(*gt_alive) if gt_alive else set()

    occluded: set[tuple[EdgeKey, int]] = set()
    for e in inst.occlusion_manifest.entries:
        if e.edge_key is not None:
            occluded.update((e.edge_key, f) for f in range(e.start_frame, e.end_frame + 1))

    clean_total = clean_hit = 0
    pred_total = spurious = ghost = 0
    for f in range(n):
        for k in gt_alive[f]:
            if (k, f) not in occluded:
                clean_total += 1
                clean_hit += k in pred[f]
        for k in pred[f]:
            pred_total += 1
            if k not in gt_alive[f]:
                spurious += 1
                ghost += k not in gt_keys_ever

    lags: list[int] = []
    for e in inst.occlusion_manifest.entries:
        k = e.edge_key
        f0 = e.end_frame + 1
        if k is None or f0 >= n or k not in gt_alive[f0]:
            continue
        lag = 0
        f = f0
        while f < n and k in gt_alive[f] and k not in pred[f]:
            lag += 1
            f += 1
        lags.append(lag)

    swap_total = swap_hit = 0
    for e in (inst.manifests.get("id_switch", {}) or {}).get("entries", []):
        tracks = set(e["track_ids"] or ())
        for f in range(e["start_frame"], min(e["end_frame"] + 1, n)):
            for k in gt_alive[f]:
                if k.subject_track_id in tracks or k.object_track_id in tracks:
                    swap_total += 1
                    swap_hit += k in pred[f]

    return {
        "diag_clean_alive_recall": clean_hit / clean_total if clean_total else NAN,
        "diag_spurious_rate": spurious / pred_total if pred_total else 0.0,
        "diag_ghost_edge_frames": float(ghost),
        "diag_recovery_lag": sum(lags) / len(lags) if lags else NAN,
        "diag_swap_recall": swap_hit / swap_total if swap_total else NAN,
    }


def compute_metrics(inst: Instance, pred_frames: list[FrameGraph]) -> dict[str, float]:
    """Compute frozen metrics (via taorg.metrics) plus diagnostics."""
    has_occ = any(e.corruption_type == "occlusion" for e in inst.occlusion_manifest.entries)
    pers = compute_persistence_f1(inst.gt_frames, pred_frames, inst.occlusion_manifest)
    churn = compute_churn_rate(pred_frames, inst.gt_frames)
    term = compute_termination_delay(inst.gt_frames, pred_frames)
    fp = compute_false_persistence(inst.gt_frames, pred_frames)
    rec: dict[str, float] = {
        "persistence_f1": pers.f1 if has_occ else NAN,
        "persistence_precision": pers.precision if has_occ else NAN,
        "persistence_recall": pers.recall if has_occ else NAN,
        "persistence_tp": float(pers.tp),
        "persistence_fp": float(pers.fp),
        "persistence_fn": float(pers.fn),
        "churn_mean": churn.mean_churn,
        "churn_total_flips": float(churn.total_flips),
        "term_delay_mean": term.mean_delay if term.num_evaluated else NAN,
        "term_delay_median": float(term.median_delay) if term.num_evaluated else NAN,
        "term_num_censored": float(term.num_censored),
        "term_num_evaluated": float(term.num_evaluated),
        "false_pers_mean": fp.mean_rate if fp.num_evaluated else NAN,
        "false_pers_num_evaluated": float(fp.num_evaluated),
    }
    rec.update(compute_diagnostics(inst, pred_frames))
    return rec


def evaluate_config(
    inst: Instance,
    method: str,
    params: dict[str, Any],
    keep_predictions: bool = False,
) -> tuple[dict[str, float], list[FrameGraph] | None]:
    """Run one configuration on one instance.

    Returns:
        (metrics, predictions or None).  ``metrics["method_runtime_s"]`` is the
        wall-clock time of the full temporal pass (reset + all steps).
    """
    model = build_model(method, params)
    t0 = time.perf_counter()
    pred = run_model(model, inst.input_frames, inst.gt_frames[0].video_id)
    runtime = time.perf_counter() - t0
    metrics = compute_metrics(inst, pred)
    metrics["method_runtime_s"] = runtime
    return metrics, (pred if keep_predictions else None)


def evaluate_latency(inst: Instance, method: str, params: dict[str, Any]) -> dict[str, float]:
    """Per-frame step() latency via the frozen latency instrumentation (§7)."""
    lat = measure_latency(build_model(method, params), inst.input_frames,
                          video_id=inst.gt_frames[0].video_id)
    return {
        "latency_mean_ms": lat.mean_ms, "latency_median_ms": lat.median_ms,
        "latency_p95_ms": lat.p95_ms, "latency_p99_ms": lat.p99_ms,
        "latency_frames": float(lat.total_frames),
    }


def is_nan(x: float) -> bool:
    """True if x is a float NaN."""
    return isinstance(x, float) and math.isnan(x)
