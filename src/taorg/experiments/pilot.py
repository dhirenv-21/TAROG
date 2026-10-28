"""Phase 3 go/no-go pilot: GT relationship streams + synthetic occlusion.

Protocol (Phases.md Phase 3, configs/experiments/pilot_action_genome.yaml):
    1. Split dataset train videos into fit / validation (fixed split_seed).
    2. Per predeclared seed: corrupt the fit videos with occlusion gaps of all
       lengths and fit every V-HPED variant marked `fit: true` (MLE, train only).
    3. Per (seed, gap length): inject occlusion into val and test videos
       (deterministic per video), run every method once at threshold 0 and
       evaluate the frozen metrics at every threshold of the sweep.
    4. Select one threshold per (seed, method) on validation (pooled over gap
       lengths, max frame-level F1); report test metrics at that threshold.
    5. Trade-off analysis: hypervolume of the test (false persistence rate,
       persistence F1) curve per (seed, gap, method); pairwise win fractions.

Nothing here decides the go/no-go gate automatically; it produces the
evidence the team reviews (Phases.md: "stable improvement ... rather than a
single cherry-picked operating point").
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from taorg.metrics.threshold_sweep import SweepAccumulator, hypervolume, video_scores
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.schemas.edges import FrameGraph
from taorg.temporal import METHODS
from taorg.temporal.vhped import VHPED, TrainerConfig, VHPEDParams, fit_vhped

logger = logging.getLogger(__name__)


@dataclass
class PilotData:
    """Pilot inputs.  Each video is a GT FrameGraph list with frame_idx == position."""

    fit_videos: list[list[FrameGraph]]
    val_videos: list[list[FrameGraph]]
    test_videos: list[list[FrameGraph]]
    num_predicates: int
    dataset_id: str


def split_train(videos: list[list[FrameGraph]], val_fraction: float,
                split_seed: int) -> tuple[list, list]:
    """Deterministic fit/val split of the dataset's train videos."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    order = list(range(len(videos)))
    random.Random(split_seed).shuffle(order)
    n_val = max(1, int(round(len(videos) * val_fraction)))
    val_idx = set(order[:n_val])
    return ([v for i, v in enumerate(videos) if i not in val_idx],
            [v for i, v in enumerate(videos) if i in val_idx])


def occlusion_config(cfg: dict[str, Any], gaps: list[int]) -> OcclusionConfig:
    occ = cfg["occlusion"]
    return OcclusionConfig(
        gap_lengths=list(gaps),
        num_gaps_per_edge=int(occ["num_gaps_per_edge"]),
        max_edges=occ.get("max_edges"),
        min_pre_gap_frames=int(occ["min_pre_gap_frames"]),
        min_post_gap_frames=int(occ["min_post_gap_frames"]),
    )


def corrupt_videos(videos: list[list[FrameGraph]], seed: int, gap_tag: int,
                   occ: OcclusionConfig) -> tuple[list[list[FrameGraph]], list[CorruptionManifest]]:
    """Inject occlusion per video with a seed derived from (seed, gap_tag, video index)."""
    out, manifests = [], []
    for i, v in enumerate(videos):
        vseed = (seed * 1_000_003 + gap_tag * 10_007 + i) % (2**31 - 1)
        frames, manifest = inject_occlusion(v, seed=vseed, config=occ)
        out.append(frames)
        manifests.append(manifest)
    return out, manifests


# ---------------------------------------------------------------------------
# Method construction
# ---------------------------------------------------------------------------

# Methods whose output at threshold t equals their threshold-0 output filtered by
# predicate_score >= t (tests/unit/test_threshold_sweep.py).  FixedWindow is
# excluded: it thresholds the best score in its window but emits another score.
SWEEPABLE_METHODS = frozenset({"no_memory", "ema", "gated_ema", "hmm", "vhped"})

def build_method(name: str, spec: dict[str, Any], cfg: dict[str, Any],
                 num_predicates: int, fitted: dict[str, dict] | None) -> Any:
    """Construct a method at score_threshold 0 from its pilot spec."""
    kind = spec["method"]
    if kind not in SWEEPABLE_METHODS:
        raise ValueError(
            f"Pilot entry {name!r} uses {kind!r}, which does not support post-hoc threshold "
            f"sweeps; allowed: {sorted(SWEEPABLE_METHODS)}"
        )
    params = {k: v for k, v in spec.items()
              if k not in ("method", "fit", "use_visibility", "use_excitation",
                           "use_cross_predicate")}
    if kind != "vhped":
        if kind not in METHODS:
            raise KeyError(f"Unknown method {kind!r} for pilot entry {name!r}")
        params.pop("score_threshold", None)
        return METHODS[kind](score_threshold=0.0, **params)
    if spec.get("fit", False):
        if fitted is None or name not in fitted:
            raise KeyError(f"No fitted parameters for {name!r}")
        vparams = VHPEDParams.from_dict(fitted[name])
    else:
        vparams = handset_params(cfg, num_predicates).apply_ablation(**ablation_flags(spec))
    return VHPED(params=vparams, score_threshold=0.0, fps=float(cfg["vhped"]["fps"]),
                 prune_below=float(cfg["vhped"]["prune_below"]))


def ablation_flags(spec: dict[str, Any]) -> dict[str, bool]:
    return {k: bool(spec.get(k, True))
            for k in ("use_visibility", "use_excitation", "use_cross_predicate")}


def handset_params(cfg: dict[str, Any], num_predicates: int) -> VHPEDParams:
    return VHPEDParams.from_config(dict(cfg["vhped"], num_predicates=num_predicates))


def fit_variants(cfg: dict[str, Any], data: PilotData, seed: int) -> dict[str, dict]:
    """Fit every `fit: true` V-HPED variant on corrupted fit videos for one seed."""
    specs = {n: s for n, s in cfg["methods"].items() if s["method"] == "vhped" and s.get("fit")}
    if not specs:
        return {}
    fit_videos = data.fit_videos
    limit = cfg["training"].get("max_videos")
    if limit is not None and len(fit_videos) > limit:
        rng = random.Random(seed)
        fit_videos = [fit_videos[i] for i in sorted(rng.sample(range(len(fit_videos)), limit))]
    corrupted, _ = corrupt_videos(fit_videos, seed, gap_tag=0,
                                  occ=occlusion_config(cfg, cfg["occlusion"]["gap_lengths"]))
    init = handset_params(cfg, data.num_predicates)
    tcfg = cfg["training"]["trainer"]
    fitted = {}
    for name, spec in specs.items():
        flags = ablation_flags(spec)
        trainer = TrainerConfig(**{**tcfg, **flags, "seed": seed})
        res = fit_vhped(corrupted, init.apply_ablation(**flags), trainer,
                        fps=float(cfg["vhped"]["fps"]))
        logger.info("seed %d fit %s: mean NLL %.4f -> %.4f (%d terms)", seed, name,
                    res.nll_history[0], res.nll_history[-1], res.num_terms)
        fitted[name] = res.params.to_dict()
        fitted[name]["_nll_history"] = [res.nll_history[0], res.nll_history[-1]]
    return fitted


# ---------------------------------------------------------------------------
# Evaluation cells (parallelisable)
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(cfg: dict[str, Any], data: PilotData) -> None:
    _WORKER["cfg"], _WORKER["data"] = cfg, data


def evaluate_cell(seed: int, gap: int, split: str,
                  fitted: dict[str, dict]) -> dict[str, Any]:
    """Run all methods on one (seed, gap, split) cell; metrics for every threshold."""
    cfg, data = _WORKER["cfg"], _WORKER["data"]
    videos = data.val_videos if split == "val" else data.test_videos
    occ = occlusion_config(cfg, [gap])
    corrupted, manifests = corrupt_videos(videos, seed, gap_tag=gap, occ=occ)
    thresholds = np.asarray(cfg["thresholds"], dtype=np.float64)
    curves: dict[str, list[dict[str, float]]] = {}
    for name, spec in cfg["methods"].items():
        acc = SweepAccumulator(thresholds)
        for gt, noisy, manifest in zip(videos, corrupted, manifests):
            model = build_method(name, spec, cfg, data.num_predicates, fitted)
            model.reset_video(gt[0].video_id)
            pred = [model.step(f) for f in noisy]
            model.finalize_video()
            acc.add(video_scores(gt, pred, manifest))
        curves[name] = acc.results()
    merged = CorruptionManifest(seed=seed, config={"type": "occlusion", "gap": gap,
                                                   "split": split})
    for m in manifests:
        merged.entries.extend(m.entries)
    return {"seed": seed, "gap": gap, "split": split, "curves": curves,
            "manifest": merged.to_dict(), "num_gaps": len(merged.entries)}


def run_pilot(cfg: dict[str, Any], data: PilotData, workers: int = 1) -> dict[str, Any]:
    """Execute the full pilot; returns all curves, selections and summaries."""
    seeds = [int(s) for s in cfg["seeds"]]
    gaps = [int(g) for g in cfg["occlusion"]["gap_lengths"]]

    fitted = {s: fit_variants(cfg, data, s) for s in seeds}
    jobs = [(s, g, split, fitted[s]) for s in seeds for g in gaps for split in ("val", "test")]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(cfg, data)) as ex:
            cells = list(ex.map(evaluate_cell, *zip(*jobs)))
    else:
        _init_worker(cfg, data)
        cells = [evaluate_cell(*j) for j in jobs]

    return {"fitted": fitted, "cells": cells, **summarize(cfg, cells)}


# ---------------------------------------------------------------------------
# Selection + summaries
# ---------------------------------------------------------------------------

def summarize(cfg: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(cfg["methods"])
    seeds = sorted({c["seed"] for c in cells})
    gaps = sorted({c["gap"] for c in cells})
    by = {(c["seed"], c["gap"], c["split"]): c["curves"] for c in cells}

    # 4. threshold selection on validation, pooled over gaps (mean frame F1).
    selected: dict[int, dict[str, int]] = {}
    for s in seeds:
        selected[s] = {}
        for m in methods:
            k_len = len(by[(s, gaps[0], "val")][m])
            score = [np.mean([by[(s, g, "val")][m][k]["frame_f1"] for g in gaps])
                     for k in range(k_len)]
            selected[s][m] = int(np.argmax(score))

    table = []
    for s in seeds:
        for g in gaps:
            for m in methods:
                row = dict(by[(s, g, "test")][m][selected[s][m]])
                row.update(seed=s, gap=g, method=m)
                table.append(row)

    # 5. trade-off hypervolume on test curves.
    hv = {(s, g, m): hypervolume([(r["false_persistence_rate"], r["persistence_f1"])
                                  for r in by[(s, g, "test")][m]])
          for s in seeds for g in gaps for m in methods}
    comparisons = []
    for a, b in cfg.get("comparisons", []):
        if a not in methods or b not in methods:
            continue
        diffs = np.array([hv[(s, g, a)] - hv[(s, g, b)] for s in seeds for g in gaps])
        comparisons.append({
            "a": a, "b": b,
            "cells": int(diffs.size),
            "a_better_frac": float((diffs > 0).mean()),
            "hv_diff_mean": float(diffs.mean()),
            "hv_diff_std": float(diffs.std()),
            "per_gap_mean_diff": {g: float(np.mean([hv[(s, g, a)] - hv[(s, g, b)] for s in seeds]))
                                  for g in gaps},
        })

    keys = [k for k in table[0] if k not in ("seed", "gap", "method")] if table else []
    aggregate = []
    for g in gaps:
        for m in methods:
            rows = [r for r in table if r["gap"] == g and r["method"] == m]
            entry: dict[str, Any] = {"gap": g, "method": m}
            for k in keys:
                vals = [r[k] for r in rows]
                entry[k] = [float(np.mean(vals)), float(np.std(vals))]
            entry["hypervolume"] = [float(np.mean([hv[(s, g, m)] for s in seeds])),
                                    float(np.std([hv[(s, g, m)] for s in seeds]))]
            aggregate.append(entry)

    return {
        "selected_threshold_index": selected,
        "test_table": table,
        "aggregate": aggregate,
        "hypervolume": [{"seed": s, "gap": g, "method": m, "hv": v} for (s, g, m), v in hv.items()],
        "comparisons": comparisons,
    }
