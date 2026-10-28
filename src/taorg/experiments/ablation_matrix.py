"""Phase 8 ablation matrix on the synthetic scenario (paired, multi-seed).

SYNTHETIC — the same matrix must be rerun on real predictions (Phase 6/7)
before any paper claim.  Config: configs/experiments/ablation_matrix_synthetic.yaml.

Per (condition, seed):
    1. generate train / val / test videos (vhped_benchmark.generate_video)
       and apply the condition's extra corruption (ID switches);
    2. fit V-HPED variants on train observation streams (MLE);
    3. every method: threshold curves on val and test; threshold selected on
       val (max frame F1); test metrics at that threshold; calibration (ECE,
       Brier) and latency from the threshold-0 test run; per-video frame F1
       and termination delay for paired statistics.
All methods see identical corrupted streams (paired design, Rules §6).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from taorg.experiments.pilot import SWEEPABLE_METHODS
from taorg.experiments.vhped_benchmark import ScenarioConfig, Video, generate_video
from taorg.metrics.calibration import CalibrationAccumulator
from taorg.metrics.threshold_sweep import SweepAccumulator, video_scores
from taorg.noise.id_switch import IDSwitchConfig, inject_id_switch
from taorg.noise.identity_confidence import IdentityScoreConfig, annotate_identity_confidence
from taorg.noise.manifest import CorruptionManifest
from taorg.noise.visibility_errors import flip_visibility, jitter_timestamps
from taorg.temporal import METHODS
from taorg.temporal.vhped import HAND_SET_CONFIG, TrainerConfig, VHPEDParams, fit_vhped
from taorg.temporal.vhped_ia import IAParams, fit_ia_vhped

logger = logging.getLogger(__name__)
NUM_PREDICATES = 3
IA_KINDS = ("vhped_ia", "vhped_ia_lazy")


def make_videos(prefix: str, seeds: range, scenario: ScenarioConfig,
                id_switch: dict | None, extra: dict | None = None) -> list[Video]:
    """Generate videos and apply the condition's corruptions to the observation stream.

    extra keys (optional): identity_score (IdentityScoreConfig kwargs; adds tracker
    identity confidence), vis_flip_prob, timestamp_jitter.  Ground truth is never changed.
    """
    extra = extra or {}
    videos = []
    for s in seeds:
        v = generate_video(f"{prefix}{s}", s, scenario)
        noisy, manifest = v.noisy, v.manifest
        if id_switch:
            noisy, man = inject_id_switch(noisy, seed=s, config=IDSwitchConfig(**id_switch))
            merged = CorruptionManifest(seed=manifest.seed, config=manifest.config)
            merged.entries = list(manifest.entries) + list(man.entries)
            manifest = merged
        if "identity_score" in extra:
            noisy = annotate_identity_confidence(
                noisy, manifest, seed=s + 7, config=IdentityScoreConfig(**extra["identity_score"]))
        if extra.get("vis_flip_prob"):
            noisy = flip_visibility(noisy, float(extra["vis_flip_prob"]), seed=s + 11)
        if extra.get("timestamp_jitter"):
            noisy = jitter_timestamps(noisy, float(extra["timestamp_jitter"]), seed=s + 13,
                                      fps=scenario.fps)
        videos.append(Video(gt=v.gt, noisy=noisy, manifest=manifest))
    return videos


def build(spec: dict[str, Any], fitted: dict[str, Any], name: str, thr: float, fps: float):
    kind = spec["method"]
    params = {k: v for k, v in spec.items() if k not in ("method", "fit", "trainer", "ia")}
    if kind in ("vhped", "vhped_lazy"):
        p = fitted[name] if spec.get("fit") else VHPEDParams.from_config(
            dict(HAND_SET_CONFIG, num_predicates=NUM_PREDICATES))
        return METHODS[kind](params=p, score_threshold=thr, fps=fps)
    if kind in IA_KINDS:
        ia = fitted[name] if spec.get("fit") else IAParams.default(NUM_PREDICATES,
                                                                   **spec.get("ia", {}))
        return METHODS[kind](ia=ia, score_threshold=thr, fps=fps)
    return METHODS[kind](score_threshold=thr, **params)


def _run(model, video: Video):
    model.reset_video(video.gt[0].video_id)
    t0 = time.perf_counter()
    out = [model.step(f) for f in video.noisy]
    ms = (time.perf_counter() - t0) * 1000.0 / max(len(video.noisy), 1)
    model.finalize_video()
    return out, ms


def method_curves(name: str, spec: dict, fitted: dict, videos: list[Video],
                  thresholds: np.ndarray, fps: float) -> dict[str, Any]:
    """Curves for all thresholds + threshold-0 calibration/latency + per-video stats hook."""
    acc = SweepAccumulator(thresholds)
    cal = CalibrationAccumulator()
    latency = []
    if spec["method"] in SWEEPABLE_METHODS or spec["method"] in ("vhped_lazy",) + IA_KINDS:
        for v in videos:
            out, ms = _run(build(spec, fitted, name, 0.0, fps), v)
            vs = video_scores(v.gt, out, v.manifest)
            acc.add(vs)
            cal.add(vs)
            latency.append(ms)
        curves = acc.results()
    else:  # not sweepable post hoc: one run per threshold
        curves = []
        for thr in thresholds:
            one = SweepAccumulator(np.array([thr]))
            for v in videos:
                out, ms = _run(build(spec, fitted, name, float(thr), fps), v)
                one.add(video_scores(v.gt, out, v.manifest))
                latency.append(ms)
            curves.append(one.results()[0])
        for v in videos:
            out, _ = _run(build(spec, fitted, name, 0.0, fps), v)
            cal.add(video_scores(v.gt, out, v.manifest))
    return {"curves": curves, "calibration": cal.result(),
            "latency_ms": float(np.mean(latency))}


def idswitch_spurious(video: Video, pred: list, window_after: int) -> tuple[int, int]:
    """(spurious, predicted) edge-frames on swapped tracks, swap start .. end + window_after.

    Frozen in research/metric_definitions.md §7: an output edge-frame is spurious if
    the edge is not observed in the uncorrupted GT at that frame.
    """
    win: dict[int, set[int]] = {}
    for e in video.manifest.entries:
        if e.corruption_type != "id_switch" or not e.track_ids:
            continue
        for f in range(e.start_frame, e.end_frame + window_after + 1):
            win.setdefault(f, set()).update(e.track_ids)
    spurious = predicted = 0
    for g, p in zip(video.gt, pred):
        tracks = win.get(g.frame_idx)
        if not tracks:
            continue
        gt = {o.edge_key for o in g.edge_observations if o.observed}
        for o in p.edge_observations:
            k = o.edge_key
            if k.subject_track_id in tracks or k.object_track_id in tracks:
                predicted += 1
                spurious += k not in gt
    return spurious, predicted


def per_video_at(name: str, spec: dict, fitted: dict, videos: list[Video], thr: float,
                 fps: float) -> list[dict[str, float]]:
    rows = []
    for v in videos:
        out, _ = _run(build(spec, fitted, name, thr, fps), v)
        acc = SweepAccumulator(np.array([thr]))
        acc.add(video_scores(v.gt, out, v.manifest))
        r = acc.results()[0]
        sp, pr = idswitch_spurious(v, out, window_after=10)
        rows.append({"video": v.gt[0].video_id, "frame_f1": r["frame_f1"],
                     "termination_delay_mean": r["termination_delay_mean"],
                     "num_terminating_edges": r["num_terminating_edges"],
                     "idswitch_spurious": sp, "idswitch_predicted": pr})
    return rows


def run_cell(cfg: dict[str, Any], condition: str, seed: int) -> dict[str, Any]:
    """One (condition, seed) cell of the matrix."""
    cond = cfg["conditions"][condition]
    scenario = ScenarioConfig(**cfg.get("scenario", {}))
    base = seed * 10_000
    n = cfg["splits"]
    ids = cond.get("id_switch")
    train = make_videos("tr", range(base, base + n["train"]), scenario, ids, cond)
    val = make_videos("va", range(base + 1000, base + 1000 + n["val"]), scenario, ids, cond)
    test = make_videos("te", range(base + 2000, base + 2000 + n["test"]), scenario, ids, cond)
    init = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=NUM_PREDICATES))
    fitted: dict[str, Any] = {}
    fit_info: dict[str, Any] = {}
    for name, spec in cfg["methods"].items():
        if not spec.get("fit"):
            continue
        tcfg = TrainerConfig(**{**cfg["trainer"], **spec.get("trainer", {}), "seed": seed})
        start = init.apply_ablation(use_visibility=tcfg.use_visibility,
                                    use_excitation=tcfg.use_excitation,
                                    use_cross_predicate=tcfg.use_cross_predicate)
        if spec["method"] in IA_KINDS:
            res = fit_ia_vhped([v.noisy for v in train], IAParams(base=start, **spec.get("ia", {})),
                               tcfg, fps=scenario.fps)
        else:
            res = fit_vhped([v.noisy for v in train], start, tcfg, fps=scenario.fps)
        fitted[name] = res.params
        fit_info[name] = {"params": res.params.to_dict(),
                          "nll_start": res.nll_history[0], "nll_end": res.nll_history[-1]}

    thresholds = np.asarray(cfg["thresholds"], dtype=np.float64)
    out: dict[str, Any] = {"condition": condition, "seed": seed, "fitted": fit_info,
                           "methods": {}}
    for name, spec in cfg["methods"].items():
        val_c = method_curves(name, spec, fitted, val, thresholds, scenario.fps)
        k = int(np.argmax([r["frame_f1"] for r in val_c["curves"]]))
        test_c = method_curves(name, spec, fitted, test, thresholds, scenario.fps)
        out["methods"][name] = {
            "selected_threshold": float(thresholds[k]),
            "test": test_c["curves"][k],
            "test_curves": test_c["curves"],
            "calibration": test_c["calibration"],
            "latency_ms": test_c["latency_ms"],
            "per_video": per_video_at(name, spec, fitted, test, float(thresholds[k]),
                                      scenario.fps),
        }
        logger.info("%s seed %d %-20s thr %.2f frameF1 %.3f delay %.2f ECE %.3f", condition,
                    seed, name, thresholds[k], test_c["curves"][k]["frame_f1"],
                    test_c["curves"][k]["termination_delay_mean"], test_c["calibration"]["ece"])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% percentile CI of the mean of paired differences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(cfg: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(cfg["methods"])
    metrics = ["frame_f1", "persistence_f1", "termination_delay_mean",
               "false_persistence_rate", "churn"]
    table = []
    for cond in cfg["conditions"]:
        cc = [c for c in cells if c["condition"] == cond]
        for m in methods:
            row: dict[str, Any] = {"condition": cond, "method": m}
            for k in metrics:
                vals = [c["methods"][m]["test"][k] for c in cc]
                row[k] = [float(np.mean(vals)), float(np.std(vals))]
            for k in ("ece", "brier"):
                vals = [c["methods"][m]["calibration"][k] for c in cc]
                row[k] = [float(np.mean(vals)), float(np.std(vals))]
            rates = []
            for c in cc:
                sp = sum(pv.get("idswitch_spurious", 0) for pv in c["methods"][m]["per_video"])
                pr = sum(pv.get("idswitch_predicted", 0) for pv in c["methods"][m]["per_video"])
                rates.append(sp / pr if pr else 0.0)
            row["idswitch_spurious_rate"] = [float(np.mean(rates)), float(np.std(rates))]
            lat = [c["methods"][m]["latency_ms"] for c in cc]
            row["latency_ms"] = [float(np.mean(lat)), float(np.std(lat))]
            thr = [c["methods"][m]["selected_threshold"] for c in cc]
            row["selected_threshold"] = [float(np.mean(thr)), float(np.std(thr))]
            table.append(row)

    ref = cfg["reference_method"]
    stats = []
    for cond in cfg["conditions"]:
        cc = [c for c in cells if c["condition"] == cond]
        for m in methods:
            if m == ref:
                continue
            entry: dict[str, Any] = {"condition": cond, "a": ref, "b": m}
            for k in metrics:
                d = np.array([c["methods"][ref]["test"][k] - c["methods"][m]["test"][k]
                              for c in cc])
                entry[f"{k}_seed_diff"] = [float(d.mean()), float(d.std())]
                entry[f"{k}_a_greater_seeds"] = int((d > 0).sum())
            for k in ("frame_f1", "termination_delay_mean"):
                d = []
                for c in cc:
                    for ra, rb in zip(c["methods"][ref]["per_video"], c["methods"][m]["per_video"]):
                        if k == "termination_delay_mean" and ra["num_terminating_edges"] == 0:
                            continue
                        d.append(ra[k] - rb[k])
                d = np.array(d)
                lo, hi = paired_bootstrap(d, cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]) \
                    if d.size else (0.0, 0.0)
                entry[f"{k}_video_diff_mean"] = float(d.mean()) if d.size else 0.0
                entry[f"{k}_video_ci95"] = [lo, hi]
                entry[f"{k}_video_units"] = int(d.size)
            stats.append(entry)
    return {"table": table, "paired_vs_reference": stats}


def run_matrix(cfg: dict[str, Any], workers: int = 1) -> dict[str, Any]:
    jobs = [(cfg, cond, int(s)) for cond in cfg["conditions"] for s in cfg["seeds"]]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            cells = list(ex.map(run_cell, *zip(*jobs)))
    else:
        cells = [run_cell(*j) for j in jobs]
    return {"cells": cells, **summarize(cfg, cells)}
