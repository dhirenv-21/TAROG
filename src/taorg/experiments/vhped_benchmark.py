"""Synthetic benchmark: V-HPED vs. temporal baselines with fitted parameters.

SYNTHETIC SANITY BENCHMARK — not paper evidence (no real detector, no real
dataset).  Its purpose is to check that the model behaves as derived and
that the train/val/test protocol works end to end before real data.

Scenario per pair (3 predicates):
    0 "holding"       alive from frame 0; terminates (both objects VISIBLE)
                      with probability p_terminate.
    1 "putting_down"  alive for the `precursor_len` frames before holding ends
                      — a cross-predicate termination cue.
    2 "looking_at"    independent, may terminate.
Observation noise:
    - per-edge detector reliability drawn from `reliabilities` (heterogeneous
      visible misses = detector flicker, the hard case for fixed decays)
    - spurious detections of dead edges with probability `fp_prob`
    - pair-level occlusion gaps (all predicates of the pair become
      OCCLUDED_OR_UNRELIABLE and unobserved), recorded in a CorruptionManifest.

Protocol: fit on train videos, choose each method's score_threshold on val
(by frame-level edge F1), report frozen metrics on test.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.termination import compute_false_persistence, compute_termination_delay
from taorg.noise.manifest import CorruptionEntry, CorruptionManifest
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.temporal import METHODS
from taorg.temporal.vhped import HAND_SET_CONFIG, VHPED, TrainerConfig, VHPEDParams, fit_vhped

logger = logging.getLogger(__name__)

VIS = VisibilityState.VISIBLE
OCC = VisibilityState.OCCLUDED_OR_UNRELIABLE
NUM_PREDICATES = 3


@dataclass
class ScenarioConfig:
    """Synthetic scenario parameters (all frame counts at `fps`)."""

    num_frames: int = 150
    num_pairs: int = 4
    fps: float = 30.0
    p_terminate: float = 0.6
    terminate_range: tuple[int, int] = (40, 120)
    precursor_len: int = 6
    reliabilities: tuple[float, ...] = (0.95, 0.75, 0.5)
    fp_prob: float = 0.02
    gap_lengths: tuple[int, ...] = (3, 5, 10, 20)
    gaps_per_pair: int = 2


@dataclass
class Video:
    gt: list[FrameGraph]
    noisy: list[FrameGraph]
    manifest: CorruptionManifest


def generate_video(video_id: str, seed: int, cfg: ScenarioConfig) -> Video:
    """Generate one deterministic synthetic video (GT stream + corrupted stream)."""
    rng = random.Random(seed)
    t_len = cfg.num_frames
    alive = np.zeros((cfg.num_pairs, NUM_PREDICATES, t_len), dtype=bool)
    for n in range(cfg.num_pairs):
        end = rng.randint(*cfg.terminate_range) if rng.random() < cfg.p_terminate else t_len - 1
        alive[n, 0, : end + 1] = True
        if end < t_len - 1:
            alive[n, 1, max(0, end - cfg.precursor_len + 1): end + 1] = True
        end2 = rng.randint(*cfg.terminate_range) if rng.random() < 0.5 else t_len - 1
        alive[n, 2, : end2 + 1] = True
    reliability = np.array([[rng.choice(cfg.reliabilities) for _ in range(NUM_PREDICATES)]
                            for _ in range(cfg.num_pairs)])

    occluded = np.zeros((cfg.num_pairs, t_len), dtype=bool)
    manifest = CorruptionManifest(seed=seed, config={"type": "pair_occlusion",
                                                     "gap_lengths": list(cfg.gap_lengths)})
    for n in range(cfg.num_pairs):
        for _ in range(cfg.gaps_per_pair):
            g = rng.choice(cfg.gap_lengths)
            start = rng.randint(5, t_len - g - 5)
            occluded[n, start: start + g] = True
    for n in range(cfg.num_pairs):
        runs = _runs(occluded[n])
        for p in range(NUM_PREDICATES):
            for s, e in runs:
                manifest.entries.append(CorruptionEntry(
                    corruption_type="occlusion", video_id=video_id,
                    edge_key_tuple=(n + 1, 100 + n, p), start_frame=s, end_frame=e,
                    duration=e - s + 1,
                ))

    gt_frames, noisy_frames = [], []
    for t in range(t_len):
        ts = t / cfg.fps
        gt_edges, noisy_edges = [], []
        for n in range(cfg.num_pairs):
            occ = occluded[n, t]
            for p in range(NUM_PREDICATES):
                key = EdgeKey(n + 1, 100 + n, p)
                is_alive = bool(alive[n, p, t])
                gt_edges.append(_edge(key, video_id, t, ts, is_alive, 1.0 if is_alive else 0.0, VIS))
                if occ:
                    seen = False
                elif is_alive:
                    seen = rng.random() < reliability[n, p]
                else:
                    seen = rng.random() < cfg.fp_prob
                score = rng.uniform(0.5, 1.0) if seen else 0.0
                noisy_edges.append(_edge(key, video_id, t, ts, seen, score, OCC if occ else VIS))
        tracks = _tracks(video_id, t, ts, cfg.num_pairs, occluded[:, t])
        gt_frames.append(FrameGraph(video_id, t, ts, track_observations=tracks,
                                    edge_observations=gt_edges))
        noisy_frames.append(FrameGraph(video_id, t, ts, track_observations=list(tracks),
                                       edge_observations=noisy_edges))
    return Video(gt=gt_frames, noisy=noisy_frames, manifest=manifest)


def _edge(key: EdgeKey, vid: str, t: int, ts: float, seen: bool, score: float,
          obj_vis: VisibilityState) -> EdgeObservation:
    return EdgeObservation(edge_key=key, video_id=vid, frame_idx=t, timestamp=ts,
                           predicate_score=score, observed=seen,
                           subject_visibility=VIS, object_visibility=obj_vis,
                           source_model="synthetic")


def _tracks(vid: str, t: int, ts: float, num_pairs: int, occluded: np.ndarray) -> list:
    """Subject n+1 (class 0) and object 100+n (class 1) for every pair; needed by ID-switch injection."""
    out = []
    for n in range(num_pairs):
        out.append(TrackObservation(vid, t, ts, n + 1, 0, (0.0, 0.0, 1.0, 1.0)))
        out.append(TrackObservation(vid, t, ts, 100 + n, 1, (0.0, 0.0, 1.0, 1.0),
                                    visibility_state=OCC if occluded[n] else VIS))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def frame_f1(gt: list[FrameGraph], pred: list[FrameGraph]) -> tuple[int, int, int]:
    """(tp, fp, fn) of predicted edge sets vs GT-alive edge sets over all frames.

    Auxiliary selection criterion — NOT one of the frozen metrics.
    """
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        gs = {o.edge_key for o in g.edge_observations if o.observed}
        ps = {o.edge_key for o in p.edge_observations}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def run_method(factory: Callable[[float], Any], videos: list[Video], threshold: float):
    preds = []
    for v in videos:
        model = factory(threshold)
        model.reset_video(v.gt[0].video_id)
        preds.append([model.step(f) for f in v.noisy])
        model.finalize_video()
    return preds


def evaluate(videos: list[Video], preds: list[list[FrameGraph]]) -> dict[str, float]:
    """Pool frozen metrics over videos (micro for F1, mean for per-edge rates)."""
    tp = fp = fn = 0
    ptp = pfp = pfn = 0
    delays, censored, n_term, fpr, churn = [], 0, 0, [], []
    for v, p in zip(videos, preds):
        a, b, c = frame_f1(v.gt, p)
        tp, fp, fn = tp + a, fp + b, fn + c
        pers = compute_persistence_f1(v.gt, p, v.manifest)
        ptp, pfp, pfn = ptp + pers.tp, pfp + pers.fp, pfn + pers.fn
        term = compute_termination_delay(v.gt, p)
        delays += list(term.per_edge_delay.values())
        censored += term.num_censored
        n_term += term.num_evaluated
        fpr += list(compute_false_persistence(v.gt, p).per_edge_rate.values())
        churn.append(compute_churn_rate(p, v.gt).mean_churn)
    return {
        "frame_f1": _f1(tp, fp, fn),
        "persistence_f1": _f1(ptp, pfp, pfn),
        "persistence_recall": ptp / (ptp + pfn) if ptp + pfn else 0.0,
        "termination_delay_mean": float(np.mean(delays)) if delays else 0.0,
        "termination_censored_frac": censored / n_term if n_term else 0.0,
        "false_persistence_rate": float(np.mean(fpr)) if fpr else 0.0,
        "churn": float(np.mean(churn)),
    }


THRESHOLDS = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]


def select_threshold(factory, val: list[Video]) -> float:
    """Pick the score_threshold maximising frame-level F1 on validation videos."""
    best_t, best = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        tp = fp = fn = 0
        for v, p in zip(val, run_method(factory, val, t)):
            a, b, c = frame_f1(v.gt, p)
            tp, fp, fn = tp + a, fp + b, fn + c
        score = _f1(tp, fp, fn)
        if score > best:
            best_t, best = t, score
    return best_t


@dataclass
class BenchmarkResult:
    seed: int
    per_method: dict[str, dict[str, float]] = field(default_factory=dict)
    fitted_params: dict[str, dict] = field(default_factory=dict)


def run_benchmark(seed: int, scenario: ScenarioConfig, n_train: int, n_val: int, n_test: int,
                  trainer: TrainerConfig) -> BenchmarkResult:
    """One full train/val/test run for a data seed."""
    base = seed * 10_000
    train = [generate_video(f"tr{i}", base + i, scenario) for i in range(n_train)]
    val = [generate_video(f"va{i}", base + 1000 + i, scenario) for i in range(n_val)]
    test = [generate_video(f"te{i}", base + 2000 + i, scenario) for i in range(n_test)]

    init = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=NUM_PREDICATES))
    fps = scenario.fps
    factories: dict[str, Callable[[float], Any]] = {
        name: (lambda t, cls=METHODS[name]: cls(score_threshold=t))
        for name in ("no_memory", "ema", "gated_ema", "hmm")
    }
    factories["vhped_handset"] = lambda t: VHPED(params=init, score_threshold=t, fps=fps)

    result = BenchmarkResult(seed=seed)
    ablations = {
        "vt_hmm_fit": dict(use_excitation=False),
        "vhped_novis_fit": dict(use_visibility=False),
        "vhped_self_fit": dict(use_cross_predicate=False),
        "vhped_full_fit": {},
    }
    for name, flags in ablations.items():
        cfg = TrainerConfig(**{**trainer.__dict__, **flags})
        start = init.apply_ablation(**flags)
        fit = fit_vhped([v.noisy for v in train], start, cfg, fps=fps)
        logger.info("seed %d %s: NLL %.4f -> %.4f", seed, name,
                    fit.nll_history[0], fit.nll_history[-1])
        result.fitted_params[name] = fit.params.to_dict()
        factories[name] = lambda t, p=fit.params: VHPED(params=p, score_threshold=t, fps=fps)

    for name, factory in factories.items():
        thr = select_threshold(factory, val)
        metrics = evaluate(test, run_method(factory, test, thr))
        metrics["selected_threshold"] = thr
        result.per_method[name] = metrics
        logger.info("seed %d %-16s %s", seed, name,
                    {k: round(v, 3) for k, v in metrics.items()})
    return result
