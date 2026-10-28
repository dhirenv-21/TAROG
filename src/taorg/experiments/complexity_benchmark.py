"""Phase 5 scaling benchmark: eager VHPED vs LazyVHPED on synthetic traces.

Varies the number of active pairs A, the fraction f of pairs with a record
in each frame (touched), and the predicate count P.  Reports per-frame
temporal-module latency (output enumeration included and also counted
separately), operation counters, heap statistics, and decision mismatches
against the eager oracle (must be 0 outside declared near-ties).
"""

from __future__ import annotations

import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.vhped import HAND_SET_CONFIG, VHPED, LazyVHPED, VHPEDParams

VIS, OCC = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE


@dataclass
class TraceConfig:
    num_pairs: int
    touch_fraction: float
    num_predicates: int = 26
    preds_per_pair: int = 2
    warmup_frames: int = 5
    frames: int = 60
    fps: float = 30.0
    occluded_fraction: float = 0.2
    seed: int = 0
    pattern: str = "random"   # "random": each pair touched w.p. f; "alternating": all pairs
                              # touched on even frames, none on odd frames (adversarial)


def make_trace(cfg: TraceConfig) -> list[FrameGraph]:
    """All pairs observed during warm-up; afterwards a random f-fraction is touched per frame."""
    rng = random.Random(cfg.seed)
    preds = {i: rng.sample(range(cfg.num_predicates), cfg.preds_per_pair)
             for i in range(cfg.num_pairs)}
    frames = []
    for t in range(cfg.warmup_frames + cfg.frames):
        ts = t / cfg.fps
        warm = t < cfg.warmup_frames
        edges = []
        for i in range(cfg.num_pairs):
            if not warm:
                if cfg.pattern == "alternating":
                    if t % 2:
                        continue
                elif rng.random() > cfg.touch_fraction:
                    continue
            vis = OCC if (not warm and rng.random() < cfg.occluded_fraction) else VIS
            for p in preds[i]:
                seen = vis == VIS and (warm or rng.random() < 0.7)
                edges.append(EdgeObservation(
                    edge_key=EdgeKey(i, 100_000 + i, p), video_id="bench", frame_idx=t,
                    timestamp=ts, predicate_score=0.9 if seen else 0.0, observed=seen,
                    subject_visibility=VIS, object_visibility=vis))
        frames.append(FrameGraph("bench", t, ts, edge_observations=edges))
    return frames


def _time_model(model, frames: list[FrameGraph], warmup: int) -> tuple[list[set], np.ndarray]:
    model.reset_video("bench")
    outs, times = [], []
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        out = model.step(f)
        dt = time.perf_counter() - t0
        outs.append({o.edge_key for o in out.edge_observations})
        if i >= warmup:
            times.append(dt)
    return outs, np.array(times) * 1000.0


def _peak_kib(model, frames: list[FrameGraph]) -> float:
    """Peak traced Python heap (KiB) while running the model over the trace."""
    tracemalloc.start()
    model.reset_video("bench")
    for f in frames:
        model.step(f)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def run_point(cfg: TraceConfig, threshold: float = 0.5,
              measure_memory: bool = False) -> dict[str, Any]:
    params = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=cfg.num_predicates,
                                          rho_unk=0.0))
    frames = make_trace(cfg)
    eager = VHPED(params=params, score_threshold=threshold, fps=cfg.fps)
    lazy = LazyVHPED(params=params, score_threshold=threshold, fps=cfg.fps)
    e_out, e_ms = _time_model(eager, frames, cfg.warmup_frames)
    l_out, l_ms = _time_model(lazy, frames, cfg.warmup_frames)
    mismatches = sum(len(a ^ b) for a, b in zip(e_out, l_out))
    c = lazy.counters
    memory = {}
    if measure_memory:  # separate pass so tracing does not distort the timings
        memory = {
            "eager_peak_kib": _peak_kib(VHPED(params=params, score_threshold=threshold,
                                              fps=cfg.fps), frames),
            "lazy_peak_kib": _peak_kib(LazyVHPED(params=params, score_threshold=threshold,
                                                 fps=cfg.fps), frames),
        }
    return {
        "pattern": cfg.pattern,
        **memory,
        "num_pairs": cfg.num_pairs,
        "touch_fraction": cfg.touch_fraction,
        "num_predicates": cfg.num_predicates,
        "eager_ms_mean": float(e_ms.mean()),
        "eager_ms_p95": float(np.percentile(e_ms, 95)),
        "lazy_ms_mean": float(l_ms.mean()),
        "lazy_ms_p95": float(np.percentile(l_ms, 95)),
        "speedup": float(e_ms.mean() / l_ms.mean()),
        "decision_mismatches": int(mismatches),
        "lazy_exact": lazy.lazy_exact,
        "pair_steps": c.pair_steps,
        "eager_equivalent_pair_steps": c.eager_equivalent_pair_steps,
        "materializations": c.materializations,
        "wakeups": c.wakeups,
        "early_wakeups": c.early_wakeups,
        "output_edges_per_frame": c.output_edges / max(c.frames, 1),
        "heap": lazy._heap.counters.to_dict(),
    }


def run_grid(pairs: list[int], fractions: list[float], predicates: list[int],
             frames: int, seed: int, pattern: str = "random",
             measure_memory: bool = False) -> list[dict[str, Any]]:
    rows = []
    for p in predicates:
        for f in fractions:
            for a in pairs:
                rows.append(run_point(TraceConfig(num_pairs=a, touch_fraction=f,
                                                  num_predicates=p, frames=frames, seed=seed,
                                                  pattern=pattern),
                                      measure_memory=measure_memory))
    return rows
