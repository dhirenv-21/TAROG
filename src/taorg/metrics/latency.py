"""Per-frame temporal-module latency instrumentation.

Measures wall-clock time of the temporal method's step() call only,
excluding base model inference.

See research/metric_definitions.md §7 for the frozen definition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from taorg.schemas.edges import FrameGraph
from taorg.temporal.base import TemporalEdgeModel


@dataclass(frozen=True)
class LatencyResult:
    """Result of latency measurement.

    Attributes:
        per_frame_ms: List of per-frame latencies in milliseconds.
        mean_ms: Mean latency in ms.
        median_ms: Median latency in ms.
        p95_ms: 95th percentile latency in ms.
        p99_ms: 99th percentile latency in ms.
        total_frames: Number of frames measured.
    """

    per_frame_ms: list[float]
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    total_frames: int


def measure_latency(
    model: TemporalEdgeModel,
    frames: list[FrameGraph],
    video_id: str = "latency_test",
    warmup_frames: int = 0,
) -> LatencyResult:
    """Measure per-frame latency of a temporal method.

    Args:
        model: Temporal method to measure.
        frames: Input frame sequence.
        video_id: Video ID for reset.
        warmup_frames: Number of initial frames to exclude from measurement.

    Returns:
        LatencyResult with per-frame and aggregate statistics.
    """
    model.reset_video(video_id)

    timings: list[float] = []
    for i, frame in enumerate(frames):
        start = time.perf_counter()
        model.step(frame)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        if i >= warmup_frames:
            timings.append(elapsed)

    model.finalize_video()

    if not timings:
        return LatencyResult(
            per_frame_ms=[], mean_ms=0.0, median_ms=0.0,
            p95_ms=0.0, p99_ms=0.0, total_frames=0,
        )

    sorted_t = sorted(timings)
    n = len(sorted_t)

    mean_ms = sum(sorted_t) / n
    median_ms = sorted_t[n // 2]
    p95_ms = sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1]
    p99_ms = sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1]

    return LatencyResult(
        per_frame_ms=timings,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        total_frames=n,
    )
