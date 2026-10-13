"""Downstream study runner: load one video's pipeline outputs, build the shared observation stream,
run a temporal method (optionally with the flag policy) and score it (prereg §4–§7)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from taorg.adapters.mot17 import FrameTracks
from taorg.adapters.vidor import VidorVideo, load_video
from taorg.experiments.downstream.io import read_csv_gz
from taorg.experiments.downstream.observations import Stream, build_stream, scoring_arrays
from taorg.metrics.downstream import video_outcomes
from taorg.schemas.edges import EdgeKey
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.flag_policy import EndpointInvalidation
from taorg.temporal.gated_variants import GatedEMAHysteresis

FB_GRID = [{"window_size": w} for w in (5, 10, 15, 30, 45, 60, 90, 120)]
B2_GRID = [{"alpha": a, "decay_visible": dv, "decay_occluded": do, "score_threshold": th, "off_ratio": 0.5}
           for a, dv, do, th in itertools.product((0.1, 0.2, 0.3, 0.5), (0.3, 0.5, 0.7, 0.9),
                                                  (0.8, 0.9, 0.95, 0.98, 0.99), (0.2, 0.3, 0.5))]
METHODS = ("FB", "FB+F", "B2", "B2+F")


@dataclass
class VideoData:
    """Everything one video contributes; ``flags`` is the only extra input of the +F variants."""

    video: VidorVideo
    frames: list[FrameTracks]
    stream: Stream
    flags: dict[int, frozenset[int]]
    labels: dict[EdgeKey, dict[str, np.ndarray]]
    missing: list[dict[str, Any]]


def load_video_data(split_dir: Path, ann_path: Path) -> VideoData:
    """Rebuild tracker outputs (detection boxes) and flags from the perception-stage artifacts."""
    v = load_video(ann_path)
    outs = read_csv_gz(split_dir / "outputs.csv.gz")
    n = v.frame_count
    per = {f: {} for f in range(1, n + 1)}
    for r in outs:
        per[int(r["frame"])][int(r["track"])] = ((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                                                 float(r["conf"]))
    frames = [FrameTracks(f, per[f], {}) for f in range(1, n + 1)]
    missing = read_csv_gz(split_dir / "missing_records.csv.gz")
    fl: dict[int, set[int]] = {}
    for r in missing:
        if r["flag"] == "True":
            fl.setdefault(int(r["frame"]), set()).add(int(r["track"]))
    stream = build_stream(v, frames)
    return VideoData(v, frames, stream, {f: frozenset(s) for f, s in fl.items()}, scoring_arrays(v, frames, stream),
                     missing)


def make_method(family: str, params: dict[str, Any], flags: dict[int, frozenset[int]] | None):
    """FB / B2 with the given parameters; wrapped by the flag policy iff ``flags`` is given."""
    inner = FixedWindow(score_threshold=0.0, **params) if family == "FB" else GatedEMAHysteresis(**params)
    return inner if flags is None else EndpointInvalidation(inner, flags)


def run_method(method, d: VideoData) -> dict[EdgeKey, np.ndarray]:
    """Per scored key, bool[T+1] output (index 0 unused)."""
    T = len(d.frames)
    out = {k: np.zeros(T + 1, bool) for k in d.stream.keys}
    method.reset_video(d.video.video_id)
    for g in d.stream.graphs:
        res = method.step(g)
        for o in res.edge_observations:
            if o.edge_key in out:
                out[o.edge_key][g.frame_idx] = True
    method.finalize_video()
    return out


def score_method(family: str, params: dict[str, Any], d: VideoData, flagged: bool) -> dict[str, float]:
    """Run and score one method on one video."""
    m = make_method(family, params, d.flags if flagged else None)
    return video_outcomes(run_method(m, d), d.labels, d.video.fps, first=1)
