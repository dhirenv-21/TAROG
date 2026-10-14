"""Final evaluation runner (research/taorg_final_preregistration.md §4-§7).

Methods (identical inputs): NM (NoMemory), FB (FixedWindow), B2 (GatedEMA + hysteresis),
VH (V-HPED M3 with certified lazy maintenance), FB+F, B2+F (endpoint-invalidation flag policy).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from taorg.adapters.mot17 import FrameTracks
from taorg.adapters.vidor import VidorVideo, load_video
from taorg.experiments.downstream.io import read_csv_gz
from taorg.experiments.final.stream import PredStream, build_predicted_stream, load_sgg, oracle_keys, scoring_labels
from taorg.metrics.downstream import genuine_outcomes, video_outcomes
from taorg.schemas.edges import EdgeKey
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.flag_policy import EndpointInvalidation
from taorg.temporal.gated_variants import GatedEMAHysteresis
from taorg.temporal.lazy_vhped import LazyVHPED
from taorg.temporal.no_memory import NoMemory

STRIDE = 3
METHODS = ("NM", "FB", "B2", "VH", "FB+F", "B2+F")
TAU_GRID = (0.005, 0.01, 0.02, 0.05, 0.1)
FB_GRID = [{"window_size": w} for w in (1, 2, 3, 5, 10, 15, 20, 30, 40)]
B2_GRID = [{"alpha": a, "decay_visible": dv, "decay_occluded": do, "score_threshold": th, "off_ratio": 0.5}
           for a, dv, do, th in itertools.product((0.1, 0.2, 0.3, 0.5), (0.3, 0.5, 0.7, 0.9),
                                                  (0.5, 0.7, 0.8, 0.9, 0.95), (0.2, 0.3, 0.5))]
VH_GRID = [{"mu": mu, "nu": nu, "gamma_o": go, "gamma_v": gv, "miss": m, "false_pos": fp, "threshold": th,
            "r_ov": 0.05, "r_vo": 0.05, "dt_ref": STRIDE / 30.0}
           for mu, nu, go, gv, m, fp, th in itertools.product((0.2, 0.5, 1.0), (0.01, 0.05), (0.5, 2.0), (1.0, 3.0),
                                                             (0.05, 0.2), (0.01, 0.05), (0.3, 0.5, 0.7))]


@dataclass
class FinalVideo:
    """One video's inputs (inference side) and labels (oracle side)."""

    video: VidorVideo
    frames: list[FrameTracks]
    sgg: dict
    sampled: list[int]
    flags: dict[int, frozenset[int]]      # sampled frame -> tracks flagged in (previous sampled, this]
    fps_eff: float


def load_final_video(perc_dir: Path, sgg_path: Path, ann_path: Path) -> FinalVideo:
    """Tracker outputs (detection boxes), EGTR person relations, per-step flags."""
    v = load_video(ann_path)
    n = v.frame_count
    per = {f: {} for f in range(1, n + 1)}
    for r in read_csv_gz(perc_dir / "outputs.csv.gz"):
        per[int(r["frame"])][int(r["track"])] = ((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                                                 float(r["conf"]))
    frames = [FrameTracks(f, per[f], {}) for f in range(1, n + 1)]
    sampled = list(range(1, n + 1, STRIDE))
    raw: dict[int, set[int]] = {}
    for r in read_csv_gz(perc_dir / "missing_records.csv.gz"):
        if r["flag"] == "True":
            raw.setdefault(int(r["frame"]), set()).add(int(r["track"]))
    flags, prev = {}, 0
    for f in sampled:
        s = set().union(*[raw.get(x, set()) for x in range(prev + 1, f + 1)])
        if s:
            flags[f] = frozenset(s)
        prev = f
    return FinalVideo(v, frames, load_sgg(sgg_path), sampled, flags, v.fps / STRIDE)


def make_method(name: str, params: dict[str, Any], flags):
    """Method factory; '+F' wraps FB / B2 with the endpoint-invalidation policy."""
    base = name.split("+")[0]
    if base == "NM":
        m = NoMemory(score_threshold=0.0)
    elif base == "FB":
        m = FixedWindow(score_threshold=0.0, **params)
    elif base == "B2":
        m = GatedEMAHysteresis(**params)
    elif base == "VH":
        m = LazyVHPED(**params, rung="m3")
    else:
        raise KeyError(name)
    return EndpointInvalidation(m, flags) if name.endswith("+F") else m


@dataclass
class Prepared:
    """Stream and labels of one video at one tau (shared by every method)."""

    d: FinalVideo
    ps: PredStream
    keys: list[EdgeKey]
    labels: dict


def prepare(d: FinalVideo, tau: float) -> Prepared:
    """Build the predicted stream at threshold tau and the oracle labels of the scored keys."""
    ps = build_predicted_stream(d.video.video_id, d.video.fps, d.frames, d.sgg, d.sampled, tau)
    _, matches = scoring_labels(d.video, d.frames, [], d.sampled)
    keys = list(dict.fromkeys(ps.predicted_keys + oracle_keys(d.video, d.frames, matches, d.sampled)))
    labels, _ = scoring_labels(d.video, d.frames, keys, d.sampled)
    return Prepared(d, ps, keys, labels)


def run_outputs(p: Prepared, name: str, params: dict[str, Any]) -> tuple[dict[EdgeKey, np.ndarray], Any]:
    """Per scored key, bool[n+1] output over the sampled steps (index 0 unused); the method object."""
    d = p.d
    n = len(d.sampled)
    pos = {f: i for i, f in enumerate(d.sampled, start=1)}
    m = make_method(name, params, d.flags)
    out = {k: np.zeros(n + 1, bool) for k in p.keys}
    m.reset_video(d.video.video_id)
    for g in p.ps.graphs:
        for o in m.step(g).edge_observations:
            if o.edge_key in out:
                out[o.edge_key][pos[g.frame_idx]] = True
    return out, m


def outcomes(p: Prepared, out: dict[EdgeKey, np.ndarray]) -> dict[str, float]:
    """Registered outcomes of one method on one video."""
    r = video_outcomes(out, p.labels, p.d.fps_eff)
    r.update(genuine_outcomes(out, p.labels, p.d.fps_eff))
    tp = sum(int((out[k] & p.labels[k]["alive"])[1:].sum()) for k in p.keys)
    fp = sum(int((out[k] & ~p.labels[k]["alive"])[1:].sum()) for k in p.keys)
    fn = sum(int((~out[k] & p.labels[k]["alive"])[1:].sum()) for k in p.keys)
    r.update({"tp": float(tp), "fp": float(fp), "fn": float(fn), "n_predicted_keys": float(len(p.ps.predicted_keys))})
    return r


def score_video(d: FinalVideo, tau: float, configs: dict[str, dict], methods=METHODS) -> dict[str, dict]:
    """Prepare once, run every method, return outcomes (VH also reports lazy counters)."""
    p = prepare(d, tau)
    res = {}
    for name in methods:
        out, m = run_outputs(p, name, configs.get(name.split("+")[0], {}))
        r = outcomes(p, out)
        if name == "VH":
            r.update({f"lazy_{k}": float(v) for k, v in m.counters().items()})
        res[name] = r
    return res


def configs_from(params: dict[str, dict]) -> dict[str, dict]:
    """Per-family parameter dicts (FB/B2/VH), reused by the +F variants."""
    return {k: dict(v) for k, v in params.items()}


def to_keystr(k: EdgeKey) -> str:
    return f"{k.subject_track_id}-{k.object_track_id}-{k.predicate_id}"
