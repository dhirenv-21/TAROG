"""Phase 5H-1 driver: registered splits, leakage guard, layout from TRAIN, smoke pipeline.

Split policy (research/phase5h1_preregistration.md §2):
    TRAIN = Stage 0 FIT; SELECT / EVAL-A = Stage 0 DEV halved with seed 6101; EVAL-B = VidOR validation.
    DEV-SMOKE = 300 TRAIN videos (seed 6103) for pipeline checks only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from taorg.hawkes.glm import Layout, build_design, fit_rung, ll_rung, n_events_per_video
from taorg.hawkes.stage0 import load_config as load_stage0_config
from taorg.hawkes.stage0 import splits as stage0_splits
from taorg.hawkes.vidor import VideoRecord, load_videos

logger = logging.getLogger(__name__)

EVAL_SPLITS = {"SELECT", "EVAL_A", "EVAL_B"}


class LeakageError(RuntimeError):
    """Raised when development code asks for registered selection/evaluation data."""


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_source_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def split_indices(n_train_videos: int, cfg: dict[str, Any], s0cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Registered index sets into the sorted training list (EVAL_B is the validation split)."""
    dev, fitv, _ = stage0_splits([None] * n_train_videos, s0cfg)   # only the count is used
    d = list(dev)
    random.Random(cfg["splits"]["dev_half_seed"]).shuffle(d)
    half = len(d) // 2
    tr = list(fitv)
    random.Random(cfg["splits"]["smoke_seed"]).shuffle(tr)
    s = cfg["splits"]
    return {"TRAIN": np.array(sorted(fitv)), "SELECT": np.array(sorted(d[:half])), "EVAL_A": np.array(sorted(d[half:])),
            "SMOKE_FIT": np.array(sorted(tr[: s["smoke_fit"]])),
            "SMOKE_CHECK": np.array(sorted(tr[s["smoke_fit"]: s["smoke_fit"] + s["smoke_check"]]))}


def guard(requested: set[str], allow_eval: bool) -> None:
    """Fail fast if evaluation splits are requested outside the registered evaluation entry point."""
    bad = requested & EVAL_SPLITS
    if bad and not allow_eval:
        raise LeakageError(f"Development code requested registered evaluation data {sorted(bad)}; "
                           "only the registered evaluation entry point may load SELECT/EVAL splits.")


def build_layout(train: list[VideoRecord], vocab: list[str], cfg: dict[str, Any]) -> tuple[Layout, list[int]]:
    """Layout and target list from TRAIN only (support rules of prereg §4)."""
    R = len(vocab)
    counts = Counter()
    cats = Counter()
    for v in train:
        cats.update(v.cats.values())
        for pr in v.pairs.values():
            for r, ivs in pr.rels.items():
                for b, e in ivs:
                    if b > 0:
                        counts[2 * r] += 1
                    if e < v.T:
                        counts[2 * r + 1] += 1
    sup = cfg["support"]
    targets = [k for k in range(2 * R) if counts[k] >= sup["target_min_events"]]
    sources = [k for k in range(2 * R) if counts[k] >= sup["source_min_events"]]
    b = cfg["bins"]
    lay = Layout(vocab, sorted(cats), b["lag_edges_s"], b["elapsed_edges_s"], b["pair_events_bins"],
                 b["active_bins"], b["duration_edges_s"], sources)
    return lay, targets


def run_smoke(cfg_path: Path, out: Path, rungs: tuple[str, ...] = ("G0", "G1", "G2", "G3"), ridge: float = 1.0
              ) -> dict[str, Any]:
    """Pipeline smoke test on DEV-SMOKE (TRAIN videos only). Outputs are NOT evidence."""
    guard({"SMOKE_FIT", "SMOKE_CHECK"}, allow_eval=False)
    cfg = load_config(cfg_path)
    s0 = load_stage0_config(Path(cfg["stage0_config"]))
    t0 = time.time()
    train, vocab, _ = load_videos(s0["data"]["root"], s0["data"]["train_dir"])
    idx = split_indices(len(train), cfg, s0)
    fit_v = [train[i] for i in idx["SMOKE_FIT"]]
    chk_v = [train[i] for i in idx["SMOKE_CHECK"]]
    assert not (set(idx["SMOKE_FIT"]) | set(idx["SMOKE_CHECK"])) - set(idx["TRAIN"]), "smoke data must be TRAIN"
    lay, targets = build_layout([train[i] for i in idx["TRAIN"]], vocab, cfg)
    tb = time.time()
    d_fit, d_chk = build_design(fit_v, lay, targets), build_design(chk_v, lay, targets)
    build_s = time.time() - tb
    res: dict[str, Any] = {"LABEL": "SMOKE TEST — pipeline check on 300 TRAIN videos; NOT scientific evidence",
                           "n_fit_videos": len(fit_v), "n_check_videos": len(chk_v),
                           "n_targets": len(targets), "n_sources": len(lay.sources), "n_columns": lay.size,
                           "n_segments_fit": int(len(d_fit.seg_dur)), "design_build_s": round(build_s, 1),
                           "ridge": ridge, "rungs": {}}
    n_ev = n_events_per_video(d_chk).sum()
    for rung in rungs:
        tf = time.time()
        fits = fit_rung(d_fit, rung, ridge, cfg["fit"]["max_iter"])
        ll1 = ll_rung(d_chk, fits, rung)
        ll2 = ll_rung(d_chk, fit_rung(d_fit, rung, ridge, cfg["fit"]["max_iter"]), rung)
        res["rungs"][rung] = {"check_ll_per_event": float(ll1.sum() / n_ev), "finite": bool(np.isfinite(ll1).all()),
                              "reproducible": bool(np.array_equal(ll1, ll2)), "fit_s": round(time.time() - tf, 1)}
        logger.info("smoke %s: %s", rung, res["rungs"][rung])
    res["n_check_events"] = int(n_ev)
    res["elapsed_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
