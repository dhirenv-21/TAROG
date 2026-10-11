"""DEVELOPMENT PILOT (dev seeds 7000-7009 only; never registered evaluation).

Feasibility check of the run-outcome termination rule before pre-registration: B2's frozen
base parameters + a small grid of (tau, kappa) per mode.  The prior is fitted on the
episode-study CALIBRATION input streams (C1-C7 x 5000-5019; observable data only).
Output: results/next_method/audit/dev_pilot.csv, prior_dev.json
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = list(range(7000, 7010))
OUT = Path("results/next_method/audit")
TAG = "_v2_visible_b2"  # iteration 2 (iteration 1 = dev_pilot.csv, visible_b2 absent/False)
GRID = list(itertools.product(("km", "static", "pooled", "geometric"), (0.2, 0.4, 0.6, 0.8), (20.0, 200.0)))


def _q():
    logging.disable(logging.WARNING)


def _task(a):
    cname, cdict, seed, world, base, metrics, b2, prior = a
    from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance
    from taorg.experiments.episodes.evaluation import compute_all, run_model
    from taorg.temporal import METHODS
    from taorg.temporal.cure import HysteresisRunPosterior
    inst = build_instance(EpisodeCondition.from_dict(cname, cdict), seed, world, base)
    rows = []
    pred = run_model(METHODS["gated_ema_hyst"](**b2), inst.input_frames, "v")
    rows.append(("B2", compute_all(inst, pred, metrics)))
    bp = dict(b2)
    for mode, tau, kappa in GRID:
        m = HysteresisRunPosterior(**bp, tau=tau, kappa=kappa, mode=mode, prior=prior)
        rows.append((f"{mode}|{tau}|{kappa}", compute_all(inst, run_model(m, inst.input_frames, "v"), metrics)))
    return cname, seed, [(n, {k: r[k] for k in ("J", "fpt_s", "frr", "std_mean_s", "rdd_mean_s", "gap_recall",
                                              "churn", "ghost_time_s", "bridge_rate")}) for n, r in rows]


def main(workers=4):
    from taorg.experiments.episodes import pipeline as P
    from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance
    from taorg.temporal.cure import fit_prior
    cfg = P.load_config(Path("configs/experiments/episodes.yaml"))
    b2 = json.loads(Path("results/episodes/configs/frozen_parameters.json").read_text(encoding="utf-8"))["selected"]["gated_ema_hyst"]["params"]
    streams = [build_instance(EpisodeCondition.from_dict(c, d), s, cfg["world"], cfg["base_corruption"]).input_frames
               for c, d in cfg["conditions"].items() if d.get("calibrate") for s in cfg["seeds"]["calibration"]]
    prior = fit_prior(streams, 3.0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prior_dev.json").write_text(json.dumps(prior), encoding="utf-8")
    tasks = [(c, cfg["conditions"][c], s, cfg["world"], cfg["base_corruption"], cfg["metrics"], b2, prior)
             for c in cfg["conditions"] for s in SEEDS]
    acc = defaultdict(lambda: defaultdict(list))
    with mp.get_context("spawn").Pool(workers, initializer=_q) as pool:
        for c, s, rows in pool.imap_unordered(_task, tasks, chunksize=1):
            for n, r in rows:
                for k, v in r.items():
                    acc[(c, n)][k].append(v)
    out = []
    for (c, n), d in sorted(acc.items()):
        out.append({"condition": c, "variant": n, **{k: float(np.nanmean(v)) for k, v in d.items()}})
    with open(OUT / f"dev_pilot{TAG}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
