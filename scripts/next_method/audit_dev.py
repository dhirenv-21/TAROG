"""Next-method failure audit on DESIGNATED DEVELOPMENT seeds (7000-7019).

Development data only: never used for any registered tuning or evaluation. Measures, for
the strongest practical baseline B2 (GatedEMA + hysteresis, frozen episode-study parameters):

  A2  headroom: per-condition best of B2's 200 sampled configurations (condition-matched,
      label-using oracle tuning) vs the frozen pooled operating point;
  A3  error decomposition of the frozen B2 point into frame-error types;
  A4  observable absence-run statistics per video (is there enough information in ONE
      video to recalibrate absence evidence online?).

Outputs: results/next_method/audit/{a2_headroom.csv, a3_errors.csv, a4_runs.csv, audit.json}
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEV_SEEDS = list(range(7000, 7020))
OUT = Path("results/next_method/audit")


def _quiet():
    logging.disable(logging.WARNING)


def _task(args):
    cname, cdict, seed, world, base, metrics, confs, frozen_b2, frozen_others = args
    from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance
    from taorg.experiments.episodes.evaluation import evaluate_config, run_model, build_model
    from taorg.schemas.tracks import VisibilityState
    cond = EpisodeCondition.from_dict(cname, cdict)
    inst = build_instance(cond, seed, world, base)
    j = []
    for cid, params in confs:
        m, _ = evaluate_config(inst, "gated_ema_hyst", params, metrics)
        j.append((cid, m["J"], m["churn"]))
    ref = {}
    for name, (meth, params) in frozen_others.items():
        m, _ = evaluate_config(inst, meth, params, metrics)
        ref[name] = m["J"]
    # A3 error decomposition of frozen B2
    pred = run_model(build_model("gated_ema_hyst", frozen_b2), inst.input_frames, "v")
    out = [{o.edge_key for o in fr.edge_observations} for fr in pred]
    n = len(out)
    win = defaultdict(set)
    for kind in ("occlusion", "termination_occlusion", "post_termination_flags"):
        for e in (inst.manifests.get(kind) or {}).get("entries", []):
            k = tuple(e["edge_key_tuple"])
            win[kind].update((k, f) for f in range(e["start_frame"], e["end_frame"] + 1))
    err = defaultdict(int)
    for k, eps in inst.gt_episodes.items():
        kt = tuple(k)
        alive = [any(a <= f <= b for a, b in eps) for f in range(n)]
        for a, b in eps:
            first = next((f for f in range(a, b + 1) if k in out[f]), b + 1)
            for f in range(a, b + 1):
                if k in out[f]:
                    continue
                if (kt, f) in win["occlusion"]:
                    err["fn_temporary_gap"] += 1
                elif (kt, f) in win["termination_occlusion"]:
                    err["fn_hidden_end_block"] += 1
                elif f < first:
                    err["fn_onset_delay"] += 1
                else:
                    err["fn_fragmentation_visible"] += 1
        for f in range(n):
            if k in out[f] and not alive[f]:
                if (kt, f) in win["termination_occlusion"]:
                    err["fp_hidden_termination"] += 1
                elif (kt, f) in win["post_termination_flags"]:
                    err["fp_misleading_flags"] += 1
                else:
                    prev_alive = any(b < f and f - b <= 30 for _, b in eps)
                    err["fp_visible_lag_or_reactivation" if prev_alive else "fp_false_birth"] += 1
    for fr in out:
        err["fp_ghost"] += sum(1 for k in fr if k not in inst.gt_episodes)
    # A4 observable absence runs (input only): runs between detections of a key
    runs = defaultdict(int)
    for k in {o.edge_key for fr in inst.input_frames for o in fr.edge_observations if o.observed}:
        seen = False
        cur = []
        for fr in inst.input_frames:
            rec = next((o for o in fr.edge_observations if o.edge_key == k), None)
            if rec is not None and rec.observed:
                if seen and cur:
                    maj = max(set(cur), key=cur.count)
                    runs[f"redetected_{maj}"] += 1
                    runs[f"redetected_len_{maj}"] += len(cur)
                seen, cur = True, []
            elif seen:
                cur.append("NONE" if rec is None else rec.edge_visibility.name)
        if seen and cur:
            runs["open_at_end"] += 1
    return cname, seed, j, ref, dict(err), dict(runs)


def main(workers: int = 4) -> None:
    sys.path.insert(0, "src")
    from taorg.experiments.episodes import pipeline as P
    from taorg.experiments.episodes.search import method_configs
    cfg = P.load_config(Path("configs/experiments/episodes.yaml"))
    assert not set(DEV_SEEDS) & (set(cfg["seeds"]["calibration"]) | set(cfg["seeds"]["evaluation"]) | set(cfg["seeds"]["smoke"]))
    frozen = json.loads(Path("results/episodes/configs/frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    confs = [(c.config_id, c.params) for c in method_configs(cfg, ["gated_ema_hyst"])]
    others = {m: (m, frozen[m]["params"]) for m in ("gated_ema", "gated_ema_hyst", "va_hmm", "hsmm", "episode")}
    tasks = [(c, cfg["conditions"][c], s, cfg["world"], cfg["base_corruption"], cfg["metrics"], confs,
              frozen["gated_ema_hyst"]["params"], others) for c in cfg["conditions"] for s in DEV_SEEDS]
    res = []
    with mp.get_context("spawn").Pool(workers, initializer=_quiet) as pool:
        for k, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(tasks)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    conds = list(cfg["conditions"])
    a2, a3, a4, summary = [], [], [], {}
    for c in conds:
        rc = [r for r in res if r[0] == c]
        J = defaultdict(list)
        for _, _, j, _, _, _ in rc:
            for cid, jj, _ in j:
                J[cid].append(jj)
        mean = {cid: float(np.mean(v)) for cid, v in J.items()}
        best = max(mean, key=mean.get)
        pooled = frozen["gated_ema_hyst"]["config_id"]
        refs = {m: float(np.mean([r[3][m] for r in rc])) for m in others}
        a2.append({"condition": c, "B2_pooled_J": mean[pooled], "B2_matched_best_J": mean[best],
                   "B2_matched_config": best, "headroom": mean[best] - mean[pooled], **{f"J_{m}": v for m, v in refs.items()}})
        errs = defaultdict(list)
        for r in rc:
            for k in ("fn_temporary_gap", "fn_hidden_end_block", "fn_onset_delay", "fn_fragmentation_visible",
                      "fp_hidden_termination", "fp_misleading_flags", "fp_visible_lag_or_reactivation",
                      "fp_false_birth", "fp_ghost"):
                errs[k].append(r[4].get(k, 0))
        a3.append({"condition": c, **{k: float(np.mean(v)) for k, v in errs.items()}})
        rk = defaultdict(list)
        for r in rc:
            for k in ("redetected_VISIBLE", "redetected_OCCLUDED_OR_UNRELIABLE", "redetected_UNKNOWN", "redetected_NONE",
                      "open_at_end"):
                rk[k].append(r[5].get(k, 0))
        a4.append({"condition": c, **{f"runs_{k}": float(np.mean(v)) for k, v in rk.items()}})
    for name, rows in (("a2_headroom", a2), ("a3_errors", a3), ("a4_runs", a4)):
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    summary = {"dev_seeds": DEV_SEEDS, "n_tasks": len(res), "b2_frozen": frozen["gated_ema_hyst"]["config_id"],
               "n_b2_configs": len(confs), "note": "development data only; not registered evaluation"}
    (OUT / "audit.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
