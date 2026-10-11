"""Phase 5H-1 runtime/memory benchmark on DEVELOPMENT data only (TRAIN videos; never SELECT/EVAL).

Measures load, design build, exact row aggregation (amendment A1) and per-target fit cost on a
TRAIN subset, checks aggregated == unaggregated fits, and extrapolates to the registered matrix.

Usage:
    python scripts/phase5h1/benchmark.py --n-videos 1000 --out D:/TAORG_runs/phase5h1/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=1000)
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import numpy as np
    from taorg.hawkes.glm import aggregate_target, build_design, fit_target, fit_target_agg, ll_agg_per_group
    from taorg.hawkes.phase5h1 import build_layout, guard, load_config, split_indices
    from taorg.hawkes.stage0 import load_config as s0load
    from taorg.hawkes.vidor import load_videos

    guard({"TRAIN"}, allow_eval=False)
    cfg = load_config(Path("configs/experiments/phase5h1.yaml"))
    s0 = s0load(Path(cfg["stage0_config"]))
    res: dict = {"LABEL": "BENCHMARK on TRAIN videos only; not evidence"}
    t = time.time()
    train, vocab, _ = load_videos(s0["data"]["root"], s0["data"]["train_dir"])
    res["load_all_training_s"] = round(time.time() - t, 1)
    res["rss_after_load_gb"] = round(rss_gb(), 2)
    idx = split_indices(len(train), cfg, s0)
    tr = [train[i] for i in idx["TRAIN"]]
    lay, targets = build_layout(tr, vocab, cfg)
    sub = list(idx["TRAIN"])
    random.Random(99).shuffle(sub)
    vids = [train[i] for i in sorted(sub[: a.n_videos])]
    del train
    t = time.time()
    d = build_design(vids, lay, targets)
    res.update(n_videos=len(vids), design_build_s=round(time.time() - t, 1), n_segments=int(len(d.seg_dur)),
               X_nnz=int(d.X.nnz), rss_after_design_gb=round(rss_gb(), 2))
    grp = np.array([v % 5 for v in d.seg_video])            # stand-in fold labels for the benchmark
    t = time.time()
    tds = {k: aggregate_target(d, k, grp) for k in targets}
    res["aggregate_all_targets_s"] = round(time.time() - t, 1)
    res["rows_total"] = int(sum(td.n_rows for td in tds.values()))
    res["rows_agg_total"] = int(sum(len(td.y) for td in tds.values()))
    res["agg_nnz_total"] = int(sum(td.X.nnz for td in tds.values()))
    res["rss_after_agg_gb"] = round(rss_gb(), 2)
    # fit timing: biggest targets by rows, G3 and G4, aggregated vs raw (raw only for G4 on a few)
    order = sorted(targets, key=lambda k: -tds[k].n_rows)
    pick = order[: a.n_targets // 2] + order[-a.n_targets // 2:]
    rows = []
    for k in pick:
        for rung in ("G3", "G4"):
            t = time.time()
            fr = fit_target_agg(tds[k], lay, rung, 1.0)
            ta = time.time() - t
            row = {"k": k, "rung": rung, "agg_s": round(ta, 2), **{x: fr.diag[x] for x in
                   ("nit", "success", "n_rows_agg", "n_params", "proj_grad_inf")}, "n_rows_raw": tds[k].n_rows}
            if rung == "G4" and k in pick[:2] + pick[-1:]:
                t = time.time()
                raw = fit_target(d, k, rung, 1.0)
                row["raw_s"] = round(time.time() - t, 2)
                row["max_abs_w_diff"] = float(np.abs(raw.w - fr.fit.w).max())
                row["eta_diff"] = abs(raw.eta - fr.fit.eta)
                la, _ = ll_agg_per_group(tds[k], fr.fit, lay, 5, True)
                lr, _ = ll_agg_per_group(tds[k], raw, lay, 5, True)
                row["ll_diff"] = float(abs(la.sum() - lr.sum()))
            rows.append(row)
            logging.info("%s", row)
    res["fits"] = rows
    res["peak_rss_gb"] = round(rss_gb(), 2)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"benchmark_{a.n_videos}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "fits"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
