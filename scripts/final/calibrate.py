"""Final evaluation, DEVELOPMENT calibration and gate (research/taorg_final_preregistration.md §5, §7).

    python scripts/final/calibrate.py [--workers 4]

tau_obs by mean NM frame F1; then FB / B2 / VH grids by mean J (unflagged). Gate G1-G3.
Outputs results/final/dev/{calibration.csv.gz, gate.json}. The flag is never used here.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.downstream.io import write_csv_gz
from taorg.experiments.final.runner import (B2_GRID, FB_GRID, TAU_GRID, VH_GRID, load_final_video, outcomes,
                                            prepare, run_outputs)
from taorg.utils.seed import set_seed

PERC = Path("D:/TAORG_runs/downstream/dev")
SGG = Path("D:/TAORG_runs/egtr/sgg/dev")
OUT = Path("results/final/dev")
_P = None


def videos():
    return [(Path(vp).stem, vp) for vp in sample(Path("data/vidor"), "training")]


def load(vid, vp):
    return load_final_video(PERC / vid, SGG / f"{vid}.npz", Path("data/vidor/training") / vp.replace(".mp4", ".json"))


def _init(tau):
    global _P
    _P = [prepare(load(v, vp), tau) for v, vp in videos()]


def _eval(job):
    fam, ci, params = job
    js, igt = [], 0.0
    for p in _P:
        out, _ = run_outputs(p, fam, params)
        r = outcomes(p, out)
        js.append(r["J"])
        igt += r["igt_s"]
    return fam, ci, float(np.nanmean(js)), igt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    set_seed(20261013)
    rec = json.loads(Path("results/final/dev/sgg_records.json").read_text(encoding="utf-8"))
    vids = videos()
    missing = [v for v, _ in vids if rec["videos"].get(v, {}).get("status") != "done"]
    if missing:
        raise SystemExit(f"G1 fails: EGTR incomplete for {missing}")
    data = [load(v, vp) for v, vp in vids]
    rows, tau_f1 = [], {}
    for tau in TAU_GRID:
        f1, npk = [], []
        for d in data:
            p = prepare(d, tau)
            r = outcomes(p, run_outputs(p, "NM", {})[0])
            f1.append(r["frame_f1"])
            npk.append(r["n_predicted_keys"])
        tau_f1[tau] = (float(np.nanmean(f1)), int(sum(x >= 1 for x in npk)))
        rows.append({"family": "tau", "config": TAU_GRID.index(tau), "params": json.dumps({"tau": tau}),
                     "mean_J": float("nan"), "mean_NM_frame_f1": tau_f1[tau][0]})
        print("tau", tau, tau_f1[tau], flush=True)
    tau = max(TAU_GRID, key=lambda t: (tau_f1[t][0], -TAU_GRID.index(t)))
    jobs = [(f, i, p) for f, g in (("FB", FB_GRID), ("B2", B2_GRID), ("VH", VH_GRID)) for i, p in enumerate(g)]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(tau,)) as ex:
        for fam, ci, mj, igt in ex.map(_eval, jobs, chunksize=4):
            res[(fam, ci)] = (mj, igt)
    best = {}
    for fam, grid in (("FB", FB_GRID), ("B2", B2_GRID), ("VH", VH_GRID)):
        scores = [res[(fam, i)][0] for i in range(len(grid))]
        bi = int(np.argmax(scores))
        best[fam] = {"config": bi, "params": grid[bi], "mean_J": scores[bi], "igt_s": res[(fam, bi)][1]}
        for i, p in enumerate(grid):
            rows.append({"family": fam, "config": i, "params": json.dumps(p, sort_keys=True), "mean_J": scores[i],
                         "mean_NM_frame_f1": float("nan")})
        print(fam, best[fam], flush=True)
    write_csv_gz(OUT / "calibration.csv.gz", rows, ["family", "config", "params", "mean_J", "mean_NM_frame_f1"])
    f1, nvid = tau_f1[tau]
    gate = {"tau_obs": tau, "tau_scan": {str(k): v for k, v in tau_f1.items()}, "selected": best,
            "G1": True, "G2": {"NM_frame_f1": f1, "videos_with_predicted_keys": nvid, "pass": bool(f1 >= 0.05 and nvid >= 20)},
            "G3": {"igt_s_B2_unflagged": best["B2"]["igt_s"], "pass": bool(best["B2"]["igt_s"] >= 5.0)}}
    gate["PASS"] = bool(gate["G2"]["pass"] and gate["G3"]["pass"])
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: gate[k] for k in ("tau_obs", "G2", "G3", "PASS")}, indent=1, default=float))


if __name__ == "__main__":
    main()
