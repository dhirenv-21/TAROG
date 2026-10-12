"""Downstream study, DEVELOPMENT calibration (prereg §7) and feasibility gate (§9).

    python scripts/downstream/calibrate.py

Unflagged FB / B2 grids on the 30 development videos; selection by mean J; gate F1-F3.
No flagged method is scored here. Outputs: results/downstream/dev/{calibration.csv.gz, gate.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.downstream.io import write_csv_gz
from taorg.experiments.downstream.runner import B2_GRID, FB_GRID, load_video_data, score_method
from taorg.metrics.episodes import runs
from taorg.utils.seed import set_seed

HEAVY = Path("D:/TAORG_runs/downstream/dev")
OUT = Path("results/downstream/dev")


def main() -> None:
    set_seed(20261013)
    rec = json.loads((OUT / "pipeline_records.json").read_text(encoding="utf-8"))
    vids = [(Path(vp).stem, vp) for vp in sample(Path("data/vidor"), "training")]
    missing = [v for v, _ in vids if rec["videos"].get(v, {}).get("status") != "done"]
    if missing:
        raise SystemExit(f"F1 fails: perception incomplete for {missing}")
    data = {}
    for vid, vp in vids:
        data[vid] = load_video_data(HEAVY / vid, Path("data/vidor/training") / vp.replace(".mp4", ".json"))
    rows, best = [], {}
    for fam, grid in (("FB", FB_GRID), ("B2", B2_GRID)):
        means = []
        for ci, p in enumerate(grid):
            js = [score_method(fam, p, data[v], flagged=False)["J"] for v, _ in vids]
            m = float(np.nanmean(js))
            means.append(m)
            rows.append({"family": fam, "config": ci, "params": json.dumps(p, sort_keys=True), "mean_J": m})
        bi = int(np.argmax(means))                                   # ties -> first in grid order
        best[fam] = {"config": bi, "params": grid[bi], "mean_J": means[bi]}
        print(fam, best[fam], flush=True)
    write_csv_gz(OUT / "calibration.csv.gz", rows, ["family", "config", "params", "mean_J"])
    # F2 exposure with the selected UNFLAGGED B2
    inc_total, vids_with, igt = 0, 0, 0.0
    per_video = {}
    for vid, _ in vids:
        d = data[vid]
        r = score_method("B2", best["B2"]["params"], d, flagged=False)
        inc = sum(len(runs(d.labels[k]["takeover"][1:].tolist())) for k in d.stream.keys)
        inc_total += inc
        vids_with += inc > 0
        igt += r["igt_s"]
        per_video[vid] = {"takeover_incidents": inc, "igt_s_B2": r["igt_s"], "n_keys": len(d.stream.keys)}
    n_miss = sum(len(d.missing) for d in data.values())
    n_avail = sum(r["appearance_available"] == "True" for d in data.values() for r in d.missing)
    gate = {"F1_pipeline_complete": True,
            "F2": {"takeover_incidents": inc_total, "videos_with_incidents": int(vids_with), "igt_s_B2_unflagged": igt,
                   "pass": bool(inc_total >= 20 and vids_with >= 5 and igt >= 10.0)},
            "F3": {"missing_records": n_miss, "appearance_available": n_avail,
                   "coverage": n_avail / n_miss if n_miss else float("nan"),
                   "pass": bool(n_miss and n_avail / n_miss >= 0.50)},
            "selected": best, "per_video": per_video}
    gate["PASS"] = bool(gate["F2"]["pass"] and gate["F3"]["pass"])
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: gate[k] for k in ("F2", "F3", "PASS")}, indent=1, default=float))


if __name__ == "__main__":
    main()
