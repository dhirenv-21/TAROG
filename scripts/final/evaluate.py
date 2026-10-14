"""Final evaluation: frozen methods on a split (research/taorg_final_preregistration.md §8-§9).

    python scripts/final/evaluate.py --split final     (registered; after the freeze)
    python scripts/final/evaluate.py --split dev       (descriptive; after the freeze)

Per video (resumable): prepare the EGTR stream at the frozen tau, run NM/FB/B2/VH/FB+F/B2+F with frozen
parameters, compute outcomes, run the E1 eager-vs-lazy V-HPED check, save method outputs.
Writes results/final/<split>/{rows.jsonl, method_outputs.csv.gz, e1.jsonl}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.downstream.io import write_csv_gz
from taorg.experiments.final.runner import METHODS, load_final_video, outcomes, prepare, run_outputs, to_keystr
from taorg.temporal.lazy_vhped import LazyVHPED
from taorg.temporal.vhped import VHPED
from taorg.utils.seed import set_seed

SPLIT = {"dev": "training", "final": "final"}


def e1_check(p, params) -> dict:
    """Eager VHPED vs LazyVHPED on the identical stream: decision mismatches and max |dscore|."""
    eager, lazy = VHPED(**params, rung="m3"), LazyVHPED(**params, rung="m3")
    eager.reset_video("v")
    lazy.reset_video("v")
    mism, near, md = 0, 0, 0.0
    for g in p.ps.graphs:
        e = {o.edge_key: o.predicate_score for o in eager.step(g).edge_observations}
        z = {o.edge_key: o.predicate_score for o in lazy.step(g).edge_observations}
        for k in set(e) | set(z):
            if (k in e) != (k in z):
                if abs(eager.belief(k) - params["threshold"]) < 1e-9:
                    near += 1
                else:
                    mism += 1
            else:
                md = max(md, abs(e[k] - z[k]))
    return {"mismatches": mism, "near_threshold": near, "max_abs_score_diff": md, **lazy.counters(),
            "eager_maintained": len(eager._x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT), required=True)
    a = ap.parse_args()
    set_seed(20261021)
    fz = json.loads(Path("results/final/freeze/frozen.json").read_text(encoding="utf-8"))
    out = Path(f"results/final/{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    rows_p, e1_p = out / "rows.jsonl", out / "e1.jsonl"
    done = set()
    if rows_p.exists():
        done = {json.loads(l)["video"] for l in rows_p.read_text(encoding="utf-8").splitlines()}
    perc = Path(f"D:/TAORG_runs/downstream/{a.split}")
    sggd = Path(f"D:/TAORG_runs/egtr/sgg/{a.split}")
    outputs = []
    for vp in sample(Path("data/vidor"), SPLIT[a.split]):
        vid = Path(vp).stem
        if vid in done:
            continue
        d = load_final_video(perc / vid, sggd / f"{vid}.npz",
                             Path("data/vidor/validation" if a.split == "final" else "data/vidor/training") / vp.replace(".mp4", ".json"))
        p = prepare(d, fz["tau_obs"])
        recs = []
        for name in METHODS:
            o, m = run_outputs(p, name, fz["params"][name.split("+")[0]] if name.split("+")[0] != "NM" else {})
            r = outcomes(p, o)
            recs.append({"video": vid, "method": name, **r})
            for k, arr in o.items():
                outputs.append({"video": vid, "method": name, "key": to_keystr(k), "on_steps": " ".join(map(str, np.where(arr)[0]))})
        e1 = {"video": vid, **e1_check(p, fz["params"]["VH"])}
        with open(rows_p, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        with open(e1_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(e1, default=float) + "\n")
        print(vid, {r["method"]: round(r["J"], 3) for r in recs}, "E1 mism", e1["mismatches"], flush=True)
    if outputs:
        write_csv_gz(out / f"method_outputs_{len(done)}.csv.gz", outputs, ["video", "method", "key", "on_steps"])
    print("done")


if __name__ == "__main__":
    main()
