"""Final evaluation: registered analysis and decisions (research/taorg_final_preregistration.md §8).

    python scripts/final/analyze.py --split final    (registered)
    python scripts/final/analyze.py --split dev      (descriptive)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.appearance.labels import frame_label, purity
from taorg.experiments.downstream.io import read_csv_gz
from taorg.experiments.downstream.observations import gt_dict
from taorg.experiments.final.runner import METHODS, load_final_video
from taorg.experiments.observation.oracle import identity_of, match
from taorg.utils.seed import set_seed

SPLIT = {"dev": "training", "final": "final"}
SEED, B = 20261021, 2000
SUMS = ("igt_s", "pgg_num", "pgg_den", "std_gen_sum", "std_gen_n", "frr_gen_num", "frr_gen_den", "rdd_gen_sum",
        "rdd_gen_n", "tp", "fp", "fn")


def pooled(T: dict[str, np.ndarray], ii: np.ndarray) -> dict[str, float]:
    s = {f: float(T[f][ii].sum()) for f in SUMS}
    dv = lambda a, b: a / b if b else float("nan")   # noqa: E731
    prec, rec = dv(s["tp"], s["tp"] + s["fp"]), dv(s["tp"], s["tp"] + s["fn"])
    return {"IGT_s": s["igt_s"], "PGG": dv(s["pgg_num"], s["pgg_den"]), "STD_gen_s": dv(s["std_gen_sum"], s["std_gen_n"]),
            "FRR_gen": dv(s["frr_gen_num"], s["frr_gen_den"]), "RDD_gen_s": dv(s["rdd_gen_sum"], s["rdd_gen_n"]),
            "precision": prec, "recall": rec, "F1": dv(2 * prec * rec, prec + rec) if prec == prec and rec == rec else float("nan"),
            "J": float(np.nanmean(T["J"][ii])), "n_gap_frames": s["pgg_den"], "n_std": s["std_gen_n"], "n_rdd": s["rdd_gen_n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT), required=True)
    a = ap.parse_args()
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = Path(f"results/final/{a.split}")
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = [Path(v).stem for v in sample(Path("data/vidor"), SPLIT[a.split])]
    got = Counter((r["video"], r["method"]) for r in rows)
    comp = {"expected_videos": len(expected), "expected_method_runs": len(expected) * len(METHODS),
            "actual_method_runs": sum(got.values()),
            "missing": [f"{v}|{m}" for v in expected for m in METHODS if got[(v, m)] == 0],
            "duplicate": [f"{v}|{m}" for (v, m), c in got.items() if c > 1], "failed": []}
    vids = [v for v in expected if all(got[(v, m)] == 1 for m in METHODS)]
    by = {(r["video"], r["method"]): r for r in rows}
    T = {m: {f: np.array([float(by[(v, m)][f]) for v in vids]) for f in SUMS + ("J",)} for m in METHODS}
    allv = np.arange(len(vids))
    point = {m: pooled(T[m], allv) for m in METHODS}
    boots = [{m: pooled(T[m], ii) for m in METHODS} for ii in (rng.integers(0, len(vids), len(vids)) for _ in range(B))]

    def ci(fn, level):
        d = np.array([fn(b) for b in boots], float)
        d = d[np.isfinite(d)]
        q = (1 - level) / 2
        return (float(np.quantile(d, q)), float(np.quantile(d, 1 - q))) if len(d) else (float("nan"), float("nan"))

    def red(p, m, b):
        return 1 - p[m]["IGT_s"] / p[b]["IGT_s"] if p[b]["IGT_s"] else float("nan")

    def ratio(p, m, b, f):
        return p[m][f] / p[b][f] if p[b][f] else float("nan")

    q1 = {}
    for c, m, b in (("C1", "FB+F", "FB"), ("C2", "B2+F", "B2")):
        e = {"ghost_reduction": red(point, m, b), "ghost_reduction_ci": ci(lambda x, m=m, b=b: red(x, m, b), 0.975),
             "PGG_ratio": ratio(point, m, b, "PGG"), "PGG_ratio_ci": ci(lambda x, m=m, b=b: ratio(x, m, b, "PGG"), 0.975)}
        for f in ("STD_gen_s", "RDD_gen_s", "F1", "J", "FRR_gen"):
            e[f"d_{f}"] = point[m][f] - point[b][f]
            e[f"d_{f}_ci"] = ci(lambda x, f=f, m=m, b=b: x[m][f] - x[b][f], 0.975)
        checks = {"benefit": bool(e["ghost_reduction"] >= 0.25 and e["ghost_reduction_ci"][0] > 0),
                  "PGG_relative": bool(e["PGG_ratio_ci"][0] >= 0.90),
                  "STD_gen": bool(e["d_STD_gen_s_ci"][1] <= 0.20), "RDD_gen": bool(e["d_RDD_gen_s_ci"][1] <= 0.20),
                  "F1": bool(e["d_F1_ci"][0] >= -0.01)}
        e.update({"checks": checks, "pass": all(checks.values())})
        q1[c] = e
    n_pass = sum(q1[c]["pass"] for c in q1)
    q1_decision = "GO" if n_pass == 2 else "NARROW" if n_pass == 1 else "STOP"
    dJ = point["VH"]["J"] - point["B2"]["J"]
    q2 = {"dJ_VH_minus_B2": dJ, "dJ_ci975": ci(lambda x: x["VH"]["J"] - x["B2"]["J"], 0.975),
          "dIGT_VH_minus_B2_ci95": ci(lambda x: x["VH"]["IGT_s"] - x["B2"]["IGT_s"], 0.95),
          "dSTD_gen_VH_minus_B2_ci95": ci(lambda x: x["VH"]["STD_gen_s"] - x["B2"]["STD_gen_s"], 0.95)}
    q2["decision"] = ("SUPPORTED" if dJ >= 0.01 and q2["dJ_ci975"][0] > 0 and q2["dIGT_VH_minus_B2_ci95"][1] <= 0
                      and q2["dSTD_gen_VH_minus_B2_ci95"][1] <= 0 else "NOT SUPPORTED")
    desc = {}
    for m, b in (("FB", "NM"), ("B2", "NM"), ("VH", "NM"), ("B2", "FB"), ("VH", "FB")):
        desc[f"{m}-{b}"] = {f: {"estimate": point[m][f] - point[b][f],
                                "ci95": ci(lambda x, f=f, m=m, b=b: x[m][f] - x[b][f], 0.95)}
                            for f in ("J", "F1", "IGT_s", "STD_gen_s", "RDD_gen_s", "PGG")}
    e1 = [json.loads(line) for line in (out / "e1.jsonl").read_text(encoding="utf-8").splitlines()]
    e1s = {"videos": len(e1), "mismatches": sum(r["mismatches"] for r in e1), "near_threshold": sum(r["near_threshold"] for r in e1),
           "max_abs_score_diff": max(r["max_abs_score_diff"] for r in e1),
           "key_updates": sum(r["key_updates"] for r in e1), "eager_key_updates_upper": None,
           "freezes": sum(r["freezes"] for r in e1), "materialisations": sum(r["mat_record"] + r["mat_pair"] + r["mat_regime"] for r in e1),
           "stale_pops": sum(r["stale_pops"] for r in e1), "valid_pops": sum(r["valid_pops"] for r in e1),
           "max_heap": max(r["heap_size"] for r in e1), "steps": sum(r["steps"] for r in e1),
           "emitted": sum(r["emitted"] for r in e1), "eager_maintained_final": sum(r["eager_maintained"] for r in e1)}
    e1s["pass"] = bool(e1s["mismatches"] == 0 and e1s["max_abs_score_diff"] < 1e-9)
    fl = Counter()
    for vp in sample(Path("data/vidor"), SPLIT[a.split]):
        vid = Path(vp).stem
        if vid not in vids:
            continue
        d = load_final_video(Path(f"D:/TAORG_runs/downstream/{a.split}") / vid, Path(f"D:/TAORG_runs/egtr/sgg/{a.split}") / f"{vid}.npz",
                             Path("data/vidor/validation" if a.split == "final" else "data/vidor/training") / vp.replace(".mp4", ".json"))
        gt = gt_dict(d.video)
        mt = match(d.frames, gt)
        g, pur = identity_of(mt), purity(mt)
        for r in read_csv_gz(Path(f"D:/TAORG_runs/downstream/{a.split}") / vid / "missing_records.csv.gz"):
            lab = frame_label(int(r["track"]), int(r["frame"]), g, pur, mt, gt)
            fl[(lab, r["flag"] == "True", r["appearance_available"] == "True")] += 1
    labs = sorted({k[0] for k in fl})
    flag_beh = {"coverage": sum(v for k, v in fl.items() if k[2]) / max(1, sum(fl.values())),
                "flag_rate": {L: sum(v for k, v in fl.items() if k[0] == L and k[1]) / max(1, sum(v for k, v in fl.items() if k[0] == L))
                              for L in labs},
                "label_counts": {L: sum(v for k, v in fl.items() if k[0] == L) for L in labs}}
    res = {"split": a.split, "completeness": comp, "n_videos": len(vids), "point": point, "Q1": q1, "Q1_decision": q1_decision,
           "Q2": q2, "descriptive": desc, "E1": e1s, "flag_behaviour": flag_beh}
    if a.split != "final":
        res["Q1_decision"] = f"descriptive (would be {q1_decision})"
        res["Q2"]["decision"] = f"descriptive (would be {q2['decision']})"
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"completeness": comp, "Q1": res["Q1_decision"], "Q2": res["Q2"]["decision"], "E1": e1s["pass"]}, indent=1))
    for m in METHODS:
        print(m, {k: round(v, 3) for k, v in point[m].items()})


if __name__ == "__main__":
    main()
