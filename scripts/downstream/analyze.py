"""Downstream study: registered analysis and decision (prereg §11).

    python scripts/downstream/analyze.py --split eval    (registered decision)
    python scripts/downstream/analyze.py --split dev     (descriptive, after the freeze)

Paired cluster bootstrap over VIDEOS (2000 resamples, seed 20261015); Bonferroni 97.5% intervals for
the two flag contrasts; non-inferiority harm margins; descriptive C3 and interaction (95%).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.downstream.io import read_csv_gz
from taorg.experiments.downstream.runner import METHODS
from taorg.utils.seed import set_seed

SPLIT = {"dev": "training", "eval": "validation"}
SEED, B = 20261015, 2000
FIELDS = ("igt_s", "pgg_num", "pgg_den", "ftt_s", "std_g_sum", "std_g_n", "std_g_censored", "frr_g_num", "frr_g_den",
          "rdd_sum", "rdd_n", "J", "n_frames", "n_keys")


def pooled(T: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    """Pooled outcomes over the videos idx (ratios of sums / pooled means)."""
    s = {f: T[f][idx].sum() for f in FIELDS if f != "J"}
    div = lambda a, b: a / b if b else float("nan")   # noqa: E731
    return {"IGT_s": s["igt_s"], "PGG": div(s["pgg_num"], s["pgg_den"]), "FTT_s": s["ftt_s"],
            "STD_g_s": div(s["std_g_sum"], s["std_g_n"]), "FRR_g": div(s["frr_g_num"], s["frr_g_den"]),
            "RDD_s": div(s["rdd_sum"], s["rdd_n"]), "J": float(np.nanmean(T["J"][idx])),
            "n_std": s["std_g_n"], "n_rdd": s["rdd_n"], "n_gap_frames": s["pgg_den"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT), required=True)
    a = ap.parse_args()
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    out = Path(f"results/downstream/{a.split}")
    rows = read_csv_gz(out / "outcomes.csv.gz")
    expected = [Path(v).stem for v in sample(Path("data/vidor"), SPLIT[a.split])]
    got = Counter((r["video"], r["method"]) for r in rows if r["status"] == "done")
    comp = {"expected_videos": len(expected), "expected_method_runs": len(expected) * len(METHODS),
            "actual_method_runs": sum(got.values()),
            "missing": [f"{v}|{m}" for v in expected for m in METHODS if got[(v, m)] == 0],
            "duplicate": [f"{v}|{m}" for (v, m), c in got.items() if c > 1],
            "failed": [r["video"] for r in rows if r["status"] != "done"]}
    vids = [v for v in expected if all(got[(v, m)] == 1 for m in METHODS)]
    T = {m: {f: np.array([float(next(r[f] for r in rows if r["video"] == v and r["method"] == m and r["status"] == "done"))
                          for v in vids]) for f in FIELDS} for m in METHODS}
    idx0 = np.arange(len(vids))
    point = {m: pooled(T[m], idx0) for m in METHODS}
    boots = []
    for _ in range(B):
        ii = rng.integers(0, len(vids), len(vids))
        boots.append({m: pooled(T[m], ii) for m in METHODS})

    def ci(fn, level):
        d = np.array([fn(b) for b in boots], float)
        d = d[np.isfinite(d)]
        q = (1 - level) / 2
        return (float(np.quantile(d, q)), float(np.quantile(d, 1 - q))) if len(d) else (float("nan"),) * 2

    def red(p, m, base):
        return 1 - p[m]["IGT_s"] / p[base]["IGT_s"] if p[base]["IGT_s"] else float("nan")

    contrasts = {}
    for name, m, base in (("C1", "FB+F", "FB"), ("C2", "B2+F", "B2")):
        c = {"ghost_reduction": red(point, m, base), "ghost_reduction_ci": ci(lambda b, m=m, base=base: red(b, m, base), 0.975)}
        for f in ("PGG", "STD_g_s", "RDD_s", "J", "FTT_s", "FRR_g"):
            c[f"d_{f}"] = point[m][f] - point[base][f]
            c[f"d_{f}_ci"] = ci(lambda b, f=f, m=m, base=base: b[m][f] - b[base][f], 0.975)
        gr_ok = bool(c["ghost_reduction"] >= 0.25 and c["ghost_reduction_ci"][0] > 0)
        harm = {"PGG": bool(c["d_PGG_ci"][0] >= -0.05), "STD_g": bool(c["d_STD_g_s_ci"][1] <= 0.20),
                "RDD": bool(c["d_RDD_s_ci"][1] <= 0.20)}
        c.update({"benefit_pass": gr_ok, "harm_pass": harm, "pass": bool(gr_ok and all(harm.values()))})
        contrasts[name] = c
    c3 = {f: {"B2+F_minus_FB+F": point["B2+F"][f] - point["FB+F"][f],
              "ci95": ci(lambda b, f=f: b["B2+F"][f] - b["FB+F"][f], 0.95)} for f in ("IGT_s", "PGG", "STD_g_s", "RDD_s", "J")}
    inter = {f: {"estimate": (point["B2+F"][f] - point["B2"][f]) - (point["FB+F"][f] - point["FB"][f]),
                 "ci95": ci(lambda b, f=f: (b["B2+F"][f] - b["B2"][f]) - (b["FB+F"][f] - b["FB"][f]), 0.95)}
             for f in ("IGT_s", "J")}
    n_pass = sum(contrasts[c]["pass"] for c in ("C1", "C2"))
    decision = "GO" if n_pass == 2 else "NARROW" if n_pass == 1 else "STOP"
    fr = read_csv_gz(out / "flag_records.csv.gz")
    lab = Counter(r["label"] for r in fr)
    fb = {"n_missing_records": len(fr), "coverage": float(np.mean([r["appearance_available"] == "True" for r in fr])) if fr else float("nan"),
          "label_counts": dict(lab),
          "flag_rate": {L: float(np.mean([r["flag"] == "True" for r in fr if r["label"] == L])) for L in lab},
          "flag_precision_takeover": (sum(r["flag"] == "True" and r["label"] == "TAKEOVER" for r in fr)
                                      / max(1, sum(r["flag"] == "True" for r in fr)))}
    res = {"split": a.split, "completeness": comp, "n_videos_analysed": len(vids), "point": point,
           "contrasts": contrasts, "C3_information_vs_method": c3, "interaction": inter,
           "flag_behaviour": fb, "decision": decision if a.split == "eval" else f"descriptive (would be {decision})"}
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"completeness": comp, "decision": res["decision"],
                      "contrasts": {k: {x: v[x] for x in ("ghost_reduction", "ghost_reduction_ci", "d_PGG_ci", "d_STD_g_s_ci",
                                                         "d_RDD_s_ci", "pass")} for k, v in contrasts.items()}}, indent=1, default=float))
    for m in METHODS:
        print(m, {k: round(v, 3) for k, v in point[m].items()})


if __name__ == "__main__":
    main()
