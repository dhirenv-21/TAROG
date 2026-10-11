"""POST-HOC, EXPLORATORY (defined after the registered STOP; never decisive; DEV seeds 7000-7019 only).

Question: at the frozen tau (0.376) and B2 base (#0091), would a weaker prior (smaller kappa) let the
online re-detection-time learning correct the S2 shift (UNKNOWN = gap) without new harms in
C1/C2/C6?  Also includes the frozen BUF and B2 for reference.
Output: results/next_method/posthoc/kappa_sensitivity_dev.csv, posthoc.md
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

SEEDS = list(range(7000, 7020))
CONDS = ["C1_gaps", "C2_hidden_termination", "C6_id_switch", "S2_mixed_unknown"]
KAPPAS = [2.0, 5.0, 20.0, 100.0, 838.0]
OUT = Path("results/next_method/posthoc")


def _q():
    logging.disable(logging.WARNING)


def _task(a):
    cname, cdict, seed, world, base, metrics, frozen, prior = a
    from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance
    from taorg.experiments.episodes.evaluation import evaluate_config
    inst = build_instance(EpisodeCondition.from_dict(cname, cdict), seed, world, base)
    rows = []
    for name in ("gated_ema_hyst", "b2_hold_buffer"):
        m, _ = evaluate_config(inst, name, frozen[name]["params"], metrics)
        rows.append((name, m))
    for k in KAPPAS:
        p = {**frozen["b2_hold_km"]["params"], "kappa": k, "prior": prior}
        m, _ = evaluate_config(inst, "b2_hold_km", p, metrics)
        rows.append((f"km_kappa{k:g}", m))
    keep = ("J", "gap_recall", "fpt_s", "ghost_time_s", "std_mean_s", "churn")
    return cname, [(n, {x: r[x] for x in keep}) for n, r in rows]


def main(workers=4):
    from taorg.experiments.next_method import pipeline as P
    cfg = P.load_config(Path("configs/experiments/next_method.yaml"))
    frozen = json.loads(Path("results/next_method/configs/frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    prior = P.load_prior(Path("results/next_method"))
    tasks = [(c, cfg["conditions"][c], s, cfg["world"], cfg["base_corruption"], cfg["metrics"], frozen, prior)
             for c in CONDS for s in SEEDS]
    acc = defaultdict(lambda: defaultdict(list))
    with mp.get_context("spawn").Pool(workers, initializer=_q) as pool:
        for c, rows in pool.imap_unordered(_task, tasks, chunksize=1):
            for n, r in rows:
                for k, v in r.items():
                    acc[(c, n)][k].append(v)
    OUT.mkdir(parents=True, exist_ok=True)
    out = [{"condition": c, "variant": n, **{k: float(np.nanmean(v)) for k, v in d.items()}} for (c, n), d in sorted(acc.items())]
    with open(OUT / "kappa_sensitivity_dev.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    names = ["gated_ema_hyst", "b2_hold_buffer"] + [f"km_kappa{k:g}" for k in KAPPAS]
    lines = ["# POST-HOC, EXPLORATORY: prior-strength sensitivity (dev seeds 7000-7019, n = 20; never decisive)", "",
             "Frozen tau 0.376 and B2 base #0091; only kappa varies. Mean J (gap recall / FPT s / ghost s).", "",
             "| Variant | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    idx = {(r["condition"], r["variant"]): r for r in out}
    for n in names:
        cells = [f"{idx[(c, n)]['J']:.3f} ({idx[(c, n)]['gap_recall']:.2f} / {idx[(c, n)]['fpt_s']:.1f} / "
                 f"{idx[(c, n)]['ghost_time_s']:.1f})" for c in CONDS]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    (OUT / "posthoc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
