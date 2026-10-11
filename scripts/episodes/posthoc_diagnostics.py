"""POST-HOC, EXPLORATORY episode-study diagnostics (defined AFTER the registered results; never decisive).

Written after checkpoint 4 (registered decision STOP) to test two explanations:

E1  Attribution of the P2-vs-P1 differences.  P1 and P2 selected DIFFERENT base
    configurations (#0176 vs #0101).  Evaluate P1 with P2's base parameters (#0101, identity
    off) so that P2#0101 - P1#0101 isolates the identity mechanism and P1#0101 - P1#0176
    isolates the base-parameter choice (e.g. the refractory term).
E2  Ghost timing in C6: ghost output frames DURING swap windows vs AFTER them, for the
    selected operating points.  Hypothesis: ghost keys born during a swap linger after it
    because the long-hold estimator treats 'no record' as weak (UNKNOWN) evidence.

Outputs: results/episodes/posthoc/{e1_attribution.csv, e2_ghost_timing.csv, posthoc.md}
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from taorg.experiments.episodes import pipeline as P
from taorg.experiments.episodes.conditions import build_instance
from taorg.experiments.episodes.evaluation import build_model, evaluate_config, run_model
from taorg.experiments.episodes.search import method_configs
from taorg.experiments.phase3.stats import paired_comparison
from taorg.experiments.phase3.conditions import derive_seed

logging.disable(logging.WARNING)
OUT = Path("results/episodes")
METRICS = ("J", "frr", "fpt_s", "ghost_time_s", "std_mean_s", "rdd_mean_s", "churn", "gap_recall", "swap_recall")


def main() -> None:
    cfg = P.load_config(Path("configs/experiments/episodes.yaml"))
    frozen = json.loads((OUT / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    E = P.load_transition(OUT)
    mc = {c.config_id: c.params for c in method_configs(cfg, ["episode", "episode_id"])}
    variants = {"P1#0176 (selected P1)": ("episode", frozen["episode"]["params"]),
                "P1#0101 (P2 base, no identity)": ("episode", mc["episode#0101"]),
                "P2#0101 (selected P2)": ("episode_id", frozen["episode_id"]["params"]),
                "B2 selected": ("gated_ema_hyst", frozen["gated_ema_hyst"]["params"])}
    assert mc["episode_id#0101"] == frozen["episode_id"]["params"]
    seeds = cfg["seeds"]["evaluation"]
    conds = P.conditions(cfg)
    vals = {(c, v): {m: [] for m in METRICS} for c in conds for v in variants}
    ghost_rows = []
    for c, cond in conds.items():
        for s in seeds:
            inst = build_instance(cond, s, cfg["world"], cfg["base_corruption"])
            for v, (meth, params) in variants.items():
                r, pred = evaluate_config(inst, meth, P.model_params(meth, params, cfg, E), cfg["metrics"])
                for m in METRICS:
                    vals[(c, v)][m].append(r[m])
            if c != "C6_id_switch":
                continue
            sw = [(e["start_frame"], e["end_frame"]) for e in inst.manifests["id_switch"]["entries"]]
            for name in ("gated_ema", "gated_ema_hyst", "gated_ema_id", "episode", "episode_id", "va_hmm", "hsmm"):
                pred = run_model(build_model(name, P.model_params(name, frozen[name]["params"], cfg, E)),
                                 inst.input_frames, "v")
                during = after = 0
                for f, fr in enumerate(pred):
                    ghosts = [o for o in fr.edge_observations if o.edge_key not in inst.gt_episodes]
                    if any(a <= f <= b for a, b in sw):
                        during += len(ghosts)
                    else:
                        after += len(ghosts)
                ghost_rows.append({"seed": s, "method": name, "ghost_frames_during_swap": during,
                                   "ghost_frames_outside_swap": after})
    out = OUT / "posthoc"
    out.mkdir(parents=True, exist_ok=True)
    rng = lambda *l: np.random.default_rng(derive_seed(int(cfg["seeds"]["bootstrap"]), "posthoc", *l))  # noqa: E731
    rows = []
    pairs = [("P2#0101 (selected P2)", "P1#0101 (P2 base, no identity)", "identity mechanism"),
             ("P1#0101 (P2 base, no identity)", "P1#0176 (selected P1)", "base-parameter choice"),
             ("P1#0101 (P2 base, no identity)", "B2 selected", "P1 with P2 base vs B2")]
    for c in conds:
        for a, b, label in pairs:
            for m in METRICS:
                st = paired_comparison(vals[(c, a)][m], vals[(c, b)][m], 5000, 0.95, rng(c, a, b, m))
                rows.append({"condition": c, "contrast": label, "a": a, "b": b, "metric": m,
                             "mean_a": st["mean_a"], "mean_b": st["mean_b"], "mean_diff": st["mean_diff"],
                             "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"], "n": st["n"]})
    with open(out / "e1_attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(out / "e2_ghost_timing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ghost_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(ghost_rows)
    lines = ["# POST-HOC, EXPLORATORY diagnostics (defined after the registered STOP; never decisive)", "",
             "## E1 — attribution (mean difference a − b, 95% bootstrap CI, n = 50 seeds)", "",
             "| Condition | Contrast | ΔJ | ΔFRR | Δghost s | ΔSTD s | ΔRDD s |", "|---|---|---|---|---|---|---|"]
    idx = {(r["condition"], r["contrast"], r["metric"]): r for r in rows}
    for c in conds:
        for _, _, label in pairs:
            cell = []
            for m in ("J", "frr", "ghost_time_s", "std_mean_s", "rdd_mean_s"):
                r = idx[(c, label, m)]
                cell.append(f"{r['mean_diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
            lines.append(f"| {c} | {label} | " + " | ".join(cell) + " |")
    lines += ["", "## E2 — C6 ghost output frames during vs outside swap windows (mean per video)", "",
              "| Method | during swap | outside swap |", "|---|---|---|"]
    for name in ("gated_ema", "gated_ema_hyst", "gated_ema_id", "va_hmm", "hsmm", "episode", "episode_id"):
        g = [r for r in ghost_rows if r["method"] == name]
        lines.append(f"| {cfg['method_labels'][name]} | {np.mean([r['ghost_frames_during_swap'] for r in g]):.1f} | "
                     f"{np.mean([r['ghost_frames_outside_swap'] for r in g]):.1f} |")
    (out / "posthoc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
