"""Phase 4 method-selection experiment — command-line entry point.

Stages (each reads only artifacts written by earlier stages):
    calibrate  evaluate every sampled configuration of a method GROUP on calibration seeds
    select     operating points + Pareto sets (calibration artifacts only)
    evaluate   held-out evaluation of the group's planned configurations
    latency    per-frame latency of primary-budget operating points (serial)
    analyze    statistics, H2/H3/H4 decisions, ablation (needs both groups; baseline-only
               mode writes baseline tables)
    plots      figures from aggregated artifacts

Registered order: baseline group (calibrate, select, freeze, evaluate, latency) BEFORE the
proposed group.

Usage:
    python scripts/phase4/run_phase4.py --group baseline --stage calibrate --workers 10
    python scripts/phase4/run_phase4.py --smoke --out <scratch> --group baseline --stage all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

STAGES = ("calibrate", "select", "evaluate", "latency")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/phase4.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase4"))
    ap.add_argument("--group", choices=("baseline", "proposed"), default="baseline")
    ap.add_argument("--stage", choices=(*STAGES, "all", "analyze", "plots", "failures"), default="all")
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Use only the first k sampled configs per method (smoke only).")
    args = ap.parse_args(argv)
    if args.max_samples is not None and not args.smoke:
        ap.error("--max-samples is only allowed with --smoke")

    from taorg.experiments.phase4 import pipeline as P

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("taorg.noise", "matplotlib", "fontTools"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)

    cfg = P.load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    P.save_resolved_config(cfg, args.out)
    todo = STAGES if args.stage == "all" else (args.stage,)
    for stage in todo:
        logging.info("=== Phase 4 stage: %s [%s] (smoke=%s) ===", stage, args.group, args.smoke)
        if stage == "calibrate":
            P.run_calibration(cfg, args.out, args.group, args.workers, smoke=args.smoke,
                              max_samples=args.max_samples)
        elif stage == "select":
            P.run_selection(cfg, args.out, args.group, smoke=args.smoke)
        elif stage == "evaluate":
            P.run_evaluation(cfg, args.out, args.group, args.workers, smoke=args.smoke)
        elif stage == "latency":
            P.run_latency(cfg, args.out, args.group, smoke=args.smoke)
        elif stage == "analyze":
            from taorg.experiments.phase4.analysis import run_analysis
            res = run_analysis(cfg, args.out)
            logging.info("Decisions: %s", res.get("decisions"))
        elif stage == "plots":
            from taorg.experiments.phase4.plots import run_plots
            run_plots(cfg, args.out)
        elif stage == "failures":
            from taorg.experiments.phase4.failures import run_failures
            run_failures(cfg, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
