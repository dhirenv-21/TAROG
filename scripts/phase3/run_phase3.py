"""Phase 3 GO/NO-GO pilot — command-line entry point.

Stages (run in order; each reads only artifacts written by earlier stages):
    calibrate  grid-evaluate every method configuration on calibration seeds
    select     budget-constrained operating points + calibration Pareto sets
    evaluate   held-out evaluation of selected + frontier configurations
    latency    per-frame latency of primary-budget operating points (serial)
    analyze    descriptive stats, paired tests, hypervolumes, H1 decision
    plots      figures A-G from aggregated artifacts
    failures   deterministic failure-case extraction

Usage:
    python scripts/phase3/run_phase3.py --stage all
    python scripts/phase3/run_phase3.py --stage calibrate --workers 24
    python scripts/phase3/run_phase3.py --smoke --out <scratch dir> --stage all

--smoke uses the smoke seeds and a strided grid; its outputs are for pipeline
integrity checks only and must never be analysed or reported.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# NOTE: heavy imports (scipy, pandas, matplotlib) are deferred into main().
# On Windows, multiprocessing "spawn" workers re-import this script; importing
# the analysis stack here made every worker load it, exhausting the paging file
# and killing workers mid-import (which leaves multiprocessing.Pool blocked).

STAGES = ("calibrate", "select", "evaluate", "latency", "analyze", "plots", "failures")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/phase3_pilot.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3"))
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 4)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid-stride", type=int, default=None,
                    help="Evaluate every k-th grid config (smoke only; default 25 in smoke mode).")
    args = ap.parse_args(argv)

    from taorg.experiments.phase3.pipeline import (
        load_phase3_config, run_calibration, run_evaluation, run_latency, run_selection,
        save_resolved_config,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("taorg.noise").setLevel(logging.WARNING)
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    stride = args.grid_stride or (25 if args.smoke else 1)
    if stride != 1 and not args.smoke:
        ap.error("--grid-stride is only allowed together with --smoke")
    cfg = load_phase3_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, args.out)

    todo = STAGES if args.stage == "all" else (args.stage,)
    for stage in todo:
        logging.info("=== Phase 3 stage: %s (smoke=%s) ===", stage, args.smoke)
        if stage == "calibrate":
            run_calibration(cfg, args.out, args.workers, smoke=args.smoke, grid_stride=stride)
        elif stage == "select":
            run_selection(cfg, args.out, smoke=args.smoke)
        elif stage == "evaluate":
            run_evaluation(cfg, args.out, args.workers, smoke=args.smoke)
        elif stage == "latency":
            run_latency(cfg, args.out, smoke=args.smoke)
        elif stage == "analyze":
            from taorg.experiments.phase3.analysis import run_analysis
            from taorg.experiments.phase3.report import run_report
            decision = run_analysis(cfg, args.out, smoke=args.smoke)
            run_report(cfg, args.out)
            logging.info("H1 decision (pre-registered rule): %s", decision["decision"])
        elif stage == "plots":
            from taorg.experiments.phase3.plots import run_plots
            run_plots(cfg, args.out)
        elif stage == "failures":
            from taorg.experiments.phase3.failures import run_failures
            run_failures(cfg, args.out, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
