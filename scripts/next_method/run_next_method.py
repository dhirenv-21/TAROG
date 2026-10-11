"""Next-method study CLI (research/taorg_next_method_preregistration.md).

Stages: fit (prior from calibration INPUT streams) -> calibrate -> select -> evaluate -> latency
        -> analyze -> figures -> failures

    python scripts/next_method/run_next_method.py fit calibrate select --workers 4
    python scripts/next_method/run_next_method.py evaluate latency analyze figures failures --workers 4
    python scripts/next_method/run_next_method.py all --smoke --max-samples 3   # pipeline check only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from taorg.experiments.next_method import pipeline as P

STAGES = ["fit", "calibrate", "select", "evaluate", "latency", "analyze", "figures", "failures"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/experiments/next_method.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    if a.workers > 4:
        ap.error("registered bound: at most 4 workers")
    if a.max_samples is not None and not a.smoke:
        ap.error("--max-samples only with --smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)
    cfg = P.load_config(Path(a.config))
    sfx = "_smoke" if a.smoke else ""
    out = Path(a.out or f"results/next_method{sfx}")
    run_dir = Path(a.run_dir or f"D:/TAORG_runs/next_method{sfx}")
    P.save_resolved_config(cfg, out)
    for st in (STAGES if "all" in a.stages else a.stages):
        logging.info("=== stage %s ===", st)
        if st == "fit":
            P.run_fit(cfg, out, a.smoke)
        elif st == "calibrate":
            P.run_calibration(cfg, out, run_dir, a.workers, a.smoke, a.max_samples)
        elif st == "select":
            P.run_selection(cfg, out, a.smoke)
        elif st == "evaluate":
            P.run_evaluation(cfg, out, run_dir, a.workers, a.smoke)
        elif st == "latency":
            P.run_latency(cfg, out, a.smoke)
        elif st == "analyze":
            from taorg.experiments.next_method.analysis import run_analysis
            run_analysis(cfg, out)
        elif st == "figures":
            from taorg.experiments.next_method.figures import run_figures
            run_figures(cfg, out)
        elif st == "failures":
            from taorg.experiments.next_method.figures import run_failures
            run_failures(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
