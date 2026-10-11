"""Episode-study pipeline CLI (research/taorg_episode_preregistration.md).

Stages (run in this order; each reads only the artifacts of earlier stages):
    fit         transition statistic E from calibration ground truth
    calibrate   all configurations x calibration instances (checkpointed, resumable)
    select      frozen operating points (pooled J) + half-sample control
    evaluate    frozen operating points x evaluation instances (checkpointed, resumable)
    latency     per-frame latency of the frozen operating points
    analyze     completeness, paired statistics, decisions, tables
    figures     registered figures
    failures    representative failures (predefined rule)

Examples:
    python scripts/episodes/run_episodes.py fit calibrate select --workers 16
    python scripts/episodes/run_episodes.py evaluate latency analyze figures failures --workers 16
    python scripts/episodes/run_episodes.py all --smoke --max-samples 3     # pipeline check only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from taorg.experiments.episodes import pipeline as P

STAGES = ["fit", "calibrate", "select", "evaluate", "latency", "analyze", "figures", "failures"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    ap.add_argument("--config", default="configs/experiments/episodes.yaml")
    ap.add_argument("--out", default=None, help="results dir (default results/episodes, smoke: results/episodes_smoke)")
    ap.add_argument("--run-dir", default=None, help="heavy checkpoint dir (default D:/TAORG_runs/episodes[_smoke])")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="smoke seeds; never analysed")
    ap.add_argument("--max-samples", type=int, default=None, help="smoke only: limit samples per method")
    a = ap.parse_args()
    if a.workers > 16:
        ap.error("registered bound: at most 16 workers")
    if a.max_samples is not None and not a.smoke:
        ap.error("--max-samples is only allowed with --smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("taorg.temporal.gated_ema").setLevel(logging.ERROR)
    cfg = P.load_config(Path(a.config))
    suffix = "_smoke" if a.smoke else ""
    out = Path(a.out or f"results/episodes{suffix}")
    run_dir = Path(a.run_dir or f"D:/TAORG_runs/episodes{suffix}")
    stages = STAGES if "all" in a.stages else a.stages
    P.save_resolved_config(cfg, out)
    for st in stages:
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
            from taorg.experiments.episodes.analysis import run_analysis
            run_analysis(cfg, out)
        elif st == "figures":
            from taorg.experiments.episodes.plots import run_figures
            run_figures(cfg, out)
        elif st == "failures":
            from taorg.experiments.episodes.failures import run_failures
            run_failures(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
