"""Run the Phase 8 ablation matrix (synthetic) and save run artifacts.

Usage: python scripts/run_ablation_matrix.py
           [--config configs/experiments/ablation_matrix_synthetic.yaml] [--workers 5]
           [--seeds ...] [--iters N]     (smoke runs only; flagged in the config)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from taorg.config import load_config
from taorg.experiments.ablation_matrix import run_matrix
from taorg.experiments.artifacts import RunArtifactWriter
from taorg.schemas.runs import RunMetadata
from taorg.utils.git import get_git_branch, get_git_commit, get_git_dirty
from taorg.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/experiments/ablation_matrix_synthetic.yaml"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config(args.config)
    smoke = args.seeds is not None or args.iters is not None
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.iters is not None:
        cfg["trainer"]["num_iters"] = args.iters
    cfg["smoke_run"] = smoke
    cfg["command"] = " ".join(sys.argv)
    set_seed(int(cfg["seeds"][0]))

    meta = RunMetadata(git_commit=get_git_commit(), git_dirty=get_git_dirty(),
                       git_branch=get_git_branch(), command=cfg["command"],
                       dataset_id="synthetic/vhped_benchmark", base_model_id="synthetic_detector",
                       temporal_method="ablation_matrix", seeds=[int(s) for s in cfg["seeds"]])
    meta.mark_started()
    writer = RunArtifactWriter(meta, output_base=args.out)
    run_dir = writer.create_run_directory()
    writer.save_config(cfg)
    t0 = time.perf_counter()
    try:
        result = run_matrix(cfg, workers=args.workers)
    except Exception as exc:
        meta.mark_failed(repr(exc))
        writer.save_metadata()
        raise
    (run_dir / "cells.json").write_text(json.dumps(result["cells"], indent=1))
    writer.save_metrics({"kind": ("SMOKE " if smoke else "") + "SYNTHETIC ablation matrix",
                         "table": result["table"],
                         "paired_vs_reference": result["paired_vs_reference"]})
    writer.save_timing({"total_seconds": time.perf_counter() - t0, "workers": args.workers})
    meta.metrics = {}
    meta.mark_completed()
    writer.save_metadata()
    print(f"\nAblation matrix finished -> {run_dir}")


if __name__ == "__main__":
    main()
