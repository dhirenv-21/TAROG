"""Run the V-HPED synthetic sanity benchmark over predeclared seeds.

SYNTHETIC — not paper evidence.  Writes a JSON artifact under
outputs/runs/vhped_synthetic/ with config, git provenance, and per-seed metrics.

Usage:
    python scripts/run_vhped_synthetic.py [--seeds 0 1 2] [--iters 200]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from taorg.experiments.vhped_benchmark import ScenarioConfig, run_benchmark
from taorg.temporal.vhped import TrainerConfig
from taorg.utils.git import get_git_commit, get_git_dirty
from taorg.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/vhped_synthetic"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scenario = ScenarioConfig()
    trainer = TrainerConfig(num_iters=args.iters)
    results = []
    for seed in args.seeds:
        set_seed(seed)
        results.append(run_benchmark(seed, scenario, args.train, args.val, args.test, trainer))

    methods = list(results[0].per_method)
    keys = list(results[0].per_method[methods[0]])
    summary = {
        m: {k: [float(np.mean([r.per_method[m][k] for r in results])),
                float(np.std([r.per_method[m][k] for r in results]))] for k in keys}
        for m in methods
    }
    header = f"{'method':<17}" + "".join(f"{k[:14]:>16}" for k in keys)
    print("\nSYNTHETIC test metrics, mean ± std over seeds", args.seeds)
    print(header)
    for m in methods:
        print(f"{m:<17}" + "".join(f"{summary[m][k][0]:>9.3f}±{summary[m][k][1]:<6.3f}" for k in keys))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"run_{stamp}.json"
    path.write_text(json.dumps({
        "kind": "SYNTHETIC sanity benchmark - not paper evidence",
        "seeds": args.seeds,
        "splits": {"train": args.train, "val": args.val, "test": args.test},
        "scenario": asdict(scenario),
        "trainer": asdict(trainer),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "summary": summary,
        "per_seed": [{"seed": r.seed, "metrics": r.per_method, "fitted_params": r.fitted_params}
                     for r in results],
    }, indent=2))
    print(f"\nArtifact: {path}")


if __name__ == "__main__":
    main()
