"""Identifiability check: simulate from known V-HPED parameters, refit, report errors.

Usage: python scripts/run_param_recovery.py [--seeds 42 123 456 789 2024] [--videos 8]
Writes outputs/runs/param_recovery/<stamp>/results.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from taorg.experiments.param_recovery import RECOVERED, SimConfig, recovery_run
from taorg.temporal.vhped import HAND_SET_CONFIG, TrainerConfig, VHPEDParams
from taorg.utils.git import get_git_commit

TRUE = dict(HAND_SET_CONFIG, num_predicates=2, mu=3.0, alpha_self=12.0, eta=-1.0,
            beta_vis=4.0, beta_occ=0.5, rho_occ=0.1, rho_unk=0.5, eps_rate=0.8, use_marks=False)
INIT = dict(HAND_SET_CONFIG, num_predicates=2, use_marks=False)  # hand-set start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/param_recovery"))
    args = ap.parse_args()
    true = VHPEDParams.from_config(TRUE)
    init = VHPEDParams.from_config(INIT)
    trainer = TrainerConfig(num_iters=args.iters, lr=0.05, l2_cross=1e-3,
                            use_cross_predicate=False)
    runs = [recovery_run(true, init, SimConfig(), args.videos, s, trainer) for s in args.seeds]
    summary = {k: {"true": runs[0]["true"][k],
                   "fitted_mean": float(np.mean([r["fitted"][k] for r in runs])),
                   "fitted_std": float(np.std([r["fitted"][k] for r in runs])),
                   "init": runs[0]["init"][k]} for k in RECOVERED}
    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "true_config": TRUE, "init_config": INIT, "args": {k: str(v) for k, v in vars(args).items()},
        "git_commit": get_git_commit(), "summary": summary, "runs": runs}, indent=2))
    print(f"{'param':>10} {'true':>8} {'init':>8} {'fitted (mean±std)':>20}")
    for k, v in summary.items():
        print(f"{k:>10} {v['true']:>8.3f} {v['init']:>8.3f} "
              f"{v['fitted_mean']:>11.3f}±{v['fitted_std']:.3f}")
    print(f"\nArtifacts: {out}")


if __name__ == "__main__":
    main()
