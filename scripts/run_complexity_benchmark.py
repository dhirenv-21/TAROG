"""Phase 5 eager-vs-lazy scaling benchmark.

Usage: python scripts/run_complexity_benchmark.py [--pairs 100 300 1000 3000]
           [--fractions 0.05 0.2 0.5 1.0] [--predicates 26] [--frames 60] [--seed 0]

Writes outputs/runs/complexity/<stamp>/{results.json, latency.pdf/png, work.pdf/png}.
Latency depends on the machine; operation counters are machine-independent.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from taorg.experiments.complexity_benchmark import run_grid  # noqa: E402
from taorg.utils.git import get_git_commit, get_git_dirty  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.2, 0.5, 1.0])
    ap.add_argument("--predicates", type=int, nargs="+", default=[26])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pattern", choices=["random", "alternating"], default="random")
    ap.add_argument("--memory", action="store_true", help="also measure peak traced memory")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/complexity"))
    args = ap.parse_args()

    rows = run_grid(args.pairs, args.fractions, args.predicates, args.frames, args.seed,
                    pattern=args.pattern, measure_memory=args.memory)
    if args.pattern == "alternating":
        args.fractions = sorted({r["touch_fraction"] for r in rows})
    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git_commit": get_git_commit(), "git_dirty": get_git_dirty(),
        "machine": platform.platform(), "python": platform.python_version(),
        "rows": rows,
    }, indent=2))

    print(f"{'P':>3} {'f':>5} {'A':>6} {'eager ms':>9} {'lazy ms':>8} {'speedup':>8} "
          f"{'steps/eager':>11} {'mismatch':>8}")
    for r in rows:
        print(f"{r['num_predicates']:>3} {r['touch_fraction']:>5.2f} {r['num_pairs']:>6} "
              f"{r['eager_ms_mean']:>9.2f} {r['lazy_ms_mean']:>8.2f} {r['speedup']:>8.2f} "
              f"{r['pair_steps'] / max(r['eager_equivalent_pair_steps'], 1):>11.3f} "
              f"{r['decision_mismatches']:>8}"
              + (f"  mem eager {r['eager_peak_kib']:.0f} KiB lazy {r['lazy_peak_kib']:.0f} KiB"
                 if "eager_peak_kib" in r else ""))

    styles = ["-o", "--s", "-.^", ":v", "-D"]
    for kind in ("latency", "work"):
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        for p in args.predicates:
            for i, f in enumerate(args.fractions):
                sub = [r for r in rows if r["touch_fraction"] == f and r["num_predicates"] == p]
                a = [r["num_pairs"] for r in sub]
                if kind == "latency":
                    ax.plot(a, [r["lazy_ms_mean"] for r in sub], styles[i % 5], color="#d95f02",
                            ms=3, label=f"lazy, f={f}")
                    if i == 0:
                        ax.plot(a, [r["eager_ms_mean"] for r in sub], "-x", color="#444444",
                                ms=4, label="eager")
                else:
                    ax.plot(a, [r["pair_steps"] / (r["eager_equivalent_pair_steps"] or 1)
                                for r in sub], styles[i % 5], color="#d95f02", ms=3,
                            label=f"f={f}")
        ax.set_xscale("log")
        ax.set_xlabel("Active pairs A")
        if kind == "latency":
            ax.set_yscale("log")
            ax.set_ylabel("Temporal update latency per frame (ms)")
        else:
            ax.set_ylabel("Pair updates, lazy / eager")
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{kind}.{ext}", dpi=200)
        plt.close(fig)
    print(f"\nArtifacts: {out}")


if __name__ == "__main__":
    main()
