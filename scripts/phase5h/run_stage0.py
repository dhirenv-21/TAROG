"""Phase 5H Stage 0 — VidOR ground-truth Hawkes signal probe (pre-registered; see
research/phase5h_stage0_preregistration.md). Writes results/phase5h/stage0/.

Usage:
    python scripts/phase5h/run_stage0.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiments/phase5h_stage0.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/phase5h/stage0"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from taorg.hawkes.stage0 import run
    s = run(a.config, a.out)
    logging.info("HAWKES SIGNAL = %s", s["decision"]["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
