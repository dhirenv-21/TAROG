"""Phase 5H-1 pipeline smoke test (TRAIN videos only; output labelled SMOKE, never evidence).

Usage:
    python scripts/phase5h1/smoke.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from taorg.hawkes.phase5h1 import run_smoke
    res = run_smoke(Path("configs/experiments/phase5h1.yaml"), Path("results/phase5h1/smoke"))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
