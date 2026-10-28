#!/usr/bin/env python3
"""Start the TAORG dashboard.

Usage (from the project root):
    PYTHONPATH=src python scripts/run_dashboard.py [--runs outputs/runs] [--port 8050]

Reads saved runs only; it never starts an experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "outputs" / "runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()

    import uvicorn

    from taorg.dashboard.app import create_app

    print(f"TAORG dashboard: http://{args.host}:{args.port}  (runs: {args.runs})")
    uvicorn.run(create_app(args.runs), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
