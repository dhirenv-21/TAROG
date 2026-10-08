"""Phase 5H-1 registered run (one stage per invocation; resumable).

Usage (in registered order; commit the freeze before evaluate):
    python scripts/phase5h1/run.py prepare
    python scripts/phase5h1/run.py tune --workers 6
    python scripts/phase5h1/run.py freeze            # then: git add results/phase5h1/full/frozen && git commit
    python scripts/phase5h1/run.py evaluate_prepare
    python scripts/phase5h1/run.py evaluate --workers 6
    python scripts/phase5h1/run.py nulls --workers 5
    python scripts/phase5h1/run.py stability --workers 5
    python scripts/phase5h1/run.py hypotheses --workers 8
    python scripts/phase5h1/run.py check

Heavy state: --run-root (default D:/TAORG_runs/phase5h1/main). Execution settings (workers, slots,
chunk) never change task specs, seeds or results; they are logged in <run-root>/invocations.jsonl.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "tune", "freeze", "evaluate_prepare", "evaluate", "nulls",
                                      "stability", "hypotheses", "check"])
    ap.add_argument("--run-root", type=Path, default=Path("D:/TAORG_runs/phase5h1/main"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=None, help="memory budget in job-weight units (default workers)")
    ap.add_argument("--chunk", type=int, default=400, help="videos per design chunk (memory bound only)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from taorg.hawkes import phase5h1_run as R
    from taorg.hawkes.runner import code_hash
    run = a.run_root
    run.mkdir(parents=True, exist_ok=True)
    inv = {"stage": a.stage, "workers": a.workers, "slots": a.slots, "chunk": a.chunk, "code_hash": code_hash(),
           "git_commit": R._git("rev-parse", "HEAD"), "git_dirty": bool(R._git("status", "--porcelain", "src")),
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    slots = a.slots or a.workers
    if a.stage == "prepare":
        out = R.stage_prepare(run)
    elif a.stage == "tune":
        out = R.stage_tune(run, a.workers, slots, a.chunk)
    elif a.stage == "freeze":
        out = R.stage_freeze(run)
    elif a.stage == "evaluate_prepare":
        out = R.stage_evaluate_prepare(run)
    elif a.stage == "evaluate":
        out = R.stage_evaluate(run, a.workers, slots, a.chunk)
    elif a.stage == "nulls":
        out = R.stage_nulls(run, a.workers, slots, a.chunk)
    elif a.stage == "stability":
        out = R.stage_stability(run, a.workers, slots, a.chunk)
    elif a.stage == "hypotheses":
        out = R.stage_hypotheses(run, a.workers, slots)
    else:
        out = R.stage_check(run)
    inv["elapsed_s"] = round(time.time() - t0, 1)
    st = out.get("status") if isinstance(out, dict) else None
    inv["job_status_counts"] = {s: list(st.values()).count(s) for s in set(st.values())} if st else None
    with open(run / "invocations.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(inv) + "\n")
    print(json.dumps(out if not st else {**{k: v for k, v in out.items() if k != "status"},
                                         "job_status_counts": inv["job_status_counts"]}, indent=1, default=str)[:4000])
    return 0 if not st or all(v == "complete" for v in st.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
