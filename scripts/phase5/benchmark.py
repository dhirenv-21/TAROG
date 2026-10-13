"""Phase 5 benchmark: eager V-HPED vs certified lazy maintenance on synthetic traces.

    python scripts/phase5/benchmark.py            (full grid; resumable; results/phase5/benchmark.jsonl)
    python scripts/phase5/benchmark.py --smoke    (one tiny configuration)

Every configuration also re-checks emitted-edge equivalence (|score difference| < 1e-9; decisions may
differ only when the eager score is within 1e-9 of the threshold). Timings are wall-clock
(time.perf_counter) on this machine and are reported with the hardware; they are not portable.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from taorg.experiments.phase5.traces import TraceConfig, make_trace
from taorg.temporal.lazy_vhped import LazyVHPED
from taorg.temporal.vhped import VHPED
from taorg.utils.seed import set_seed

PARAMS = dict(mu=0.5, nu=0.01, gamma_o=0.5, gamma_v=3.0, miss=0.05, false_pos=0.01, threshold=0.5,
              r_ov=0.05, r_vo=0.05)          # rung M3 (lazy applies); M5 is an eager fallback by design
BASE = TraceConfig(n_frames=1800, n_tracks=60)
GRID = (
    [("A", {"n_keys": n}) for n in (100, 400, 1600, 6400)]
    + [("obs_rate", {"n_keys": 1600, "p_on": p}) for p in (0.01, 0.2)]
    + [("visibility", {"n_keys": 1600, "p_occ": p}) for p in (0.0, 0.1)]
    + [("expiry_rate", {"n_keys": 1600, "p_retire": p}) for p in (0.0005, 0.02)]
    + [("predicates", {"n_keys": 1600, "n_predicates": p}) for p in (2, 20)]
    + [("jitter", {"n_keys": 1600, "p_jitter": p}) for p in (0.01, 0.1)]
)
SEEDS = (53001, 53002)


def run(model, frames):
    model.reset_video("b")
    t0 = time.perf_counter()
    outs = [{o.edge_key: o.predicate_score for o in model.step(fr).edge_observations} for fr in frames]
    return time.perf_counter() - t0, outs


def one(family: str, over: dict, seed: int) -> dict:
    set_seed(seed)
    cfg = replace(BASE, **over)
    frames = make_trace(cfg, np.random.default_rng(seed))
    n_records = sum(len(f.edge_observations) for f in frames)
    eager, lazy = VHPED(**PARAMS), LazyVHPED(**PARAMS)
    te, oe = run(eager, frames)
    eager_updates = 0
    eager.reset_video("b")
    for fr in frames:                                   # count eager key updates (= |A_t| per frame)
        eager.step(fr)
        eager_updates += len(eager._x)
    tl, ol = run(lazy, frames)
    mism, near, maxdiff = 0, 0, 0.0
    for e, z in zip(oe, ol):
        for k in set(e) | set(z):
            if (k in e) != (k in z):
                mism += 1
            else:
                maxdiff = max(maxdiff, abs(e[k] - z[k]))
    c = lazy.counters()
    emitted = sum(len(o) for o in oe)
    return {"family": family, "override": over, "seed": seed, "config": asdict(cfg), "frames": cfg.n_frames,
            "records": n_records, "eager_s": te, "lazy_s": tl, "speedup": te / tl if tl else float("nan"),
            "eager_key_updates": eager_updates, "lazy_key_updates": c["key_updates"],
            "final_keys": len(eager._x), "emitted_total": emitted, "decision_mismatches": mism,
            "max_score_diff": maxdiff, "lazy_counters": c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path("results/phase5/benchmark_smoke.jsonl" if a.smoke else "results/phase5/benchmark.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.smoke:
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["family"], json.dumps(r["override"], sort_keys=True), r["seed"]))
    else:
        out.write_text("", encoding="utf-8")
    grid = [("smoke", {"n_keys": 50, "n_frames": 200})] if a.smoke else GRID
    seeds = SEEDS[:1] if a.smoke else SEEDS
    meta = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "params": PARAMS}
    Path(out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for fam, over in grid:
        for seed in seeds:
            key = (fam, json.dumps(over, sort_keys=True), seed)
            if key in done:
                continue
            r = one(fam, over, seed)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=float) + "\n")
            print(fam, over, seed, f"eager {r['eager_s']:.2f}s lazy {r['lazy_s']:.2f}s x{r['speedup']:.1f}",
                  f"updates {r['eager_key_updates']}->{r['lazy_key_updates']}", "mism", r["decision_mismatches"],
                  f"maxdiff {r['max_score_diff']:.1e}", flush=True)


if __name__ == "__main__":
    main()
