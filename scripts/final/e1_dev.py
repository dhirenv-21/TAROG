"""E1 engineering check on REAL streams (development videos; labelled; not a method comparison).

The registered final evaluation stopped at its development gate, so E1 is run on the 30 development
EGTR streams at the diagnostic 90th-percentile threshold (results/final/dev/diagnosis.json), where the
streams are populated. Eager VHPED vs LazyVHPED with the VH calibration selection (results/final/dev/
gate.json): decision mismatches, max |dscore|, and the lazy counters (real-stream ADSA evidence).
    python scripts/final/e1_dev.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from taorg.adapters.vidor import sample
from taorg.experiments.final.runner import load_final_video, prepare
from taorg.temporal.lazy_vhped import LazyVHPED
from taorg.temporal.vhped import VHPED


def main() -> None:
    tau = json.loads(Path("results/final/dev/diagnosis.json").read_text())["quantile_tau_nm_f1"]["0.9"]["tau"]
    params = json.loads(Path("results/final/dev/gate.json").read_text())["selected"]["VH"]["params"]
    rows = []
    for v, vp in [(Path(vp).stem, vp) for vp in sample(Path("data/vidor"), "training")]:
        d = load_final_video(Path("D:/TAORG_runs/downstream/dev") / v, Path("D:/TAORG_runs/egtr/sgg/dev") / f"{v}.npz",
                             Path("data/vidor/training") / vp.replace(".mp4", ".json"))
        p = prepare(d, tau)
        eager, lazy = VHPED(**params, rung="m3"), LazyVHPED(**params, rung="m3")
        res = {}
        for name, m in (("eager", eager), ("lazy", lazy)):
            m.reset_video(v)
            t0 = time.perf_counter()
            res[name] = [{o.edge_key: o.predicate_score for o in m.step(g).edge_observations} for g in p.ps.graphs]
            res[name + "_s"] = time.perf_counter() - t0
        mism = md = 0
        for e, z in zip(res["eager"], res["lazy"]):
            for k in set(e) | set(z):
                if (k in e) != (k in z):
                    mism += 1
                else:
                    md = max(md, abs(e[k] - z[k]))
        eu = sum(len(eager._x) for _ in [0])
        rows.append({"video": v, "steps": len(p.ps.graphs), "records": sum(len(g.edge_observations) for g in p.ps.graphs),
                     "mismatches": mism, "max_abs_score_diff": md, "eager_s": res["eager_s"], "lazy_s": res["lazy_s"],
                     "eager_keys_final": eu, **lazy.counters()})
        print(v, rows[-1]["steps"], "mism", mism, "maxdiff", md, flush=True)
    tot = {k: sum(r[k] for r in rows) for k in ("steps", "records", "mismatches", "key_updates", "emitted", "freezes",
                                                 "valid_pops", "stale_pops", "eager_s", "lazy_s")}
    tot["max_abs_score_diff"] = max(r["max_abs_score_diff"] for r in rows)
    out = {"note": "engineering check on development streams at the diagnostic 90th-percentile tau; not a comparison",
           "tau": tau, "params": params, "totals": tot, "videos": rows}
    Path("results/final/dev/e1_real_streams.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(tot, indent=1, default=float))


if __name__ == "__main__":
    main()
