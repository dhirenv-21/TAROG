"""Phase 5 figures from results/phase5/benchmark.jsonl (scripts only; no hand-transcribed numbers).

    python scripts/phase5/figures.py
Fig 1  per-frame key updates (eager vs lazy) and emitted edges per frame vs candidate keys A
Fig 2  wall-clock speed-up eager/lazy per benchmark setting (each seed shown)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.episodes.plots import _save, _setup  # noqa: E402

P = Path("results/phase5/plots")


def main() -> None:
    _setup()
    R = [json.loads(line) for line in Path("results/phase5/benchmark.jsonl").read_text(encoding="utf-8").splitlines()]
    A = sorted({r["override"]["n_keys"] for r in R if r["family"] == "A"})
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for lab, f, c, m in (("eager key updates", "eager_key_updates", "#999999", "s"),
                         ("lazy key updates", "lazy_key_updates", "#D55E00", "^"),
                         ("emitted edges (output floor)", "emitted_total", "#0072B2", "o")):
        ys = [np.mean([r[f] / r["frames"] for r in R if r["family"] == "A" and r["override"]["n_keys"] == a]) for a in A]
        ax[0].plot(A, ys, color=c, marker=m, label=lab)
    ax[0].set(xscale="log", yscale="log", xlabel="candidate keys A", ylabel="per frame",
              title="Work per frame (M3, synthetic)")
    ax[0].legend(fontsize=7)
    g = defaultdict(list)
    for r in R:
        lab = r["family"] + " " + ",".join(f"{k}={v}" for k, v in r["override"].items() if k != "n_keys" or r["family"] == "A")
        g[lab].append(r["speedup"])
    labs = list(g)
    for i, lab in enumerate(labs):
        ax[1].scatter([g[lab][j] for j in range(len(g[lab]))], [i] * len(g[lab]), color="#D55E00", s=18)
    ax[1].axvline(1.0, color="k", lw=0.7, ls="--")
    ax[1].set_yticks(range(len(labs)), labs, fontsize=7)
    ax[1].set(xlabel="wall-clock speed-up (eager / lazy)", title="Runtime (2 seeds per setting)")
    fig.tight_layout()
    print("\n".join(_save(fig, P / "fig1_work_and_speedup")))


if __name__ == "__main__":
    main()
