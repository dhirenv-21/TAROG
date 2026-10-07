"""Showcase figures from SAVED artifacts only (no experiment is re-run).

fig_stage0_gain_vs_null: Stage 0 observed cross-history gain (bootstrap CI) vs the 19
    circular-shift null values and the registered C3 practical line (results/phase5h/stage0/summary.json).
fig_fit_support: Phase 5H-1 fitting-split event imbalance (results/phase5h1/audit/).

Usage:
    python scripts/phase5h1/showcase_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("results/phase5h1/showcase")
FAM_COL = {"SPATIAL": "#0072B2", "MOTION": "#E69F00", "ATTENTION_COMMUNICATION": "#009E73",
           "VEHICLE_SUPPORT": "#CC79A7", "CONTACT_MANIPULATION": "#D55E00"}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    s = json.loads(Path("results/phase5h/stage0/summary.json").read_text(encoding="utf-8"))
    g = s["test"]["G_cross"]
    null = np.array(s["c1_null"])
    d = s["decision"]
    c3_line = d["null_median"] + 0.10 * s["test"]["G_self"]["mean"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rng = np.random.default_rng(0)
    ax.scatter(null, 0.5 + rng.uniform(-0.12, 0.12, len(null)), color="#777777", s=28,
               label=f"shuffled-timing null (19 runs): {null.min():.3f}–{null.max():.3f}")
    ax.errorbar([g["mean"]], [1.4], xerr=[[g["mean"] - g["ci_lo"]], [g["ci_hi"] - g["mean"]]], fmt="o",
                color="#D55E00", capsize=5, ms=8, label=f"observed {g['mean']:.3f} [95% CI {g['ci_lo']:.3f}, {g['ci_hi']:.3f}]")
    ax.axvline(c3_line, color="k", ls="--", lw=1)
    ax.text(c3_line + 0.0008, 1.85, f"registered 'practically meaningful' line\n= null median + 10% of within-relation gain "
            f"= {c3_line:.3f}", fontsize=8, va="top")
    ax.annotate("", xy=(g["mean"], 1.05), xytext=(d["null_median"], 1.05),
                arrowprops=dict(arrowstyle="<->", color="#D55E00"))
    ax.text((g["mean"] + d["null_median"]) / 2, 0.88, f"timing-specific excess {d['excess']:.3f}", ha="center",
            fontsize=8, color="#D55E00")
    ax.set_yticks([0.5, 1.4])
    ax.set_yticklabels(["null", "observed"])
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("held-out gain of cross-relation history (HK2 − HK1), nats per event")
    ax.set_title("Stage 0 (VidOR ground-truth annotations): real signal above the null, but below the practical bar\n"
                 f"→ HAWKES SIGNAL = {d['label']}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig_stage0_gain_vs_null")

    rows = list(csv.DictReader(open("results/phase5h1/audit/predicate_support.csv", encoding="utf-8")))
    audit = json.loads(Path("results/phase5h1/audit/audit.json").read_text(encoding="utf-8"))
    fams = audit["family_event_counts"]
    tot = sum(fams.values())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    names = sorted(fams, key=lambda f: -fams[f])
    a1.barh(range(len(names)), [100 * fams[f] / tot for f in names], color=[FAM_COL[f] for f in names])
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels([f.replace("_", " ").lower() for f in names])
    a1.invert_yaxis()
    a1.set_xlabel("% of onset+offset events")
    a1.set_title("Event share by predicate family", fontsize=10)
    for i, f in enumerate(names):
        a1.text(100 * fams[f] / tot + 1, i, f"{100 * fams[f] / tot:.1f}%", va="center", fontsize=8)
    rows.sort(key=lambda r: -(int(r["onsets"]) + int(r["offsets"])))
    top = rows[:15]
    a2.bar(range(len(top)), [int(r["onsets"]) + int(r["offsets"]) for r in top],
           color=[FAM_COL[r["family"]] for r in top])
    a2.set_yscale("log")
    a2.set_xticks(range(len(top)))
    a2.set_xticklabels([r["predicate"] for r in top], rotation=60, ha="right", fontsize=8)
    a2.set_ylabel("onset + offset events (log)")
    a2.set_title(f"15 most frequent predicates (top 6 = {100 * audit['top6_event_share']:.0f}% of events)", fontsize=10)
    fig.suptitle(f"VidOR fitting split only ({audit['n_videos']} videos, {audit['n_events']:,} events): strong imbalance",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_fit_support")
    print("written to", OUT)


if __name__ == "__main__":
    main()
