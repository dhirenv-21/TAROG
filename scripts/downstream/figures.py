"""Downstream study figures and predefined examples (prereg §12; timelines only, no frames).

    python scripts/downstream/figures.py --split eval

Fig 1  primary outcomes per method (pooled) with video-bootstrap intervals for the flag contrasts
Fig 2  flag behaviour on VidOR missing records (flag rate by oracle label) vs time since output
Fig 3  example timelines (a) ghost suppressed, (b) false termination, (c) delayed recurrence (B2 vs B2+F)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.adapters.vidor import sample  # noqa: E402
from taorg.experiments.downstream.io import read_csv_gz  # noqa: E402
from taorg.experiments.downstream.runner import load_video_data, make_method, run_method  # noqa: E402
from taorg.experiments.episodes.plots import _save, _setup  # noqa: E402
from taorg.metrics.episodes import runs  # noqa: E402

SPLIT = {"dev": "training", "eval": "validation"}
COL = {"FB": "#999999", "FB+F": "#56B4E9", "B2": "#E69F00", "B2+F": "#D55E00"}
MK = {"FB": "s", "FB+F": "D", "B2": "o", "B2+F": "^"}


def fig_outcomes(R: dict, P: Path) -> list:
    pt = R["point"]
    mets = [("IGT_s", "identity ghost time (s, pooled)"), ("PGG", "persistence in genuine gaps"),
            ("STD_g_s", "stable termination delay (s)"), ("RDD_s", "recurrence detection delay (s)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (m, lab) in zip(axes, mets):
        for i, meth in enumerate(("FB", "FB+F", "B2", "B2+F")):
            ax.scatter(i, pt[meth][m], color=COL[meth], marker=MK[meth], s=50)
        ax.set_xticks(range(4), ["FB", "FB+F", "B2", "B2+F"])
        ax.set_title(lab, fontsize=9)
    fig.suptitle(f"{R['split']}: pooled outcomes over {R['n_videos_analysed']} videos (decision: {R['decision']})", fontsize=9)
    fig.tight_layout()
    return _save(fig, P / "fig1_outcomes")


def fig_contrasts(R: dict, P: Path) -> list:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    specs = [("ghost_reduction", "ghost reduction (1 − IGT ratio)", 0.25, ">="), ("d_PGG", "Δ PGG", -0.05, ">="),
             ("d_STD_g_s", "Δ STD-g (s)", 0.20, "<="), ("d_RDD_s", "Δ RDD (s)", 0.20, "<=")]
    for ax, (k, lab, thr, _) in zip(axes, specs):
        for i, c in enumerate(("C1", "C2")):
            e = R["contrasts"][c][k]
            lo, hi = R["contrasts"][c][k + "_ci"]
            col = COL["FB+F"] if c == "C1" else COL["B2+F"]
            ax.errorbar(i, e, yerr=[[e - lo], [hi - e]], color=col, marker="o", capsize=4)
        ax.axhline(thr, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks([0, 1], ["FB+F vs FB", "B2+F vs B2"])
        ax.set_title(lab + " (97.5%)", fontsize=9)
    fig.tight_layout()
    return _save(fig, P / "fig2_flag_contrasts")


def fig_flag(split: str, P: Path) -> list:
    fr = read_csv_gz(Path(f"results/downstream/{split}/flag_records.csv.gz"))
    bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.01]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for L, c, m in (("TAKEOVER", "#D55E00", "^"), ("GAP", "#0072B2", "o"), ("GONE", "#999999", "s")):
        ys = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [r["flag"] == "True" for r in fr if r["label"] == L and lo < float(r["dt"]) <= hi]
            ys.append(np.mean(sel) if sel else np.nan)
        ax.plot([0.5 * (a + b) for a, b in zip(bins[:-1], bins[1:])], ys, color=c, marker=m, label=L)
    ax.set(xlabel="time since last output (s)", ylabel="share of missing-record frames flagged", ylim=(0, 1),
           title=f"Frozen flag on VidOR ({split})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, P / "fig3_flag_behaviour")


def examples(split: str, P: Path) -> tuple[dict, list]:
    fz = json.loads(Path("results/downstream/freeze/frozen.json").read_text(encoding="utf-8"))
    cand = {"a": [], "b": [], "c": []}
    cache = {}
    for vp in sample(Path("data/vidor"), SPLIT[split]):
        vid = Path(vp).stem
        d = load_video_data(Path(f"D:/TAORG_runs/downstream/{split}") / vid,
                            Path(f"data/vidor/{SPLIT[split]}") / vp.replace(".mp4", ".json"))
        o0 = run_method(make_method("B2", fz["params"]["B2"], None), d)
        o1 = run_method(make_method("B2", fz["params"]["B2"], d.flags), d)
        cache[vid] = (d, o0, o1)
        for k in d.stream.keys:
            L = d.labels[k]
            for (a, b) in runs(L["takeover"].tolist()):
                r = int((o0[k][a:b + 1] & ~o1[k][a:b + 1]).sum())
                if r > 0:
                    cand["a"].append((r, vid, k, a, b))
            for (a, b) in runs(L["gap"].tolist()):
                r = int((o0[k][a:b + 1] & ~o1[k][a:b + 1]).sum())
                if r > 0:
                    cand["b"].append((b - a + 1, vid, k, a, b))
            ep = runs(L["alive"].tolist())
            for (a, b) in ep[1:]:
                f0 = next((f for f in range(a, b + 1) if o0[k][f]), b + 1)
                f1 = next((f for f in range(a, b + 1) if o1[k][f]), b + 1)
                if f1 > f0:
                    cand["c"].append((f1 - f0, vid, k, a, b))
    chosen = {}
    for c, lst in cand.items():
        if not lst:
            chosen[c] = None
            continue
        lst.sort(key=lambda e: (e[1], tuple(e[2]), e[3]))
        med = sorted(e[0] for e in lst)[(len(lst) - 1) // 2]
        e = next(x for x in lst if x[0] == med)
        chosen[c] = {"n_candidates": len(lst), "value": e[0], "video": e[1], "key": list(e[2]), "start": e[3], "end": e[4]}
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    titles = {"a": "(a) identity ghost suppressed", "b": "(b) false termination in a genuine gap",
              "c": "(c) recurrence detected later"}
    for ax, c in zip(axes, "abc"):
        e = chosen[c]
        ax.set_yticks([])
        if e is None:
            ax.set_title(titles[c] + ": none", fontsize=9)
            continue
        d, o0, o1 = cache[e["video"]]
        k = next(x for x in d.stream.keys if list(x) == e["key"])
        lo, hi = max(1, e["start"] - 60), min(len(o0[k]) - 1, e["end"] + 60)
        x = np.arange(lo, hi + 1)
        L = d.labels[k]
        rows = [("annotated alive*", L["alive"], "#009E73"), ("observed", L["observed"], "#000000"),
                ("endpoint TAKEOVER", L["takeover"], "#CC79A7"), ("B2 output", o0[k], COL["B2"]),
                ("B2+F output", o1[k], COL["B2+F"])]
        for j, (nm, arr, col) in enumerate(rows):
            ax.fill_between(x, j + 0.1, j + 0.9, where=arr[lo:hi + 1], color=col, step="mid", alpha=0.8)
            ax.text(lo - 2, j + 0.5, nm, ha="right", va="center", fontsize=7)
        fl = [f for f in x if any(t in d.flags.get(f, ()) for t in (k.subject_track_id, k.object_track_id))]
        ax.scatter(fl, [len(rows) + 0.3] * len(fl), marker="|", color="k", s=20)
        ax.set_xlim(lo - 1, hi + 1)
        ax.set_title(f"{titles[c]}: video {e['video']}, key {tuple(e['key'])} (median of {e['n_candidates']}; "
                     f"ticks = flag on an endpoint)", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    return chosen, _save(fig, P / "fig4_examples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    _setup()
    P = Path(f"results/downstream/plots_{a.split}")
    P.mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(f"results/downstream/{a.split}/analysis.json").read_text(encoding="utf-8"))
    files = fig_outcomes(R, P) + fig_contrasts(R, P) + fig_flag(a.split, P)
    chosen, f4 = examples(a.split, P)
    Path(f"results/downstream/{a.split}/examples.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    print("\n".join(files + f4))


if __name__ == "__main__":
    main()
