"""Phase 9: generate every paper table, figure and in-text number from run artifacts.

No number in the paper is typed by hand: tables (LaTeX + CSV), figures
(PDF + PNG) and `numbers.tex` (\\newcommand macros used in the text) are
written to paper/generated/ together with `provenance.json` listing the run
directories they came from.

Usage:
    python scripts/make_paper_artifacts.py [--matrix outputs/runs/<id>]
        [--complexity outputs/runs/complexity/<stamp>] [--pilot outputs/runs/<id>]
Defaults: latest non-smoke ablation-matrix run, latest complexity run, latest
non-smoke AG pilot run (skipped if none exists).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.pilot_figures import STYLE  # noqa: E402

OUT = Path("paper/generated")
STYLE = {**STYLE,
         "fixed_window": dict(label="Fixed window", color="#a6761d", marker="h", ls=":"),
         "vhped_lazyexact_fit": dict(label="V-HPED lazy-exact", color="#d95f02", marker="s",
                                     ls=(0, (1, 1)))}
METRIC_LABEL = {
    "frame_f1": r"Frame F1 $\uparrow$",
    "persistence_f1": r"Pers.\ F1 $\uparrow$",
    "termination_delay_mean": r"Term.\ delay $\downarrow$",
    "false_persistence_rate": r"False pers.\ $\downarrow$",
    "churn": r"Churn $\downarrow$",
    "ece": r"ECE $\downarrow$",
    "latency_ms": r"ms/frame",
}
LOWER_BETTER = {"termination_delay_mean", "false_persistence_rate", "churn", "ece", "latency_ms"}


def _latest(pattern_ok) -> Path | None:
    runs = sorted(Path("outputs/runs").glob("*/metrics.json"), key=lambda p: p.stat().st_mtime)
    for m in reversed(runs):
        kind = json.loads(m.read_text(encoding="utf-8")).get("kind", "")
        if pattern_ok(kind):
            return m.parent
    return None


def _label(m: str) -> str:
    return STYLE.get(m, {}).get("label", m)


def _cmd(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------

def matrix_tables(run: Path, numbers: list[str]) -> None:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    cells = json.loads((run / "cells.json").read_text(encoding="utf-8"))
    table = metrics["table"]
    conds = list(dict.fromkeys(r["condition"] for r in table))
    keys = list(METRIC_LABEL)

    with open(OUT / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "method"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for r in table:
            w.writerow([r["condition"], r["method"]] + [x for k in keys for x in r[k]])

    for cond in conds:
        rows = [r for r in table if r["condition"] == cond]
        best = {}
        for k in keys[:-1]:
            vals = [r[k][0] for r in rows]
            best[k] = min(vals) if k in LOWER_BETTER else max(vals)
        lines = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
                 "Method & " + " & ".join(METRIC_LABEL[k] for k in keys) + r" \\", r"\midrule"]
        for r in rows:
            cells_txt = []
            for k in keys:
                mean, std = r[k]
                fmt = f"{mean:.2f}" if k in ("termination_delay_mean", "latency_ms") \
                    else f"{mean:.3f}"
                txt = f"{fmt}$\\pm${std:.2f}" if k != "latency_ms" else fmt
                if k in best and np.isclose(mean, best[k]):
                    txt = r"\textbf{" + txt + "}"
                cells_txt.append(txt)
            lines.append(_label(r["method"]) + " & " + " & ".join(cells_txt) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / f"ablation_{cond}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # paired statistics vs reference
    stats = metrics["paired_vs_reference"]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Condition & vs.\ method & $\Delta$ frame F1 [95\% CI] & $\Delta$ term.\ delay "
             r"[95\% CI] & seeds F1$>$ \\", r"\midrule"]
    for s in stats:
        f1, f1ci = s["frame_f1_video_diff_mean"], s["frame_f1_video_ci95"]
        dl, dlci = s["termination_delay_mean_video_diff_mean"], s["termination_delay_mean_video_ci95"]
        n_seeds = len([c for c in cells if c["condition"] == s["condition"]])
        lines.append(
            f"{s['condition'].replace('_', ' ')} & {_label(s['b'])} & "
            f"{f1:+.3f} [{f1ci[0]:+.3f}, {f1ci[1]:+.3f}] & "
            f"{dl:+.2f} [{dlci[0]:+.2f}, {dlci[1]:+.2f}] & "
            f"{s['frame_f1_a_greater_seeds']}/{n_seeds}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "paired_stats.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # numbers used in text (reference vs strongest simple baseline)
    def row(cond, m):
        return next(r for r in table if r["condition"] == cond and r["method"] == m)
    c0 = conds[0]
    for m, tag in (("vhped_full_fit", "Full"), ("gated_ema", "Gated"), ("vt_hmm_fit", "Vthmm"),
                   ("vhped_novis_fit", "Novis"), ("vhped_lazyexact_fit", "Lazyexact")):
        r = row(c0, m)
        numbers.append(_cmd(f"syn{tag}FrameFone", f"{r['frame_f1'][0]:.3f}"))
        numbers.append(_cmd(f"syn{tag}Delay", f"{r['termination_delay_mean'][0]:.2f}"))
        numbers.append(_cmd(f"syn{tag}PersFone", f"{r['persistence_f1'][0]:.3f}"))
        numbers.append(_cmd(f"syn{tag}FalsePers", f"{r['false_persistence_rate'][0]:.3f}"))
        numbers.append(_cmd(f"syn{tag}ECE", f"{r['ece'][0]:.3f}"))
    numbers.append(_cmd("synSeeds", str(len({c['seed'] for c in cells}))))
    numbers.append(_cmd("synRunId", run.name.replace("_", r"\_")))

    results_text(table, metrics["paired_vs_reference"], conds)

    # trade-off figure per condition (test curves averaged over seeds)
    methods = list(dict.fromkeys(r["method"] for r in table))
    for cond in conds:
        fig, ax = plt.subplots(figsize=(4.4, 3.5))
        cc = [c for c in cells if c["condition"] == cond]
        for m in methods:
            x = np.mean([[r["false_persistence_rate"] for r in c["methods"][m]["test_curves"]]
                         for c in cc], axis=0)
            y = np.mean([[r["frame_f1"] for r in c["methods"][m]["test_curves"]]
                         for c in cc], axis=0)
            st = STYLE.get(m, {})
            ax.plot(x, y, color=st.get("color"), ls=st.get("ls", "-"), marker=st.get("marker"),
                    ms=3, lw=1.1, label=_label(m))
        ax.set_xlabel("False persistence rate after termination (↓)")
        ax.set_ylabel("Frame-level edge F1 (↑)")
        ax.set_title(f"Synthetic: {cond.replace('_', ' ')} (mean of {len(cc)} seeds)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5, loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"tradeoff_{cond}.{ext}", dpi=200)
        plt.close(fig)

    # termination-delay distribution (per-video means, pooled over seeds)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    data = [[pv["termination_delay_mean"] for c in cells if c["condition"] == c0
             for pv in c["methods"][m]["per_video"] if pv["num_terminating_edges"]]
            for m in methods]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([_label(m) for m in methods], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Termination delay per video (frames, ↓)")
    ax.set_title(f"Synthetic: {c0.replace('_', ' ')}", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"termination_delay.{ext}", dpi=200)
    plt.close(fig)

    # churn comparison
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    width = 0.8 / len(conds)
    for i, cond in enumerate(conds):
        rows = [row(cond, m) for m in methods]
        ax.bar(np.arange(len(methods)) + i * width, [r["churn"][0] for r in rows], width,
               yerr=[r["churn"][1] for r in rows], capsize=2,
               color=["#bbbbbb", "#555555"][i % 2], hatch=["", "//"][i % 2],
               label=cond.replace("_", " "))
    ax.set_xticks(np.arange(len(methods)) + width * (len(conds) - 1) / 2)
    ax.set_xticklabels([_label(m) for m in methods], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Churn rate (↓)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"churn.{ext}", dpi=200)
    plt.close(fig)


def results_text(table: list[dict], stats: list[dict], conds: list[str]) -> None:
    """Generate the results paragraph; comparison words are derived from the data."""
    def r(cond, m):
        return next(x for x in table if x["condition"] == cond and x["method"] == m)

    def st(cond, b):
        return next(s for s in stats if s["condition"] == cond and s["b"] == b)

    def ci_word(ci, higher_good=True):
        lo, hi = ci
        if lo > 0:
            return "higher" if higher_good else "longer"
        if hi < 0:
            return "lower" if higher_good else "shorter"
        return "not significantly different"

    def pm(x, d=3):
        return f"{x[0]:.{d}f}$\\pm${x[1]:.{d}f}"

    c0 = conds[0]
    full, gated, vt = r(c0, "vhped_full_fit"), r(c0, "gated_ema"), r(c0, "vt_hmm_fit")
    novis, selfm, lazy = r(c0, "vhped_novis_fit"), r(c0, "vhped_self_fit"), \
        r(c0, "vhped_lazyexact_fit")
    hand = r(c0, "vhped_handset")
    sg, sv, ss = st(c0, "gated_ema"), st(c0, "vt_hmm_fit"), st(c0, "vhped_self_fit")
    best_f1 = max((x for x in table if x["condition"] == c0), key=lambda x: x["frame_f1"][0])
    best_delay = min((x for x in table if x["condition"] == c0 and x["method"] != "no_memory"),
                     key=lambda x: x["termination_delay_mean"][0])
    lines = [
        "% Generated by scripts/make_paper_artifacts.py from run artifacts - do not edit.",
        rf"\paragraph{{Visibility gating vs.\ V-HPED.}} Under occlusion and detector flicker "
        rf"(Table~\ref{{tab:abl}}), fitted V-HPED reaches frame F1 {pm(full['frame_f1'])} vs.\ "
        rf"{pm(gated['frame_f1'])} for visibility-gated EMA; the paired per-video difference is "
        rf"{sg['frame_f1_video_diff_mean']:+.3f} (95\% CI "
        rf"[{sg['frame_f1_video_ci95'][0]:+.3f}, {sg['frame_f1_video_ci95'][1]:+.3f}], "
        rf"{ci_word(sg['frame_f1_video_ci95'])}). Termination delay is "
        rf"{pm(full['termination_delay_mean'], 2)} vs.\ {pm(gated['termination_delay_mean'], 2)} "
        rf"frames (paired difference {sg['termination_delay_mean_video_diff_mean']:+.2f}, CI "
        rf"[{sg['termination_delay_mean_video_ci95'][0]:+.2f}, "
        rf"{sg['termination_delay_mean_video_ci95'][1]:+.2f}]), false persistence "
        rf"{pm(full['false_persistence_rate'])} vs.\ {pm(gated['false_persistence_rate'])}, "
        rf"and churn {pm(full['churn'])} vs.\ {pm(gated['churn'])}.",
        rf"\paragraph{{Ablations.}} Removing visibility collapses persistence through occlusion "
        rf"(persistence F1 {pm(novis['persistence_f1'])} vs.\ {pm(full['persistence_f1'])}). "
        rf"Against the fitted VT-HMM (visibility, no Hawkes terms) the paired frame-F1 "
        rf"difference is {sv['frame_f1_video_diff_mean']:+.3f} (CI "
        rf"[{sv['frame_f1_video_ci95'][0]:+.3f}, {sv['frame_f1_video_ci95'][1]:+.3f}]) and the "
        rf"termination-delay difference {sv['termination_delay_mean_video_diff_mean']:+.2f} (CI "
        rf"[{sv['termination_delay_mean_video_ci95'][0]:+.2f}, "
        rf"{sv['termination_delay_mean_video_ci95'][1]:+.2f}]) frames. Cross-predicate terms vs.\ "
        rf"self-only: frame-F1 difference {ss['frame_f1_video_diff_mean']:+.3f} (CI "
        rf"[{ss['frame_f1_video_ci95'][0]:+.3f}, {ss['frame_f1_video_ci95'][1]:+.3f}]), delay "
        rf"difference {ss['termination_delay_mean_video_diff_mean']:+.2f} (CI "
        rf"[{ss['termination_delay_mean_video_ci95'][0]:+.2f}, "
        rf"{ss['termination_delay_mean_video_ci95'][1]:+.2f}]). Fitting vs.\ hand-set "
        rf"parameters: frame F1 {pm(full['frame_f1'])} vs.\ {pm(hand['frame_f1'])}, delay "
        rf"{pm(full['termination_delay_mean'], 2)} vs.\ {pm(hand['termination_delay_mean'], 2)}. "
        rf"The lazy-exact variant ($\gamma=0$, $\rho_{{\textsc{{unk}}}}=0$) reaches frame F1 "
        rf"{pm(lazy['frame_f1'])} and delay {pm(lazy['termination_delay_mean'], 2)}.",
        rf"\paragraph{{Best methods.}} The highest frame F1 in this condition is obtained by "
        rf"{_label(best_f1['method'])} ({pm(best_f1['frame_f1'])}); the shortest termination "
        rf"delay among memory methods by {_label(best_delay['method'])} "
        rf"({pm(best_delay['termination_delay_mean'], 2)} frames). "
        rf"Calibration (ECE) of the alive probability: V-HPED {pm(full['ece'])}, "
        rf"HMM {pm(r(c0, 'hmm')['ece'])}.",
    ]
    if len(conds) > 1:
        c1 = conds[1]
        f1, g1 = r(c1, "vhped_full_fit"), r(c1, "gated_ema")
        s1 = st(c1, "gated_ema")
        best1 = max((x for x in table if x["condition"] == c1), key=lambda x: x["frame_f1"][0])
        loses = s1["frame_f1_video_ci95"][1] < 0
        (OUT / "id_finding.tex").write_text(
            ("Under synthetic ID switches, however, fitted V-HPED has lower frame F1 than "
             "visibility-gated EMA and is poorly calibrated."
             if loses else
             "Under synthetic ID switches fitted V-HPED is not worse than visibility-gated EMA "
             "in frame F1.") + "\n", encoding="utf-8")
        lines.append(
            rf"\paragraph{{ID switches.}} With ID switches added (Table~\ref{{tab:ablid}}), "
            rf"frame F1 is {pm(f1['frame_f1'])} for V-HPED vs.\ {pm(g1['frame_f1'])} for gated EMA "
            rf"(paired difference {s1['frame_f1_video_diff_mean']:+.3f}, CI "
            rf"[{s1['frame_f1_video_ci95'][0]:+.3f}, {s1['frame_f1_video_ci95'][1]:+.3f}]), and "
            rf"termination delay {pm(f1['termination_delay_mean'], 2)} vs.\ "
            rf"{pm(g1['termination_delay_mean'], 2)} frames. The highest frame F1 here is "
            rf"obtained by {_label(best1['method'])} ({pm(best1['frame_f1'])}); fitted V-HPED's "
            rf"ECE rises to {pm(f1['ece'])} (hand-set: {pm(r(c1, 'vhped_handset')['ece'])}).")
    (OUT / "results_text.tex").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def complexity_tables(run: Path, numbers: list[str]) -> None:
    res = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rows = res["rows"]
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$f$ & $A$ & eager ms & lazy ms & speed-up & lazy/eager pair updates \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['touch_fraction']:.2f} & {r['num_pairs']} & {r['eager_ms_mean']:.2f} & "
                     f"{r['lazy_ms_mean']:.2f} & {r['speedup']:.2f} & "
                     f"{r['pair_steps'] / max(r['eager_equivalent_pair_steps'], 1):.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "efficiency.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ext in ("pdf", "png"):
        for kind in ("latency", "work"):
            src = run / f"{kind}.{ext}"
            if src.exists():
                shutil.copy(src, OUT / f"complexity_{kind}.{ext}")
    big = max(r["num_pairs"] for r in rows)
    sp = {r["touch_fraction"]: r["speedup"] for r in rows if r["num_pairs"] == big}
    fr = sorted(sp)
    numbers.append(_cmd("cxMaxPairs", str(big)))
    numbers.append(_cmd("cxSpeedupSparse", f"{sp[fr[0]]:.1f}"))
    numbers.append(_cmd("cxFracSparse", f"{fr[0]:.2f}"))
    numbers.append(_cmd("cxSpeedupDense", f"{sp[fr[-1]]:.2f}"))
    numbers.append(_cmd("cxMismatches", str(sum(r["decision_mismatches"] for r in rows))))
    numbers.append(_cmd("cxMachine", res["machine"].replace("_", r"\_")))


# ---------------------------------------------------------------------------
# Qualitative sequence (deterministic synthetic example)
# ---------------------------------------------------------------------------

def qualitative_figure(matrix_run: Path | None) -> None:
    from taorg.experiments.vhped_benchmark import ScenarioConfig, generate_video
    from taorg.temporal import METHODS
    from taorg.temporal.vhped import VHPED, VHPEDParams

    params = None
    if matrix_run is not None:
        cells = json.loads((matrix_run / "cells.json").read_text(encoding="utf-8"))
        params = VHPEDParams.from_dict(cells[0]["fitted"]["vhped_full_fit"]["params"])
    # Deterministic search for an illustrative case: an edge that is occluded
    # and later truly terminates while visible.
    for qseed in range(2_000_000, 2_000_500):
        v = generate_video("qual", qseed, ScenarioConfig(num_frames=150, num_pairs=2))
        ends, occ_frames = {}, {}
        for f in v.gt:
            for o in f.edge_observations:
                if o.observed:
                    ends[o.edge_key] = f.frame_idx
        for f in v.noisy:
            for o in f.edge_observations:
                if o.edge_visibility.value != "visible":
                    occ_frames.setdefault(o.edge_key, []).append(f.frame_idx)
        key = next((k for k, e in sorted(ends.items())
                    if k.predicate_id == 0 and 60 <= e <= 120
                    and any(e - 40 <= x < e - 5 for x in occ_frames.get(k, []))), None)
        if key is not None:
            break
    models = {"V-HPED (full)": VHPED(params=params, score_threshold=0.0) if params else VHPED(),
              "Vis-Gated EMA": METHODS["gated_ema"](score_threshold=0.0)}
    T = len(v.noisy)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    gt = [any(o.edge_key == key and o.observed for o in f.edge_observations) for f in v.gt]
    obs = [any(o.edge_key == key and o.observed for o in f.edge_observations) for f in v.noisy]
    occ = [any(o.pair_key == key[:2] and o.edge_visibility.value != "visible"
               for o in f.edge_observations) for f in v.noisy]
    t = np.arange(T)
    axes[0].fill_between(t, 0, 1, where=gt, color="#222222", alpha=0.25, step="mid", label="GT alive")
    axes[0].fill_between(t, 0, 1, where=occ, facecolor="none", edgecolor="#4c78a8", hatch="///",
                         step="mid", label="occluded")
    axes[0].plot(t[np.array(obs)], np.full(sum(obs), 0.5), "|", color="black", ms=8,
                 label="detection")
    axes[0].set_yticks([])
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    for name, m in models.items():
        m.reset_video("qual")
        s = []
        for f in v.noisy:
            out = {o.edge_key: o.predicate_score for o in m.step(f).edge_observations}
            s.append(out.get(key, 0.0))
        st = STYLE["vhped_full_fit" if "V-HPED" in name else "gated_ema"]
        axes[1].plot(t, s, color=st["color"], ls=st["ls"], lw=1.2, label=name)
    end = max(i for i, g in enumerate(gt) if g)
    axes[1].axvline(end + 0.5, color="#b2182b", ls="--", lw=1, label="true termination")
    axes[1].set_ylabel("score / P(alive)")
    axes[1].set_xlabel("frame")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"qualitative.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=None)
    ap.add_argument("--complexity", type=Path, default=None)
    ap.add_argument("--pilot", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    matrix = args.matrix or _latest(lambda k: "ablation matrix" in k and "SMOKE" not in k)
    cx_runs = sorted(Path("outputs/runs/complexity").glob("*/results.json"))
    complexity = args.complexity or (cx_runs[-1].parent if cx_runs else None)
    pilot = args.pilot or _latest(lambda k: k == "Phase 3 pilot")

    numbers: list[str] = ["% Generated by scripts/make_paper_artifacts.py - do not edit.\n"]
    if matrix:
        matrix_tables(matrix, numbers)
    if complexity:
        complexity_tables(complexity, numbers)
    qualitative_figure(matrix)
    numbers.append(_cmd("pilotAvailable", "1" if pilot else "0"))
    if pilot:
        for fig in (pilot / "figures").glob("*"):
            shutil.copy(fig, OUT / f"pilot_{fig.name}")
        shutil.copy(pilot / "metrics.json", OUT / "pilot_metrics.json")
        pm = json.loads((pilot / "metrics.json").read_text(encoding="utf-8"))
        rows = [rf"{_label(c['a'])} vs.\ {_label(c['b'])} & {c['a_better_frac']:.2f} & "
                rf"{c['hv_diff_mean']:+.3f}$\pm${c['hv_diff_std']:.3f} \\" for c in pm["comparisons"]]
        (OUT / "pilot_text.tex").write_text(
            "% Generated - do not edit.\n"
            rf"Hypervolume of the (false persistence, persistence F1) trade-off on Action Genome "
            rf"test videos (run {pilot.name.replace('_', chr(92) + '_')}); fraction of "
            r"(seed, gap) cells where the first method is better." "\n\n"
            r"\begin{tabular}{lrr}\toprule Comparison & a better & $\Delta$HV \\ \midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n"
            r"\includegraphics[width=0.48\linewidth]{generated/pilot_robustness.pdf}" "\n",
            encoding="utf-8")
    (OUT / "numbers.tex").write_text("".join(numbers), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "ablation_matrix_run": str(matrix) if matrix else None,
        "complexity_run": str(complexity) if complexity else None,
        "ag_pilot_run": str(pilot) if pilot else None,
    }, indent=2))
    print(json.dumps({"matrix": str(matrix), "complexity": str(complexity),
                      "pilot": str(pilot)}, indent=2))
    print(f"Wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
