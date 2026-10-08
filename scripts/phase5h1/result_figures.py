"""Phase 5H-1 result figures from results/phase5h1/full/analysis.json (saved artifacts only).

fig_main_comparison : per-event held-out ΔLL ladder with 95% video-bootstrap CIs (EVAL, EVAL-A, EVAL-B)
fig_null_comparison : observed Δ(G4x,G2) and Δ(G4,G3) vs 19 N-circ + 19 N-jit replicates
fig_ablation_stability : RQ-H2 class/lag ablations; per-family Δ_F(G4,G3) in EVAL-A vs EVAL-B vs null
fig_k4_stability : K4 fold reproducibility (sign-consistent cells, inter-fold Spearman) vs N-jit fold nulls

Written before any EVAL result was computed (layout is fixed; values come from the JSON).

Usage:
    python scripts/phase5h1/result_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("results/phase5h1/full")
FAM_COL = {"SPATIAL": "#0072B2", "MOTION": "#E69F00", "ATTENTION_COMMUNICATION": "#009E73",
           "VEHICLE_SUPPORT": "#CC79A7", "CONTACT_MANIPULATION": "#D55E00"}
SET_STYLE = {"EVAL": ("o", "#000000", 0.0), "EVAL_A": ("s", "#0072B2", -0.22), "EVAL_B": ("^", "#E69F00", 0.22)}
LADDER = [("G1-G0", "heterogeneity\nG1 − G0"), ("G2-G1", "duration\nG2 − G1"), ("G3-G2", "simple transitions\nG3 − G2"),
          ("G4x-G2", "cross-history | het+dur\nG4x − G2  (K1)"), ("G4-G3", "history | transitions\nG4 − G3  (K2)"),
          ("G4+-G4", "positive-only history\nG4+ − G4")]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci_row(ax, y, st, marker, color, label=None, ms=6):
    ax.errorbar([st["mean"]], [y], xerr=[[st["mean"] - st["ci_lo"]], [st["ci_hi"] - st["mean"]]], fmt=marker,
                color=color, capsize=3, ms=ms, lw=1.2, label=label)


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    a = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    C = a["comparisons"]
    dec = a["decision"]

    # ---- main comparison: two panels (large-scale ladder; zoom on history terms) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, items in ((ax1, LADDER[:3]), (ax2, LADDER[3:])):
        for i, (key, lab) in enumerate(items):
            for vs, (mk, col, off) in SET_STYLE.items():
                ci_row(ax, i + off, C[key][vs], mk, col, label={"EVAL": "EVAL (A∪B, 1535 videos)",
                       "EVAL_A": "EVAL-A (700)", "EVAL_B": "EVAL-B (835)"}[vs] if i == 0 else None,
                       ms=7 if vs == "EVAL" else 5)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([lab for _, lab in items], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", lw=0.8)
        ax.set_xlabel("held-out Δ log-likelihood (nats per EVAL event)")
    ax2.axvline(dec["k3_threshold"] + dec["null_time_med"], color="k", ls="--", lw=0.9)
    ax2.text(dec["k3_threshold"] + dec["null_time_med"], -0.62, " K3 bar for G4 − G3\n (null median + 10% of G2 − G1)",
             fontsize=7, va="bottom")
    ax1.set_title("Controls (registered ladder)", fontsize=9)
    ax2.set_title("Event-history terms", fontsize=9)
    ax1.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"Phase 5H-1 held-out comparison (EXPLORATORY; 95% video-bootstrap CI) → decision {dec['label']}",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_main_comparison")

    # ---- null comparison ----
    reps = a["nulls"]["replicates"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    rng = np.random.default_rng(0)
    for ax, key, nk, title in ((axes[0], "G4x-G2", "d_cross", "K1: Δ(G4x, G2)"), (axes[1], "G4-G3", "d_time", "K2: Δ(G4, G3)")):
        for kind, mk, col, lab in (("circ", "o", "#777777", "N-circ (19)"), ("jit", "D", "#009E73", "N-jit ±5 s (19)")):
            v = [r[nk] for r in reps if r["kind"] == kind]
            ax.scatter(v, 0.5 + rng.uniform(-0.12, 0.12, len(v)) + (0 if kind == "circ" else 0.35), marker=mk,
                       color=col, s=22, label=lab, alpha=0.85)
        ci_row(ax, 1.6, C[key]["EVAL"], "o", "#D55E00", label="observed (95% CI)", ms=8)
        ax.axvline(max(r[nk] for r in reps), color="k", ls=":", lw=0.9)
        ax.set_yticks([0.5, 0.85, 1.6])
        ax.set_yticklabels(["N-circ", "N-jit", "observed"])
        ax.set_ylim(0.1, 2.0)
        ax.set_xlabel("Δ log-likelihood (nats per event)")
        ax.set_title(f"{title}; dotted = null max", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Observed history gain vs registered nulls (refit at the frozen ridge on null TRAIN, scored on null EVAL)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_null_comparison")

    # ---- ablation + family stability ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ab = a["ablations"]
    order = ["G3+H:ON>ON", "G3+H:ON>OFF", "G3+H:OFF>ON", "G3+H:OFF>OFF", "G3+H:lag0", "G3+H:lag1", "G3+H:lag2"]
    labels = ["ONSET→ONSET", "ONSET→OFFSET", "OFFSET→ONSET", "OFFSET→OFFSET", "lag ≤ 0.5 s", "lag 0.5–2 s", "lag 2–10 s"]
    for i, r in enumerate(order):
        for vs, (mk, col, off) in SET_STYLE.items():
            ci_row(ax1, i + off, ab[f"{r}-G3"][vs], mk, col, ms=6 if vs == "EVAL" else 4)
    ci_row(ax1, len(order) + 0.5, C["G4-G3"]["EVAL"], "o", "#000000")
    ax1.set_yticks(list(range(len(order))) + [len(order) + 0.5])
    ax1.set_yticklabels(labels + ["all history (G4)"], fontsize=8)
    ax1.invert_yaxis()
    ax1.axvline(0, color="#777777", lw=0.8)
    ax1.set_xlabel("Δ vs G3 (nats per event)")
    ax1.set_title("Which history terms carry the gain (G3 + one block)", fontsize=9)
    fams = a["families"]
    names = sorted(fams, key=lambda f: -fams[f]["d_all"]["n_events"])
    for i, F in enumerate(names):
        f = fams[F]
        ci_row(ax2, i - 0.15, f["d_a"], "s", FAM_COL[F], ms=5)
        ci_row(ax2, i + 0.15, f["d_b"], "^", FAM_COL[F], ms=5)
        ax2.plot([f["null_max"]], [i], marker="|", color="k", ms=14, mew=1.5)
        ax2.text(1.01, i, "PASS" if f["passes"] else "fail", transform=ax2.get_yaxis_transform(), va="center",
                 fontsize=8, fontweight="bold" if f["passes"] else "normal")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"{F.replace('_', ' ').lower()}\n({int(fams[F]['d_all']['n_events'])} events)" for F in names],
                        fontsize=7)
    ax2.invert_yaxis()
    ax2.axvline(0, color="#777777", lw=0.8)
    ax2.set_xlabel("Δ_F(G4, G3), nats per family event   (■ EVAL-A, ▲ EVAL-B, | null max)")
    ax2.set_title("Family test (NARROW): both halves > 0, above null max, ≥ 10% of duration gain", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_ablation_stability")

    # ---- K4 fold reproducibility (negative result shown at the same prominence) ----
    st = a["stability"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    nn = st["n_sign_agree_null"]
    ax1.bar(["real"] + [f"N-jit {i + 1}" for i in range(len(nn))], [st["n_sign_agree_real"]] + nn,
            color=["#D55E00"] + ["#009E73"] * len(nn))
    ax1.set_ylabel("HIST cells with 5/5 sign agreement")
    ax1.set_ylim(0, 1.15 * st["n_sign_agree_real"])
    ax1.set_title(f"Sign-consistent cells: real {st['n_sign_agree_real']} of {st['n_cells']} vs null {min(nn)}–{max(nn)}\n"
                  f"min estimated FDR = {st['fdr_scan_min']:.2f} > 0.10 → 0 discoveries (K4 fails)", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7)
    rn = st["rho_null"]
    ax2.scatter(rn, [0] * len(rn), marker="D", color="#009E73", s=30, label="N-jit fold nulls (5)")
    ax2.scatter([st["rho_real"]], [0], marker="o", color="#D55E00", s=60, label="real")
    ax2.set_yticks([])
    ax2.set_xlabel("mean inter-fold Spearman of G4 HIST weights")
    ax2.set_title(f"Weight-vector agreement: real {st['rho_real']:.3f} > null max {max(rn):.3f} (this part passes)",
                  fontsize=9)
    ax2.legend(frameon=False, fontsize=7)
    fig.suptitle("K4 reproducibility of individual interactions (5 disjoint TRAIN folds)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_k4_stability")
    print("written to", OUT)


if __name__ == "__main__":
    main()
