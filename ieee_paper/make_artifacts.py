"""Build every table and figure of the IEEE paper from SAVED run artifacts.

Reads only files under ../outputs/runs (and ../paper/generated for figures that
were already rendered from those runs).  It runs no experiment and no model.

Usage (from ieee_paper/):  python make_artifacts.py
Outputs: tables/*.tex, figures/*.pdf|png, supplementary/provenance.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "outputs" / "runs"
GEN = ROOT / "paper" / "generated"

BASELINE = "20261027_202753_a7314fe5"      # Phase 8 ablation matrix (2 conditions)
PROBE_PRIOR = "20261028_044818_696b2840"   # H-ID probe (prior pinned)
PROBE_RHO = "20261028_045023_e10fb99f"     # H-ID2 probe (rho_unk pinned)
IA_V1 = "20261028_050001_0c54b136"         # IA-v1 matrix (8 conditions)
COMPLEXITY = "complexity/20261027T202322Z"
RECOVERY = "param_recovery/20261028T051308Z"

LABEL = {
    "no_memory": "No memory", "fixed_window": "Fixed window", "ema": "EMA",
    "gated_ema": "Vis.-gated EMA", "hmm": "HMM (hand-set)", "vhped_handset": "V-HPED hand-set",
    "vt_hmm_fit": "VT-HMM (fit)", "vhped_novis_fit": "V-HPED w/o vis. (fit)",
    "vhped_self_fit": "V-HPED self-only (fit)", "vhped_full_fit": "V-HPED (fit)",
    "vhped_lazyexact_fit": "V-HPED lazy-exact (fit)",
    "vhped_full_fit_prior05": r"V-HPED fit, $\pi_0{=}0.5$ pinned",
    "vhped_fit_rhounk05": r"V-HPED fit, $\rho_{\mathrm{unk}}{=}0.5$ pinned",
    "vhped_fit_rhounk1": r"V-HPED fit, $\rho_{\mathrm{unk}}{=}1$ pinned",
    "ia_full_fit": "IA-v1 (fit)", "ia_nobirth_fit": "IA-v1 w/o id.\\ births",
    "ia_nopost_fit": "IA-v1 w/o posterior $w$", "ia_lazyexact_fit": "IA-v1 lazy-exact",
}


def load(run: str, name: str = "metrics.json") -> dict:
    return json.loads((RUNS / run / name).read_text(encoding="utf-8"))


def row(table: list[dict], cond: str, method: str) -> dict:
    return next(r for r in table if r["condition"] == cond and r["method"] == method)


def pm(x: list[float], d: int = 3) -> str:
    return f"{x[0]:.{d}f}\\,$\\pm$\\,{x[1]:.{d}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def bold_best(rows, key, lower):
    vals = [r[key][0] for r in rows]
    best = min(vals) if lower else max(vals)
    return {id(r): abs(r[key][0] - best) < 1e-12 for r in rows}


def main_table(table, cond, methods, fname):
    rows = [row(table, cond, m) for m in methods]
    cols = [("frame_f1", False, 3), ("persistence_f1", False, 3),
            ("termination_delay_mean", True, 2), ("false_persistence_rate", True, 3),
            ("churn", True, 3), ("ece", True, 3)]
    best = {k: bold_best(rows, k, low) for k, low, _ in cols}
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Method & F1$_{\mathrm{fr}}\uparrow$ & F1$_{\mathrm{pers}}\uparrow$ & "
             r"Delay$\downarrow$ & FPR$\downarrow$ & Churn$\downarrow$ & ECE$\downarrow$ \\",
             r"\midrule"]
    for r in rows:
        cells = []
        for k, _, d in cols:
            txt = pm(r[k], d)
            cells.append(r"\textbf{" + txt + "}" if best[k][id(r)] else txt)
        lines.append(LABEL[r["method"]] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")


def paired_table(metrics, fname):
    keep = ["gated_ema", "vt_hmm_fit", "vhped_self_fit", "vhped_handset", "vhped_novis_fit"]
    lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Cond. & vs.\ method & $\Delta$F1$_{\mathrm{fr}}$ [95\% CI] & "
             r"$\Delta$Delay [95\% CI] \\", r"\midrule"]
    for s in metrics["paired_vs_reference"]:
        if s["b"] not in keep:
            continue
        c = "A" if s["condition"] == "occlusion_flicker" else "B"
        f, fc = s["frame_f1_video_diff_mean"], s["frame_f1_video_ci95"]
        d, dc = s["termination_delay_mean_video_diff_mean"], s["termination_delay_mean_video_ci95"]
        lines.append(f"{c} & {LABEL[s['b']]} & {f:+.3f} [{fc[0]:+.3f}, {fc[1]:+.3f}] & "
                     f"{d:+.2f} [{dc[0]:+.2f}, {dc[1]:+.2f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")


def probe_table(fname):
    a = load(PROBE_PRIOR)["table"]
    b = load(PROBE_RHO)["table"]
    cond = "occlusion_flicker_idswitch"
    rows = [row(a, cond, "gated_ema"), row(a, cond, "vhped_handset"),
            row(a, cond, "vhped_full_fit"), row(a, cond, "vhped_full_fit_prior05"),
            row(b, cond, "vhped_fit_rhounk05"), row(b, cond, "vhped_fit_rhounk1")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Method (cond.\ B) & F1$_{\mathrm{fr}}\uparrow$ & F1$_{\mathrm{pers}}\uparrow$ & "
             r"ECE$\downarrow$ & $\bar\tau$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{LABEL[r['method']]} & {pm(r['frame_f1'])} & {pm(r['persistence_f1'])} & "
                     f"{pm(r['ece'])} & {r['selected_threshold'][0]:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")


def ia_table(fname):
    m = load(IA_V1)
    t = m["table"]
    conds = [("swaps0", "0 swaps"), ("swaps2_k1", r"2 swaps, $s{=}1$"),
             ("swaps2_k0", r"2 swaps, $s{=}0$"), ("swaps4_k1", r"4 swaps, $s{=}1$"),
             ("swaps2_k1_visflip", r"2 sw., vis.\ flips"), ("swaps2_k1_jitter", r"2 sw., jitter")]
    methods = ["gated_ema", "vhped_handset", "vhped_full_fit", "vhped_fit_rhounk05", "ia_full_fit"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}", r"\toprule",
             "Condition & " + " & ".join(LABEL[x] for x in methods) + r" \\", r"\midrule"]
    for c, lab in conds:
        vals = [row(t, c, x)["frame_f1"][0] for x in methods]
        best = max(vals)
        cells = [(r"\textbf{%.3f}" % v) if abs(v - best) < 1e-12 else f"{v:.3f}" for v in vals]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")

    # dose-response figure: frame F1 vs number of swaps (informative score)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    style = {"gated_ema": ("#1f77b4", "--", "^"), "vhped_handset": ("#d95f02", ":", "x"),
             "vhped_full_fit": ("#d95f02", "-", "o"), "vhped_fit_rhounk05": ("#7570b3", "-.", "s"),
             "ia_full_fit": ("#e7298a", "-", "D")}
    xs = [0, 2, 4]
    cs = ["swaps0", "swaps2_k1", "swaps4_k1"]
    for x in methods:
        col, ls, mk = style[x]
        ys = [row(t, c, x)["frame_f1"][0] for c in cs]
        es = [row(t, c, x)["frame_f1"][1] for c in cs]
        ax.errorbar(xs, ys, yerr=es, color=col, ls=ls, marker=mk, ms=3.5, lw=1.1, capsize=2,
                    label=LABEL[x].replace("$\\pi_0{=}0.5$", "pi0=0.5")
                    .replace(r"$\rho_{\mathrm{unk}}{=}0.5$", "rho_unk=0.5"))
    ax.set_xticks(xs)
    ax.set_xlabel("ID switches per video (synthetic)")
    ax.set_ylabel("Frame-level edge F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.5, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / "figures" / f"idswitch_dose.{ext}", dpi=250)
    plt.close(fig)


def efficiency_table(fname):
    res = json.loads((RUNS / COMPLEXITY / "results.json").read_text(encoding="utf-8"))
    rows = [r for r in res["rows"] if r["num_pairs"] in (300, 3000)]
    lines = [r"\begin{tabular}{@{}rrrrrr@{}}", r"\toprule",
             r"$f$ & $A$ & Eager (ms) & Lazy (ms) & Speed-up & Lazy/eager updates \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['touch_fraction']:.2f} & {r['num_pairs']} & {r['eager_ms_mean']:.2f} & "
                     f"{r['lazy_ms_mean']:.2f} & {r['speedup']:.2f}$\\times$ & "
                     f"{r['pair_steps'] / r['eager_equivalent_pair_steps']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")
    return res


def recovery_table(fname):
    s = json.loads((RUNS / RECOVERY / "results.json").read_text(encoding="utf-8"))["summary"]
    names = {"mu": r"$\mu$", "alpha_diag": r"$\alpha_{pp}$", "eta": r"$\eta$",
             "beta_vis": r"$\beta_{\mathrm{vis}}$", "beta_occ": r"$\beta_{\mathrm{occ}}$",
             "rho_occ": r"$\rho_{\mathrm{occ}}$", "eps_rate": r"$\varepsilon$"}
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Param. & True & Init. & Fitted (mean\,$\pm$\,std) \\", r"\midrule"]
    for k, lab in names.items():
        v = s[k]
        lines.append(f"{lab} & {v['true']:.2f} & {v['init']:.2f} & "
                     f"{v['fitted_mean']:.2f}\\,$\\pm$\\,{v['fitted_std']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(HERE / "tables" / fname, "\n".join(lines) + "\n")


def copy_figures():
    for name in ("tradeoff_occlusion_flicker", "tradeoff_occlusion_flicker_idswitch",
                 "qualitative", "complexity_latency", "complexity_work", "termination_delay"):
        for ext in ("pdf", "png"):
            shutil.copy(GEN / f"{name}.{ext}", HERE / "figures" / f"{name}.{ext}")


def main() -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    base = load(BASELINE)
    t = base["table"]
    meth = ["no_memory", "fixed_window", "ema", "gated_ema", "hmm", "vhped_handset",
            "vt_hmm_fit", "vhped_novis_fit", "vhped_self_fit", "vhped_full_fit",
            "vhped_lazyexact_fit"]
    main_table(t, "occlusion_flicker", meth, "results_condA.tex")
    main_table(t, "occlusion_flicker_idswitch", meth, "results_condB.tex")
    paired_table(base, "paired.tex")
    probe_table("probes.tex")
    ia_table("ia_v1.tex")
    cx = efficiency_table("efficiency.tex")
    recovery_table("recovery.tex")
    copy_figures()
    prov = {"baseline_matrix": BASELINE, "probe_prior": PROBE_PRIOR, "probe_rho_unk": PROBE_RHO,
            "ia_v1_matrix": IA_V1, "complexity": COMPLEXITY, "param_recovery": RECOVERY,
            "complexity_machine": cx["machine"],
            "figures_copied_from": "paper/generated (rendered from the runs above by "
                                   "scripts/make_paper_artifacts.py)",
            "note": "All runs synthetic. git_commit = unknown (checkout not under git)."}
    write(HERE / "supplementary" / "provenance.json", json.dumps(prov, indent=2) + "\n")
    print("tables:", sorted(p.name for p in (HERE / "tables").iterdir()))
    print("figures:", sorted(p.name for p in (HERE / "figures").iterdir()))


if __name__ == "__main__":
    main()
