"""Pilot figures (Design.md §5–6): trade-off curves and occlusion-length robustness.

Both figures use test-split results.  Curves average each threshold's metrics
over seeds; robustness uses the validation-selected threshold, mean ± std
over seeds.  Exported as PDF (vector) + PNG preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed method order / styles across all figures.  V-HPED variants share the accent hue.
STYLE: dict[str, dict[str, Any]] = {
    "no_memory": dict(label="No memory", color="#7f7f7f", marker="o", ls=":"),
    "ema": dict(label="EMA", color="#8c6d31", marker="s", ls="--"),
    "gated_ema": dict(label="Vis-Gated EMA", color="#1f77b4", marker="^", ls="--"),
    "hmm": dict(label="HMM", color="#9467bd", marker="v", ls=":"),
    "vhped_handset": dict(label="V-HPED (hand-set)", color="#d95f02", marker="x", ls=":"),
    "vt_hmm_fit": dict(label="VT-HMM (fit, no Hawkes)", color="#e7298a", marker="D", ls="-."),
    "vhped_novis_fit": dict(label="V-HPED w/o visibility", color="#d95f02", marker="P", ls="-."),
    "vhped_self_fit": dict(label="V-HPED self-only", color="#d95f02", marker="*", ls="--"),
    "vhped_full_fit": dict(label="V-HPED (full)", color="#d95f02", marker="o", ls="-"),
}


def _style(m: str) -> dict[str, Any]:
    return STYLE.get(m, dict(label=m, color="black", marker=".", ls="-"))


def plot_tradeoff(result: dict[str, Any], methods: list[str], out_dir: Path) -> list[Path]:
    """One figure per gap: x = false persistence rate, y = persistence F1."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [c for c in result["cells"] if c["split"] == "test"]
    gaps = sorted({c["gap"] for c in cells})
    paths = []
    for g in gaps:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        for m in methods:
            curves = [c["curves"][m] for c in cells if c["gap"] == g]
            x = np.mean([[r["false_persistence_rate"] for r in cv] for cv in curves], axis=0)
            y = np.mean([[r["persistence_f1"] for r in cv] for cv in curves], axis=0)
            st = _style(m)
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3,
                    lw=1.2, label=st["label"])
        ax.set_xlabel("False persistence rate after termination (↓)")
        ax.set_ylabel("Persistence F1 in occlusion gaps (↑)")
        ax.set_title(f"Occlusion gap = {g} annotated frames (mean of {len(curves)} seeds)",
                     fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = out_dir / f"tradeoff_gap{g}.{ext}"
            fig.savefig(p, dpi=200)
            paths.append(p)
        plt.close(fig)
    return paths


def plot_robustness(result: dict[str, Any], methods: list[str], out_dir: Path) -> list[Path]:
    """Persistence F1 at the validation-selected threshold vs gap length."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = result["aggregate"]
    gaps = sorted({a["gap"] for a in agg})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for m in methods:
        rows = [next(a for a in agg if a["gap"] == g and a["method"] == m) for g in gaps]
        st = _style(m)
        for ax, key in zip(axes, ("persistence_f1", "false_persistence_rate")):
            mean = np.array([r[key][0] for r in rows])
            std = np.array([r[key][1] for r in rows])
            ax.errorbar(gaps, mean, yerr=std, color=st["color"], ls=st["ls"],
                        marker=st["marker"], ms=3, lw=1.2, capsize=2, label=st["label"])
    axes[0].set_ylabel("Edge persistence F1 (↑)")
    axes[1].set_ylabel("False persistence rate (↓)")
    for ax in axes:
        ax.set_xlabel("Injected occlusion gap (annotated frames)")
        ax.set_xscale("log")
        ax.set_xticks(gaps)
        ax.set_xticklabels([str(g) for g in gaps])
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6)
    fig.suptitle("Validation-selected threshold; mean ± std over seeds", fontsize=8)
    fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"robustness.{ext}"
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths
