"""Phase 4 failure-case extraction (deterministic; rule fixed before V-HPED evaluation).

For each H2 failure condition and each H3 condition, at gap 10 and the primary
budget, compute M5's per-seed REGRET against the better of GatedEMA (M0) and
VA-HMM (M1) on the condition's registered primary quantities:

    default frontier:  regret = (F1_best - F1_M5) + (delay_M5 - delay_best) / 40
    identity frontier: regret = (ghost_M5 - ghost_best) / 300 + (swaprec_best - swaprec_M5)

where "best" is the per-seed better value of M0/M1 for each term.  The WORST
and the lower-median positive-regret seeds are saved (JSON + timeline figure of
the three edges with most M5 output errors).  "No failure observed" if no seed
has positive regret.  For H3 conditions the treatment shown is M6 and the
comparator set is {M5}.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from taorg.experiments.phase3.evaluation import build_model  # noqa: E402
from taorg.experiments.phase3.pipeline import read_csv_gz  # noqa: E402
from taorg.experiments.phase3.plots import _setup  # noqa: E402
from taorg.experiments.phase4.pipeline import _build, _slim_cfg, load_context, model_params, specs  # noqa: E402
from taorg.experiments.phase4.plots import STYLE  # noqa: E402
from taorg.schemas.edges import EdgeKey  # noqa: E402

logger = logging.getLogger(__name__)
GAP = 10


def _outs(method: str, params: dict, inst) -> list[set[EdgeKey]]:
    m = build_model(method, params)
    m.reset_video(inst.gt_frames[0].video_id)
    out = [{e.edge_key for e in m.step(f).edge_observations} for f in inst.input_frames]
    m.finalize_video()
    return out


def _plot(inst, outs: dict[str, list[set[EdgeKey]]], labels: dict[str, str], title: str, path: Path) -> list[str]:
    _setup()
    gt = [{o.edge_key for o in f.edge_observations if o.observed} for f in inst.gt_frames]
    treat = list(outs)[-1]
    keys = sorted({k for f in gt for k in f} | {k for f in outs[treat] for k in f})
    err = {k: sum((k in gt[t]) != (k in outs[treat][t]) for t in range(len(gt))) for k in keys}
    show = [k for k in sorted(keys, key=lambda k: -err[k]) if err[k] > 0][:3] or keys[:1]
    n = len(gt)
    fig, axes = plt.subplots(len(show), 1, figsize=(7.2, 1.1 + 1.5 * len(show)), squeeze=False)
    for ax, k in zip(axes[:, 0], show):
        rows = [("GT alive", [k in gt[t] for t in range(n)], "#222222")]
        vis = []
        for t in range(n):
            rec = next((o for o in inst.input_frames[t].edge_observations if o.edge_key == k), None)
            vis.append("NONE" if rec is None else ("OBS" if rec.observed else rec.edge_visibility.name))
        rows.append(("Input observed", [v == "OBS" for v in vis], "#999999"))
        for m, o in outs.items():
            rows.append((labels[m], [k in o[t] for t in range(n)], STYLE[m]["color"]))
        for t, v in enumerate(vis):
            c = {"OCCLUDED_OR_UNRELIABLE": "#CFE2F3", "UNKNOWN": "#EAD1DC", "NONE": "#F4E3C1"}.get(v)
            if c:
                ax.axvspan(t - 0.5, t + 0.5, color=c, zorder=0, linewidth=0)
        for i, (name, on, color) in enumerate(rows):
            y = len(rows) - i
            ax.broken_barh([(t - 0.5, 1) for t, a in enumerate(on) if a], (y - 0.35, 0.7), color=color)
        ax.set_yticks([len(rows) - i for i in range(len(rows))])
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(f"edge (subj, obj, pred) = {tuple(k)}; M-treatment output errors: {err[k]} frames",
                     fontsize=7, loc="left")
        ax.set_xlim(-0.5, n - 0.5)
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel("Frame (30 fps) — shading: blue OCCLUDED, pink UNKNOWN, tan no record (absence frames)")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(path.with_suffix(".pdf")), str(path.with_suffix(".png"))]


def run_failures(cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """Extract and save representative failure cases."""
    sp = specs(cfg)
    ctx = load_context(out)
    B = str(cfg["primary_budget_frames"])
    sel: dict[str, Any] = {}
    for g in ("baseline", "proposed"):
        for c, v in json.loads((out / "configs" / f"selected_{g}.json").read_text(encoding="utf-8")).items():
            sel.setdefault(c, {}).update(v)
    ev = pd.concat([pd.DataFrame(read_csv_gz(out / "raw" / f"evaluation_{g}.csv.gz")) for g in ("baseline", "proposed")])
    for c in ("persistence_f1", "term_delay_mean", "diag_ghost_edge_frames", "diag_swap_recall", "gap", "seed"):
        ev[c] = ev[c].astype(float)
    labels = cfg["method_labels"]
    index = []
    conds = [c for c, s in sp.items() if s.role == "h2_failure"] + list(cfg["criteria"]["h3"]["benefit_conditions"])
    for cond in conds:
        h3 = cond in cfg["criteria"]["h3"]["benefit_conditions"]
        treat = "vhped_m6" if h3 else "vhped_m5"
        comps = ["vhped_m5"] if h3 else ["gated_ema", "va_hmm"]
        ids = {m: sel[cond][m][B]["config_id"] for m in [treat] + comps}
        e = ev[(ev.condition == cond) & (ev.gap == GAP)]
        per = {m: e[e.config_id == ids[m]].set_index("seed").sort_index() for m in ids}
        identity = cond in cfg["identity_conditions"]
        if identity:
            ghost_best = np.minimum.reduce([per[m].diag_ghost_edge_frames for m in comps])
            sr_best = np.maximum.reduce([per[m].diag_swap_recall for m in comps])
            regret = (per[treat].diag_ghost_edge_frames - ghost_best) / 300 + (sr_best - per[treat].diag_swap_recall)
        else:
            f1_best = np.maximum.reduce([per[m].persistence_f1 for m in comps])
            d_best = np.minimum.reduce([per[m].term_delay_mean for m in comps])
            regret = (f1_best - per[treat].persistence_f1) + (per[treat].term_delay_mean - d_best) / 40
        regret = regret.fillna(0.0)
        pos = regret[regret > 1e-12].sort_values()
        entry = {"condition": cond, "gap": GAP, "treatment": treat, "comparators": comps,
                 "regret_kind": "identity" if identity else "default", "n_seeds": int(len(regret)),
                 "n_positive": int(len(pos)), "mean_regret": float(regret.mean()),
                 "per_seed_regret": {str(int(k)): float(v) for k, v in regret.items()}, "cases": []}
        if len(pos):
            picks = [("worst", int(pos.index[-1])), ("typical", int(pos.index[(len(pos) - 1) // 2]))]
            for tag, seed in picks:
                inst = _build(_slim_cfg(cfg), sp[cond], GAP, seed)
                outs = {m: _outs(m, model_params(m, sel[cond][m][B]["params"], ctx[cond]), inst)
                        for m in comps + [treat]}
                vals = {m: {k: float(per[m].loc[seed, k]) for k in ("persistence_f1", "term_delay_mean",
                                                                   "diag_ghost_edge_frames", "diag_swap_recall")}
                        for m in ids}
                title = f"{cond} gap {GAP} seed {seed} ({tag}); {labels[treat]} regret {regret[seed]:.3f}"
                figs = _plot(inst, outs, labels, title, out / "failures" / f"{cond}__{tag}")
                case = {"tag": tag, "seed": seed, "regret": float(regret[seed]), "metrics": vals, "figures": figs}
                (out / "failures" / f"{cond}__{tag}.json").write_text(json.dumps(case, indent=1), encoding="utf-8")
                entry["cases"].append(case)
        index.append(entry)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    (out / "failures" / "failure_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index
