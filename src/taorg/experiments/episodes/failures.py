"""Representative failures (prereg §9, rule fixed before evaluation).

For T in each BEN condition and in C4: per-seed J regret = J(best comparator) - J(T),
where the best comparator is the comparator with the highest MEAN J in that condition.
Save the worst seed and the lower-median positive-regret seed as timelines of the
three keys with the most T output errors (GT alive, input, best comparator, T).
"No failure observed" if no seed has positive regret.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.episodes import decision as D  # noqa: E402
from taorg.experiments.episodes.analysis import Data  # noqa: E402
from taorg.experiments.episodes.conditions import build_instance  # noqa: E402
from taorg.experiments.episodes.evaluation import build_model, run_model  # noqa: E402
from taorg.experiments.episodes.pipeline import conditions, load_transition, model_params, read_csv_gz, seeds  # noqa: E402
from taorg.experiments.episodes.plots import STYLE, _save, _setup  # noqa: E402


def _outs(cfg, method, params, inst, E):
    pred = run_model(build_model(method, model_params(method, params, cfg, E)), inst.input_frames, "v")
    return [{o.edge_key for o in f.edge_observations} for f in pred]


def _plot(cfg, inst, outs: dict[str, list[set]], title: str, path: Path) -> list[str]:
    _setup()
    n = len(inst.input_frames)
    gt = {k: [any(a <= f <= b for a, b in eps) for f in range(n)] for k, eps in inst.gt_episodes.items()}
    T = list(outs)[-1]
    keys = sorted(set(gt) | {k for s in outs[T] for k in s})
    err = {k: sum((gt.get(k, [False] * n)[f]) != (k in outs[T][f]) for f in range(n)) for k in keys}
    show = [k for k in sorted(keys, key=lambda k: -err[k]) if err[k] > 0][:3] or keys[:1]
    fig, axes = plt.subplots(len(show), 1, figsize=(7.2, 0.9 + 1.4 * len(show)), squeeze=False)
    lab = cfg["method_labels"]
    for ax, k in zip(axes[:, 0], show):
        vis, det = [], []
        for f in range(n):
            rec = next((o for o in inst.input_frames[f].edge_observations if o.edge_key == k), None)
            det.append(rec is not None and rec.observed)
            vis.append("NONE" if rec is None else ("OBS" if rec.observed else rec.edge_visibility.name))
        rows = [("GT alive", gt.get(k, [False] * n), "#222222"), ("input detection", det, "#999999")]
        rows += [(lab[m], [k in o[f] for f in range(n)], STYLE[m][0]) for m, o in outs.items()]
        for f, v in enumerate(vis):
            c = {"OCCLUDED_OR_UNRELIABLE": "#CFE2F3", "UNKNOWN": "#EAD1DC", "NONE": "#F4E3C1"}.get(v)
            if c:
                ax.axvspan(f - 0.5, f + 0.5, color=c, zorder=0, lw=0)
        for i, (name, on, col) in enumerate(rows):
            y = len(rows) - i
            ax.broken_barh([(f - 0.5, 1) for f, a in enumerate(on) if a], (y - 0.35, 0.7), color=col)
        ax.set_yticks([len(rows) - i for i in range(len(rows))])
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_title(f"key (subj, obj, pred) = {tuple(k)}; T output errors {err[k]} frames", fontsize=7, loc="left")
    axes[-1, 0].set_xlabel("frame (30 fps); shading: blue OCC flag, pink UNKNOWN, tan no record")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def run_failures(cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    """Apply the predefined failure-selection rule and save JSON + figures."""
    an = json.loads((out / "aggregated" / "analysis.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "configs" / "frozen_parameters.json").read_text(encoding="utf-8"))["selected"]
    E = load_transition(out)
    ev_seeds = seeds(cfg, "evaluation", False)
    data = Data(read_csv_gz(out / "raw" / "evaluation.csv.gz"), ev_seeds)
    T = an["T"]
    crit = cfg["criteria"]
    index = []
    for c in crit["benefit_conditions"] + ["C4_recurrence"]:
        comps = D.comparators_for(c, crit)
        best = max(comps, key=lambda y: np.nanmean(data.get(c, y, "J")))
        reg = data.get(c, best, "J") - data.get(c, T, "J")
        pos = [(r, s) for r, s in zip(reg, ev_seeds) if np.isfinite(r) and r > 0]
        rec = {"condition": c, "treatment": T, "best_comparator": best, "n_positive_regret": len(pos),
               "mean_regret": float(np.nanmean(reg))}
        if not pos:
            rec["result"] = "No failure observed"
            index.append(rec)
            continue
        pos.sort()
        picks = {"worst": pos[-1], "median": pos[(len(pos) - 1) // 2]}
        rec["cases"] = {}
        cond = conditions(cfg)[c]
        for tag, (r, s) in picks.items():
            inst = build_instance(cond, s, cfg["world"], cfg["base_corruption"])
            outs = {best: _outs(cfg, best, frozen[best]["params"], inst, E), T: _outs(cfg, T, frozen[T]["params"], inst, E)}
            files = _plot(cfg, inst, outs, f"{c} seed {s}: J regret of {cfg['method_labels'][T]} vs "
                                           f"{cfg['method_labels'][best]} = {r:.3f} ({tag})",
                          out / "failures" / f"{c}__{tag}")
            rec["cases"][tag] = {"seed": s, "regret_J": float(r), "files": files}
        index.append(rec)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    (out / "failures" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index
