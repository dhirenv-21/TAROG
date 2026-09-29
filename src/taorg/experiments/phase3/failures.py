"""Deterministic extraction of representative GatedEMA failure cases.

For every pre-registered failure category (research/phase3_preregistration.md
§9) we evaluate the primary-budget GatedEMA operating point on every
evaluation seed, compute a per-unit severity score, and save:

    - the WORST case (max severity; ties -> lowest seed, then unit order),
    - the TYPICAL case (lower median severity among units with severity > 0),

as a per-frame timeline JSON plus a timeline figure.  If no unit has positive
severity the category is recorded as "no failure observed" (with the number
of units examined) — absence of failure is also a result.

Severities (per unit):
    gap_fn                 edge: GT-alive frames inside its occlusion intervals not output
    term_delay             terminating edge: frozen termination-delay definition
    gap_fn_plus_recovery   edge: gap_fn + frames after the gap until re-output
    flips                  edge: output-state flips inside the GT-alive window
    id_switch              swap: ghost edge-frames on swapped tracks (from swap
                           start to sequence end) + GT-alive frames of edges on
                           swapped tracks missed during the swap
The same severity is also reported for the other methods on the same unit so
the reader can see whether a failure is specific to visibility gating.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from taorg.experiments.phase3.conditions import Instance, build_instance  # noqa: E402
from taorg.experiments.phase3.evaluation import build_model  # noqa: E402
from taorg.experiments.phase3.pipeline import condition_specs  # noqa: E402
from taorg.experiments.phase3.plots import STYLE, _setup  # noqa: E402
from taorg.schemas.edges import EdgeKey  # noqa: E402

logger = logging.getLogger(__name__)

CATEGORIES: list[dict[str, Any]] = [
    {"name": "short_occlusion", "condition": "C0_oracle", "gap": 1, "severity": "gap_fn"},
    {"name": "long_occlusion", "condition": "C0_oracle", "gap": 20, "severity": "gap_fn"},
    {"name": "true_termination", "condition": "C0_oracle", "gap": 5, "severity": "term_delay"},
    {"name": "termination_under_occlusion", "condition": "S1_termination_under_occlusion",
     "gap": 5, "severity": "term_delay"},
    {"name": "ambiguous_visibility_unknown", "condition": "S4_unknown_all", "gap": 5,
     "severity": "term_delay"},
    {"name": "visibility_returns_after_long_absence", "condition": "S5_long_absence", "gap": 60,
     "severity": "gap_fn_plus_recovery"},
    {"name": "repeated_occlusion", "condition": "S3_repeated_occlusion", "gap": 10,
     "severity": "gap_fn"},
    {"name": "id_switch", "condition": "S2_id_switch_d20", "gap": 0, "severity": "id_switch"},
    {"name": "visibility_flag_noise", "condition": "R1_visnoise_0.2", "gap": 10, "severity": "gap_fn"},
    {"name": "base_model_flicker", "condition": "R2_flicker_0.1", "gap": 10, "severity": "flips"},
]
METHODS_SHOWN = ("no_memory", "fixed_window", "ema", "gated_ema", "hmm")


def run_with_trace(method: str, params: dict, inst: Instance) -> dict[str, Any]:
    """Run a method recording per-frame output sets and internal scores."""
    model = build_model(method, params)
    model.reset_video(inst.gt_frames[0].video_id)
    outs: list[set[EdgeKey]] = []
    scores: list[dict[EdgeKey, float]] = []
    for f in inst.input_frames:
        o = model.step(f)
        outs.append({e.edge_key for e in o.edge_observations})
        state = getattr(model, "_edge_scores", None)
        if state is None:
            state = getattr(model, "_belief", None)
        scores.append(dict(state) if state is not None else {})
    model.finalize_video()
    return {"outs": outs, "scores": scores}


def _gt(inst: Instance) -> list[set[EdgeKey]]:
    return [{o.edge_key for o in f.edge_observations if o.observed} for f in inst.gt_frames]


def _occ_frames(inst: Instance) -> dict[EdgeKey, set[int]]:
    occ: dict[EdgeKey, set[int]] = {}
    for e in inst.occlusion_manifest.entries:
        if e.edge_key is not None:
            occ.setdefault(e.edge_key, set()).update(range(e.start_frame, e.end_frame + 1))
    return occ


def unit_severities(kind: str, inst: Instance, outs: list[set[EdgeKey]]) -> list[tuple[str, float, dict]]:
    """Return (unit_id, severity, unit_info) for every unit of a category."""
    gt = _gt(inst)
    n = len(gt)
    res: list[tuple[str, float, dict]] = []
    if kind in ("gap_fn", "gap_fn_plus_recovery"):
        for k, frames in sorted(_occ_frames(inst).items()):
            fn = sum(1 for f in frames if k in gt[f] and k not in outs[f])
            sev = float(fn)
            if kind == "gap_fn_plus_recovery":
                f = max(frames) + 1
                while f < n and k in gt[f] and k not in outs[f]:
                    sev += 1
                    f += 1
            res.append((str(tuple(k)), sev, {"edges": [list(k)], "start": min(frames),
                                             "end": max(frames)}))
    elif kind == "term_delay":
        for lt in inst.lifetimes:
            if not lt.terminates:
                continue
            k, t_end = lt.edge_key, lt.end_frame
            t_drop = next((f for f in range(t_end + 1, n) if k not in outs[f]), n)
            res.append((str(tuple(k)), float(t_drop - t_end),
                        {"edges": [list(k)], "start": t_end, "end": t_end}))
    elif kind == "flips":
        for lt in inst.lifetimes:
            k = lt.edge_key
            flips = sum(1 for f in range(lt.start_frame + 1, lt.end_frame + 1)
                        if (k in outs[f]) != (k in outs[f - 1]))
            res.append((str(tuple(k)), float(flips),
                        {"edges": [list(k)], "start": lt.start_frame, "end": lt.end_frame}))
    elif kind == "id_switch":
        gt_ever = set().union(*gt)
        for j, e in enumerate((inst.manifests.get("id_switch") or {}).get("entries", [])):
            tracks = set(e["track_ids"])
            s, t = e["start_frame"], min(e["end_frame"], n - 1)
            ghost = sum(1 for f in range(s, n) for k in outs[f]
                        if k not in gt[f] and (k.subject_track_id in tracks or k.object_track_id in tracks))
            missed = sum(1 for f in range(s, t + 1) for k in gt[f]
                         if (k.subject_track_id in tracks or k.object_track_id in tracks)
                         and k not in outs[f])
            edges = sorted({k for f in range(s, n) for k in outs[f] | gt[f]
                            if k.subject_track_id in tracks or k.object_track_id in tracks})
            res.append((f"swap{j}:{sorted(tracks)}", float(ghost + missed),
                        {"edges": [list(k) for k in edges], "start": s, "end": t,
                         "ghost_frames": ghost, "missed_frames": missed,
                         "ghost_keys": [list(k) for k in edges if k not in gt_ever]}))
    else:
        raise ValueError(f"Unknown severity kind {kind}")
    return res


def _timeline(inst: Instance, traces: dict[str, dict], info: dict, pad_before: int = 15,
              pad_after: int = 45) -> dict[str, Any]:
    n = len(inst.gt_frames)
    lo, hi = max(0, info["start"] - pad_before), min(n - 1, info["end"] + pad_after)
    gt = _gt(inst)
    edges = [EdgeKey(*e) for e in info["edges"]]
    tl: dict[str, Any] = {"frame_range": [lo, hi], "edges": {}}
    for k in edges:
        inp = {}
        for f in range(lo, hi + 1):
            rec = next((o for o in inst.input_frames[f].edge_observations if o.edge_key == k), None)
            inp[f] = rec
        tl["edges"][str(tuple(k))] = {
            "gt_alive": [k in gt[f] for f in range(lo, hi + 1)],
            "input_observed": [bool(inp[f] and inp[f].observed) for f in range(lo, hi + 1)],
            "input_visibility": [inp[f].edge_visibility.name if inp[f] else "ABSENT_RECORD"
                                 for f in range(lo, hi + 1)],
            "methods": {
                m: {"output": [k in tr["outs"][f] for f in range(lo, hi + 1)],
                    "internal_score": [tr["scores"][f].get(k) for f in range(lo, hi + 1)]}
                for m, tr in traces.items()
            },
        }
    return tl


def _plot_timeline(tl: dict[str, Any], title: str, labels: dict[str, str],
                   thresholds: dict[str, float], path: Path) -> list[str]:
    _setup()
    lo, hi = tl["frame_range"]
    xs = list(range(lo, hi + 1))
    edge_items = list(tl["edges"].items())[:3]
    rows_per_edge = 2 + len(METHODS_SHOWN)
    fig, axes = plt.subplots(len(edge_items), 1, figsize=(7.0, 1.0 + 1.55 * len(edge_items)),
                             squeeze=False)
    vis_color = {"VISIBLE": None, "OCCLUDED_OR_UNRELIABLE": "#CFE2F3", "UNKNOWN": "#EAD1DC"}
    for ax, (ek, d) in zip(axes[:, 0], edge_items):
        for i, (x, v, obs) in enumerate(zip(xs, d["input_visibility"], d["input_observed"])):
            c = vis_color.get(v)
            if c and not obs:
                ax.axvspan(x - 0.5, x + 0.5, color=c, zorder=0, linewidth=0)
        y = rows_per_edge
        ax.broken_barh([(x - 0.5, 1) for x, a in zip(xs, d["gt_alive"]) if a], (y - 0.35, 0.7),
                       color="#222222")
        y -= 1
        ax.broken_barh([(x - 0.5, 1) for x, a in zip(xs, d["input_observed"]) if a],
                       (y - 0.35, 0.7), color="#999999")
        ytick = [rows_per_edge, rows_per_edge - 1]
        ylab = ["GT alive", "Input observed"]
        for m in METHODS_SHOWN:
            if m not in d["methods"]:
                continue
            y -= 1
            ax.broken_barh([(x - 0.5, 1) for x, a in zip(xs, d["methods"][m]["output"]) if a],
                           (y - 0.35, 0.7), color=STYLE[m]["color"],
                           hatch="//" if m == "gated_ema" else None, edgecolor="white", linewidth=0)
            ytick.append(y)
            ylab.append(labels[m])
        ax.set_yticks(ytick)
        ax.set_yticklabels(ylab)
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_ylim(y - 0.8, rows_per_edge + 0.8)
        ax.set_title(f"edge (subj, obj, pred) = {ek}", fontsize=7, loc="left")
        ax.tick_params(axis="y", length=0)
    axes[-1, 0].set_xlabel("Frame index (30 fps)   —   shading: blue = input flagged OCCLUDED, "
                           "pink = UNKNOWN (absence frames)")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(path.with_suffix(".pdf")), str(path.with_suffix(".png"))]


def run_failures(cfg: dict[str, Any], out_dir: Path, smoke: bool = False) -> list[dict[str, Any]]:
    """Extract and save failure cases for every category."""
    specs = condition_specs(cfg)
    selected = json.loads((out_dir / "configs" / "selected_params.json").read_text(encoding="utf-8"))
    B = str(cfg["primary_budget_frames"])
    seeds = cfg["seeds"]["smoke" if smoke else "evaluation"]
    fdir = out_dir / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        if cat["condition"] not in specs:
            logger.warning("Failure category %s skipped: condition %s not configured",
                           cat["name"], cat["condition"])
            continue
        spec = specs[cat["condition"]]
        src = spec.name if spec.calibrate else spec.tuned_on
        ops = {m: selected[src][m][B] for m in METHODS_SHOWN}
        candidates = []
        cache: dict[int, tuple[Instance, dict]] = {}
        for seed in seeds:
            inst = build_instance(spec, cat["gap"], seed, cfg["world"], cfg["occlusion"])
            traces = {m: run_with_trace(m, ops[m]["params"], inst) for m in METHODS_SHOWN}
            cache[seed] = (inst, traces)
            other = {m: {u: s for u, s, _ in unit_severities(cat["severity"], inst, traces[m]["outs"])}
                     for m in METHODS_SHOWN}
            for unit, sev, info in unit_severities(cat["severity"], inst, traces["gated_ema"]["outs"]):
                candidates.append({"seed": seed, "unit": unit, "severity": sev, "info": info,
                                   "severity_by_method": {m: other[m].get(unit) for m in METHODS_SHOWN}})
        positive = sorted([c for c in candidates if c["severity"] > 0],
                          key=lambda c: (c["severity"], -c["seed"]))
        entry = {"category": cat["name"], "condition": cat["condition"], "gap": cat["gap"],
                 "severity_kind": cat["severity"], "budget": int(B), "tuning_source": src,
                 "operating_points": {m: ops[m]["config_id"] for m in METHODS_SHOWN},
                 "n_units": len(candidates), "n_units_positive": len(positive),
                 "gated_mean_severity": (sum(c["severity"] for c in candidates) / len(candidates)
                                         if candidates else float("nan")),
                 "mean_severity_by_method": {
                     m: (sum((c["severity_by_method"][m] or 0.0) for c in candidates) / len(candidates)
                         if candidates else float("nan")) for m in METHODS_SHOWN},
                 "cases": []}
        picks = []
        if positive:
            worst = max(positive, key=lambda c: (c["severity"], -c["seed"]))
            typical = positive[(len(positive) - 1) // 2]
            picks = [("worst", worst), ("typical", typical)]
        for tag, c in picks:
            inst, traces = cache[c["seed"]]
            tl = _timeline(inst, traces, c["info"])
            case_id = f"{cat['name']}__{tag}"
            payload = {"case_id": case_id, **{k: v for k, v in c.items()},
                       "instance_id": inst.instance_id, "manifest_sha256": inst.manifest_sha256,
                       "timeline": tl}
            (fdir / f"{case_id}.json").write_text(json.dumps(payload, indent=1, default=str),
                                                  encoding="utf-8")
            thr = {m: ops[m]["params"].get("score_threshold", float("nan")) for m in METHODS_SHOWN}
            title = (f"{cat['name']} ({tag}) — {inst.instance_id}; GatedEMA severity "
                     f"{c['severity']:.0f} ({cat['severity']}); budget {B} frames")
            figs = _plot_timeline(tl, title, cfg["method_labels"], thr, fdir / case_id)
            entry["cases"].append({"tag": tag, "case_id": case_id, "seed": c["seed"],
                                   "unit": c["unit"], "severity": c["severity"],
                                   "severity_by_method": c["severity_by_method"], "figures": figs})
        index.append(entry)
    (fdir / "failure_index.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    return index
