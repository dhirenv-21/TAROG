"""Phase 3 go/no-go pilot on Action Genome (GT relations + synthetic occlusion).

Usage:
    python scripts/run_pilot_ag.py --config configs/experiments/pilot_action_genome.yaml
        [--root data/action_genome] [--workers 4]
        [--max-videos N] [--seeds 42 123] [--iters 20]     (smoke runs only)

--max-videos / --seeds / --iters are for smoke runs: a run using them is
flagged `smoke_run: true` and must not be reported as the pilot.
Writes outputs/runs/<run_id>/ (config, metadata, metrics summary, all curves,
corruption manifests, fitted parameters, figures).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

from taorg.config import load_config
from taorg.data.action_genome import ActionGenomeConfig, load_action_genome
from taorg.experiments.artifacts import RunArtifactWriter
from taorg.experiments.pilot import PilotData, run_pilot, split_train
from taorg.experiments.pilot_figures import plot_robustness, plot_tradeoff
from taorg.schemas.runs import RunMetadata
from taorg.utils.git import get_git_branch, get_git_commit, get_git_dirty
from taorg.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/experiments/pilot_action_genome.yaml"))
    ap.add_argument("--root", type=Path, default=None, help="override dataset root")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=None, help="SMOKE: cap videos per split")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="SMOKE: override seeds")
    ap.add_argument("--iters", type=int, default=None, help="SMOKE: trainer iterations")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = load_config(args.config)
    ds_cfg = load_config(Path(cfg["dataset_config"]))["action_genome"]
    if args.root is not None:
        ds_cfg["root"] = str(args.root)
    smoke = any(x is not None for x in (args.max_videos, args.seeds, args.iters))
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.iters is not None:
        cfg["training"]["trainer"]["num_iters"] = args.iters
    cfg["smoke_run"] = smoke
    cfg["resolved_dataset"] = ds_cfg
    cfg["command"] = " ".join(sys.argv)
    set_seed(int(cfg["seeds"][0]))

    ag = ActionGenomeConfig(root=Path(ds_cfg["root"]), annotation_dir=ds_cfg["annotation_dir"],
                            fps=float(ds_cfg["fps"]), min_frames=int(ds_cfg["min_frames"]))
    train, vocab, train_stats = load_action_genome(ag, "train")
    test, _, test_stats = load_action_genome(ag, "test")
    if args.max_videos is not None:
        train, test = train[: args.max_videos], test[: args.max_videos]
    fit, val = split_train(train, float(cfg["val_fraction"]), int(cfg["split_seed"]))
    data = PilotData(fit_videos=fit, val_videos=val, test_videos=test,
                     num_predicates=vocab.num_predicates, dataset_id="action_genome/gt")

    meta = RunMetadata(
        git_commit=get_git_commit(), git_dirty=get_git_dirty(), git_branch=get_git_branch(),
        command=cfg["command"],
        dataset_id=f"action_genome (fit {len(fit)}, val {len(val)}, test {len(test)} videos)",
        base_model_id="ground_truth_relations", temporal_method="pilot_sweep",
        seeds=[int(s) for s in cfg["seeds"]],
    )
    meta.mark_started()
    writer = RunArtifactWriter(meta, output_base=args.out)
    run_dir = writer.create_run_directory()
    writer.save_config(cfg)
    t0 = time.perf_counter()
    try:
        result = run_pilot(cfg, data, workers=args.workers)
    except Exception as exc:
        meta.mark_failed(repr(exc))
        writer.save_metadata()
        raise
    elapsed = time.perf_counter() - t0

    (run_dir / "load_stats.json").write_text(json.dumps(
        {"train": train_stats.to_dict(), "test": test_stats.to_dict(),
         "relationship_classes": vocab.relationship_classes}, indent=2))
    (run_dir / "fitted_params.json").write_text(json.dumps(result["fitted"], indent=2))
    mdir = run_dir / "manifests"
    mdir.mkdir()
    curve_rows = []
    for c in result["cells"]:
        with gzip.open(mdir / f"seed{c['seed']}_gap{c['gap']}_{c['split']}.json.gz", "wt") as f:
            json.dump(c["manifest"], f)
        for m, rows in c["curves"].items():
            for r in rows:
                curve_rows.append({"seed": c["seed"], "gap": c["gap"], "split": c["split"],
                                   "method": m, **r})
    _write_csv(run_dir / "curves.csv", curve_rows)
    _write_csv(run_dir / "test_selected.csv", result["test_table"])
    writer.save_metrics({
        "kind": "SMOKE RUN - not the pilot" if smoke else "Phase 3 pilot",
        "aggregate": result["aggregate"],
        "comparisons": result["comparisons"],
        "selected_threshold_index": result["selected_threshold_index"],
    })
    writer.save_timing({"total_seconds": elapsed, "workers": args.workers})
    methods = list(cfg["methods"])
    plot_tradeoff(result, methods, run_dir / "figures")
    plot_robustness(result, methods, run_dir / "figures")
    meta.metrics = {}  # the full summary lives in metrics.json
    meta.mark_completed()
    writer.save_metadata()

    print(f"\n{'SMOKE RUN' if smoke else 'PILOT'} finished in {elapsed:.0f}s -> {run_dir}")
    print("Hypervolume comparisons (test; fraction of seed x gap cells where a > b):")
    for comp in result["comparisons"]:
        print(f"  {comp['a']:>16} vs {comp['b']:<16} a_better={comp['a_better_frac']:.2f} "
              f"dHV={comp['hv_diff_mean']:+.3f}+-{comp['hv_diff_std']:.3f}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
