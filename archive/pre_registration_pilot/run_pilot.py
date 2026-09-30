"""Phase 3 Go/No-Go Pilot Runner.

Sweeps all Phase 1 baselines across controlled corruption gaps and
multiple seeds to test hypothesis H1: visibility-aware gating produces
a measurable persistence-termination Pareto improvement.

Output: JSON results file + CSV summary in outputs/pilot/

This is a PILOT — NOT final paper evidence.
Results are for internal go/no-go assessment only.
"""

from __future__ import annotations

import json
import csv
import logging
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from taorg.metrics.persistence import compute_persistence_f1
from taorg.metrics.churn import compute_churn_rate
from taorg.metrics.termination import (
    compute_false_persistence,
    compute_termination_delay,
)
from taorg.noise.occlusion import OcclusionConfig, inject_occlusion
from taorg.schemas.edges import EdgeKey, FrameGraph
from taorg.temporal import METHODS
from taorg.utils.git import get_git_commit, get_git_dirty, get_git_branch
from taorg.utils.seed import set_seed

# Import pilot fixtures
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from fixtures.pilot_fixtures import make_multi_edge_video, make_long_single_edge_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

GAP_LENGTHS = [1, 2, 3, 5, 10, 20]
SEEDS = [42, 123, 456, 789, 2024]
METHOD_NAMES = ["no_memory", "fixed_window", "ema", "gated_ema", "hmm"]
SCORE_THRESHOLD = 0.1

SCENARIOS = {
    "multi_edge_100": {
        "factory": "make_multi_edge_video",
        "kwargs": {"num_frames": 100},
    },
    "long_single_200": {
        "factory": "make_long_single_edge_video",
        "kwargs": {"num_frames": 200, "edge_end_frame": 149},
    },
    "long_single_persistent": {
        "factory": "make_long_single_edge_video",
        "kwargs": {"num_frames": 200, "edge_end_frame": None},
    },
}

FACTORIES = {
    "make_multi_edge_video": make_multi_edge_video,
    "make_long_single_edge_video": make_long_single_edge_video,
}


def _run_method_on_corrupted(
    method_name: str,
    gt_frames: list[FrameGraph],
    corrupted_frames: list[FrameGraph],
    video_id: str,
) -> list[FrameGraph]:
    """Instantiate and run a temporal method on corrupted input."""
    cls = METHODS[method_name]
    try:
        model = cls(score_threshold=SCORE_THRESHOLD)
    except TypeError:
        model = cls()

    model.reset_video(video_id)
    pred_frames: list[FrameGraph] = []
    for frame in corrupted_frames:
        out = model.step(frame)
        pred_frames.append(out)
    model.finalize_video()
    return pred_frames


def run_pilot() -> dict[str, Any]:
    """Execute the full Phase 3 pilot sweep.

    Returns:
        Complete results dictionary.
    """
    start_time = time.time()

    provenance = {
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "git_branch": get_git_branch(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gap_lengths": GAP_LENGTHS,
        "seeds": SEEDS,
        "methods": METHOD_NAMES,
        "score_threshold": SCORE_THRESHOLD,
    }

    all_results: list[dict[str, Any]] = []
    total_runs = (
        len(SCENARIOS) * len(GAP_LENGTHS) * len(SEEDS) * len(METHOD_NAMES)
    )
    run_count = 0

    for scenario_name, scenario_cfg in SCENARIOS.items():
        factory = FACTORIES[scenario_cfg["factory"]]
        gt_frames = factory(**scenario_cfg["kwargs"])
        video_id = gt_frames[0].video_id

        for gap_length in GAP_LENGTHS:
            for seed in SEEDS:
                set_seed(seed)

                # Apply corruption
                config = OcclusionConfig(
                    gap_lengths=[gap_length],
                    num_gaps_per_edge=1,
                    min_pre_gap_frames=2,
                    min_post_gap_frames=2,
                )
                corrupted, manifest = inject_occlusion(
                    gt_frames, seed=seed, config=config
                )

                for method_name in METHOD_NAMES:
                    run_count += 1
                    if run_count % 50 == 0:
                        logger.info(
                            "Progress: %d/%d runs (%.0f%%)",
                            run_count, total_runs,
                            100.0 * run_count / total_runs,
                        )

                    # Run method
                    pred_frames = _run_method_on_corrupted(
                        method_name, gt_frames, corrupted, video_id,
                    )

                    # Compute metrics
                    persistence = compute_persistence_f1(
                        gt_frames, pred_frames, manifest
                    )
                    churn = compute_churn_rate(pred_frames, gt_frames)
                    termination = compute_termination_delay(
                        gt_frames, pred_frames
                    )
                    false_pers = compute_false_persistence(
                        gt_frames, pred_frames
                    )

                    result = {
                        "scenario": scenario_name,
                        "gap_length": gap_length,
                        "seed": seed,
                        "method": method_name,
                        "persistence_tp": persistence.tp,
                        "persistence_fp": persistence.fp,
                        "persistence_fn": persistence.fn,
                        "persistence_precision": persistence.precision,
                        "persistence_recall": persistence.recall,
                        "persistence_f1": persistence.f1,
                        "churn_mean": churn.mean_churn,
                        "churn_total_flips": churn.total_flips,
                        "term_delay_mean": termination.mean_delay,
                        "term_delay_median": termination.median_delay,
                        "term_num_censored": termination.num_censored,
                        "term_num_evaluated": termination.num_evaluated,
                        "false_pers_mean": false_pers.mean_rate,
                        "false_pers_num_evaluated": false_pers.num_evaluated,
                        "num_corruption_entries": len(manifest.entries),
                    }
                    all_results.append(result)

    elapsed = time.time() - start_time

    return {
        "provenance": provenance,
        "elapsed_seconds": elapsed,
        "total_runs": total_runs,
        "results": all_results,
    }


def aggregate_results(results: list[dict]) -> dict[str, Any]:
    """Compute per-method, per-gap-length aggregates across seeds.

    Returns nested dict: method -> gap_length -> {metric: {mean, std, values}}
    """
    import statistics

    # Group by (method, gap_length)
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        key = (r["method"], r["gap_length"])
        groups.setdefault(key, []).append(r)

    aggregated: dict[str, dict[int, dict]] = {}
    for (method, gap_length), runs in sorted(groups.items()):
        if method not in aggregated:
            aggregated[method] = {}

        metrics_to_agg = [
            "persistence_f1", "persistence_recall", "persistence_precision",
            "churn_mean", "term_delay_mean", "false_pers_mean",
        ]
        agg: dict[str, Any] = {"n_runs": len(runs)}
        for metric in metrics_to_agg:
            values = [r[metric] for r in runs]
            agg[f"{metric}_mean"] = statistics.mean(values)
            agg[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            agg[f"{metric}_values"] = values

        aggregated[method][gap_length] = agg

    return aggregated


def compute_h1_assessment(aggregated: dict) -> dict[str, Any]:
    """Assess hypothesis H1: does visibility gating help?

    Compares gated_ema vs. ema across all gap lengths.
    H1 passes if gated_ema has STRICTLY better persistence_f1
    and equal-or-better false_pers_mean for the majority of gap lengths.
    """
    if "gated_ema" not in aggregated or "ema" not in aggregated:
        return {"h1_decision": "INCONCLUSIVE", "reason": "Missing methods"}

    gap_lengths = sorted(
        set(aggregated["ema"].keys()) & set(aggregated["gated_ema"].keys())
    )

    comparisons = []
    persistence_wins = 0
    persistence_ties = 0
    persistence_losses = 0
    false_pers_wins = 0
    false_pers_ties = 0
    false_pers_losses = 0

    for gl in gap_lengths:
        ema_f1 = aggregated["ema"][gl]["persistence_f1_mean"]
        gema_f1 = aggregated["gated_ema"][gl]["persistence_f1_mean"]
        ema_fp = aggregated["ema"][gl]["false_pers_mean_mean"]
        gema_fp = aggregated["gated_ema"][gl]["false_pers_mean_mean"]
        ema_churn = aggregated["ema"][gl]["churn_mean_mean"]
        gema_churn = aggregated["gated_ema"][gl]["churn_mean_mean"]

        # F1 comparison (higher is better)
        f1_diff = gema_f1 - ema_f1
        if abs(f1_diff) < 0.01:
            persistence_ties += 1
            f1_verdict = "TIE"
        elif f1_diff > 0:
            persistence_wins += 1
            f1_verdict = "GATED_EMA_WINS"
        else:
            persistence_losses += 1
            f1_verdict = "EMA_WINS"

        # False persistence (lower is better)
        fp_diff = ema_fp - gema_fp
        if abs(fp_diff) < 0.01:
            false_pers_ties += 1
            fp_verdict = "TIE"
        elif fp_diff > 0:
            false_pers_wins += 1
            fp_verdict = "GATED_EMA_WINS"
        else:
            false_pers_losses += 1
            fp_verdict = "EMA_WINS"

        comparisons.append({
            "gap_length": gl,
            "ema_f1": round(ema_f1, 4),
            "gated_ema_f1": round(gema_f1, 4),
            "f1_diff": round(f1_diff, 4),
            "f1_verdict": f1_verdict,
            "ema_false_pers": round(ema_fp, 4),
            "gated_ema_false_pers": round(gema_fp, 4),
            "fp_verdict": fp_verdict,
            "ema_churn": round(ema_churn, 4),
            "gated_ema_churn": round(gema_churn, 4),
        })

    # H1 decision: gated_ema must win persistence F1 in majority of gap
    # lengths AND not be significantly worse on false persistence
    total = len(gap_lengths)
    h1_persistence_pass = persistence_wins > total / 2
    h1_false_pers_pass = false_pers_losses <= total / 3

    if h1_persistence_pass and h1_false_pers_pass:
        decision = "GO"
        reason = (
            f"GatedEMA wins persistence F1 in {persistence_wins}/{total} "
            f"gap lengths with acceptable false persistence "
            f"({false_pers_losses}/{total} losses)."
        )
    elif h1_persistence_pass and not h1_false_pers_pass:
        decision = "MODIFY"
        reason = (
            f"GatedEMA wins persistence F1 in {persistence_wins}/{total} "
            f"gap lengths but has worse false persistence in "
            f"{false_pers_losses}/{total} gap lengths. "
            f"Visibility gating helps persistence but hurts termination."
        )
    else:
        decision = "NO-GO"
        reason = (
            f"GatedEMA wins persistence F1 in only {persistence_wins}/{total} "
            f"gap lengths. Visibility gating does not provide consistent "
            f"improvement."
        )

    return {
        "h1_decision": decision,
        "reason": reason,
        "persistence_wins": persistence_wins,
        "persistence_ties": persistence_ties,
        "persistence_losses": persistence_losses,
        "false_pers_wins": false_pers_wins,
        "false_pers_ties": false_pers_ties,
        "false_pers_losses": false_pers_losses,
        "comparisons": comparisons,
    }


def save_results(
    pilot_data: dict,
    aggregated: dict,
    h1_assessment: dict,
    output_dir: Path,
) -> None:
    """Save all pilot outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results JSON
    raw_path = output_dir / "pilot_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(pilot_data, f, indent=2, default=str)
    logger.info("Raw results: %s", raw_path)

    # H1 assessment JSON
    h1_path = output_dir / "h1_assessment.json"
    with open(h1_path, "w") as f:
        json.dump(h1_assessment, f, indent=2, default=str)
    logger.info("H1 assessment: %s", h1_path)

    # Summary CSV
    csv_path = output_dir / "pilot_summary.csv"
    fieldnames = [
        "method", "gap_length",
        "persistence_f1_mean", "persistence_f1_std",
        "persistence_recall_mean", "persistence_recall_std",
        "churn_mean_mean", "churn_mean_std",
        "term_delay_mean_mean", "term_delay_mean_std",
        "false_pers_mean_mean", "false_pers_mean_std",
        "n_runs",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in sorted(aggregated.keys()):
            for gl in sorted(aggregated[method].keys()):
                row = {"method": method, "gap_length": gl}
                for field in fieldnames[2:]:
                    row[field] = round(aggregated[method][gl].get(field, 0), 6)
                writer.writerow(row)
    logger.info("Summary CSV: %s", csv_path)

    # Per-method comparison table (human-readable)
    table_path = output_dir / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PHASE 3 PILOT — PERSISTENCE F1 (mean ± std across seeds)\n")
        f.write("=" * 100 + "\n\n")

        header = f"{'Gap':>4} | "
        header += " | ".join(f"{m:>16}" for m in sorted(aggregated.keys()))
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        all_gaps = sorted(
            set(gl for m in aggregated.values() for gl in m.keys())
        )
        for gl in all_gaps:
            row = f"{gl:>4} | "
            cells = []
            for method in sorted(aggregated.keys()):
                if gl in aggregated[method]:
                    mean = aggregated[method][gl]["persistence_f1_mean"]
                    std = aggregated[method][gl]["persistence_f1_std"]
                    cells.append(f"{mean:.4f}±{std:.4f}")
                else:
                    cells.append("N/A")
            row += " | ".join(f"{c:>16}" for c in cells)
            f.write(row + "\n")

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("FALSE PERSISTENCE RATE (mean ± std — lower is better)\n")
        f.write("=" * 100 + "\n\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for gl in all_gaps:
            row = f"{gl:>4} | "
            cells = []
            for method in sorted(aggregated.keys()):
                if gl in aggregated[method]:
                    mean = aggregated[method][gl]["false_pers_mean_mean"]
                    std = aggregated[method][gl]["false_pers_mean_std"]
                    cells.append(f"{mean:.4f}±{std:.4f}")
                else:
                    cells.append("N/A")
            row += " | ".join(f"{c:>16}" for c in cells)
            f.write(row + "\n")

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("CHURN RATE (mean ± std — lower is better)\n")
        f.write("=" * 100 + "\n\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for gl in all_gaps:
            row = f"{gl:>4} | "
            cells = []
            for method in sorted(aggregated.keys()):
                if gl in aggregated[method]:
                    mean = aggregated[method][gl]["churn_mean_mean"]
                    std = aggregated[method][gl]["churn_mean_std"]
                    cells.append(f"{mean:.4f}±{std:.4f}")
                else:
                    cells.append("N/A")
            row += " | ".join(f"{c:>16}" for c in cells)
            f.write(row + "\n")

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("H1 ASSESSMENT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Decision: {h1_assessment['h1_decision']}\n")
        f.write(f"Reason: {h1_assessment['reason']}\n\n")
        for comp in h1_assessment.get("comparisons", []):
            f.write(
                f"  Gap {comp['gap_length']:>2}: "
                f"EMA F1={comp['ema_f1']:.4f} vs "
                f"GatedEMA F1={comp['gated_ema_f1']:.4f} "
                f"({comp['f1_verdict']}), "
                f"FP: EMA={comp['ema_false_pers']:.4f} vs "
                f"GatedEMA={comp['gated_ema_false_pers']:.4f} "
                f"({comp['fp_verdict']})\n"
            )

    logger.info("Comparison table: %s", table_path)


def main() -> None:
    output_dir = Path("outputs/pilot")

    logger.info("=" * 60)
    logger.info("PHASE 3 GO/NO-GO PILOT")
    logger.info("=" * 60)
    logger.info("Methods: %s", METHOD_NAMES)
    logger.info("Gap lengths: %s", GAP_LENGTHS)
    logger.info("Seeds: %s", SEEDS)
    logger.info("Scenarios: %s", list(SCENARIOS.keys()))

    pilot_data = run_pilot()
    logger.info(
        "Pilot complete: %d runs in %.1f seconds",
        pilot_data["total_runs"],
        pilot_data["elapsed_seconds"],
    )

    aggregated = aggregate_results(pilot_data["results"])
    h1_assessment = compute_h1_assessment(aggregated)

    save_results(pilot_data, aggregated, h1_assessment, output_dir)

    logger.info("=" * 60)
    logger.info("H1 DECISION: %s", h1_assessment["h1_decision"])
    logger.info("Reason: %s", h1_assessment["reason"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
