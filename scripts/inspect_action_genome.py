"""Inspect real Action Genome annotation files and check loader assumptions.

Run this BEFORE the pilot (Rules §7).  It prints a JSON report and writes it to
outputs/dataset_reports/action_genome_report.json.  Review in particular:
    unknown_object_classes / unknown_relationships  (must be empty)
    frames_with_duplicate_classes                   (track-per-class assumption)
    object_dict_keys, visible_counts, split_counts  (field assumptions)
    annotated_frame_step_median                     (keyframe sparsity)

Usage: python scripts/inspect_action_genome.py --root data/action_genome
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taorg.data.action_genome import ActionGenomeConfig, inspect_action_genome, load_action_genome


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--annotation-dir", default="annotations")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/dataset_reports/action_genome_report.json"))
    args = ap.parse_args()
    cfg = ActionGenomeConfig(root=args.root, annotation_dir=args.annotation_dir)
    report = inspect_action_genome(cfg)
    for split in ("train", "test"):
        _, _, stats = load_action_genome(cfg, split)
        report[f"load_stats_{split}"] = stats.to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
