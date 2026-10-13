"""VidOR annotation adapter for the downstream study (person–person relationships).

ANNOTATION ACCESS. ``load_video`` reads ground truth (person boxes, relation intervals). In the
downstream study it is used (i) by the sampling rule (annotation-only), (ii) by the ORACLE relation
predictor that constructs the observation stream, and (iii) for scoring. Temporal methods never
receive anything from this module directly (research/taorg_downstream_preregistration.md §4).

Frame indices are 0-based; relation intervals are [begin_fid, end_fid) as in the VidOR files.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PERSON_CATEGORIES = frozenset({"adult", "child", "baby"})
SAMPLE_RULE = {"min_persons": 3, "max_duration_s": 60.0, "min_pp_relations": 1}
SAMPLE_SEEDS = {"training": (30, 20261013), "validation": (60, 20261014)}


@dataclass(frozen=True)
class VidorVideo:
    """Person-only view of one VidOR annotation file."""

    video_id: str
    video_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    boxes: list[dict[int, tuple[float, float, float, float]]]   # frame -> {person tid: xyxy}
    relations: list[tuple[int, int, str, int, int]]             # (subj tid, obj tid, predicate, begin, end_excl)


def load_video(path: Path) -> VidorVideo:
    """Parse one annotation JSON, keeping persons and person–person relations only."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cat = {o["tid"]: o["category"] for o in d["subject/objects"]}
    pers = {t for t, c in cat.items() if c in PERSON_CATEGORIES}
    boxes = []
    for fr in d["trajectories"]:
        boxes.append({b["tid"]: (float(b["bbox"]["xmin"]), float(b["bbox"]["ymin"]), float(b["bbox"]["xmax"]),
                                 float(b["bbox"]["ymax"])) for b in fr if b["tid"] in pers})
    rels = [(r["subject_tid"], r["object_tid"], r["predicate"], int(r["begin_fid"]), int(r["end_fid"]))
            for r in d["relation_instances"] if r["subject_tid"] in pers and r["object_tid"] in pers]
    return VidorVideo(str(d["video_id"]), d["video_path"], float(d["fps"]), int(d["width"]), int(d["height"]),
                      int(d["frame_count"]), boxes, rels)


def eligible(v: VidorVideo, n_persons: int) -> bool:
    """Annotation-only sampling rule (prereg §2)."""
    return (n_persons >= SAMPLE_RULE["min_persons"] and len(v.relations) >= SAMPLE_RULE["min_pp_relations"]
            and v.frame_count / v.fps <= SAMPLE_RULE["max_duration_s"])


def sample(ann_root: Path, split: str) -> list[str]:
    """Seeded sample of eligible annotation files (relative paths, sorted pool order)."""
    k, seed = SAMPLE_SEEDS[split]
    pool = []
    for fn in sorted(Path(ann_root, split).glob("*/*.json")):
        d = json.loads(fn.read_text(encoding="utf-8"))
        cat = {o["tid"]: o["category"] for o in d["subject/objects"]}
        n = sum(c in PERSON_CATEGORIES for c in cat.values())
        if n < SAMPLE_RULE["min_persons"] or d["frame_count"] / d["fps"] > SAMPLE_RULE["max_duration_s"]:
            continue
        if not any(cat[r["subject_tid"]] in PERSON_CATEGORIES and cat[r["object_tid"]] in PERSON_CATEGORIES
                   for r in d["relation_instances"]):
            continue
        pool.append(d["video_path"])
    return random.Random(seed).sample(pool, k)


def active_relations(v: VidorVideo, f: int) -> set[tuple[int, int, str]]:
    """Annotated person–person relations active at frame f."""
    return {(s, o, p) for s, o, p, a, b in v.relations if a <= f < b}


def gt_boxes_array(v: VidorVideo, f: int) -> tuple[list[int], np.ndarray]:
    """Person tids and (N, 4) xyxy boxes annotated at frame f."""
    tids = sorted(v.boxes[f])
    return tids, np.asarray([v.boxes[f][t] for t in tids], float).reshape(-1, 4)
