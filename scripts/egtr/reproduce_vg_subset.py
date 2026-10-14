"""EGTR source-reproduction check on a seeded VG test subset (Phase 6).

Run in the EGTR venv:
    D:/TAORG_runs/egtr/venv/Scripts/python scripts/egtr/reproduce_vg_subset.py

1. OFFICIAL path: EGTR's VGDataset + collate_fn + evaluate() (evaluate_egtr.py) on the subset, with the
   adapter-built model, giving SGDet R@K / mR@K / AP50 for comparison with the authors' full-test JSON
   (R@20/50/100 = 23.5/30.2/34.4; mR@50 7.9; AP50 30.8 on 26,446 images).
2. EQUALITY: the official per-image single-prediction entries (captured from the official evaluator) are
   compared with ``EGTRPredictor.predict`` on the same images (batch size 1).

Declared deviation: lib.fpn.box_intersections_cpu.bbox (an uncompiled Cython module) is replaced by a
NumPy port with the same +1-pixel convention; it is used only by the official evaluator.
Output: results/egtr/reproduction.json
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

from taorg.adapters.egtr import DEFAULT_SRC, FEATURE_EXTRACTOR, EGTRPredictor
from taorg.utils.seed import set_seed

SUBSET = Path("D:/TAORG_runs/egtr/vg_subset")


def bbox_overlaps(boxes, query_boxes):
    """NumPy port of EGTR's lib/fpn/box_intersections_cpu/bbox.pyx (float64, +1 pixel convention)."""
    b = np.ascontiguousarray(boxes, dtype=np.float64)
    q = np.ascontiguousarray(query_boxes, dtype=np.float64)
    iw = np.minimum(b[:, None, 2], q[None, :, 2]) - np.maximum(b[:, None, 0], q[None, :, 0]) + 1
    ih = np.minimum(b[:, None, 3], q[None, :, 3]) - np.maximum(b[:, None, 1], q[None, :, 1]) + 1
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    area_q = (q[:, 2] - q[:, 0] + 1) * (q[:, 3] - q[:, 1] + 1)
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    ua = area_b[:, None] + area_q[None, :] - inter
    return np.where(inter > 0, inter / ua, 0.0)


def main() -> None:
    set_seed(20261013)
    shim = types.ModuleType("lib.fpn.box_intersections_cpu.bbox")
    shim.bbox_overlaps = bbox_overlaps
    sys.modules["lib.fpn.box_intersections_cpu.bbox"] = shim
    pred = EGTRPredictor()                                         # inserts the EGTR source on sys.path
    import torch
    from torch.utils.data import DataLoader

    import evaluate_egtr as E
    import lib.evaluation.sg_eval as SG
    from data.visual_genome import VGDataset
    from model.deformable_detr import DeformableDetrFeatureExtractor
    from train_egtr import collate_fn
    fe = DeformableDetrFeatureExtractor(**FEATURE_EXTRACTOR)
    ds = VGDataset(data_folder=str(SUBSET), feature_extractor=fe, split="test", num_object_queries=200)
    dl = DataLoader(ds, collate_fn=lambda x: collate_fn(x, fe), batch_size=1, num_workers=0)
    captured = []
    orig = SG.BasicSceneGraphEvaluator.evaluate_scene_graph_entry

    def capture(self, gt_entry, pred_entry, *a, **k):
        if getattr(self, "_taorg_main", False):
            captured.append({k2: np.array(v) for k2, v in pred_entry.items()})
        return orig(self, gt_entry, pred_entry, *a, **k)
    SG.BasicSceneGraphEvaluator.evaluate_scene_graph_entry = capture
    single = SG.BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    for ev in single.values():
        ev._taorg_main = True
    coco = E.CocoEvaluator(ds.coco, ["bbox"])
    metrics = E.evaluate(pred.model, dl, max(range(150)) + 1, None, single, None, coco, fe)
    SG.BasicSceneGraphEvaluator.evaluate_scene_graph_entry = orig
    # adapter equality, image by image (same order as the official loader)
    from PIL import Image
    max_err = {"boxes": 0.0, "obj_scores": 0.0, "rel_scores": 0.0}
    same_cls = same_pairs = 0
    for idx, entry in enumerate(captured):
        info = ds.coco.loadImgs(ds.ids[idx])[0]
        rgb = np.asarray(Image.open(SUBSET / "images" / info["file_name"]).convert("RGB"))
        g = pred.predict([rgb])[0]
        max_err["boxes"] = max(max_err["boxes"], float(np.abs(g.boxes - entry["pred_boxes"]).max()))
        max_err["obj_scores"] = max(max_err["obj_scores"], float(np.abs(g.obj_scores - entry["obj_scores"]).max()))
        same_cls += bool((g.classes == entry["pred_classes"]).all())
        same_pairs += bool((g.pairs == entry["pred_rel_inds"]).all())
        max_err["rel_scores"] = max(max_err["rel_scores"], float(np.abs(g.rel_scores - entry["rel_scores"]).max()))
    reference = json.loads(next(Path("D:/TAORG_runs/egtr/ckpt").rglob("*__test__26446__*.json")).read_text())
    out = {"n_images": len(ds), "subset_seed": 20261013, "official_metrics_on_subset": metrics,
           "authors_full_test_reference": {k: v for k, v in reference.items() if k != "eval_arg"},
           "adapter_vs_official": {"images": len(captured), "identical_classes": same_cls,
                                   "identical_ranked_pairs": same_pairs, "max_abs_error": max_err},
           "torch": torch.__version__, "deviation": "Cython bbox_overlaps replaced by NumPy port (evaluator only); "
           "backbone built with pretrained=False then strict checkpoint load; local feature-extractor parameters"}
    Path("results/egtr").mkdir(parents=True, exist_ok=True)
    Path("results/egtr/reproduction.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
