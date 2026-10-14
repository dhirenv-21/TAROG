"""EGTR scene-graph predictor adapter (Phase 6; research/taorg_egtr_integration.md).

EGTR (Im et al., CVPR 2024; https://github.com/naver-ai/egtr, Apache-2.0) is a one-stage
Deformable-DETR scene-graph generator trained on Visual Genome (150 object, 50 predicate classes).
This adapter runs the OFFICIAL model code from a pinned clone (env ``TAORG_EGTR_SRC``) with the
official VG checkpoint, and reproduces the official single-prediction post-processing of
``train_egtr.evaluate_batch`` (SGDet, ``eval_single_preds``):

    obj_scores, classes = max softmax over the 150 object classes, per query
    pair score         = obj_score[s] * obj_score[o]   (s != o)
    rel scores         = clamp(pred_rel) * clamp(pred_connectivity)       (per pair, 50 predicates)
    ranking            = max_p rel[s, o, p] * pair score; top-k pairs (k = 100)
    boxes              = cxcywh (normalised) rescaled to original image pixels (xyxy)

Deviations from the source, all declared: the timm backbone is built with ``pretrained=False``
(every weight is then loaded from the checkpoint with ``strict=True``); the feature extractor is
constructed locally with the parameters of ``SenseTime/deformable-detr`` instead of downloading
them; the deformable-attention CUDA kernel is not compiled (EGTR's own PyTorch fallback is used).

Confidence semantics: object scores are softmax probabilities over 150 classes + no-object; relation
scores are independent sigmoid outputs multiplied by a sigmoid connectivity score. Neither is
calibrated for video, and the product is a ranking score, not a probability.
Runs in the EGTR environment (``D:/TAORG_runs/egtr/venv``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EGTR_COMMIT = "7f87450f32758ed8583948847a8186f2ee8b21e3"
CKPT_SHA256 = "8926640facecd2784160c221c048dd4209df7d623ceb3c9eda289a11cb1f84d5"
DEFAULT_SRC = Path(os.environ.get("TAORG_EGTR_SRC", "D:/TAORG_runs/egtr/src"))
DEFAULT_ARTIFACT = Path(os.environ.get("TAORG_EGTR_ARTIFACT", "D:/TAORG_runs/egtr/ckpt"))
FEATURE_EXTRACTOR = {"format": "coco_detection", "do_resize": True, "size": 800, "max_size": 1333,
                     "do_normalize": True, "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
TOPK = 100
VOCAB = Path(__file__).resolve().parents[3] / "configs" / "egtr" / "vg_vocab.json"


PERSON_CLASSES = ("boy", "child", "girl", "guy", "kid", "lady", "man", "men", "people", "person", "player",
                  "skier", "woman")
# VidOR person-person predicate <- VG predicate column(s) (fixed before any comparison; max over columns)
PREDICATE_MAP = {"behind": ("behind",), "in_front_of": ("in front of",), "next_to": ("near",),
                 "watch": ("watching", "looking at")}
MAPPED = tuple(sorted(PREDICATE_MAP))
MIN_PERSON_SCORE = 0.1
MAX_PERSONS = 30


@dataclass
class PersonRelations:
    """Person queries of one frame and their ordered-pair scores for the mapped predicates."""

    boxes: np.ndarray        # (N, 4) xyxy pixels
    obj_scores: np.ndarray   # (N,)
    classes: np.ndarray      # (N,) VG class index
    scores: np.ndarray       # (N, N, len(MAPPED)) triplet score rel[p] * obj_s * obj_o; diagonal 0


@dataclass
class SceneGraph:
    """Post-processed EGTR output for one image (all arrays numpy)."""

    boxes: np.ndarray        # (Q, 4) xyxy pixels, original image coordinates
    classes: np.ndarray      # (Q,) object class index 0..149
    obj_scores: np.ndarray   # (Q,) softmax probability of that class
    pairs: np.ndarray        # (K, 2) query indices (subject, object), ranked
    rel_scores: np.ndarray   # (K, 50) predicate scores of each ranked pair
    pair_scores: np.ndarray  # (K,) ranking score = max_p rel * obj_s * obj_o


def find_artifact(root: Path) -> tuple[Path, Path]:
    """(artifact dir containing config.json, checkpoint path) inside the extracted archive."""
    cfgs = sorted(Path(root).rglob("config.json"))
    if not cfgs:
        raise FileNotFoundError(f"no EGTR config.json under {root}; extract egtr_vg.tar.gz there")
    art = cfgs[-1].parent
    ckpts = sorted(art.glob("checkpoints/epoch=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {art}/checkpoints")
    return art, ckpts[-1]


class EGTRPredictor:
    """Official EGTR model + official SGDet single-prediction post-processing."""

    def __init__(self, src: Path = DEFAULT_SRC, artifact_root: Path = DEFAULT_ARTIFACT, device: str = "cuda",
                 verify_checksum: bool = True):
        import hashlib

        import torch
        if not (Path(src) / "model" / "egtr.py").exists():
            raise FileNotFoundError(f"EGTR source not found at {src} (set TAORG_EGTR_SRC)")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import timm

        import model.deformable_detr as dd
        from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
        from model.egtr import DetrForSceneGraphGeneration
        dd.create_model = lambda *a, **k: timm.create_model(*a, **{**k, "pretrained": False})
        # No compiled kernel: call EGTR's own PyTorch fallback directly (identical computation, without
        # the per-call exception + print of the source's try/except path).
        dd.MultiScaleDeformableAttentionFunction.apply = staticmethod(
            lambda value, shapes, start, loc, weights, step: dd.ms_deform_attn_core_pytorch(value, shapes, loc, weights))
        art, ckpt = find_artifact(artifact_root)
        if verify_checksum:
            h = hashlib.sha256()
            with open(ckpt, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            if h.hexdigest() != CKPT_SHA256:
                raise RuntimeError(f"EGTR checkpoint checksum mismatch: {ckpt}")
        self.config = DeformableDetrConfig.from_pretrained(str(art))
        self.config.logit_adjustment = False
        self.config.logit_adj_tau = 0.3
        self.model = DetrForSceneGraphGeneration(self.config)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[6:]: v for k, v in sd.items()}                      # strip "model."
        self.model.load_state_dict(sd, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.fe = DeformableDetrFeatureExtractor(**FEATURE_EXTRACTOR)
        import json
        vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
        self.class_names = vocab["object_classes"]            # index c <-> VG category id c + 1
        self.predicate_names = vocab["predicates"]            # column p <-> rel_categories[p + 1]
        self.num_labels = len(self.class_names)
        if self.config.num_labels != self.num_labels or self.config.num_rel_labels != len(self.predicate_names):
            raise RuntimeError("EGTR config does not match the VG vocabulary (150 classes / 50 predicates)")
        self.torch = torch

    def raw(self, images_rgb: list[np.ndarray]):
        """Model outputs for a batch of RGB uint8 images (official preprocessing + padding)."""
        torch = self.torch
        from PIL import Image
        enc = self.fe(images=[Image.fromarray(im) for im in images_rgb], return_tensors="pt")
        pix = enc["pixel_values"].to(self.device)
        mask = enc["pixel_mask"].to(self.device)
        with torch.inference_mode():
            return self.model(pixel_values=pix, pixel_mask=mask, output_attentions=False,
                              output_attention_states=True, output_hidden_states=True)

    def postprocess(self, outputs, j: int, orig_hw: tuple[int, int], topk: int = TOPK) -> SceneGraph:
        """Official single-prediction SGDet post-processing for batch item j."""
        torch = self.torch
        logits = outputs["logits"][j]
        obj_scores, classes = torch.max(logits.softmax(-1)[:, :self.num_labels], -1)
        pair = torch.outer(obj_scores, obj_scores)
        q = logits.size(0)
        pair[torch.arange(q), torch.arange(q)] = 0.0
        rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)
        if "pred_connectivity" in outputs:
            rel = torch.mul(rel, torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0))
        trip = torch.mul(rel.max(-1)[0], pair).cpu().numpy()
        order = np.column_stack(np.unravel_index(np.argsort(-trip.ravel()), trip.shape))[:topk]
        relc = rel.cpu().numpy()
        h, w = orig_hw
        b = outputs["pred_boxes"][j].float().cpu().numpy()
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        return SceneGraph(xyxy, classes.cpu().numpy(), obj_scores.cpu().numpy(), order,
                          relc[order[:, 0], order[:, 1]], trip[order[:, 0], order[:, 1]])

    def person_relations(self, outputs, j: int, orig_hw: tuple[int, int]) -> PersonRelations:
        """Every person-class query (score >= MIN_PERSON_SCORE, top MAX_PERSONS) and the full ordered-pair
        score matrix for the mapped predicates. Uses the same quantities as the official ranking
        (connectivity-weighted relation score times both object scores), without the top-100 cut, so
        pairs outside the official top-k still count as evaluated."""
        torch = self.torch
        logits = outputs["logits"][j]
        obj_scores, classes = torch.max(logits.softmax(-1)[:, :self.num_labels], -1)
        pidx = [self.class_names.index(c) for c in PERSON_CLASSES]
        is_p = torch.isin(classes, torch.tensor(pidx, device=classes.device)) & (obj_scores >= MIN_PERSON_SCORE)
        q = torch.nonzero(is_p).flatten()
        q = q[torch.argsort(-obj_scores[q])][:MAX_PERSONS]
        rel = torch.clamp(outputs["pred_rel"][j], 0.0, 1.0)
        if "pred_connectivity" in outputs:
            rel = torch.mul(rel, torch.clamp(outputs["pred_connectivity"][j], 0.0, 1.0))
        rel = rel[q][:, q]                                            # (N, N, 50)
        cols = [[self.predicate_names.index(v) for v in PREDICATE_MAP[m]] for m in MAPPED]
        mapped = torch.stack([rel[:, :, c].max(-1)[0] for c in cols], -1)
        pair = torch.outer(obj_scores[q], obj_scores[q])
        n = len(q)
        pair[torch.arange(n), torch.arange(n)] = 0.0
        sc = (mapped * pair[:, :, None]).float().cpu().numpy()
        h, w = orig_hw
        b = outputs["pred_boxes"][j][q].float().cpu().numpy().reshape(-1, 4)
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        return PersonRelations(xyxy * np.array([w, h, w, h], dtype=np.float32), obj_scores[q].cpu().numpy(),
                               classes[q].cpu().numpy(), sc)

    def predict(self, images_rgb: list[np.ndarray], topk: int = TOPK) -> list[SceneGraph]:
        """Scene graphs for a batch of RGB images."""
        out = self.raw(images_rgb)
        return [self.postprocess(out, j, im.shape[:2], topk) for j, im in enumerate(images_rgb)]
