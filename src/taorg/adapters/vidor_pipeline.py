"""Perception pipeline for the downstream study (INFERENCE ONLY; prereg §3).

video (mp4) -> JPEG frames -> COCO person detections -> OSNet embeddings -> V3 BoT-SORT
            -> missing records + cues -> frozen takeover flag

Never reads annotations (enforced by an AST test). Frames are 1-based internally.
Heavy imports (torch, torchvision, boxmot) are lazy; the module must run in the BoxMOT venv.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from taorg.adapters.mot17 import SeqInfo

FLAG_ARTIFACT_SHA256 = "98d7e46a6ea2cc44a4e989165a8783618ae16683e93074b3644a085995427f5a"
FLAG_ENTRY = ("V3", "T+A")
DETECTOR_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"
PERSON_LABEL = 1
MIN_SCORE = 0.1
JPEG_QUALITY = 95


def extract_frames(video: Path, out_dir: Path) -> tuple[int, tuple[int, int], float]:
    """Decode every frame to out_dir/{f:06d}.jpg (f = 1..n). Returns (n, (h, w), container fps)."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n, shape = 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        shape = fr.shape[:2]
        p = out_dir / f"{n:06d}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), fr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    cap.release()
    return n, shape, fps


def load_detector(device: str = "cuda"):
    """torchvision Faster R-CNN v2 COCO_V1 (weights checksum verified)."""
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(w.url).name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != DETECTOR_SHA256:
        raise RuntimeError(f"detector weights checksum mismatch: {path}")
    m = fasterrcnn_resnet50_fpn_v2(weights=w).eval().to(device)
    return m


def detect(model, img_dir: Path, n: int, device: str = "cuda", batch: int = 4) -> tuple[dict[int, np.ndarray], float]:
    """Person detections per frame as (N, 5) [x1, y1, x2, y2, score], score >= MIN_SCORE."""
    import cv2
    import torch
    dets: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(1, n + 1, batch):
            fs = list(range(s, min(n, s + batch - 1) + 1))
            ims = [torch.from_numpy(cv2.imread(str(img_dir / f"{f:06d}.jpg"))[:, :, ::-1].copy()).permute(2, 0, 1)
                   .float().div(255).to(device) for f in fs]
            for f, o in zip(fs, model(ims)):
                keep = (o["labels"] == PERSON_LABEL) & (o["scores"] >= MIN_SCORE)
                b = o["boxes"][keep].cpu().numpy()
                sc = o["scores"][keep].cpu().numpy()
                dets[f] = np.column_stack([b, sc]).astype(float).reshape(-1, 5)
    return dets, time.perf_counter() - t0


def load_flag_model(path: Path) -> tuple[Any, float]:
    """The frozen appearance-audit model V3|T+A and its theta (checksum verified)."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FLAG_ARTIFACT_SHA256:
        raise RuntimeError("frozen flag artifact checksum mismatch")
    model, theta = pickle.loads(raw)["models"][FLAG_ENTRY]
    return model, float(theta)


def flag_rows(cues: list[dict[str, Any]], model, theta: float) -> list[dict[str, Any]]:
    """Score every missing record with the frozen model; flag iff a_loc defined and P >= theta."""
    from taorg.experiments.appearance.stats import score
    p = score(model, cues, "T+A") if cues else np.zeros(0)
    out = []
    for r, q in zip(cues, p):
        avail = not math.isnan(r["a_loc"])
        out.append({**r, "p_takeover": float(q), "appearance_available": avail, "flag": bool(avail and q >= theta)})
    return out


def seqinfo(video_id: str, fps: float, width: int, height: int, n: int) -> SeqInfo:
    """SeqInfo for the tracker adapters (fps from the annotation-independent container)."""
    return SeqInfo(video_id, fps, width, height, n)
