"""EGTR on VidOR video frames (relationship-prediction stage; inference only, no annotations).

Run in the EGTR venv, one video at a time, resumable:
    D:/TAORG_runs/egtr/venv/Scripts/python scripts/egtr/run_video_sgg.py --split final
Uses the frames extracted by the perception stage (D:/TAORG_runs/downstream/<split>/<vid>/frames/,
internal 1-based indices). EGTR runs on frames f = 1, 1 + STRIDE, ... (registered stride 3 = 10 fps
at 30 fps). Saves D:/TAORG_runs/egtr/sgg/<split>/<vid>.npz and results/final/<split>/sgg_records.json
(content hash = SHA-256 over the saved arrays' bytes in key order; .npz zip timestamps are ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from taorg.adapters.egtr import CKPT_SHA256, EGTR_COMMIT, MAPPED, PREDICATE_MAP, EGTRPredictor
from taorg.adapters.vidor import sample
from taorg.utils.seed import set_seed

STRIDE = 3
SPLITS = {"dev": "training", "final": "final"}


def array_hash(d: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    a = ap.parse_args()
    set_seed(20261020)
    import cv2
    import torch
    frames_root = Path(f"D:/TAORG_runs/downstream/{a.split}")
    out_dir = Path(f"D:/TAORG_runs/egtr/sgg/{a.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_p = Path(f"results/final/{a.split}/sgg_records.json")
    rec_p.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.exists() else {"videos": {}}
    rec.update({"stride": STRIDE, "egtr_commit": EGTR_COMMIT, "ckpt_sha256": CKPT_SHA256, "mapped": MAPPED,
                "predicate_map": PREDICATE_MAP, "torch": torch.__version__,
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "git_dirty_src": bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/egtr"],
                                                     capture_output=True, text=True).stdout.strip())})
    pred = EGTRPredictor()
    for vp in sample(Path("data/vidor"), SPLITS[a.split]):
        vid = Path(vp).stem
        if rec["videos"].get(vid, {}).get("status") == "done":
            continue
        fdir = frames_root / vid / "frames"
        n = len(list(fdir.glob("*.jpg")))
        if n == 0:
            raise SystemExit(f"no frames for {vid} under {fdir}; run the perception stage first")
        t0 = time.perf_counter()
        fr, offs, B, S, C, R = [], [0], [], [], [], []
        for f in range(1, n + 1, STRIDE):
            rgb = cv2.imread(str(fdir / f"{f:06d}.jpg"))[:, :, ::-1].copy()
            out = pred.raw([rgb])
            pr = pred.person_relations(out, 0, rgb.shape[:2])
            fr.append(f)
            B.append(pr.boxes)
            S.append(pr.obj_scores)
            C.append(pr.classes)
            R.append(pr.scores.reshape(-1, len(MAPPED)))
            offs.append(offs[-1] + len(pr.obj_scores))
        dt = time.perf_counter() - t0
        arrays = {"frames": np.asarray(fr, np.int32), "offsets": np.asarray(offs, np.int64),
                  "boxes": np.concatenate(B).astype(np.float32).reshape(-1, 4),
                  "obj_scores": np.concatenate(S).astype(np.float32),
                  "classes": np.concatenate(C).astype(np.int16),
                  "pair_scores": np.concatenate(R).astype(np.float32).reshape(-1, len(MAPPED))}
        np.savez_compressed(out_dir / f"{vid}.npz", **arrays)
        rec["videos"][vid] = {"status": "done", "frames_total": n, "frames_sgg": len(fr),
                              "persons_per_frame": float(np.mean(np.diff(offs))) if fr else 0.0,
                              "s_per_frame": dt / max(1, len(fr)), "content_sha256": array_hash(arrays)}
        rec_p.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        print(vid, len(fr), "sgg frames", round(dt / max(1, len(fr)), 3), "s/frame", flush=True)
    print("done", len(rec["videos"]))


if __name__ == "__main__":
    main()
