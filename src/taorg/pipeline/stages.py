"""End-to-end pipeline stages (docs/PIPELINE.md). Each stage reads/writes one work directory and is
idempotent; heavy imports are lazy because stages run in different pinned environments:

    perception  [perception env: BoxMOT]  video -> frames, person detections, OSNet, V3 tracks, flags
    sgg         [EGTR env]                frames -> EGTR person relations (every STRIDE-th frame)
    maintain    [main env]                tracks + relations -> canonical stream -> temporal method -> graphs.json
    view        [main env]                graphs.json (+ frames) -> viewer.html

No stage reads annotations. Work-directory layout:
    frames/000001.jpg ...   perception/{outputs.csv.gz, missing_records.csv.gz, meta.json}
    sgg/relations.npz       graphs.json      stream.jsonl.gz      viewer.html
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

STRIDE = 3


class StageError(RuntimeError):
    """A pipeline stage cannot run; the message says what is missing and how to fix it."""


def perception(video: Path, work: Path, flag_artifact: Path, reid_weights: Path) -> dict[str, Any]:
    """Frames, detections, tracks, missing records and the frozen takeover flag (perception env)."""
    os.environ.setdefault("TORCH_HOME", str(Path(os.environ.get("TAORG_TORCH_HOME", work.parent / "torch_home"))))
    try:
        import torch
        from boxmot.reid.core.runtime import ReID
    except ImportError as e:
        raise StageError(f"perception stage needs the perception environment (BoxMOT): {e}. See docs/SETUP.md") from e
    from taorg.adapters.appearance import compute_cues, compute_embeddings, run_botsort
    from taorg.adapters.mot17 import build_records
    from taorg.adapters.vidor_pipeline import detect, extract_frames, flag_rows, load_detector, load_flag_model, seqinfo
    from taorg.experiments.downstream.io import write_csv_gz
    if not Path(video).exists():
        raise StageError(f"video not found: {video}")
    if not Path(reid_weights).exists():
        raise StageError(f"ReID weights not found: {reid_weights} (osnet_x1_0_msmt17.pt; docs/SETUP.md)")
    out = work / "perception"
    if (out / "meta.json").exists():
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n, shape, fps = extract_frames(Path(video), work / "frames")
    if n == 0:
        raise StageError(f"no frames decoded from {video}")
    info = seqinfo(Path(video).stem, fps, shape[1], shape[0], n)
    dets, t_det = detect(load_detector(), work / "frames", n)
    reid = ReID(weights=reid_weights, device="0" if torch.cuda.is_available() else "cpu")
    embs, _ = compute_embeddings(info, dets, work / "frames", reid)
    run = run_botsort(info, dets, work / "frames", embs, use_embeddings=True)
    model, theta = load_flag_model(flag_artifact)
    cues = flag_rows(compute_cues(info, run, build_records(info, run.frames), dets, embs), model, theta)
    outs = [{"frame": fr.frame, "track": t, "det": run.out_det[i][t], "x1": fr.outputs[t][0][0],
             "y1": fr.outputs[t][0][1], "x2": fr.outputs[t][0][2], "y2": fr.outputs[t][0][3], "conf": fr.outputs[t][1]}
            for i, fr in enumerate(run.frames) for t in sorted(fr.outputs)]
    write_csv_gz(out / "outputs.csv.gz", outs, ["frame", "track", "det", "x1", "y1", "x2", "y2", "conf"])
    write_csv_gz(out / "missing_records.csv.gz", cues, ["track", "frame", "last_frame", "dt", "a_glob", "a_loc", "n_cand",
                                                        "n_loc", "has_s", "p_takeover", "appearance_available", "flag"])
    meta = {"video": str(video), "frames": n, "fps": fps, "width": shape[1], "height": shape[0],
            "n_outputs": len(outs), "n_flagged": int(sum(r["flag"] for r in cues)), "wall_s": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def sgg(work: Path) -> dict[str, Any]:
    """EGTR person relations on frames 1, 1+STRIDE, ... (EGTR env)."""
    try:
        import cv2

        from taorg.adapters.egtr import MAPPED, EGTRPredictor
    except ImportError as e:
        raise StageError(f"sgg stage needs the EGTR environment: {e}. See docs/SETUP.md") from e
    dst = work / "sgg" / "relations.npz"
    if dst.exists():
        return {"cached": True}
    frames = sorted((work / "frames").glob("*.jpg"))
    if not frames:
        raise StageError(f"no frames in {work / 'frames'}; run the perception stage first")
    pred = EGTRPredictor()
    fr, offs, B, S, C, R = [], [0], [], [], [], []
    t0 = time.time()
    for f in range(1, len(frames) + 1, STRIDE):
        rgb = cv2.imread(str(work / "frames" / f"{f:06d}.jpg"))[:, :, ::-1].copy()
        pr = pred.person_relations(pred.raw([rgb]), 0, rgb.shape[:2])
        fr.append(f)
        B.append(pr.boxes)
        S.append(pr.obj_scores)
        C.append(pr.classes)
        R.append(pr.scores.reshape(-1, len(MAPPED)))
        offs.append(offs[-1] + len(pr.obj_scores))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, frames=np.asarray(fr, np.int32), offsets=np.asarray(offs, np.int64),
                        boxes=np.concatenate(B).astype(np.float32).reshape(-1, 4),
                        obj_scores=np.concatenate(S).astype(np.float32), classes=np.concatenate(C).astype(np.int16),
                        pair_scores=np.concatenate(R).astype(np.float32).reshape(-1, len(MAPPED)))
    return {"sgg_frames": len(fr), "s_per_frame": (time.time() - t0) / max(1, len(fr))}


def maintain(work: Path, method: str, params: dict[str, Any], tau: float, use_flag: bool) -> dict[str, Any]:
    """Canonical stream -> temporal method -> graphs.json (+ stream.jsonl.gz). Main env; no annotations."""
    from taorg.adapters.egtr import MAPPED
    from taorg.adapters.mot17 import FrameTracks
    from taorg.experiments.downstream.io import read_csv_gz, write_csv_gz
    from taorg.experiments.final.runner import make_method
    from taorg.experiments.final.stream import build_predicted_stream, load_sgg
    meta_p = work / "perception" / "meta.json"
    rel_p = work / "sgg" / "relations.npz"
    for p, stage in ((meta_p, "perception"), (rel_p, "sgg")):
        if not p.exists():
            raise StageError(f"{p} missing: run the {stage} stage first")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    n = meta["frames"]
    per = {f: {} for f in range(1, n + 1)}
    for r in read_csv_gz(work / "perception" / "outputs.csv.gz"):
        per[int(r["frame"])][int(r["track"])] = ((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                                                 float(r["conf"]))
    frames = [FrameTracks(f, per[f], {}) for f in range(1, n + 1)]
    sampled = list(range(1, n + 1, STRIDE))
    raw: dict[int, set[int]] = {}
    for r in read_csv_gz(work / "perception" / "missing_records.csv.gz"):
        if r["flag"] == "True":
            raw.setdefault(int(r["frame"]), set()).add(int(r["track"]))
    flags, prev = {}, 0
    for f in sampled:
        s = set().union(*[raw.get(x, set()) for x in range(prev + 1, f + 1)])
        if s:
            flags[f] = frozenset(s)
        prev = f
    ps = build_predicted_stream(Path(meta["video"]).stem, meta["fps"], frames, load_sgg(rel_p), sampled, tau)
    name = f"{method}+F" if use_flag else method
    if use_flag and method not in ("FB", "B2"):
        raise StageError("the flag policy is implemented for FB and B2 only")
    m = make_method(name, params, flags)
    m.reset_video(Path(meta["video"]).stem)
    graphs, rows = [], []
    for g in ps.graphs:
        res = m.step(g)
        graphs.append({"frame": g.frame_idx, "t": g.timestamp,
                       "tracks": {str(t): [round(v, 1) for v in b] for t, (b, _) in per[g.frame_idx].items()},
                       "flagged": sorted(flags.get(g.frame_idx, ())),
                       "edges": [{"s": o.edge_key.subject_track_id, "o": o.edge_key.object_track_id,
                                  "p": MAPPED[o.edge_key.predicate_id], "score": round(o.predicate_score, 4),
                                  "observed": o.observed} for o in res.edge_observations]})
        for o in g.edge_observations:
            rows.append({"frame": g.frame_idx, "s": o.edge_key.subject_track_id, "o": o.edge_key.object_track_id,
                         "p": MAPPED[o.edge_key.predicate_id], "score": round(o.predicate_score, 4),
                         "observed": o.observed})
    out = {"method": name, "params": params, "tau": tau, "stride": STRIDE, "fps": meta["fps"],
           "width": meta["width"], "height": meta["height"], "frames": graphs}
    (work / "graphs.json").write_text(json.dumps(out), encoding="utf-8")
    write_csv_gz(work / "stream.csv.gz", rows, ["frame", "s", "o", "p", "score", "observed"])
    return {"method": name, "steps": len(graphs), "edges_emitted": sum(len(g["edges"]) for g in graphs),
            "records": len(rows)}
