"""Downstream study, PERCEPTION stage (inference only; no annotations): per sampled video
frames -> person detections -> OSNet embeddings -> V3 BoT-SORT -> missing records, cues -> frozen flag.

Run in the BoxMOT venv, one video at a time (low free RAM), resumable:
    D:/TAORG_runs/appearance/venv/Scripts/python scripts/downstream/run_pipeline.py --split training
Heavy state: D:/TAORG_runs/downstream/<split>/<video_id>/ ; record: results/downstream/<split>/pipeline_records.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TORCH_HOME", "D:/TAORG_runs/downstream/torch_home")

import numpy as np  # noqa: E402

from taorg.adapters.appearance import (  # noqa: E402
    BOTSORT_SETTINGS,
    compute_cues,
    compute_embeddings,
    run_botsort,
)
from taorg.adapters.mot17 import build_records  # noqa: E402
from taorg.adapters.vidor import sample  # noqa: E402
from taorg.adapters.vidor_pipeline import (  # noqa: E402
    FLAG_ARTIFACT_SHA256,
    detect,
    extract_frames,
    flag_rows,
    load_detector,
    load_flag_model,
    seqinfo,
)
from taorg.experiments.downstream.io import write_csv_gz  # noqa: E402
from taorg.utils.seed import set_seed  # noqa: E402

VIDEOS = Path("D:/TAORG_runs/downstream/videos")
WEIGHTS = Path("D:/TAORG_runs/appearance/weights/osnet_x1_0_msmt17.pt")
FLAG = Path("results/appearance/freeze/frozen_models.pkl")
SPLIT_NAME = {"training": "dev", "validation": "eval"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_NAME), required=True)
    a = ap.parse_args()
    set_seed(20261012)
    import torch
    from boxmot.reid.core.runtime import ReID
    name = SPLIT_NAME[a.split]
    heavy = Path(f"D:/TAORG_runs/downstream/{name}")
    out = Path(f"results/downstream/{name}")
    out.mkdir(parents=True, exist_ok=True)
    recp = out / "pipeline_records.json"
    rec = json.loads(recp.read_text(encoding="utf-8")) if recp.exists() else {"videos": {}}
    rec.update({"split": a.split, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                                text=True).stdout.strip(),
                "git_dirty_src": bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/downstream"],
                                                     capture_output=True, text=True).stdout.strip()),
                "botsort_settings": BOTSORT_SETTINGS, "flag_artifact_sha256": FLAG_ARTIFACT_SHA256})
    model, theta = load_flag_model(FLAG)
    rec["flag_theta"] = theta
    det_model = load_detector()
    reid = ReID(weights=WEIGHTS, device="0")
    for vp in sample(Path("data/vidor"), a.split):
        vid = Path(vp).stem
        if vid in rec["videos"] and rec["videos"][vid].get("status") == "done":
            continue                                                   # resume
        t0 = time.time()
        vd = heavy / vid
        n, shape, fps = extract_frames(VIDEOS / vp, vd / "frames")
        info = seqinfo(vid, fps, shape[1], shape[0], n)
        dets, t_det = detect(det_model, vd / "frames", n)
        embs, et = compute_embeddings(info, dets, vd / "frames", reid)
        run = run_botsort(info, dets, vd / "frames", embs, use_embeddings=True)
        recs = build_records(info, run.frames)
        cues = flag_rows(compute_cues(info, run, recs, dets, embs), model, theta)
        np.savez_compressed(vd / "detections.npz", **{str(f): d for f, d in dets.items()})
        outs = [{"frame": fr.frame, "track": t, "det": run.out_det[i][t],
                 "x1": fr.outputs[t][0][0], "y1": fr.outputs[t][0][1], "x2": fr.outputs[t][0][2],
                 "y2": fr.outputs[t][0][3], "conf": fr.outputs[t][1]}
                for i, fr in enumerate(run.frames) for t in sorted(fr.outputs)]
        h_out = write_csv_gz(vd / "outputs.csv.gz", outs, ["frame", "track", "det", "x1", "y1", "x2", "y2", "conf"])
        h_cue = write_csv_gz(vd / "missing_records.csv.gz", cues,
                             ["track", "frame", "last_frame", "dt", "a_glob", "a_loc", "n_cand", "n_loc", "has_s",
                              "p_takeover", "appearance_available", "flag"])
        rec["videos"][vid] = {"video_path": vp, "status": "done", "frames": n, "size": list(shape), "fps": fps,
                              "n_detections": int(sum(len(d) for d in dets.values())), "n_tracks": len(recs),
                              "n_outputs": len(outs), "n_missing_records": len(cues),
                              "n_flagged": int(sum(r["flag"] for r in cues)),
                              "n_appearance_available": int(sum(r["appearance_available"] for r in cues)),
                              "detect_ms_per_frame": 1000 * t_det / n, "reid_ms_per_frame": et["reid_ms_per_frame"],
                              "track_ms_per_frame": run.timing["track_ms_per_frame"],
                              "outputs_content_sha256": h_out, "missing_records_content_sha256": h_cue,
                              "wall_s": time.time() - t0}
        recp.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")   # checkpoint
        print(vid, n, "frames", rec["videos"][vid]["n_missing_records"], "missing,",
              rec["videos"][vid]["n_flagged"], "flagged", round(time.time() - t0), "s", flush=True)
        del dets, embs, run, recs, cues, outs
        gc.collect()
        torch.cuda.empty_cache()
    print("done", len(rec["videos"]))


if __name__ == "__main__":
    main()
