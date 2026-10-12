"""Appearance audit, RUN stage (no ground truth): embeddings, V1/V2/V3 tracker runs, records, cues,
and the label-free feasibility checks (prereg §6).

Must run in the isolated venv (BoxMOT):
    D:/TAORG_runs/appearance/venv/Scripts/python scripts/appearance/run_trackers.py --split dev

Heavy state  -> D:/TAORG_runs/appearance/<split>/<seq>-<det>/
Compact logs -> results/appearance/<split>/run_records.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from taorg.adapters.appearance import (
    BOTSORT_SETTINGS,
    HIGH_CONF,
    check_alignment,
    compute_cues,
    compute_embeddings,
    run_botsort,
    run_bytetrack_v1,
    stored_appearance_series,
    unit,
)
from taorg.adapters.mot17 import build_records, load_detections
from taorg.utils.seed import set_seed

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
FRAMES = Path("D:/TAORG_runs/appearance/MOT17Det/train")
WEIGHTS = Path("D:/TAORG_runs/appearance/weights/osnet_x1_0_msmt17.pt")
WEIGHTS_SHA = "b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2"
SPLITS = {"dev": ["02", "09", "11"], "eval": ["04", "05", "10", "13"]}
DETS = ["FRCNN", "SDP"]
TRACKERS = ["V1", "V2", "V3"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_embs(p: Path, embs: dict[int, np.ndarray]) -> None:
    keys = sorted(embs)
    np.savez_compressed(p, frames=np.array(keys), counts=np.array([len(embs[k]) for k in keys]),
                        emb=np.concatenate([embs[k] for k in keys]) if keys else np.zeros((0, 512), np.float16))


def load_embs(p: Path) -> dict[int, np.ndarray]:
    z = np.load(p)
    emb = z["emb"]   # decompress once (indexing z["emb"] per frame re-reads the whole array)
    out, o = {}, 0
    for f, c in zip(z["frames"], z["counts"]):
        out[int(f)] = emb[o:o + c]
        o += c
    return out


def write_csv_gz(p: Path, rows: list[dict]) -> None:
    with gzip.open(p, "wt", encoding="utf-8", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def feasibility_v3(run, dets, embs, max_rows: int = 50000) -> dict:
    """Label-free checks: appearance changes the cost; s_X == smooth_feat; raw emb == 1 - s_X . e."""
    ser = stored_appearance_series(run, dets, embs, include_creation=True)   # mirror of smooth_feat (A2)
    log = run.assoc_log
    res: dict = {"assoc_rows": int(len(log)), "n_created_recorded": len(run.created)}
    if len(log):
        changed = np.abs(log[:, 6] - log[:, 4]) > 1e-9
        res["share_final_cost_ne_geom"] = float(changed.mean())
        # raw-emb equality on a deterministic subsample
        idx = np.linspace(0, len(log) - 1, min(max_rows, len(log))).astype(int)
        ok = n = 0
        maxerr = 0.0
        for f, tid, j, _, _, raw, _ in log[idx]:
            f, tid, j = int(f), int(tid), int(j)
            prev = [s for (g, s) in ser.get(tid, []) if g < f]
            if not prev or prev[-1] is None:
                continue
            ext = 1.0 - float(unit(embs[f][j]) @ prev[-1])
            err = abs(max(0.0, ext) - float(raw))
            maxerr = max(maxerr, err)
            ok += err < 1e-4
            n += 1
        res.update({"raw_emb_checked": n, "raw_emb_equal_share": ok / n if n else float("nan"),
                    "raw_emb_max_abs_err": maxerr})
    ok = n = 0
    mincos = 1.0
    for (f, tid), sf in run.smooth_feats.items():
        prev = [s for (g, s) in ser.get(tid, []) if g < f]
        if not prev or prev[-1] is None:
            continue
        c = float(unit(sf) @ prev[-1])
        mincos = min(mincos, c)
        ok += c >= 0.999
        n += 1
    res.update({"smooth_feat_checked": n, "smooth_feat_equal_share": ok / n if n else float("nan"),
                "smooth_feat_min_cos": mincos})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    a = ap.parse_args()
    set_seed(20261012)
    import torch
    from boxmot import __version__ as boxmot_version
    from boxmot.reid.core.runtime import ReID

    if sha256(WEIGHTS) != WEIGHTS_SHA:
        raise SystemExit("ReID weights checksum mismatch")
    heavy = Path(f"D:/TAORG_runs/appearance/{a.split}")
    out = Path(f"results/appearance/{a.split}")
    heavy.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/appearance"],
                                capture_output=True, text=True).stdout.strip())
    reid = ReID(weights=WEIGHTS, device="0" if torch.cuda.is_available() else "cpu")
    records: dict = {"split": a.split, "git_commit": commit, "git_dirty_src": dirty, "seed": 20261012,
                     "python": sys.version.split()[0], "torch": torch.__version__, "boxmot": boxmot_version,
                     "cuda": torch.cuda.is_available(), "platform": platform.platform(),
                     "botsort_settings": BOTSORT_SETTINGS, "high_conf": HIGH_CONF, "units": {}}
    for seq in SPLITS[a.split]:
        for det in DETS:
            unit_name = f"{seq}-{det}"
            t_unit = time.time()
            ud = heavy / unit_name
            ud.mkdir(exist_ok=True)
            info, dets = load_detections(Z, seq, det)
            img_dir = FRAMES / f"MOT17-{seq}" / "img1"
            align = check_alignment(info, img_dir)
            if not align["ok"]:
                raise SystemExit(f"frame alignment failed: {align}")
            n_det = sum(len(d) for d in dets.values())
            ep = ud / "embeddings.npz"
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            embs, et = compute_embeddings(info, dets, img_dir, reid)
            save_embs(ep, embs)
            embs = load_embs(ep)   # the float16 cache is what every consumer sees
            emb_norms = np.concatenate([np.linalg.norm(e.astype(np.float32), axis=1) for e in embs.values() if len(e)])
            u = {"seq": seq, "det": det, "fps": info.fps, "length": info.length, "size": [info.width, info.height],
                 "alignment": align, "n_detections": n_det,
                 "n_embeddings": int(sum(len(e) for e in embs.values())),
                 "emb_norm_min": float(emb_norms.min()), "emb_norm_max": float(emb_norms.max()),
                 "embeddings_sha256": sha256(ep), "reid_timing": et,
                 "gpu_peak_mb_reid": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None,
                 "runs": {}}
            if u["n_embeddings"] != n_det:
                raise SystemExit(f"{unit_name}: embedding count {u['n_embeddings']} != detections {n_det}")
            out_dets = {}
            for name in TRACKERS:   # one run in memory at a time (low free RAM)
                if name == "V1":
                    run = run_bytetrack_v1(info, dets)
                else:
                    run = run_botsort(info, dets, img_dir, embs if name == "V3" else None,
                                      use_embeddings=(name == "V3"))
                out_dets[name] = run.out_det
                outs = [{"frame": fr.frame, "track": t, "det": run.out_det[i][t]}
                        for i, fr in enumerate(run.frames) for t in sorted(fr.outputs)]
                write_csv_gz(ud / f"{name}_outputs.csv.gz", outs)
                lost = [{"frame": fr.frame, "track": t, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "since": s}
                        for fr in run.frames for t, (b, s) in sorted(fr.internal.items())]
                write_csv_gz(ud / f"{name}_internal.csv.gz", lost)
                if run.assoc_log is not None:
                    np.savez_compressed(ud / f"{name}_assoc_log.npz", log=run.assoc_log)
                recs = build_records(info, run.frames)
                cues = compute_cues(info, run, recs, dets, embs)
                write_csv_gz(ud / f"{name}_cues.csv.gz", cues)
                r = {"timing": run.timing, "n_tracks": len(recs), "n_outputs": len(outs),
                     "n_missing_records": len(cues),
                     "outputs_sha256": sha256(ud / f"{name}_outputs.csv.gz"),
                     "cues_sha256": sha256(ud / f"{name}_cues.csv.gz")}
                if name == "V3":
                    r["feasibility"] = feasibility_v3(run, dets, embs)
                u["runs"][name] = r
                del run, recs, cues, outs, lost
                gc.collect()
            f2, f3 = out_dets["V2"], out_dets["V3"]
            u["frames_V2_V3_outputs_differ"] = int(sum(a_ != b_ for a_, b_ in zip(f2, f3)))
            u["share_frames_V2_V3_differ"] = u["frames_V2_V3_outputs_differ"] / info.length
            u["unit_wall_s"] = time.time() - t_unit
            records["units"][unit_name] = u
            print(unit_name, json.dumps({k: u[k] for k in ("n_detections", "share_frames_V2_V3_differ")}),
                  json.dumps(u["runs"]["V3"]["feasibility"], default=float), flush=True)
            del embs, out_dets
            gc.collect()
    try:
        import psutil
        records["peak_rss_mb"] = psutil.Process(os.getpid()).memory_info().peak_wset / 1e6
    except Exception as exc:  # psutil optional; report why the number is missing
        records["peak_rss_mb"] = f"unavailable: {exc!r}"
    (out / "run_records.json").write_text(json.dumps(records, indent=1, default=float), encoding="utf-8")
    print("wrote", out / "run_records.json")


if __name__ == "__main__":
    main()
