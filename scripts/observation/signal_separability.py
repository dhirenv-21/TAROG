"""EXPLORATORY, DESCRIPTIVE (defined after the pre-stated NOT_USABLE verdict; never a decision input).

Development sequences only.  For every absence frame of every confirmed track, computes the
directly available / derivable continuous signals and their single-signal ROC AUC for
separating real gaps (OCCLUDED_GAP + VISIBLE_MISS) from identity-related missing records
(TAKEOVER), and from GONE:
    t_since_s            time since the track's last detection (direct)
    in_buffer            tracker still holds the track (direct: LOST vs REMOVED)
    iou_new              max IoU of the predicted box with any NEWER detected track
    iou_old              max IoU of the predicted box with any OLDER detected track
    cover_any            max fraction of the predicted box covered by any other detected track
    outside              fraction of the predicted box outside the image
Also: P(hold correct | time since detection) curve (what a fixed buffer can exploit).
Outputs: results/observation/dev_audit/separability.json, hold_vs_time.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from taorg.adapters.mot17 import (build_records, cover, iou, load_detections, outside_fraction, run_bytetrack)
from taorg.experiments.observation.oracle import HOLD, categorise, identity_of, load_gt, match

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
OUT = Path("results/observation/dev_audit")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    rows = []
    for seq in ("02", "09", "11"):
        for det in ("FRCNN", "SDP"):
            info, dets = load_detections(Z, seq, det)
            frames = run_bytetrack(info, dets)
            recs = build_records(info, frames)
            gt = load_gt(Z, seq, info.length)
            m = match(frames, gt)
            g = identity_of(m)
            first_out = {}
            for fr in frames:
                for t in fr.outputs:
                    first_out.setdefault(t, fr.frame)
            for x, rr in recs.items():
                last_det = None
                for r in rr:
                    if r.detected:
                        last_det = r.frame_idx
                        continue
                    fr = frames[r.frame_idx - 1]
                    p = r.predicted_bbox
                    others = [(y, b) for y, (b, _) in fr.outputs.items() if y != x]
                    new = [iou(p, b) for y, b in others if first_out[y] > last_det]
                    old = [iou(p, b) for y, b in others if first_out[y] <= last_det]
                    rows.append({"det": det, "seq": seq, "cat": categorise(x, r.frame_idx, g, m, gt),
                                 "t_since_s": r.time_since_detection_s,
                                 "in_buffer": 1.0 if r.tracker_state.name == "LOST" else 0.0,
                                 "iou_new": max(new, default=0.0), "iou_old": max(old, default=0.0),
                                 "cover_any": max([cover(p, b) for _, b in others], default=0.0),
                                 "outside": outside_fraction(p, info.width, info.height)})
    sigs = ["t_since_s", "in_buffer", "iou_new", "iou_old", "cover_any", "outside"]
    res = {}
    for det in ("FRCNN", "SDP", "ALL"):
        R = [r for r in rows if det == "ALL" or r["det"] == det]
        hold = [r for r in R if r["cat"] in HOLD]
        res[det] = {"n_hold": len(hold), "n_takeover": sum(r["cat"] == "TAKEOVER" for r in R),
                    "n_gone": sum(r["cat"] == "GONE" for r in R)}
        for s in sigs:
            h = np.array([r[s] for r in hold])
            res[det][f"auc_takeover_vs_hold__{s}"] = auc(np.array([r[s] for r in R if r["cat"] == "TAKEOVER"]), h)
            res[det][f"auc_gone_vs_hold__{s}"] = auc(np.array([r[s] for r in R if r["cat"] == "GONE"]), h)
    (OUT / "separability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    bins = [0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.01]
    with open(OUT / "hold_vs_time.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["det", "t_lo", "t_hi", "n", "hold_correct_fraction", "takeover_fraction", "gone_fraction", "false_track_fraction"])
        for det in ("FRCNN", "SDP"):
            for lo, hi in zip(bins[:-1], bins[1:]):
                R = [r for r in rows if r["det"] == det and lo < r["t_since_s"] <= hi]
                n = len(R)
                if n:
                    w.writerow([det, lo, hi, n, sum(r["cat"] in HOLD for r in R) / n,
                                sum(r["cat"] == "TAKEOVER" for r in R) / n, sum(r["cat"] == "GONE" for r in R) / n,
                                sum(r["cat"] == "FALSE_TRACK" for r in R) / n])
    for det in ("FRCNN", "SDP"):
        print(det, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[det].items()})


if __name__ == "__main__":
    main()
