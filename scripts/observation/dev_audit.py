"""Observation audit on the DEVELOPMENT sequences only (02, 09, 11) x detectors (FRCNN, SDP).

Compares inference-time missing reasons (adapter heuristics + tracker state) with ORACLE
categories derived from the benchmark's manual annotations; evaluates the pre-stated criteria
U1-U3 (research/taorg_observation_audit.md §5); counts required phenomena; saves examples.
Outputs: results/observation/dev_audit/*  (absence-frame table gzipped; summaries; figures)
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from taorg.adapters.mot17 import build_records, load_detections, run_bytetrack
from taorg.experiments.observation.oracle import HOLD, categorise, identity_of, load_gt, match

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
DEV = ["02", "09", "11"]
DETS = ["FRCNN", "SDP"]
OUT = Path("results/observation/dev_audit")
REASONS = ["OUT_OF_VIEW_EVIDENCE", "IDENTITY_SUSPECT", "OCCLUSION_EVIDENCE", "TRACK_LOST", "TRACK_TERMINATED"]
CATS = ["OCCLUDED_GAP", "VISIBLE_MISS", "TAKEOVER", "GONE", "FALSE_TRACK"]


def runs_of(flags):
    out, s = [], None
    for i, v in enumerate(flags):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def audit_one(seq: str, det: str) -> tuple[list[dict], dict]:
    info, dets = load_detections(Z, seq, det)
    frames = run_bytetrack(info, dets)
    recs = build_records(info, frames)
    gt = load_gt(Z, seq, info.length)
    matches = match(frames, gt)
    g = identity_of(matches)
    rows = []
    for x, rr in recs.items():
        for r in rr:
            if r.detected:
                continue
            rows.append({"seq": seq, "det": det, "track": x, "frame": r.frame_idx, "reason": r.missing_reason.name,
                         "source": r.missing_reason_source.name, "tracker_state": r.tracker_state.name,
                         "t_since_s": round(r.time_since_detection_s, 4), "category": categorise(x, r.frame_idx, g, matches, gt)})
    # phenomena
    ph = Counter()
    gt_ids = {i for f in gt.values() for i in f}
    for i in gt_ids:
        pres = [i in gt[f] for f in range(1, info.length + 1)]
        if len(runs_of(pres)) >= 2:
            ph["gt_ids_leave_and_reenter"] += 1
        tids = {t for m in matches.values() for t, v in m.items() if v == i}
        if len(tids) == 1:
            ph["gt_ids_single_track_id"] += 1
        elif len(tids) >= 2:
            ph["gt_ids_multiple_track_ids"] += 1
    ph["gt_boxes_vis_lt_0.5"] = sum(1 for f in gt.values() for b in f.values() if b[1] < 0.5)
    ph["gt_boxes"] = sum(len(f) for f in gt.values())
    ph["tracks"] = len(recs)
    ph["false_tracks"] = sum(1 for x in recs if x not in g)
    by = defaultdict(list)
    for r in rows:
        by[r["track"]].append(r)
    for x, rr in by.items():
        fr = sorted(r["frame"] for r in rr)
        segs = runs_of([f in set(fr) for f in range(min(fr), max(fr) + 1)])
        ph["absence_segments"] += len(segs)
        resumed = {f for f, _ in [(t.frame_idx, 0) for t in recs[x] if t.detected]}
        for s, e in segs:
            if (min(fr) + e + 1) in resumed:
                ph["temporary_losses_resumed_same_id"] += 1
    return rows, dict(ph)


def summarise(rows: list[dict]) -> dict:
    C = Counter((r["reason"], r["category"]) for r in rows)
    tot_take = sum(v for (a, c), v in C.items() if c == "TAKEOVER")
    flag_id = sum(v for (a, c), v in C.items() if a == "IDENTITY_SUSPECT")
    tp = C[("IDENTITY_SUSPECT", "TAKEOVER")]
    hold = sum(v for (a, c), v in C.items() if c in HOLD)
    hold_dropflag = sum(v for (a, c), v in C.items() if c in HOLD and a in ("IDENTITY_SUSPECT", "OUT_OF_VIEW_EVIDENCE"))
    gone = sum(v for (a, c), v in C.items() if c == "GONE")
    exit_flag = sum(v for (a, c), v in C.items() if a == "OUT_OF_VIEW_EVIDENCE")
    unspec = sum(v for (a, c), v in C.items() if a in ("TRACK_LOST", "TRACK_TERMINATED"))
    return {"n_absence_frames": len(rows), "category_counts": dict(Counter(r["category"] for r in rows)),
            "reason_counts": dict(Counter(r["reason"] for r in rows)),
            "identity_precision": tp / flag_id if flag_id else float("nan"),
            "identity_recall": tp / tot_take if tot_take else float("nan"),
            "hold_frames_flagged_drop_rate": hold_dropflag / hold if hold else float("nan"),
            "exit_precision_for_gone": C[("OUT_OF_VIEW_EVIDENCE", "GONE")] / exit_flag if exit_flag else float("nan"),
            "exit_recall_for_gone": C[("OUT_OF_VIEW_EVIDENCE", "GONE")] / gone if gone else float("nan"),
            "occlusion_precision_for_hold": sum(C[("OCCLUSION_EVIDENCE", c)] for c in HOLD) / max(1, sum(
                v for (a, c), v in C.items() if a == "OCCLUSION_EVIDENCE")),
            "unspecified_coverage": unspec / len(rows) if rows else float("nan"),
            "confusion": {f"{a}|{c}": v for (a, c), v in sorted(C.items())}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, phen = [], {}
    for s in DEV:
        for d in DETS:
            rows, ph = audit_one(s, d)
            all_rows += rows
            phen[f"{s}-{d}"] = ph
    with gzip.open(OUT / "absence_frames.csv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    summ = {"pooled": summarise(all_rows),
            "by_detector": {d: summarise([r for r in all_rows if r["det"] == d]) for d in DETS},
            "by_unit": {f"{s}-{d}": summarise([r for r in all_rows if r["seq"] == s and r["det"] == d])
                        for s in DEV for d in DETS},
            "phenomena": phen}
    ok = {}
    for d in DETS:
        S = summ["by_detector"][d]
        u1 = S["identity_precision"] >= 0.70 and S["identity_recall"] >= 0.50
        u2 = S["hold_frames_flagged_drop_rate"] < 0.10
        ok[d] = {"U1": bool(u1), "U2": bool(u2),
                 "partial_U1": bool(S["identity_precision"] >= 0.70 and S["identity_recall"] >= 0.30)}
    U3 = all(v["U1"] and v["U2"] for v in ok.values())
    if U3:
        verdict = "USABLE"
    elif all(v["U2"] for v in ok.values()) and (any(v["U1"] for v in ok.values()) or all(v["partial_U1"] for v in ok.values())):
        verdict = "PARTIAL"
    else:
        verdict = "NOT_USABLE"
    summ["criteria"] = {"per_detector": ok, "U3_both_detectors": U3, "verdict": verdict}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("criteria",)}, indent=1))
    for d in DETS:
        S = summ["by_detector"][d]
        print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items() if k not in ("confusion",)})
    print(json.dumps(phen, indent=0))


if __name__ == "__main__":
    main()
