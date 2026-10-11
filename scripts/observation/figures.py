"""Observation-audit figures and representative examples (development data; artifacts only).

Example rule (fixed in this script, applied mechanically): absence SEGMENTS (maximal runs of
undetected frames of one track) with >= 10 frames; label = majority oracle category and
majority inference-time reason.
    success_identity   majority TAKEOVER and majority IDENTITY_SUSPECT (first by seq, det, track)
    success_occlusion  majority OCCLUDED_GAP and majority OCCLUSION_EVIDENCE (first)
    fail_takeover      majority TAKEOVER but majority OCCLUSION_EVIDENCE or OUT_OF_VIEW (longest)
    fail_realgap       majority OCCLUDED_GAP/VISIBLE_MISS but majority OUT_OF_VIEW (longest)
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.experiments.episodes.plots import _save, _setup  # noqa: E402

A = Path("results/observation/dev_audit")
P = Path("results/observation/plots")
REASONS = ["IDENTITY_SUSPECT", "OCCLUSION_EVIDENCE", "OUT_OF_VIEW_EVIDENCE", "TRACK_LOST", "TRACK_TERMINATED"]
CATS = ["TAKEOVER", "OCCLUDED_GAP", "VISIBLE_MISS", "GONE", "FALSE_TRACK"]
CCOL = {"TAKEOVER": "#D55E00", "OCCLUDED_GAP": "#0072B2", "VISIBLE_MISS": "#56B4E9", "GONE": "#999999",
        "FALSE_TRACK": "#000000"}


def main() -> None:
    _setup()
    rows = list(csv.DictReader(gzip.open(A / "absence_frames.csv.gz", "rt", encoding="utf-8")))
    files = []
    # Fig 1: confusion (row-normalised by oracle category)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, det in zip(axes, ("FRCNN", "SDP")):
        R = [r for r in rows if r["det"] == det]
        M = np.array([[sum(1 for r in R if r["category"] == c and r["reason"] == a) for a in REASONS] for c in CATS], float)
        Mn = M / np.maximum(1, M.sum(1, keepdims=True))
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(CATS)):
            for j in range(len(REASONS)):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Mn[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(REASONS)))
        ax.set_xticklabels(["identity\nsuspect", "occlusion\nevidence", "out of\nview", "tracker:\nlost", "tracker:\ndeleted"],
                           fontsize=6)
        ax.set_yticks(range(len(CATS)))
        ax.set_yticklabels([f"{c} (n={int(M[i].sum())})" for i, c in enumerate(CATS)], fontsize=6)
        ax.set_title(f"{det} public detections + ByteTrack", fontsize=8)
    axes[0].set_ylabel("oracle category (benchmark annotations)")
    fig.suptitle("Fig. 1 — Inference-time missing reason vs oracle category (row-normalised; dev 02/09/11)", fontsize=9, y=1.03)
    fig.tight_layout()
    files += _save(fig, P / "fig1_reason_vs_oracle")
    # Fig 2: single-signal AUCs
    sep = json.loads((A / "separability.json").read_text(encoding="utf-8"))
    sigs = ["t_since_s", "outside", "iou_new", "iou_old", "in_buffer", "cover_any"]
    names = ["time since\ndetection", "predicted box\noutside image", "IoU with\nnewer track", "IoU with\nolder track",
             "still in\ntracker buffer", "covered by\nother track"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k, (det, col) in enumerate((("FRCNN", "#0072B2"), ("SDP", "#E69F00"))):
        v = [sep[det][f"auc_takeover_vs_hold__{s}"] for s in sigs]
        ax.bar(np.arange(len(sigs)) + (k - 0.5) * 0.35, v, 0.35, color=col, label=det)
    ax.axhline(0.5, color="black", lw=0.7, ls="--")
    ax.set_xticks(range(len(sigs)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0.2, 0.8)
    ax.set_ylabel("AUC: TAKEOVER vs real gap")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Fig. 2 — Single-signal separability (0.5 = chance; < 0.5 = reversed). Exploratory, dev only", fontsize=8.5)
    files += _save(fig, P / "fig2_signal_auc")
    # Fig 3: what a fixed buffer can exploit
    hv = list(csv.DictReader(open(A / "hold_vs_time.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, det in zip(axes, ("FRCNN", "SDP")):
        H = [r for r in hv if r["det"] == det]
        x = [0.5 * (float(r["t_lo"]) + float(r["t_hi"])) for r in H]
        for key, lab, col in (("hold_correct_fraction", "real gap (hold correct)", "#0072B2"),
                              ("takeover_fraction", "identity takeover", "#D55E00"),
                              ("gone_fraction", "object gone", "#999999"), ("false_track_fraction", "false track", "#000000")):
            ax.plot(x, [float(r[key]) for r in H], marker="o", ms=3, color=col, label=lab)
        ax.set_xlabel("time since last detection (s)")
        ax.set_title(det, fontsize=8)
    axes[0].set_ylabel("share of absence frames")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Fig. 3 — Composition of missing records vs elapsed time (the information a fixed buffer uses)", fontsize=9, y=1.03)
    fig.tight_layout()
    files += _save(fig, P / "fig3_composition_vs_time")
    # examples
    seg = defaultdict(list)
    for r in rows:
        seg[(r["seq"], r["det"], int(r["track"]))].append(r)
    segs = []
    for key, rr in seg.items():
        rr.sort(key=lambda r: int(r["frame"]))
        cur = [rr[0]]
        for r in rr[1:]:
            if int(r["frame"]) == int(cur[-1]["frame"]) + 1:
                cur.append(r)
            else:
                segs.append((key, cur))
                cur = [r]
        segs.append((key, cur))
    def lab(s):
        return Counter(r["category"] for r in s).most_common(1)[0][0], Counter(r["reason"] for r in s).most_common(1)[0][0]
    long = [(k, s) for k, s in segs if len(s) >= 10]
    rules = {"success_identity": lambda c, a: c == "TAKEOVER" and a == "IDENTITY_SUSPECT",
             "success_occlusion": lambda c, a: c == "OCCLUDED_GAP" and a == "OCCLUSION_EVIDENCE",
             "fail_takeover": lambda c, a: c == "TAKEOVER" and a in ("OCCLUSION_EVIDENCE", "OUT_OF_VIEW_EVIDENCE"),
             "fail_realgap": lambda c, a: c in ("OCCLUDED_GAP", "VISIBLE_MISS") and a == "OUT_OF_VIEW_EVIDENCE"}
    ex = {}
    for name, f in rules.items():
        cand = [(k, s) for k, s in long if f(*lab(s))]
        if not cand:
            ex[name] = "none found"
            continue
        k, s = cand[0] if name.startswith("success") else max(cand, key=lambda t: len(t[1]))
        ex[name] = {"seq": k[0], "det": k[1], "track": k[2], "frames": [int(s[0]["frame"]), int(s[-1]["frame"])],
                    "n_candidates": len(cand), "categories": dict(Counter(r["category"] for r in s)),
                    "reasons": dict(Counter(r["reason"] for r in s))}
    ex["n_segments_ge10"] = len(long)
    ex["segment_majority_table"] = {f"{c}|{a}": n for (c, a), n in Counter(lab(s) for _, s in long).most_common()}
    (A / "examples.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
    # Fig 4: example timelines
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.2))
    for ax, name in zip(axes, rules):
        e = ex[name]
        if isinstance(e, str):
            ax.text(0.5, 0.5, f"{name}: none found", ha="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        s = [r for r in seg[(e["seq"], e["det"], e["track"])] if e["frames"][0] <= int(r["frame"]) <= e["frames"][1]]
        f0 = int(s[0]["frame"])
        for i, r in enumerate(s):
            ax.add_patch(plt.Rectangle((int(r["frame"]) - f0, 1), 1, 0.8, color=CCOL[r["category"]]))
            ax.add_patch(plt.Rectangle((int(r["frame"]) - f0, 0), 1, 0.8,
                                       color={"IDENTITY_SUSPECT": "#D55E00", "OCCLUSION_EVIDENCE": "#0072B2",
                                              "OUT_OF_VIEW_EVIDENCE": "#999999"}.get(r["reason"], "#DDDDDD")))
        ax.set_xlim(0, len(s))
        ax.set_ylim(0, 2)
        ax.set_yticks([0.4, 1.4])
        ax.set_yticklabels(["inferred reason", "oracle category"], fontsize=6)
        ax.set_title(f"{name}: MOT17-{e['seq']}-{e['det']} track {e['track']}, frames {e['frames'][0]}-{e['frames'][1]}",
                     fontsize=7, loc="left")
    axes[-1].set_xlabel("frames into the absence segment (colours: orange identity/takeover, blue occlusion/gap, grey out-of-view/gone, light grey tracker-only)")
    fig.suptitle("Fig. 4 — Representative examples (predefined rule; development data)", fontsize=9)
    fig.tight_layout()
    files += _save(fig, P / "fig4_examples")
    print(json.dumps({k: v for k, v in ex.items() if k != "segment_majority_table"}, indent=0)[:2500])
    print(ex["segment_majority_table"])


if __name__ == "__main__":
    main()
