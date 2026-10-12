"""Appearance-audit figures and the predefined representative examples (prereg §10).

    python scripts/appearance/figures.py --split dev     (main env; needs the frames for fig 4)

Fig 1  coverage: a_loc availability (and takeover share) vs elapsed time, per tracker
Fig 2  frame-level AUC of T, A, T+A per detector, and ΔAUC(T+A − T) with bootstrap intervals
Fig 3  incident-level AUC vs elapsed time tau (V3), with the RETROSPECTIVE reference
Fig 4  annotated example frames (V3 FRCNN; one incident per TP/TN/FN/FP cell; median length)
Frames: MOT17 (MOTChallenge), CC BY-NC-SA 3.0 — cropped, annotated, attributed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.adapters.mot17 import load_detections  # noqa: E402
from taorg.experiments.episodes.plots import _save, _setup  # noqa: E402

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
FRAMES = Path("D:/TAORG_runs/appearance/MOT17Det/train")
TRK = {"V1": ("ByteTrack (V1)", "#999999", "s", ":"), "V2": ("BoT-SORT, appearance off (V2)", "#0072B2", "o", "--"),
       "V3": ("BoT-SORT, appearance on (V3)", "#D55E00", "^", "-")}
FS = {"T": ("elapsed time (T)", "#999999", "s"), "A": ("appearance (A)", "#56B4E9", "o"), "T+A": ("combined (T+A)", "#D55E00", "^")}
ATTR = "Frames: MOT17 (MOTChallenge), CC BY-NC-SA 3.0"


def read(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_coverage(R: dict, P: Path) -> list:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for t, (lab, c, m, ls) in TRK.items():
        cb = R["trackers"][t]["coverage_by_dt"]
        x = [0.5 * (b["lo"] + b["hi"]) for b in cb]
        ax[0].plot(x, [b["a_loc_available"] for b in cb], color=c, marker=m, ls=ls, label=lab)
        ax[1].plot(x, [b["takeover_share"] for b in cb], color=c, marker=m, ls=ls, label=lab)
    ax[0].set(xlabel="time since last output (s)", ylabel="share with appearance evidence (a_loc)", ylim=(0, 1.02),
              title="When is appearance evidence available?")
    ax[1].set(xlabel="time since last output (s)", ylabel="TAKEOVER share of GAP+TAKEOVER", ylim=(0, 1.02),
              title="Label composition")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return _save(fig, P / "fig1_coverage")


def fig_comparison(R: dict, P: Path) -> list:
    dets = ["FRCNN", "SDP", "pooled"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (fs, (lab, c, m)) in enumerate(FS.items()):
        ax[0].scatter(np.arange(3) + (k - 1) * 0.18, [R["trackers"]["V3"]["frames"][d][fs]["auc"] for d in dets],
                      color=c, marker=m, s=40, label=lab)
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[0].set(xticks=range(3), xticklabels=dets, ylabel="out-of-fold ROC-AUC (TAKEOVER vs GAP)", ylim=(0.4, 1.0),
              title="V3 records: frame-level discrimination")
    ax[0].legend(fontsize=7)
    for k, (t, (lab, c, m, ls)) in enumerate(TRK.items()):
        for j, d in enumerate(dets):
            e = R["trackers"][t]["frames"][d]["delta_T+A_minus_T"]
            x = j + (k - 1) * 0.2
            ax[1].errorbar(x, e["estimate"], yerr=[[e["estimate"] - e["lo"]], [e["hi"] - e["estimate"]]], color=c,
                           marker=m, capsize=3, label=lab if j == 0 else None)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].axhline(0.03, color="k", lw=0.6, ls="--")
    ax[1].text(2.45, 0.031, "gate 0.03", fontsize=7, ha="right", va="bottom")
    ax[1].set(xticks=range(3), xticklabels=dets, ylabel="ΔAUC (T+A − T)",
              title="Gain over elapsed time (97.5% cluster bootstrap)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, P / "fig2_auc_comparison")


def fig_incidents(R: dict, P: Path) -> list:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for a, d in zip(ax, ["FRCNN", "SDP"]):
        I = R["trackers"]["V3"]["incidents"][d]
        taus = [0.2, 0.5, 1.0]
        for fs, (lab, c, m) in FS.items():
            a.plot(taus, [I[f"tau_{t}"][fs] for t in taus], color=c, marker=m, label=lab)
        a.axhline(I["RETROSPECTIVE_min_a_loc_auc"], color="k", ls="--", lw=0.8, label="RETROSPECTIVE min a_loc")
        a.axhline(0.5, color="k", lw=0.5, ls=":")
        ns = ", ".join(str(I[f"tau_{t}"]["n"]) for t in taus)
        a.set(xlabel="elapsed time into the gap τ (s)", title=f"V3 {d} incidents (n = {ns})", ylim=(0.3, 1.0))
    ax[0].set_ylabel("incident ROC-AUC (TAKEOVER vs GAP)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, P / "fig3_incident_delay", )


def pick_examples(split: str) -> dict:
    """Prereg §10 rule on V3 FRCNN incidents."""
    out = Path(f"results/appearance/{split}")
    fr = read(out / "frames_V3.csv.gz")
    inc = [r for r in read(out / "incidents_V3.csv.gz") if r["unit"].endswith("FRCNN")]
    key = {(r["unit"], int(r["track"]), int(r["frame"])): r for r in fr}
    cells: dict[str, list] = {"TP": [], "TN": [], "FN": [], "FP": []}
    for i in inc:
        if i["label"] not in ("GAP", "TAKEOVER") or int(i["length"]) < 10:
            continue
        u, t, s0, n = i["unit"], int(i["track"]), int(i["start"]), int(i["length"])
        row = next((key[(u, t, f)] for f in range(s0, s0 + n) if float(key[(u, t, f)]["dt"]) >= 0.5 - 1e-9), None)
        if row is None:
            continue
        flag = float(row["s_T+A"]) >= float(row["th_T+A"])
        c = ("TP" if flag else "FN") if i["label"] == "TAKEOVER" else ("FP" if flag else "TN")
        cells[c].append({"unit": u, "track": t, "start": s0, "length": n, "decision_frame": int(row["frame"]),
                         "score": float(row["s_T+A"]), "theta": float(row["th_T+A"]), "a_loc": float(row["a_loc"]),
                         "dt": float(row["dt"]), "label": i["label"]})
    chosen = {}
    for c, lst in cells.items():
        if not lst:
            chosen[c] = None
            continue
        lst.sort(key=lambda e: (e["length"], e["unit"], e["track"], e["start"]))
        med = sorted(e["length"] for e in lst)[(len(lst) - 1) // 2]
        chosen[c] = {**next(e for e in sorted(lst, key=lambda e: (e["unit"], e["track"], e["start"]))
                            if e["length"] == med), "n_in_cell": len(lst)}
    return chosen


def fig_examples(split: str, chosen: dict, P: Path) -> list:
    import cv2
    heavy = Path(f"D:/TAORG_runs/appearance/{split}")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for col, c in enumerate(["TP", "TN", "FN", "FP"]):
        e = chosen.get(c)
        for r in range(2):
            axes[r, col].axis("off")
        if e is None:
            axes[0, col].set_title(f"{c}: no incident")
            continue
        seq, det = e["unit"].split("-")
        info, dets = load_detections(Z, seq, det)
        outs = read(heavy / e["unit"] / "V3_outputs.csv.gz")
        last = max((int(o["frame"]), int(o["det"])) for o in outs
                   if int(o["track"]) == e["track"] and int(o["frame"]) < e["start"])
        lb = dets[last[0]][last[1], :4]
        h = lb[3] - lb[1]
        cx, cy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
        R = h * (1 + 2 * e["dt"])
        x0, x1 = int(max(0, cx - R - h)), int(min(info.width, cx + R + h))
        y0, y1 = int(max(0, cy - R - h)), int(min(info.height, cy + R + h))
        for r, f in enumerate([last[0], e["decision_frame"]]):
            img = cv2.imread(str(FRAMES / f"MOT17-{seq}" / "img1" / f"{f:06d}.jpg"))[:, :, ::-1]
            a = axes[r, col]
            a.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
            a.add_patch(plt.Rectangle((lb[0], lb[1]), lb[2] - lb[0], h, fill=False, ec="#D55E00", lw=1.5,
                                      ls="-" if r == 0 else "--"))
            if r == 1:
                a.add_patch(plt.Circle((cx, cy), R, fill=False, ec="#F0E442", lw=1.2, ls=":"))
                for d_ in dets[f]:
                    if d_[4] > 0.5:
                        a.add_patch(plt.Rectangle((d_[0], d_[1]), d_[2] - d_[0], d_[3] - d_[1], fill=False,
                                                  ec="#56B4E9", lw=0.8))
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)
            a.axis("off")
            if r == 0:
                a.set_title(f"{c} ({e['label']}; {e['n_in_cell']} in cell)\n{e['unit']} track {e['track']}, "
                            f"last seen f{f}", fontsize=8)
            else:
                a.set_title(f"decision f{f} (+{e['dt']:.2f} s): a_loc={e['a_loc']:.2f}\n"
                            f"P={e['score']:.2f} vs θ={e['theta']:.2f}", fontsize=8)
    fig.suptitle("Predefined examples (V3 FRCNN, T+A at τ = 0.5 s). Orange: track's last box; yellow: a_loc gate; "
                 f"blue: candidates (conf > 0.5).  {ATTR}", fontsize=8)
    fig.tight_layout()
    return _save(fig, P / "fig4_examples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--no-examples", action="store_true", help="the prereg example rule is development-only")
    a = ap.parse_args()
    _setup()
    out = Path(f"results/appearance/{a.split}")
    P = Path(f"results/appearance/plots_{a.split}")
    P.mkdir(parents=True, exist_ok=True)
    R = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    files = fig_coverage(R, P) + fig_comparison(R, P) + fig_incidents(R, P)
    if a.no_examples:
        print("\n".join(str(f) for f in files))
        return
    chosen = pick_examples(a.split)
    (out / "examples.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    files += fig_examples(a.split, chosen, P)
    print("\n".join(str(f) for f in files))


if __name__ == "__main__":
    main()
