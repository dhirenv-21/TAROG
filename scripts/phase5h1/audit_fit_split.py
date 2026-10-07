"""Phase 5H-1 fitting-split data audit (Stage 0 FIT videos ONLY; no model, no effect estimates).

Writes results/phase5h1/audit/: predicate_support.csv, transition_support.csv, audit.json,
fig_support.(png|pdf), fig_durations.(png|pdf).

Usage:
    python scripts/phase5h1/audit_fit_split.py
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from taorg.hawkes.families import FAMILIES, family_of  # noqa: E402
from taorg.hawkes.stage0 import load_config, splits, streams_of  # noqa: E402
from taorg.hawkes.vidor import load_videos  # noqa: E402

OUT = Path("results/phase5h1/audit")
LAG_S = 10.0          # transition-support window (support counts only)
TARGET_MIN = 200      # registered support rule (see preregistration §4)
SOURCE_MIN = 50


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(Path("configs/experiments/phase5h_stage0.yaml"))
    train, vocab, _ = load_videos(cfg["data"]["root"], cfg["data"]["train_dir"])
    _, fitv, _ = splits(train, cfg)
    R, K = len(vocab), 2 * len(vocab)
    fit_videos = [train[i] for i in fitv]
    streams = streams_of(fit_videos, K)
    on, off = np.zeros(R, int), np.zeros(R, int)
    vids, pairs = defaultdict(set), defaultdict(set)
    durs = defaultdict(list)
    cens = np.zeros(R, int)
    trans = Counter()                     # (src_type, tgt_type) -> # target events with src event in (0, LAG_S]
    pair_both = Counter()                 # (r_src, r_tgt) -> # pairs where both predicates occur
    for v, sl in zip(fit_videos, streams):
        for (u, w), pr in v.pairs.items():
            for r in pr.rels:
                vids[r].add(v.vid)
                pairs[r].add((v.vid, u, w))
            present = sorted(pr.rels)
            for a in present:
                for b in present:
                    if a != b:
                        pair_both[(a, b)] += 1
        for ps in sl:
            ev = [(t, c) for t, c in ps.events if c < K]
            for t, c in ev:
                (on if c % 2 == 0 else off)[c // 2] += 1
            for r, d, known, obs in ps.ages:
                durs[r].append(d)
                cens[r] += (not known) or (not obs)
            for i, (ti, ci) in enumerate(ev):
                for tj, cj in ev[:i]:
                    if 0 < ti - tj <= LAG_S and cj // 2 != ci // 2:
                        trans[(cj, ci)] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    total = on.sum() + off.sum()
    rows = []
    for r in range(R):
        d = np.array(durs[r]) if durs[r] else np.array([np.nan])
        rows.append({"predicate": vocab[r], "family": family_of(vocab[r]), "onsets": int(on[r]),
                     "offsets": int(off[r]), "intervals": len(durs[r]), "videos": len(vids[r]),
                     "pairs": len(pairs[r]), "event_share": round(float((on[r] + off[r]) / total), 6),
                     "dur_p10_s": round(float(np.nanpercentile(d, 10)), 3),
                     "dur_median_s": round(float(np.nanmedian(d)), 3),
                     "dur_p90_s": round(float(np.nanpercentile(d, 90)), 3),
                     "censored_fraction": round(float(cens[r] / max(1, len(durs[r]))), 4),
                     "onset_target": bool(on[r] >= TARGET_MIN), "offset_target": bool(off[r] >= TARGET_MIN)})
    with open(OUT / "predicate_support.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    name = lambda c: f"{'ON' if c % 2 == 0 else 'OFF'}({vocab[c // 2]})"
    cls = lambda s, t: f"{'ON' if s % 2 == 0 else 'OFF'}->{'ON' if t % 2 == 0 else 'OFF'}"
    trows = []
    for (s, t), n in sorted(trans.items(), key=lambda kv: -kv[1]):
        trows.append({"source": name(s), "target": name(t), "class": cls(s, t),
                      "source_family": family_of(vocab[s // 2]), "target_family": family_of(vocab[t // 2]),
                      "target_events_with_source_within_10s": n,
                      "pairs_with_both_predicates": pair_both[(s // 2, t // 2)]})
    with open(OUT / "transition_support.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(trows[0]))
        wr.writeheader()
        wr.writerows(trows)
    by_class = Counter()
    for r_ in trows:
        by_class[r_["class"]] += r_["target_events_with_source_within_10s"]
    shares = np.sort((on + off) / total)[::-1]
    summ = {
        "split": "Stage 0 FIT only", "n_videos": len(fit_videos),
        "n_pairs": sum(len(v.pairs) for v in fit_videos),
        "n_events": int(total), "n_onsets": int(on.sum()), "n_offsets": int(off.sum()),
        "vocab": vocab, "families": FAMILIES,
        "family_event_counts": {fam: int(sum(on[r] + off[r] for r in range(R) if family_of(vocab[r]) == fam))
                                for fam in FAMILIES},
        "top6_event_share": float(shares[:6].sum()),
        "n_onset_targets": int((on >= TARGET_MIN).sum()), "n_offset_targets": int((off >= TARGET_MIN).sum()),
        "n_source_types": int(((on >= SOURCE_MIN).sum() + (off >= SOURCE_MIN).sum())),
        "transition_support_by_class_10s": dict(by_class),
        "target_min": TARGET_MIN, "source_min": SOURCE_MIN, "lag_s": LAG_S,
        "stage0_hypotheses_support": {
            f"{h[0]} -> {h[1]}": next((x for x in trows if (x["source"], x["target"]) == h), None)
            for h in [("ON(hold)", "OFF(grab)"), ("ON(release)", "OFF(hold)"),
                      ("ON(get_off)", "OFF(ride)"), ("ON(beneath)", "OFF(above)")]},
    }
    (OUT / "audit.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    # ---- figures ----
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"SPATIAL": "#0072B2", "MOTION": "#E69F00", "ATTENTION_COMMUNICATION": "#009E73",
              "VEHICLE_SUPPORT": "#CC79A7", "CONTACT_MANIPULATION": "#D55E00"}
    order = np.argsort(-(on + off))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(R)
    ax.bar(x, on[order], color=[colors[family_of(vocab[r])] for r in order])
    ax.set_yscale("log")
    ax.axhline(TARGET_MIN, color="k", ls="--", lw=0.8)
    ax.text(R - 1, TARGET_MIN * 1.15, f"target support rule ≥ {TARGET_MIN} onsets", ha="right", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([vocab[r] for r in order], rotation=90, fontsize=6.5)
    ax.set_ylabel("ONSET events (log scale)")
    ax.set_title(f"VidOR fitting split (Stage 0 FIT, {len(fit_videos)} videos): onset events per predicate — "
                 f"top 6 predicates = {100 * shares[:6].sum():.0f}% of all onset+offset events", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f.replace("_", " ").lower()) for f, c in colors.items()],
              frameon=False, fontsize=7, ncol=5, loc="upper right", bbox_to_anchor=(1, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "fig_support.png", dpi=200)
    fig.savefig(OUT / "fig_support.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fams = list(FAMILIES)
    data = [np.concatenate([durs[r] for r in range(R) if family_of(vocab[r]) == f]) for f in fams]
    ax.boxplot(data, vert=False, whis=(10, 90), showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#dddddd"})
    ax.set_xscale("log")
    ax.set_yticks(range(1, len(fams) + 1))
    ax.set_yticklabels([f"{f.replace('_', ' ').lower()} (n={len(d)})" for f, d in zip(fams, data)])
    ax.set_xlabel("relation interval duration (s, log scale); box = IQR, whiskers = p10–p90")
    ax.set_title("Relation durations by predicate family (Stage 0 FIT split)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_durations.png", dpi=200)
    fig.savefig(OUT / "fig_durations.pdf")
    plt.close(fig)
    print(json.dumps({k: summ[k] for k in ("n_videos", "n_pairs", "n_events", "family_event_counts",
                                           "top6_event_share", "n_onset_targets", "n_offset_targets",
                                           "transition_support_by_class_10s")}, indent=1))


if __name__ == "__main__":
    main()
