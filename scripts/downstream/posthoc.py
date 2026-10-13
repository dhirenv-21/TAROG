"""POST-HOC / EXPLORATORY diagnostics (not registered; cannot change the decision).

D1  relative harm: relative PGG loss; ghost-seconds removed per added false-termination second.
D2  event audit: are scored terminations / recurrences genuine relation changes or matching flicker?
    A scored termination at b (both endpoints output at b+1) is GENUINE iff the annotated relation
    between the persons the endpoints represented at b is inactive at b+1; otherwise it ended only
    because an endpoint's IoU match changed (FLICKER). A recurrence start a is GENUINE iff that
    relation was inactive at a-1. STD-g and RDD are recomputed on genuine events only, with the same
    paired video bootstrap (95%).
    python scripts/downstream/posthoc.py --split eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import active_relations, sample
from taorg.experiments.downstream.runner import METHODS, load_video_data, make_method, run_method
from taorg.experiments.observation.oracle import identity_of
from taorg.metrics.episodes import runs

SPLIT = {"dev": "training", "eval": "validation"}
MIN_DEAD, SUSTAINED = 10, 15


def ident(d, g, x, f):
    if x in d.frames[f - 1].outputs:
        return d.stream.matches[f].get(x)
    return g.get(x)


def video_diag(d, fz) -> dict:
    g = identity_of(d.stream.matches)
    outs = {m: run_method(make_method(m.split("+")[0], fz["params"][m.split("+")[0]],
                                      d.flags if m.endswith("+F") else None), d) for m in METHODS}
    fps = d.video.fps
    res = {m: {"std_sum": 0.0, "std_n": 0, "rdd_sum": 0.0, "rdd_n": 0} for m in METHODS}
    cnt = {"term_genuine": 0, "term_flicker": 0, "rec_genuine": 0, "rec_flicker": 0}
    for k in d.stream.keys:
        L = d.labels[k]
        p = d.stream.pred_names[k.predicate_id]
        alive = L["alive"][1:].tolist()
        n = len(alive)
        ep = runs(alive)
        for idx, (a, b) in enumerate(ep):
            # termination (registered STD-g eligibility), frames are 0-based offsets of internal f = i + 1
            if b < n - 1:
                dlast = (ep[idx + 1][0] - 1) if idx + 1 < len(ep) else n - 1
                D = dlast - b
                if D >= MIN_DEAD and L["both_out"][b + 2]:
                    i, j = ident(d, g, k.subject_track_id, b + 1), ident(d, g, k.object_track_id, b + 1)
                    genuine = (i, j, p) not in active_relations(d.video, b + 1)
                    cnt["term_genuine" if genuine else "term_flicker"] += 1
                    if genuine:
                        for m in METHODS:
                            o = outs[m][k][1:]
                            ts = next((f for f in range(b + 1, dlast + 1)
                                       if not o[f:min(f + SUSTAINED - 1, dlast) + 1].any()), None)
                            res[m]["std_sum"] += ((D + 1) if ts is None else (ts - b)) / fps
                            res[m]["std_n"] += 1
            if idx >= 1:
                i, j = ident(d, g, k.subject_track_id, a + 1), ident(d, g, k.object_track_id, a + 1)
                genuine = a >= 1 and (i, j, p) not in active_relations(d.video, a - 1)
                cnt["rec_genuine" if genuine else "rec_flicker"] += 1
                if genuine:
                    for m in METHODS:
                        o = outs[m][k][1:]
                        c = ep[idx - 1][1] + 1
                        if a > c and o[c:a].all():
                            continue                                   # bridged
                        first = next((f for f in range(a, b + 1) if o[f]), None)
                        res[m]["rdd_sum"] += ((b - a + 1) if first is None else (first - a)) / fps
                        res[m]["rdd_n"] += 1
    return {"methods": res, "counts": cnt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    a = ap.parse_args()
    fz = json.loads(Path("results/downstream/freeze/frozen.json").read_text(encoding="utf-8"))
    R = json.loads(Path(f"results/downstream/{a.split}/analysis.json").read_text(encoding="utf-8"))
    per = []
    for vp in sample(Path("data/vidor"), SPLIT[a.split]):
        vid = Path(vp).stem
        d = load_video_data(Path(f"D:/TAORG_runs/downstream/{a.split}") / vid,
                            Path(f"data/vidor/{SPLIT[a.split]}") / vp.replace(".mp4", ".json"))
        per.append(video_diag(d, fz))
        print(vid, per[-1]["counts"], flush=True)
    cnt = {k: sum(p["counts"][k] for p in per) for k in per[0]["counts"]}
    rng = np.random.default_rng(20261016)

    def pooled(ii, m, f):
        s = sum(per[i]["methods"][m][f + "_sum"] for i in ii)
        n = sum(per[i]["methods"][m][f + "_n"] for i in ii)
        return s / n if n else float("nan")

    allv = list(range(len(per)))
    boots = [rng.integers(0, len(per), len(per)) for _ in range(2000)]
    gen = {}
    for f in ("std", "rdd"):
        gen[f] = {m: pooled(allv, m, f) for m in METHODS}
        for c, m, b in (("C1", "FB+F", "FB"), ("C2", "B2+F", "B2")):
            dd = np.array([pooled(ii, m, f) - pooled(ii, b, f) for ii in boots])
            dd = dd[np.isfinite(dd)]
            gen[f][f"{c}_delta"] = pooled(allv, m, f) - pooled(allv, b, f)
            gen[f][f"{c}_ci95"] = [float(np.quantile(dd, 0.025)), float(np.quantile(dd, 0.975))]
    pt = R["point"]
    d1 = {c: {"relative_PGG_loss": 1 - pt[m]["PGG"] / pt[b]["PGG"],
              "ghost_s_removed": pt[b]["IGT_s"] - pt[m]["IGT_s"], "ftt_s_added": pt[m]["FTT_s"] - pt[b]["FTT_s"],
              "ghost_s_removed_per_ftt_s_added": (pt[b]["IGT_s"] - pt[m]["IGT_s"]) / (pt[m]["FTT_s"] - pt[b]["FTT_s"])}
          for c, m, b in (("C1", "FB+F", "FB"), ("C2", "B2+F", "B2"))}
    out = {"note": "POST-HOC / EXPLORATORY; not registered; cannot change the decision", "split": a.split,
           "D1_relative_harm": d1, "D2_event_counts": cnt, "D2_genuine_only": gen}
    Path(f"results/downstream/{a.split}/posthoc.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
