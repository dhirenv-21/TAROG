"""EXPLORATORY diagnosis of the failed final development gate (development data only; cannot change the
registered STOP). Why did EGTR yield almost no person-person predictions at the registered thresholds?

1. EGTR person queries per frame, and the share of tracker outputs matched to an EGTR person (IoU >= 0.5)
2. distribution of mapped-predicate scores over matched tracker pairs
3. ranking signal: AUC of the EGTR score for "annotated relation active" among matched pairs, per predicate
4. NM frame F1 at score quantile thresholds (what a quantile-based tau grid would have given)
    python scripts/final/diagnose_dev.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from taorg.adapters.egtr import MAPPED
from taorg.adapters.vidor import active_relations, sample
from taorg.experiments.appearance.stats import auc
from taorg.experiments.downstream.observations import gt_dict
from taorg.experiments.final.runner import load_final_video, outcomes, prepare, run_outputs
from taorg.experiments.final.stream import assign
from taorg.experiments.observation.oracle import match


def main() -> None:
    vids = [(Path(vp).stem, vp) for vp in sample(Path("data/vidor"), "training")]
    data = [load_final_video(Path("D:/TAORG_runs/downstream/dev") / v, Path("D:/TAORG_runs/egtr/sgg/dev") / f"{v}.npz",
                             Path("data/vidor/training") / vp.replace(".mp4", ".json")) for v, vp in vids]
    n_out = n_matched = n_frames = 0
    persons = []
    scores = {m: [] for m in MAPPED}
    labels = {m: [] for m in MAPPED}
    for d in data:
        mt = match(d.frames, gt_dict(d.video))
        for f in d.sampled:
            boxes, _, sc = d.sgg[f]
            outs = d.frames[f - 1].outputs
            q2t = assign(boxes, outs)
            persons.append(len(boxes))
            n_out += len(outs)
            n_matched += len(q2t)
            n_frames += 1
            act = active_relations(d.video, f - 1)
            for qa, x in q2t.items():
                for qb, y in q2t.items():
                    if x == y:
                        continue
                    gi, gj = mt[f].get(x), mt[f].get(y)
                    for mi, m in enumerate(MAPPED):
                        scores[m].append(float(sc[qa, qb, mi]))
                        labels[m].append(int(gi is not None and gj is not None and (gi, gj, m) in act))
    res = {"note": "EXPLORATORY; development data only; cannot change the registered STOP",
           "frames": n_frames, "egtr_person_queries_per_frame": float(np.mean(persons)),
           "tracker_outputs_per_frame": n_out / n_frames, "share_tracker_outputs_matched_to_egtr": n_matched / max(1, n_out),
           "per_predicate": {}}
    allsc = np.concatenate([np.asarray(scores[m]) for m in MAPPED])
    for m in MAPPED:
        s, y = np.asarray(scores[m]), np.asarray(labels[m])
        res["per_predicate"][m] = {"pairs": int(len(s)), "positives": int(y.sum()),
                                   "score_quantiles": {q: float(np.quantile(s, q)) for q in (0.5, 0.9, 0.99, 0.999)} if len(s) else {},
                                   "auc_active_vs_inactive": auc(y, s) if len(s) else float("nan"),
                                   "mean_score_active": float(s[y == 1].mean()) if y.sum() else float("nan"),
                                   "mean_score_inactive": float(s[y == 0].mean()) if (y == 0).any() else float("nan")}
    res["quantile_tau_nm_f1"] = {}
    for q in (0.9, 0.99, 0.999):
        tau = float(np.quantile(allsc, q))
        f1 = []
        for d in data:
            p = prepare(d, tau)
            f1.append(outcomes(p, run_outputs(p, "NM", {})[0])["frame_f1"])
        res["quantile_tau_nm_f1"][str(q)] = {"tau": tau, "mean_NM_frame_f1": float(np.nanmean(f1))}
        print(q, res["quantile_tau_nm_f1"][str(q)], flush=True)
    Path("results/final/dev/diagnosis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "quantile_tau_nm_f1"}, indent=1, default=float))


if __name__ == "__main__":
    main()
