"""EXPLORATORY (not registered; cannot change the decision): is the appearance gain explained by
mere proximity of a detection to X's last observed position?

Geometric analogue of a_loc without appearance:
    g_loc = min over candidates (conf > 0.5) of centre distance / (h * (1 + 2 dt))   (1.0 if none)
Models (same logistic family): T+G, T+A (registered), T+G+A.
Development: leave-one-sequence-out. Held-back: fitted on all development frames (no tuning).
Output: results/appearance/exploratory_proximity.json
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from taorg.adapters.mot17 import load_detections
from taorg.experiments.appearance import stats as S

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
SPLITS = {"dev": ["02", "09", "11"], "eval": ["04", "05", "10", "13"]}
S.FEATURES["G"] = ["g_loc"]
S.FEATURES["T+G"] = S.FEATURES["T"] + ["g_loc"]
S.FEATURES["T+G+A"] = S.FEATURES["T+A"] + ["g_loc"]
_orig = S.feature_matrix


def feature_matrix(rows, names):
    base = [n for n in names if n != "g_loc"]
    X = _orig(rows, base) if base else np.zeros((len(rows), 0))
    if "g_loc" in names:
        X = np.column_stack([X, [r["g_loc"] for r in rows]])
    return X


S.feature_matrix = feature_matrix


def read(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for(split: str) -> list[dict]:
    fr = read(Path(f"results/appearance/{split}/frames_V3.csv.gz"))
    heavy = Path(f"D:/TAORG_runs/appearance/{split}")
    cache: dict = {}
    out = []
    for r in fr:
        if r["label"] not in ("GAP", "TAKEOVER"):
            continue
        u = r["unit"]
        if u not in cache:
            seq, det = u.split("-")
            info, dets = load_detections(Z, seq, det)
            last: dict = {}
            hist: dict = {}
            for o in read(heavy / u / "V3_outputs.csv.gz"):
                hist.setdefault(int(o["track"]), []).append((int(o["frame"]), int(o["det"])))
            cache[u] = (dets, hist)
        dets, hist = cache[u]
        t, f, dt = int(r["track"]), int(r["frame"]), float(r["dt"])
        lf, j = max(x for x in hist[t] if x[0] < f)
        lb = dets[lf][j, :4]
        h = lb[3] - lb[1]
        c0 = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
        d = dets[f][dets[f][:, 4] > 0.5]
        g = 1.0
        if len(d):
            cc = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], axis=1)
            g = float(min(1.0, (np.linalg.norm(cc - c0, axis=1) / (h * (1 + 2 * dt))).min()))
        a_glob = float(r["a_glob"])
        out.append({"unit": u, "seq": u.split("-")[0], "det": u.split("-")[1], "track": t, "dt": dt, "a_glob": a_glob,
                    "a_loc": float(r["a_loc"]), "g_loc": g, "y": int(r["label"] == "TAKEOVER")})
    return out


def main() -> None:
    res: dict = {"note": "EXPLORATORY; not registered; cannot change the decision"}
    dev = rows_for("dev")
    y = np.array([r["y"] for r in dev])
    sets = ["T", "G", "T+G", "T+A", "T+G+A"]
    sc = {fs: S.loso_scores(dev, y, SPLITS["dev"], fs)[0] for fs in sets}
    res["dev_loso_auc"] = {d: {fs: S.auc(y[m], sc[fs][m]) for fs in sets}
                           for d in ("FRCNN", "SDP") for m in [np.array([r["det"] == d for r in dev])]}
    ev_path = Path("results/appearance/eval/frames_V3.csv.gz")
    if ev_path.exists():
        ev = rows_for("eval")
        ye = np.array([r["y"] for r in ev])
        res["eval_dev_fit_auc"] = {}
        for d in ("FRCNN", "SDP"):
            m = np.array([r["det"] == d for r in ev])
            res["eval_dev_fit_auc"][d] = {fs: S.auc(ye[m], S.score(S.fit(dev, y, fs)[0], [ev[i] for i in np.where(m)[0]], fs))
                                          for fs in sets}
    Path("results/appearance/exploratory_proximity.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
