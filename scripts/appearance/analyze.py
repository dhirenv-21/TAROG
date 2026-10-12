"""Appearance audit, ANALYSIS stage: oracle labels, registered predictors and the gate/decision.

Development (prereg §7):   python scripts/appearance/analyze.py --split dev
Evaluation (prereg §8; only after the freeze commit; uses the frozen models):
                           python scripts/appearance/analyze.py --split eval --frozen results/appearance/freeze/frozen_models.pkl

Inputs: run-stage artifacts in D:/TAORG_runs/appearance/<split>/ (checked against run_records.json).
Outputs: results/appearance/<split>/{analysis.json, frames.csv.gz, incidents.csv.gz, tracking.json}
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from taorg.adapters.mot17 import FrameTracks, load_detections
from taorg.experiments.appearance import stats as S
from taorg.experiments.appearance.labels import common_unit, frame_label, incidents, purity
from taorg.experiments.observation.oracle import identity_of, load_gt, match
from taorg.utils.seed import set_seed

Z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
SPLITS = {"dev": ["02", "09", "11"], "eval": ["04", "05", "10", "13"]}
DETS = ["FRCNN", "SDP"]
TRACKERS = ["V1", "V2", "V3"]
PRIMARY = "V3"
TAUS = [0.2, 0.5, 1.0]
SEED = 20261012
B = 2000


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv_gz(p: Path) -> list[dict]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v: str) -> float:
    return float("nan") if v in ("", "nan") else float(v)


def load_unit(heavy: Path, unit: str, seq: str, det: str, tracker: str, rec: dict):
    """Rebuild FrameTracks (detection boxes) and cue rows of one run; verify checksums."""
    ud = heavy / unit
    for kind in ("outputs", "cues"):
        if sha256(ud / f"{tracker}_{kind}.csv.gz") != rec["runs"][tracker][f"{kind}_sha256"]:
            raise SystemExit(f"checksum mismatch {unit} {tracker} {kind}")
    info, dets = load_detections(Z, seq, det)
    outs: dict[int, dict] = {f: {} for f in range(1, info.length + 1)}
    for r in read_csv_gz(ud / f"{tracker}_outputs.csv.gz"):
        f, t, j = int(r["frame"]), int(r["track"]), int(r["det"])
        outs[f][t] = (tuple(float(v) for v in dets[f][j, :4]), float(dets[f][j, 4]))
    frames = [FrameTracks(f, outs[f], {}) for f in range(1, info.length + 1)]
    cues = []
    for r in read_csv_gz(ud / f"{tracker}_cues.csv.gz"):
        cues.append({"unit": unit, "seq": seq, "det": det, "track": int(r["track"]), "frame": int(r["frame"]),
                     "last_frame": int(r["last_frame"]), "dt": float(r["dt"]), "a_glob": fnum(r["a_glob"]),
                     "a_loc": fnum(r["a_loc"]), "n_cand": int(r["n_cand"]), "n_loc": int(r["n_loc"])})
    return info, frames, cues


def tracking_metrics(seq: str, frames: list[FrameTracks]) -> dict:
    """HOTA / IDF1 / MOTA / IDSW with trackers.eval (TrackEval parity, MOT17 distractors)."""
    from trackers.eval import evaluate_mot_sequence
    with tempfile.TemporaryDirectory() as td:
        gtp, trp = Path(td) / "gt.txt", Path(td) / "trk.txt"
        with zipfile.ZipFile(Z) as z:
            gtp.write_bytes(z.read(f"train/MOT17-{seq}-FRCNN/gt/gt.txt"))
        with open(trp, "w", encoding="utf-8") as f:
            for fr in frames:
                for t, (b, c) in sorted(fr.outputs.items()):
                    f.write(f"{fr.frame},{t + 1},{b[0]:.2f},{b[1]:.2f},{b[2] - b[0]:.2f},{b[3] - b[1]:.2f},{c:.4f},-1,-1,-1\n")
        r = evaluate_mot_sequence(gtp, trp, metrics=["CLEAR", "HOTA", "Identity"])
    return {"HOTA": float(r.HOTA.HOTA), "IDF1": float(r.Identity.IDF1), "MOTA": float(r.CLEAR.MOTA),
            "IDSW": int(r.CLEAR.IDSW)}


def label_rows(frames, cues, gt) -> tuple[list[dict], dict]:
    matches = match(frames, gt)
    g = identity_of(matches)
    pur = purity(matches)
    for r in cues:
        r["label"] = frame_label(r["track"], r["frame"], g, pur, matches, gt)
    return cues, common_unit(matches, gt)


def summarise_predictors(rows, y, s: dict, th: dict, prim: np.ndarray, rng, level_primary: float) -> dict:
    """Frame-level metrics for T, A, T+A on primary rows, per detector, with bootstrap Δ intervals."""
    out: dict = {}
    for det in DETS + ["pooled"]:
        m = prim & (np.array([r["det"] for r in rows]) == det if det != "pooled" else prim)
        ii = np.where(m)[0]
        yy = y[ii]
        d: dict = {"n_frames": int(len(ii)), "n_takeover": int(yy.sum()), "n_gap": int((yy == 0).sum()),
                   "n_tracks": len({(rows[i]["unit"], rows[i]["track"]) for i in ii})}
        for fs in S.FEATURES:
            ss = s[fs][ii]
            d[fs] = {"auc": S.auc(yy, ss), "ap": S.average_precision(yy, ss), "brier": S.brier(yy, ss),
                     **S.operating_point(yy, ss, th[fs][ii])}
        cl = S.cluster_index([rows[i]["unit"] for i in ii], [rows[i]["track"] for i in ii])
        for a, b in (("T+A", "T"), ("A", "T")):
            draws = S.bootstrap_delta_auc(yy, s[a][ii], s[b][ii], cl, rng, B)
            lev = level_primary if (a, b) == ("T+A", "T") else 0.95
            lo, hi = S.interval(draws, lev)
            d[f"delta_{a}_minus_{b}"] = {"estimate": d[a]["auc"] - d[b]["auc"], "level": lev, "lo": lo, "hi": hi}
        cov = np.array([not math.isnan(rows[i]["a_loc"]) for i in ii])
        d["coverage_a_loc"] = float(cov.mean()) if len(cov) else float("nan")
        d["coverage_a_glob"] = float(np.mean([not math.isnan(rows[i]["a_glob"]) for i in ii])) if len(ii) else float("nan")
        for name, mm in (("covered", cov), ("uncovered", ~cov)):
            d[f"auc_{name}"] = {fs: S.auc(yy[mm], s[fs][ii][mm]) for fs in S.FEATURES}
            d[f"n_{name}"] = int(mm.sum())
        out[det] = d
    return out


def incident_summary(rows, incs, s: dict) -> dict:
    """Incident-level AUC at elapsed tau (scores at the first frame with dt >= tau); RETROSPECTIVE a_loc min."""
    out: dict = {"counts": dict(Counter(i["label"] for i in incs))}
    for det in DETS + ["pooled"]:
        sel = [k for k, inc in enumerate(incs) if det == "pooled" or rows[inc["rows"][0]]["det"] == det]
        d: dict = {}
        for tau in TAUS:
            pairs = [(k, i) for k, i in S.incident_at([incs[k] for k in sel], rows, tau)]
            ks = [sel[k] for k, _ in pairs]
            idx = np.array([i for _, i in pairs], int)
            yy = np.array([incs[k]["label"] == "TAKEOVER" for k in ks], int)
            e = {"n": int(len(idx)), "n_takeover": int(yy.sum())}
            for fs in S.FEATURES:
                e[fs] = S.auc(yy, s[fs][idx]) if len(idx) else float("nan")
            e["delta_T+A_minus_T"] = e["T+A"] - e["T"]
            d[f"tau_{tau}"] = e
        lab = [k for k in sel if incs[k]["label"] in ("GAP", "TAKEOVER")]
        yy = np.array([incs[k]["label"] == "TAKEOVER" for k in lab], int)
        retro = np.array([-np.nanmin([rows[i]["a_loc"] for i in incs[k]["rows"]] + [1.0]) for k in lab])
        d["RETROSPECTIVE_min_a_loc_auc"] = S.auc(yy, retro)
        d["delay_first_tau_A_auc_ge_0.70"] = next((t for t in TAUS if d[f"tau_{t}"]["A"] >= 0.70), None)
        out[det] = d
    return out


def coverage_by_dt(rows, prim) -> list[dict]:
    bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.01]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ii = [i for i in np.where(prim)[0] if lo < rows[i]["dt"] <= hi]
        out.append({"lo": lo, "hi": hi, "n": len(ii),
                    "a_loc_available": float(np.mean([not math.isnan(rows[i]["a_loc"]) for i in ii])) if ii else None,
                    "takeover_share": float(np.mean([rows[i]["label"] == "TAKEOVER" for i in ii])) if ii else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--frozen", type=Path, default=None)
    a = ap.parse_args()
    if a.split == "eval" and a.frozen is None:
        raise SystemExit("evaluation requires the frozen models (prereg §9)")
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    heavy = Path(f"D:/TAORG_runs/appearance/{a.split}")
    out = Path(f"results/appearance/{a.split}")
    rec = json.loads((out / "run_records.json").read_text(encoding="utf-8"))
    seqs = SPLITS[a.split]
    expected = {f"{q}-{d}" for q in seqs for d in DETS}
    got = set(rec["units"])
    completeness = {"expected_units": len(expected) * len(TRACKERS),
                    "actual": sum(len(rec["units"][u]["runs"]) for u in got & expected),
                    "missing": sorted(expected - got), "unexpected": sorted(got - expected),
                    "failed": [u for u in got if set(rec["units"][u]["runs"]) != set(TRACKERS)]}
    if completeness["missing"] or completeness["failed"]:
        raise SystemExit(f"incomplete runs: {completeness}")
    frozen = pickle.loads(a.frozen.read_bytes()) if a.frozen else None
    result: dict = {"split": a.split, "completeness": completeness, "trackers": {}}
    all_rows: dict[str, list] = {t: [] for t in TRACKERS}
    tracking: dict = {t: {} for t in TRACKERS}
    for q in seqs:
        for det in DETS:
            unit = f"{q}-{det}"
            for t in TRACKERS:
                info, frames, cues = load_unit(heavy, unit, q, det, t, rec["units"][unit])
                gt = load_gt(Z, q, info.length)
                cues, cu = label_rows(frames, cues, gt)
                all_rows[t] += cues
                tracking[t][unit] = {**tracking_metrics(q, frames), "common_unit": cu,
                                     "missing_record_labels": dict(Counter(r["label"] for r in cues))}
    models_out = {}
    for t in TRACKERS:
        rows = all_rows[t]
        lab = np.array([r["label"] for r in rows])
        prim = np.isin(lab, ["GAP", "TAKEOVER"])
        y = (lab == "TAKEOVER").astype(int)
        s, th = {}, {}
        for fs in S.FEATURES:
            if frozen is None:
                s[fs], th[fs] = S.loso_scores(rows, y, seqs, fs, fit_mask=prim)
                pr = [rows[i] for i in np.where(prim)[0]]
                models_out[(t, fs)] = S.fit(pr, y[prim], fs)       # all-development fit (for the freeze)
            else:
                m, theta = frozen["models"][(t, fs)]
                s[fs] = S.score(m, rows, fs)
                th[fs] = np.full(len(rows), theta)
        incs = incidents(rows)
        res = {"label_counts": dict(Counter(lab.tolist())),
               "frames": summarise_predictors(rows, y, s, th, prim, rng, 0.975),
               "incidents": incident_summary(rows, incs, s),
               "coverage_by_dt": coverage_by_dt(rows, prim),
               "per_sequence": {f"{q}-{det}": {fs: S.auc(y[m_], s[fs][m_]) for fs in S.FEATURES}
                                for q in seqs for det in DETS
                                for m_ in [prim & (np.array([r["unit"] for r in rows]) == f"{q}-{det}")]}}
        result["trackers"][t] = res
        with gzip.open(out / f"frames_{t}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["unit", "track", "frame", "dt", "a_glob", "a_loc", "n_loc", "label"] + [f"s_{k}" for k in S.FEATURES] + [f"th_{k}" for k in S.FEATURES])
            for i, r in enumerate(rows):
                w.writerow([r["unit"], r["track"], r["frame"], f"{r['dt']:.4f}", f"{r['a_glob']:.5f}", f"{r['a_loc']:.5f}",
                            r["n_loc"], r["label"]] + [f"{s[k][i]:.5f}" for k in S.FEATURES] + [f"{th[k][i]:.5f}" for k in S.FEATURES])
        with gzip.open(out / f"incidents_{t}.csv.gz", "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["unit", "track", "start", "length", "label"])
            for inc in incs:
                w.writerow([inc["unit"], inc["track"], inc["start"], inc["length"], inc["label"]])
    result["tracking"] = tracking
    P = result["trackers"][PRIMARY]["frames"]
    if a.split == "dev":
        g1 = {d: bool(P[d]["delta_T+A_minus_T"]["estimate"] >= 0.03 and P[d]["delta_T+A_minus_T"]["lo"] > 0) for d in DETS}
        g2 = bool(P["pooled"]["coverage_a_loc"] >= 0.50)
        result["gate"] = {"G1_per_detector": g1, "G2_coverage": g2,
                          "PASS": bool(all(g1.values()) and g2)}
        if result["gate"]["PASS"]:
            (out / "candidate_models.pkl").write_bytes(pickle.dumps({"models": models_out}))
    else:
        inc = result["trackers"][PRIMARY]["incidents"]
        go, narrow = {}, {}
        for d in DETS:
            dd = P[d]
            go[d] = bool(dd["delta_T+A_minus_T"]["estimate"] >= 0.05 and dd["delta_T+A_minus_T"]["lo"] > 0
                         and dd["T+A"]["takeover_recall"] >= 0.50 and dd["T+A"]["gap_rejection"] <= 0.15
                         and inc[d]["tau_0.5"]["A"] >= 0.70 and dd["coverage_a_loc"] >= 0.80)
            narrow[d] = bool(dd["delta_T+A_minus_T"]["estimate"] >= 0.03 and dd["delta_T+A_minus_T"]["lo"] > 0)
        result["decision"] = {"GO_per_detector": go, "NARROW_per_detector": narrow,
                              "decision": "GO" if all(go.values()) else "NARROW" if all(narrow.values()) else "STOP"}
    (out / "analysis.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("gate", "decision") if k in result}, indent=1))
    for t in TRACKERS:
        F = result["trackers"][t]["frames"]
        print(t, {d: (F[d]["n_frames"], round(F[d]["T"]["auc"], 3), round(F[d]["A"]["auc"], 3), round(F[d]["T+A"]["auc"], 3),
                      [round(F[d]["delta_T+A_minus_T"][k], 3) for k in ("estimate", "lo", "hi")], round(F[d]["coverage_a_loc"], 3))
                  for d in DETS + ["pooled"]})


if __name__ == "__main__":
    main()
