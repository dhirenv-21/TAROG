"""Phase 5H-1 registered execution (research/phase5h1_preregistration.md; amendments A1-A3).

Stages, in registered order:
    prepare   reduce raw VidOR TRAIN/SELECT to cached records; layout and folds from TRAIN only
    tune      fit every rung x ridge on TRAIN, score SELECT, select ridge per rung; fit RQ-H2 variants
    freeze    write selected parameters into the repository (to be committed BEFORE evaluate)
    evaluate  refuses unless the freeze file is committed and unchanged; scores EVAL-A/EVAL-B
    nulls     19 N-circ + 19 N-jit replicates, refit at the frozen ridge on null TRAIN, scored on null EVAL
    stability G4 on 5 TRAIN folds (real) and 5 N-jit fold replicates; registered support counts
    hypotheses model-free H-a..H-d on EVAL with 199 N-jit replicates
    check     expected/actual/missing/duplicate/failed task accounting

Heavy state lives under a run root outside the repository (default D:/TAORG_runs/phase5h1/main).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from taorg.hawkes.glm import (
    RUNGS, Layout, TargetFit, aggregate_target, build_design, fit_target_agg, ll_agg_per_group,
    merge_targets, target_columns, td_arrays, td_from_arrays,
)
from taorg.hawkes.hyp1 import counts as hyp_counts
from taorg.hawkes.nulls1 import jitter_shift
from taorg.hawkes.phase5h1 import build_layout, guard, load_config, split_indices
from taorg.hawkes.runner import Checkpoints, Job, atomic_write_text, code_hash, run_jobs, spec_id
from taorg.hawkes.stage0 import load_config as s0load
from taorg.hawkes.vidor import VideoRecord, circular_shift, load_videos

logger = logging.getLogger(__name__)

CFG = Path("configs/experiments/phase5h1.yaml")
MAIN_RUNGS = ("G0", "G1", "G2", "G3", "G4x", "G4", "G4+")
TUNED_RUNGS = ("G1", "G2", "G3", "G4x", "G4", "G4+")
VARIANTS = tuple(r for r in RUNGS if r.startswith("G3+H:"))
NULL_RUNGS = ("G2", "G3", "G4x", "G4")
FROZEN_DIR = Path("results/phase5h1/full/frozen")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _meta(run: Path) -> dict[str, Any]:
    return json.loads((run / "data" / "meta.json").read_text(encoding="utf-8"))


def _layout(m: dict[str, Any]) -> Layout:
    L = m["layout"]
    return Layout(L["vocab"], L["cats"], L["lag_edges"], L["elapsed_edges"], L["count_bins"], L["active_bins"],
                  L["dur_edges"], L["sources"])


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def ds_name(part: str, null: list | None) -> str:
    return part if not null else f"{part}__{null[0]}{null[1]}"


def load_part(run: Path, part: str) -> list[VideoRecord]:
    """Cached records of a split. EVAL parts need the unlock token written by `evaluate` after the
    freeze check (structural leakage guard, in addition to taorg.hawkes.phase5h1.guard)."""
    if part in ("EVAL", "EVAL_A", "EVAL_B"):
        tok = run / "data" / "EVAL_UNLOCK.json"
        if not tok.exists():
            guard({"EVAL_A", "EVAL_B"} if part == "EVAL" else {part}, allow_eval=False)   # raises LeakageError
        t = json.loads(tok.read_text(encoding="utf-8"))
        if t["frozen_sha256"] != _sha_file(FROZEN_DIR / "frozen.json"):
            raise RuntimeError("EVAL unlock token does not match the committed freeze file")
    if part == "EVAL":
        return load_part(run, "EVAL_A") + load_part(run, "EVAL_B")
    with open(run / "data" / f"{part}.pkl", "rb") as fh:
        return pickle.load(fh)


def n_groups(m: dict[str, Any], part: str) -> int:
    n = m["n_videos"]
    return n["EVAL_A"] + n["EVAL_B"] if part == "EVAL" else (5 if part == "TRAIN" else n[part])


def apply_null(videos: list[VideoRecord], null: list, part: str, cfg: dict, diag: dict) -> list[VideoRecord]:
    kind, seed = null[0], null[1]
    if kind == "circ":                     # registered Stage 0 function; EVAL stream = seed + 100000 (as Stage 0)
        return circular_shift(videos, seed if part == "TRAIN" else seed + 100000)
    if kind == "jit":
        return jitter_shift(videos, seed, part, float(cfg["null_control"]["jitter_s"]), diag)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# worker functions (module level: picklable)
# ---------------------------------------------------------------------------

def build_ds(run: str, part: str, null: list | None, chunk: int) -> dict[str, Any]:
    """Aggregated per-target data of one (possibly null-transformed) split -> run/cache/<name>/t<k>.npz."""
    run = Path(run)
    m = _meta(run)
    lay = _layout(m)
    cfg = load_config(CFG)
    name = ds_name(part, null)
    cdir = run / "cache" / name
    done = cdir / "DONE.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    if cdir.exists():
        shutil.rmtree(cdir)                # partial build from an interrupted attempt: rebuild
    cdir.mkdir(parents=True)
    t0 = time.time()
    videos = load_part(run, part)
    diag: dict[str, int] = {}
    if null:
        videos = apply_null(videos, null, part, cfg, diag)
    targets = m["targets"]
    fold = m["fold_of_vid"] if part == "TRAIN" else None
    n_chunks = 0
    ev_total = Counter()
    for ci, s in enumerate(range(0, len(videos), chunk)):
        vs = videos[s: s + chunk]
        d = build_design(vs, lay, targets)
        gmap = np.array([fold[v.vid] for v in vs]) if fold else np.arange(s, s + len(vs))
        seg_group = gmap[d.seg_video] if len(d.seg_video) else d.seg_video
        for k in targets:
            td = aggregate_target(d, k, seg_group)
            ev_total[k] += int(td.y.sum())
            np.savez(cdir / f"c{ci}_t{k}.npz", **td_arrays(td))
        n_chunks += 1
        del d
    rows = {}
    for k in targets:
        parts = []
        for ci in range(n_chunks):
            with np.load(cdir / f"c{ci}_t{k}.npz") as a:
                parts.append(td_from_arrays(a))
        td = merge_targets(parts, lay.size, reaggregate=(part == "TRAIN"))
        np.savez(cdir / f"t{k}.npz", **td_arrays(td))
        rows[k] = [td.n_rows, int(len(td.y))]
        for ci in range(n_chunks):
            (cdir / f"c{ci}_t{k}.npz").unlink()
    out = {"name": name, "part": part, "null": null, "n_videos": len(videos), "null_diag": diag,
           "events_per_target": {str(k): ev_total[k] for k in targets},
           "rows_raw_agg": {str(k): v for k, v in rows.items()}, "build_s": round(time.time() - t0, 1)}
    atomic_write_text(done, json.dumps(out, indent=1))
    return out


def _load_td(run: Path, name: str, k: int):
    with np.load(run / "cache" / name / f"t{k}.npz") as a:
        return td_from_arrays(a)


def _fit_spec(stage: str, train_ds: str, k: int, s: dict, score: list[str], code: str, m: dict) -> dict:
    return {"stage": stage, "train_ds": train_ds, "k": k, "rung": s["rung"], "ridge": s["ridge"],
            "groups": s.get("groups"), "score": score, "max_iter": m["max_iter"], "config_sha": m["config_sha"],
            "layout_sha": m["layout_sha"], "code": code}


def fit_job(run: str, stage: str, train_ds: str, k: int, specs: list[dict], score: list[str], code: str
            ) -> dict[str, Any]:
    """Fit target k for each spec on train_ds; score per group on each score data set. Checkpointed."""
    run = Path(run)
    m = _meta(run)
    lay = _layout(m)
    ck = Checkpoints(run / "ckpt" / stage)
    td = None
    sc: dict[str, Any] = {}
    n_new = 0
    for s in specs:
        spec = _fit_spec(stage, train_ds, k, s, score, code, m)
        if ck.get(spec):
            continue
        if td is None:
            td = _load_td(run, train_ds, k)
        t0 = time.time()
        fr = fit_target_agg(td, lay, s["rung"], s["ridge"], m["max_iter"],
                            None if s.get("groups") is None else np.array(s["groups"]))
        arr = {"eta": np.array(fr.fit.eta), "w": fr.fit.w, "v": fr.fit.v, "cols": fr.fit.cols}
        for sd in score:
            if sd not in sc:
                sc[sd] = _load_td(run, sd, k)
            part = sd.split("__")[0]
            ll, ev = ll_agg_per_group(sc[sd], fr.fit, lay, n_groups(m, part), "DUR" in RUNGS[s["rung"]])
            arr[f"ll__{sd}"], arr[f"ev__{sd}"] = ll, ev
        ck.put(spec, {"diag": fr.diag, "fit_s": round(time.time() - t0, 3)}, arr)
        n_new += 1
    return {"k": k, "specs": len(specs), "new": n_new}


def frozen_fit(fz: Any, rung: str, k: int) -> TargetFit:
    p = f"{rung}|{k}|"
    return TargetFit(k, fz[p + "cols"], float(fz[p + "eta"]), fz[p + "w"], fz[p + "v"])


def score_job(run: str, k: int, parts: list[str], code: str) -> dict[str, Any]:
    """Held-out per-video log-likelihood of every frozen model for target k (evaluation stage)."""
    run = Path(run)
    m = _meta(run)
    lay = _layout(m)
    ck = Checkpoints(run / "ckpt" / "evaluate")
    fz_sha = _sha_file(FROZEN_DIR / "params.npz")
    fz = np.load(FROZEN_DIR / "params.npz")
    n_new = 0
    tds = {}
    for rung in MAIN_RUNGS + VARIANTS:
        for part in parts:
            spec = {"stage": "evaluate", "k": k, "rung": rung, "part": part, "frozen": fz_sha, "code": code,
                    "layout_sha": m["layout_sha"]}
            if ck.get(spec):
                continue
            if part not in tds:
                tds[part] = _load_td(run, part, k)
            ll, ev = ll_agg_per_group(tds[part], frozen_fit(fz, rung, k), lay, n_groups(m, part), "DUR" in RUNGS[rung])
            ck.put(spec, {}, {"ll": ll, "ev": ev})
            n_new += 1
    return {"k": k, "new": n_new}


def next_event_job(run: str, part: str, chunk: int, code: str) -> dict[str, Any]:
    """Secondary metric: next-event-type log-loss / top-1 among at-risk modelled types on the same pair."""
    run = Path(run)
    m = _meta(run)
    lay = _layout(m)
    ck = Checkpoints(run / "ckpt" / "evaluate")
    fz_sha = _sha_file(FROZEN_DIR / "params.npz")
    spec = {"stage": "evaluate", "metric": "next_event", "part": part, "frozen": fz_sha, "code": code}
    if ck.get(spec):
        return {"skipped": True}
    fz = np.load(FROZEN_DIR / "params.npz")
    videos = load_part(run, part)
    targets = m["targets"]
    nv = len(videos)
    arr = {f"{r}|{q}": np.zeros(nv) for r in MAIN_RUNGS for q in ("logloss", "top1")}
    arr["n"] = np.zeros(nv)
    for s in range(0, nv, chunk):
        vs = videos[s: s + chunk]
        d = build_design(vs, lay, targets)
        ev_seg = np.zeros(len(d.seg_dur), dtype=bool)
        for k in targets:
            ev_seg[d.rows[k][d.y[k] > 0]] = True
        segs = np.flatnonzero(ev_seg)
        pos = -np.ones(len(d.seg_dur), dtype=np.int64)
        pos[segs] = np.arange(len(segs))
        first = True
        for rung in MAIN_RUNGS:
            Z = np.full((len(segs), len(targets)), -np.inf)
            Yk = np.zeros((len(segs), len(targets)))
            for j, k in enumerate(targets):
                rr = d.rows[k]
                sel = pos[rr] >= 0
                if not sel.any():
                    continue
                f = frozen_fit(fz, rung, k)
                z = f.eta + d.X[rr[sel]][:, f.cols] @ f.w
                if "DUR" in RUNGS[rung]:
                    z = z + f.v[d.dur[k][sel]]
                Z[pos[rr[sel]], j] = z
                Yk[pos[rr[sel]], j] = d.y[k][sel]
            mx = Z.max(1, keepdims=True)
            lse = (mx + np.log(np.exp(Z - mx).sum(1, keepdims=True))).ravel()
            am = Z.argmax(1)
            si, tj = np.nonzero(Yk > 0)
            vid = d.seg_video[segs[si]] + s
            np.add.at(arr[f"{rung}|logloss"], vid, -(Z[si, tj] - lse[si]) * Yk[si, tj])
            np.add.at(arr[f"{rung}|top1"], vid, (am[si] == tj) * Yk[si, tj])
            if first:
                np.add.at(arr["n"], vid, Yk[si, tj])
                first = False
        del d
    ck.put(spec, {}, arr)
    return {"part": part, "events": float(arr["n"].sum())}


def null_job(run: str, kind: str, seed: int, ridges: dict[str, float], chunk: int, code: str,
             keep_cache: bool = False) -> dict[str, Any]:
    """One null replicate end to end: null TRAIN and EVAL, refit NULL_RUNGS at frozen ridge, score EVAL."""
    run = Path(run)
    m = _meta(run)
    null = [kind, seed]
    tr, ev = build_ds(str(run), "TRAIN", null, chunk), build_ds(str(run), "EVAL", null, chunk)
    specs = [{"rung": r, "ridge": ridges[r]} for r in NULL_RUNGS]
    for k in m["targets"]:
        fit_job(str(run), "nulls", tr["name"], k, specs, [ev["name"]], code)
    if not keep_cache:
        for n in (tr["name"], ev["name"]):
            shutil.rmtree(run / "cache" / n, ignore_errors=True)
    return {"null": null, "train": {k: tr[k] for k in ("n_videos", "null_diag", "build_s")},
            "eval": {k: ev[k] for k in ("n_videos", "null_diag", "build_s")},
            "train_events": tr["events_per_target"], "eval_events": ev["events_per_target"]}


def fold_job(run: str, null: list | None, ridge: float, chunk: int, code: str) -> dict[str, Any]:
    """G4 on each of the 5 TRAIN folds (real data or one N-jit fold-null replicate)."""
    run = Path(run)
    m = _meta(run)
    tr = build_ds(str(run), "TRAIN", null, chunk)
    specs = [{"rung": "G4", "ridge": ridge, "groups": [f]} for f in range(5)]
    for k in m["targets"]:
        fit_job(str(run), "stability", tr["name"], k, specs, [], code)
    if null:
        shutil.rmtree(run / "cache" / tr["name"], ignore_errors=True)
    return {"null": null, "train_events": tr["events_per_target"], "null_diag": tr["null_diag"]}


def support_job(run: str, code: str) -> dict[str, Any]:
    """Registered support (prereg §7): TRAIN source events of type c with target k at risk just after."""
    run = Path(run)
    m = _meta(run)
    ck = Checkpoints(run / "ckpt" / "stability")
    spec = {"stage": "stability", "metric": "support", "code": code, "layout_sha": m["layout_sha"]}
    if ck.get(spec):
        return {"skipped": True}
    targets = np.array(m["targets"])
    R = len(m["layout"]["vocab"])
    K = 2 * R
    S = np.zeros((len(targets), K))
    tr_pred = targets // 2
    tr_off = targets % 2 == 1
    for v in load_part(run, "TRAIN"):
        for pr in v.pairs.values():
            w0, w1 = pr.window
            evs = [(b, 2 * r) for r, ivs in pr.rels.items() for b, e in ivs if b > 0] + \
                  [(e, 2 * r + 1) for r, ivs in pr.rels.items() for b, e in ivs if e < v.T]
            for t, c in evs:
                if not (w0 <= t < w1):
                    continue
                on = np.zeros(R, dtype=bool)
                for r, ivs in pr.rels.items():
                    on[r] = any(b <= t < e for b, e in ivs)        # state just after t
                risk = np.where(tr_off, on[tr_pred], ~on[tr_pred]) & (tr_pred != c // 2)
                S[risk, c] += 1
    ck.put(spec, {}, {"support": S, "targets": targets})
    return {"total": float(S.sum())}


def hyp_job(run: str, reps: list[int], code: str) -> dict[str, Any]:
    """H-a..H-d summed counts on EVAL (rep -1 = observed, per video) and on N-jit replicates."""
    run = Path(run)
    m = _meta(run)
    cfg = load_config(CFG)
    h = cfg["hypotheses"]
    vocab = m["layout"]["vocab"]
    ck = Checkpoints(run / "ckpt" / "hypotheses")
    eval_v = load_part(run, "EVAL")
    n_new = 0
    for rep in reps:
        spec = {"stage": "hypotheses", "rep": rep, "seed": h["null_seed"], "J": cfg["null_control"]["jitter_s"],
                "window": h["window_s"], "pairs": h["pairs"], "code": code}
        if ck.get(spec):
            continue
        vs = eval_v if rep < 0 else jitter_shift(eval_v, f"{h['null_seed']}:{rep}", "EVAL",
                                                 float(cfg["null_control"]["jitter_s"]))
        arr = {}
        for s, g in h["pairs"]:
            c = hyp_counts(vs, vocab.index(s), vocab.index(g), float(h["window_s"]))
            arr[f"{s}>{g}"] = c if rep < 0 else c.sum(0)
        ck.put(spec, {}, arr)
        n_new += 1
    return {"reps": len(reps), "new": n_new}


# ---------------------------------------------------------------------------
# stages (main process)
# ---------------------------------------------------------------------------

def stage_prepare(run: Path) -> dict[str, Any]:
    """Reduce raw TRAIN/SELECT annotations; layout, targets and folds from TRAIN only."""
    guard({"TRAIN", "SELECT"}, allow_eval=False)
    cfg = load_config(CFG)
    s0 = s0load(Path(cfg["stage0_config"]))
    (run / "data").mkdir(parents=True, exist_ok=True)
    train, vocab, stats = load_videos(s0["data"]["root"], s0["data"]["train_dir"])
    idx = split_indices(len(train), cfg, s0)
    tr = [train[i] for i in idx["TRAIN"]]
    sel = [train[i] for i in idx["SELECT"]]
    lay, targets = build_layout(tr, vocab, cfg)
    order = list(range(len(tr)))
    import random
    random.Random(cfg["splits"]["fold_seed"]).shuffle(order)
    fold = {tr[i].vid: j % cfg["splits"]["n_folds"] for j, i in enumerate(order)}
    L = {"vocab": lay.vocab, "cats": lay.cats, "lag_edges": lay.lag_edges, "elapsed_edges": lay.elapsed_edges,
         "count_bins": lay.count_bins, "active_bins": lay.active_bins, "dur_edges": lay.dur_edges,
         "sources": lay.sources}
    meta = {"layout": L, "layout_sha": hashlib.sha256(json.dumps(L, sort_keys=True).encode()).hexdigest()[:16],
            "targets": targets, "config_sha": cfg["_sha256"], "max_iter": cfg["fit"]["max_iter"],
            "fold_of_vid": fold, "load_stats_training": stats,
            "split_vids": {"TRAIN": [v.vid for v in tr], "SELECT": [v.vid for v in sel],
                           "EVAL_A": [train[i].vid for i in idx["EVAL_A"]]},
            "n_videos": {"TRAIN": len(tr), "SELECT": len(sel), "EVAL_A": len(idx["EVAL_A"])},
            "git_commit": _git("rev-parse", "HEAD"), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name, vs in (("TRAIN", tr), ("SELECT", sel)):
        with open(run / "data" / f"{name}.pkl", "wb") as fh:
            pickle.dump(vs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_text(run / "data" / "meta.json", json.dumps(meta))
    return {k: meta[k] for k in ("n_videos", "layout_sha", "config_sha", "git_commit")} | {
        "n_targets": len(targets), "n_sources": len(lay.sources), "n_columns": lay.size}


def tune_specs(m: dict[str, Any], cfg: dict[str, Any]) -> list[dict]:
    out = [{"rung": "G0", "ridge": 0.0}]
    for r in TUNED_RUNGS:
        out += [{"rung": r, "ridge": float(lam)} for lam in cfg["fit"]["ridge_grid"]]
    return out


def stage_tune(run: Path, workers: int, slots: int, chunk: int) -> dict[str, Any]:
    cfg = load_config(CFG)
    m = _meta(run)
    code = code_hash()
    log = run / "attempts.jsonl"
    b = [Job("build:TRAIN", build_ds, {"run": str(run), "part": "TRAIN", "null": None, "chunk": chunk}, weight=2),
         Job("build:SELECT", build_ds, {"run": str(run), "part": "SELECT", "null": None, "chunk": chunk})]
    specs = tune_specs(m, cfg)
    fits = [Job(f"tune:{k}", fit_job, {"run": str(run), "stage": "tune", "train_ds": "TRAIN", "k": k,
                                        "specs": specs, "score": ["SELECT"], "code": code},
                deps=["build:TRAIN", "build:SELECT"]) for k in m["targets"]]
    st = run_jobs(b + fits, workers, log, slots)
    if any(v != "complete" for v in st.values()):
        return {"status": st, "selection": None}
    sel = select_ridges(run, m, cfg, code)
    vspecs = [{"rung": r, "ridge": sel["selected"]["G4"]} for r in VARIANTS]
    st2 = run_jobs([Job(f"variants:{k}", fit_job, {"run": str(run), "stage": "tune", "train_ds": "TRAIN", "k": k,
                                                     "specs": vspecs, "score": [], "code": code})
                    for k in m["targets"]], workers, log, slots)
    return {"status": {**st, **st2}, "selection": sel}


def select_ridges(run: Path, m: dict, cfg: dict, code: str) -> dict[str, Any]:
    """Ridge per rung by total SELECT log-likelihood over all targets (prereg §4)."""
    ck = Checkpoints(run / "ckpt" / "tune")
    rows = []
    for s in tune_specs(m, cfg):
        tot, n_ev = 0.0, 0.0
        for k in m["targets"]:
            rec = ck.get(_fit_spec("tune", "TRAIN", k, s, ["SELECT"], code, m))
            if rec is None:
                raise RuntimeError(f"missing tune fit {s} k={k}")
            with ck.arrays(rec["task_id"]) as a:
                tot += float(a["ll__SELECT"].sum())
                n_ev += float(a["ev__SELECT"].sum())
        rows.append({**s, "select_ll": tot, "select_ll_per_event": tot / n_ev, "select_events": n_ev})
    selected = {"G0": 0.0}
    for r in TUNED_RUNGS:
        cand = [x for x in rows if x["rung"] == r]
        best = max(cand, key=lambda x: x["select_ll"])
        selected[r] = best["ridge"]
    out = {"rows": rows, "selected": selected, "grid": cfg["fit"]["ridge_grid"],
           "at_grid_boundary": {r: selected[r] in (min(cfg["fit"]["ridge_grid"]), max(cfg["fit"]["ridge_grid"]))
                                for r in TUNED_RUNGS}}
    atomic_write_text(run / "out" / "selection.json", json.dumps(out, indent=1))
    return out


def stage_freeze(run: Path) -> dict[str, Any]:
    """Write the selected parameters into the repository. Commit them before `evaluate`."""
    cfg = load_config(CFG)
    m = _meta(run)
    code = code_hash()
    sel = json.loads((run / "out" / "selection.json").read_text(encoding="utf-8"))
    ck = Checkpoints(run / "ckpt" / "tune")
    arrays: dict[str, np.ndarray] = {}
    tasks: dict[str, dict[str, str]] = {}
    diag_bad = []
    for rung in MAIN_RUNGS + VARIANTS:
        lam = sel["selected"]["G4"] if rung in VARIANTS else sel["selected"][rung]
        score = [] if rung in VARIANTS else ["SELECT"]
        tasks[rung] = {}
        for k in m["targets"]:
            rec = ck.get(_fit_spec("tune", "TRAIN", k, {"rung": rung, "ridge": lam}, score, code, m))
            if rec is None:
                raise RuntimeError(f"missing fit {rung} k={k}")
            tasks[rung][str(k)] = rec["task_id"]
            if not rec["diag"]["success"] or not rec["diag"]["finite"]:
                diag_bad.append({"rung": rung, "k": k, **rec["diag"]})
            with ck.arrays(rec["task_id"]) as a:
                for x in ("eta", "w", "v", "cols"):
                    arrays[f"{rung}|{k}|{x}"] = np.asarray(a[x])
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FROZEN_DIR / "params.npz", **arrays)
    fz = {"LABEL": "Phase 5H-1 frozen parameters (selected on SELECT; fitted on TRAIN). Commit before evaluate.",
          "selected_ridge": sel["selected"], "variant_ridge": sel["selected"]["G4"],
          "at_grid_boundary": sel["at_grid_boundary"], "selection_rows": sel["rows"], "task_ids": tasks,
          "params_sha256": _sha_file(FROZEN_DIR / "params.npz"), "code_hash": code, "config_sha": cfg["_sha256"],
          "layout_sha": m["layout_sha"], "targets": m["targets"], "non_converged_or_nonfinite": diag_bad,
          "git_commit_at_freeze": _git("rev-parse", "HEAD"),
          "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_text(FROZEN_DIR / "frozen.json", json.dumps(fz, indent=1))
    return {"selected": sel["selected"], "non_converged": len(diag_bad)}


def check_frozen_committed() -> str:
    """The freeze files must be tracked by git with no uncommitted change; returns the commit."""
    for f in ("frozen.json", "params.npz"):
        p = str(FROZEN_DIR / f).replace("\\", "/")
        if _git("ls-files", p) != p:
            raise RuntimeError(f"{p} is not committed; commit the freeze before evaluation")
        if _git("status", "--porcelain", p):
            raise RuntimeError(f"{p} has uncommitted changes")
    fz = json.loads((FROZEN_DIR / "frozen.json").read_text(encoding="utf-8"))
    if fz["params_sha256"] != _sha_file(FROZEN_DIR / "params.npz"):
        raise RuntimeError("params.npz does not match frozen.json")
    return _git("log", "-n", "1", "--format=%H", "--", str(FROZEN_DIR / "frozen.json").replace("\\", "/"))


def stage_evaluate_prepare(run: Path) -> dict[str, Any]:
    """Registered evaluation entry point, part 1 (own process): reduce EVAL-A/EVAL-B after the freeze check."""
    commit = check_frozen_committed()
    guard({"EVAL_A", "EVAL_B"}, allow_eval=True)
    cfg = load_config(CFG)
    s0 = s0load(Path(cfg["stage0_config"]))
    m = _meta(run)
    if not (run / "data" / "EVAL_B.pkl").exists():
        train, vocab, _ = load_videos(s0["data"]["root"], s0["data"]["train_dir"])
        assert vocab == m["layout"]["vocab"]
        ids = set(m["split_vids"]["EVAL_A"])
        ev_a = [v for v in train if v.vid in ids]
        del train
        ev_b, _, st_b = load_videos(s0["data"]["root"], s0["data"]["test_dir"], vocab)
        assert [v.vid for v in ev_a] == m["split_vids"]["EVAL_A"]
        for name, vs in (("EVAL_A", ev_a), ("EVAL_B", ev_b)):
            with open(run / "data" / f"{name}.pkl", "wb") as fh:
                pickle.dump(vs, fh, protocol=pickle.HIGHEST_PROTOCOL)
        m["n_videos"]["EVAL_B"] = len(ev_b)
        m["split_vids"]["EVAL_B"] = [v.vid for v in ev_b]
        m["load_stats_validation"] = st_b
        atomic_write_text(run / "data" / "meta.json", json.dumps(m))
    atomic_write_text(run / "data" / "EVAL_UNLOCK.json", json.dumps(
        {"frozen_commit": commit, "frozen_sha256": _sha_file(FROZEN_DIR / "frozen.json"),
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    m = _meta(run)
    return {"frozen_commit": commit, "n_videos": m["n_videos"]}


def stage_evaluate(run: Path, workers: int, slots: int, chunk: int) -> dict[str, Any]:
    """Registered evaluation entry point, part 2: score every frozen model on EVAL-A / EVAL-B."""
    commit = check_frozen_committed()
    guard({"EVAL_A", "EVAL_B"}, allow_eval=True)
    if not (run / "data" / "EVAL_UNLOCK.json").exists():
        raise RuntimeError("run the evaluate_prepare stage first")
    m = _meta(run)
    code = code_hash()
    jobs = [Job(f"build:{p}", build_ds, {"run": str(run), "part": p, "null": None, "chunk": chunk})
            for p in ("EVAL_A", "EVAL_B")]
    jobs += [Job(f"score:{k}", score_job, {"run": str(run), "k": k, "parts": ["EVAL_A", "EVAL_B"], "code": code},
                 deps=["build:EVAL_A", "build:EVAL_B"]) for k in m["targets"]]
    jobs += [Job(f"next_event:{p}", next_event_job, {"run": str(run), "part": p, "chunk": chunk, "code": code},
                 weight=2) for p in ("EVAL_A", "EVAL_B")]
    st = run_jobs(jobs, workers, run / "attempts.jsonl", slots)
    return {"frozen_commit": commit, "status": st}


def _frozen() -> dict[str, Any]:
    return json.loads((FROZEN_DIR / "frozen.json").read_text(encoding="utf-8"))


def stage_nulls(run: Path, workers: int, slots: int, chunk: int) -> dict[str, Any]:
    check_frozen_committed()
    cfg = load_config(CFG)
    ridges = _frozen()["selected_ridge"]
    code = code_hash()
    nc = cfg["null_control"]
    jobs = [Job(f"null:circ{s}", null_job, {"run": str(run), "kind": "circ", "seed": s, "ridges": ridges,
                                             "chunk": chunk, "code": code}, weight=2) for s in nc["circ_seeds"]]
    jobs += [Job(f"null:jit{s}", null_job, {"run": str(run), "kind": "jit", "seed": s, "ridges": ridges,
                                             "chunk": chunk, "code": code}, weight=2) for s in nc["jitter_seeds"]]
    return {"status": run_jobs(jobs, workers, run / "attempts.jsonl", slots)}


def stage_stability(run: Path, workers: int, slots: int, chunk: int) -> dict[str, Any]:
    check_frozen_committed()
    cfg = load_config(CFG)
    lam = _frozen()["selected_ridge"]["G4"]
    code = code_hash()
    jobs = [Job("fold:real", fold_job, {"run": str(run), "null": None, "ridge": lam, "chunk": chunk, "code": code}),
            Job("support", support_job, {"run": str(run), "code": code})]
    jobs += [Job(f"fold:jit{s}", fold_job, {"run": str(run), "null": ["jit", s], "ridge": lam, "chunk": chunk,
                                             "code": code}, weight=2) for s in cfg["null_control"]["fold_null_seeds"]]
    return {"status": run_jobs(jobs, workers, run / "attempts.jsonl", slots)}


def stage_hypotheses(run: Path, workers: int, slots: int) -> dict[str, Any]:
    check_frozen_committed()
    cfg = load_config(CFG)
    code = code_hash()
    n = int(cfg["hypotheses"]["null_reps"])
    reps = [-1] + list(range(n))
    groups = [reps[i::max(workers, 1)] for i in range(max(workers, 1))]
    jobs = [Job(f"hyp:{i}", hyp_job, {"run": str(run), "reps": g, "code": code}) for i, g in enumerate(groups) if g]
    return {"status": run_jobs(jobs, workers, run / "attempts.jsonl", slots)}


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------

def expected_tasks(run: Path) -> dict[str, list[dict]]:
    """Every registered task spec (the expected experiment matrix)."""
    cfg = load_config(CFG)
    m = _meta(run)
    code = code_hash()
    fz = _frozen()
    sel = fz["selected_ridge"]
    T = m["targets"]
    out: dict[str, list[dict]] = {"tune": [], "evaluate": [], "nulls": [], "stability": [], "hypotheses": []}
    for k in T:
        for s in tune_specs(m, cfg):
            out["tune"].append(_fit_spec("tune", "TRAIN", k, s, ["SELECT"], code, m))
        for r in VARIANTS:
            out["tune"].append(_fit_spec("tune", "TRAIN", k, {"rung": r, "ridge": sel["G4"]}, [], code, m))
    fz_sha = _sha_file(FROZEN_DIR / "params.npz")
    for k in T:
        for rung in MAIN_RUNGS + VARIANTS:
            for part in ("EVAL_A", "EVAL_B"):
                out["evaluate"].append({"stage": "evaluate", "k": k, "rung": rung, "part": part, "frozen": fz_sha,
                                        "code": code, "layout_sha": m["layout_sha"]})
    for part in ("EVAL_A", "EVAL_B"):
        out["evaluate"].append({"stage": "evaluate", "metric": "next_event", "part": part, "frozen": fz_sha,
                                "code": code})
    nc = cfg["null_control"]
    for kind, seeds in (("circ", nc["circ_seeds"]), ("jit", nc["jitter_seeds"])):
        for s in seeds:
            for k in T:
                for r in NULL_RUNGS:
                    out["nulls"].append(_fit_spec("nulls", f"TRAIN__{kind}{s}", k, {"rung": r, "ridge": sel[r]},
                                                  [f"EVAL__{kind}{s}"], code, m))
    for null in [None] + [["jit", s] for s in nc["fold_null_seeds"]]:
        for k in T:
            for f in range(5):
                out["stability"].append(_fit_spec("stability", ds_name("TRAIN", null), k,
                                                  {"rung": "G4", "ridge": sel["G4"], "groups": [f]}, [], code, m))
    out["stability"].append({"stage": "stability", "metric": "support", "code": code, "layout_sha": m["layout_sha"]})
    h = cfg["hypotheses"]
    for rep in [-1] + list(range(int(h["null_reps"]))):
        out["hypotheses"].append({"stage": "hypotheses", "rep": rep, "seed": h["null_seed"],
                                  "J": cfg["null_control"]["jitter_s"], "window": h["window_s"], "pairs": h["pairs"],
                                  "code": code})
    return out


def stage_check(run: Path) -> dict[str, Any]:
    """Expected / complete / missing / duplicate / failed accounting for every stage."""
    exp = expected_tasks(run)
    res: dict[str, Any] = {}
    for stage, specs in exp.items():
        ids = [spec_id(s) for s in specs]
        ck = Checkpoints(run / "ckpt" / stage)
        present = {p.stem for p in (run / "ckpt" / stage).glob("*.json")}
        missing = [i for i, s in zip(ids, specs) if ck.get(s) is None]
        extra = sorted(present - set(ids))
        nonconv = 0
        for i, s in zip(ids, specs):
            p = ck.path(i)
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8")).get("diag")
                if d and (not d["success"] or not d["finite"]):
                    nonconv += 1
        res[stage] = {"expected": len(ids), "unique_expected": len(set(ids)), "complete": len(ids) - len(missing),
                      "missing": len(missing), "missing_ids": missing[:20],
                      "unexpected_checkpoints": len(extra), "non_converged_or_nonfinite": nonconv}
    att = [json.loads(x) for x in (run / "attempts.jsonl").read_text(encoding="utf-8").splitlines()] \
        if (run / "attempts.jsonl").exists() else []
    ends = [a for a in att if a.get("event") == "end"]
    res["attempts"] = {"job_ends": len(ends), "failed": [{"job": a["job"], "error": a.get("error"), "utc": a["utc"]}
                                                          for a in ends if a["status"] != "complete"],
                       "blocked": [a["job"] for a in att if a.get("event") == "blocked"]}
    res["all_complete"] = all(v["missing"] == 0 and v["expected"] == v["unique_expected"]
                              for k, v in res.items() if k in exp)
    atomic_write_text(run / "out" / "completeness.json", json.dumps(res, indent=1))
    return res
