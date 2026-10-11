"""Phase 5H-1 execution infrastructure: exact aggregation, N-jit null, runner, serial/parallel agreement."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest

from taorg.hawkes.glm import (
    Layout, aggregate_target, build_design, fit_target, fit_target_agg, ll_agg_per_group, merge_targets,
    target_columns, td_arrays, td_from_arrays,
)
from taorg.hawkes.nulls1 import jitter_shift, jitter_video
from taorg.hawkes.runner import Checkpoints, Job, run_jobs
from taorg.hawkes.simulate import simulate
from taorg.hawkes.vidor import PairRecord, VideoRecord

TOL = 1e-6          # declared serial/parallel and aggregated/raw tolerance (parameters and log-likelihoods)


def lay2() -> Layout:
    return Layout(["p0", "p1"], [], [0.0, 0.5, 2.0, 10.0], [2.0, 10.0], [0, 1, 2, 3, 5, 9], [0, 1, 2, 3],
                  [1.0, 4.0, 16.0], sources=[0, 1, 2, 3])


def test_aggregated_fit_equals_raw_fit():
    vids = simulate(0.6, 3, n_videos=12)
    lay = lay2()
    d = build_design(vids, lay, [0, 1, 2, 3])
    grp = d.seg_video % 3
    for k in (1, 2):
        td = aggregate_target(d, k, grp)
        assert len(td.y) < td.n_rows and td.y.sum() == d.y[k].sum()
        assert abs(td.e.sum() - d.seg_dur[d.rows[k]].sum()) < 1e-6
        for rung in ("G2", "G4", "G4+"):
            raw = fit_target(d, k, rung, 1.0)
            agg = fit_target_agg(td, lay, rung, 1.0).fit
            assert np.allclose(raw.w, agg.w, atol=TOL) and abs(raw.eta - agg.eta) < TOL
        # per-group LL on aggregated rows == per-row LL summed by group
        td_v = aggregate_target(d, k, d.seg_video)
        f = fit_target_agg(td, lay, "G4", 1.0).fit
        ll, ev = ll_agg_per_group(td_v, f, lay, len(vids), True)
        from taorg.hawkes.glm import target_ll_per_video
        assert np.allclose(ll, target_ll_per_video(d, f, True), atol=1e-8)
        assert ev.sum() == d.y[k].sum()


def test_chunked_merge_equals_whole_and_roundtrip():
    vids = simulate(0.6, 4, n_videos=10)
    lay = lay2()
    whole = build_design(vids, lay, [0, 1, 2, 3])
    gw = whole.seg_video % 2
    parts = []
    for s in (0, 4, 7):
        d = build_design(vids[s: {0: 4, 4: 7, 7: 10}[s]], lay, [0, 1, 2, 3])
        parts.append(td_from_arrays(td_arrays(aggregate_target(d, 1, (d.seg_video + s) % 2))))
    merged = merge_targets(parts, lay.size)
    ref = aggregate_target(whole, 1, gw)
    assert len(merged.y) == len(ref.y)
    a, b = fit_target_agg(merged, lay, "G4", 1.0).fit, fit_target_agg(ref, lay, "G4", 1.0).fit
    assert np.allclose(a.w, b.w, atol=TOL)
    fa = fit_target_agg(merged, lay, "G4", 1.0, groups=np.array([0])).fit
    fb = fit_target_agg(ref, lay, "G4", 1.0, groups=np.array([0])).fit
    assert np.allclose(fa.w, fb.w, atol=TOL)


def test_hist_class_and_lag_variants():
    lay = lay2()
    full = set(target_columns(lay, "G4", 0)) - set(target_columns(lay, "G3", 0))
    on_on = set(target_columns(lay, "G3+H:ON>ON", 0)) - set(target_columns(lay, "G3", 0))
    off_on = set(target_columns(lay, "G3+H:OFF>ON", 0)) - set(target_columns(lay, "G3", 0))
    assert on_on | off_on == full and not on_on & off_on
    assert set(target_columns(lay, "G3+H:ON>OFF", 0)) == set(target_columns(lay, "G3", 0))   # onset target
    lags = [set(target_columns(lay, f"G3+H:lag{m}", 0)) - set(target_columns(lay, "G3", 0)) for m in range(3)]
    assert set().union(*lags) == full and sum(len(x) for x in lags) == len(full)


def _toy_video() -> VideoRecord:
    T = 3000
    pr = PairRecord((0, T), {0: [(0, 400), (900, 1300)], 1: [(500, 1000), (2500, T)], 2: [(30, 60)]})
    pr2 = PairRecord((600, 2400), {1: [(700, 900)], 3: [(1000, 2400)]})
    return VideoRecord("toy", 30.0, T, {(1, 2): pr, (3, 4): pr2})


def test_jitter_respects_censoring_window_and_durations():
    v = _toy_video()
    diag: dict[str, int] = {}
    for seed in range(40):
        j = jitter_video(v, seed, "TRAIN", 5.0, diag)
        for key, pr in v.pairs.items():
            jp = j.pairs[key]
            assert jp.window == pr.window                                   # risk window unchanged
            for r, ivs in jp.rels.items():
                for b, e in ivs:
                    assert pr.window[0] <= b < e <= pr.window[1]
                orig = pr.rels[r]
                if orig[0][0] == 0:
                    assert ivs[0][0] == 0                                    # left-censored stays censored
                if orig[-1][1] == v.T:
                    assert ivs[-1][1] == v.T                                 # right-censored stays censored
            # uncensored intervals away from the window edges keep their duration and move by <= 5 s
            o, n = pr.rels.get(1, []), jp.rels.get(1, [])
            if key == (1, 2) and len(n) == 2:
                assert n[0][1] - n[0][0] == 500 and abs(n[0][0] - 500) <= 150
    assert diag["intervals"] == 40 * 7


def test_jitter_is_order_and_chunk_independent():
    vids = simulate(0.3, 9, n_videos=6)
    a = jitter_shift(vids, 6300, "TRAIN", 5.0)
    rev = list(reversed(vids))
    b = list(reversed(jitter_shift(rev, 6300, "TRAIN", 5.0)))
    c = jitter_shift(vids[:3], 6300, "TRAIN", 5.0) + jitter_shift(vids[3:], 6300, "TRAIN", 5.0)
    for x, y, z in zip(a, b, c):
        assert {k: p.rels for k, p in x.pairs.items()} == {k: p.rels for k, p in y.pairs.items()} \
            == {k: p.rels for k, p in z.pairs.items()}
    d = jitter_shift(vids, 6301, "TRAIN", 5.0)
    assert any(x.pairs != y.pairs for x, y in zip(a, d))


# ---- runner -----------------------------------------------------------------

def ok_job(path: str, x: int) -> dict:
    Path(path).write_text(str(x * x))
    return {"x": x}


def bad_job(path: str) -> dict:
    raise ValueError("deliberate failure")


def test_runner_records_failures_blocks_dependents_and_rejects_duplicates(tmp_path):
    jobs = [Job("a", ok_job, {"path": str(tmp_path / "a"), "x": 3}),
            Job("b", bad_job, {"path": str(tmp_path / "b")}),
            Job("c", ok_job, {"path": str(tmp_path / "c"), "x": 4}, deps=["b"]),
            Job("d", ok_job, {"path": str(tmp_path / "d"), "x": 5}, deps=["a"])]
    st = run_jobs(jobs, 2, tmp_path / "log.jsonl")
    assert st == {"a": "complete", "b": "failed", "c": "blocked", "d": "complete"}
    log = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any(r.get("status") == "failed" and "deliberate" in r.get("error", "") for r in log)
    with pytest.raises(ValueError):
        run_jobs([jobs[0], jobs[0]], 1, tmp_path / "log2.jsonl")


def test_checkpoint_spec_mismatch_is_detected(tmp_path):
    ck = Checkpoints(tmp_path)
    spec = {"a": 1}
    tid = ck.put(spec, {"v": 1})
    assert ck.get(spec)["v"] == 1
    rec = json.loads(ck.path(tid).read_text())
    rec["spec"] = {"a": 2}
    ck.path(tid).write_text(json.dumps(rec))
    with pytest.raises(RuntimeError):
        ck.get(spec)


def _mini_run(root: Path) -> None:
    """A synthetic run root shaped like `prepare` output (SYNTHETIC; never evidence)."""
    from taorg.hawkes.phase5h1 import load_config
    tr, sel = simulate(0.6, 11, n_videos=20), simulate(0.6, 12, n_videos=8)
    (root / "data").mkdir(parents=True)
    for n, vs in (("TRAIN", tr), ("SELECT", sel)):
        with open(root / "data" / f"{n}.pkl", "wb") as fh:
            pickle.dump(vs, fh)
    order = list(range(len(tr)))
    random.Random(6102).shuffle(order)
    L = {"vocab": ["p0", "p1"], "cats": [], "lag_edges": [0.0, 0.5, 2.0, 10.0], "elapsed_edges": [2.0, 10.0],
         "count_bins": [0, 1, 2, 3, 5, 9], "active_bins": [0, 1, 2, 3], "dur_edges": [1.0, 4.0, 16.0],
         "sources": [0, 1, 2, 3]}
    cfg = load_config(Path("configs/experiments/phase5h1.yaml"))
    meta = {"layout": L, "layout_sha": "synthetic", "targets": [0, 1, 2, 3], "config_sha": cfg["_sha256"],
            "max_iter": 500, "fold_of_vid": {tr[i].vid: j % 5 for j, i in enumerate(order)},
            "n_videos": {"TRAIN": 20, "SELECT": 8, "EVAL_A": 0}}
    (root / "data" / "meta.json").write_text(json.dumps(meta))


def test_serial_and_parallel_tuning_agree_on_synthetic_data(tmp_path):
    from taorg.hawkes import phase5h1_run as R
    roots = {}
    for mode, workers in (("serial", 0), ("parallel", 3)):
        root = tmp_path / mode
        _mini_run(root)
        out = R.stage_tune(root, workers, workers or 1, chunk=7)
        assert all(v == "complete" for v in out["status"].values()), out["status"]
        roots[mode] = root
    sa, sb = Checkpoints(roots["serial"] / "ckpt" / "tune"), Checkpoints(roots["parallel"] / "ckpt" / "tune")
    ids_a = sorted(p.stem for p in sa.root.glob("*.json"))
    ids_b = sorted(p.stem for p in sb.root.glob("*.json"))
    assert ids_a == ids_b and len(ids_a) == 4 * (1 + 6 * 4 + 7)              # same specs => same task ids
    for tid in ids_a:
        with sa.arrays(tid) as x, sb.arrays(tid) as y:
            for key in x.files:
                assert np.allclose(x[key], y[key], atol=TOL, rtol=0), (tid, key)
    sel_a = json.loads((roots["serial"] / "out" / "selection.json").read_text())
    sel_b = json.loads((roots["parallel"] / "out" / "selection.json").read_text())
    assert sel_a["selected"] == sel_b["selected"]
    # resume: a second invocation does no new work
    out = R.stage_tune(roots["parallel"], 2, 2, chunk=7)
    assert sum(1 for p in sb.root.glob("*.json")) == len(ids_b)


def test_eval_data_locked_before_freeze(tmp_path):
    from taorg.hawkes import phase5h1_run as R
    from taorg.hawkes.phase5h1 import LeakageError
    _mini_run(tmp_path)
    for part in ("EVAL", "EVAL_A", "EVAL_B"):
        with pytest.raises(LeakageError):
            R.load_part(tmp_path, part)
    assert len(R.load_part(tmp_path, "TRAIN")) == 20
