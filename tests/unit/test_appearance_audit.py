"""Tests for the appearance audit: cues (causal, gated), labels, statistics, no-GT inference path."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from taorg.adapters.appearance import TrackerRun, compute_cues, stored_appearance_series, unit
from taorg.adapters.mot17 import FrameTracks, SeqInfo, build_records
from taorg.experiments.appearance.labels import common_unit, frame_label, incidents, purity
from taorg.experiments.appearance.stats import (
    auc,
    average_precision,
    bootstrap_delta_auc,
    cluster_index,
    feature_matrix,
    loso_scores,
    operating_point,
)

ROOT = Path(__file__).resolve().parents[2]
ZIP = Path("D:/TAORG_runs/observation/MOT17Labels.zip")


def _toy():
    """Track 1 detected frames 1-3 then missing; track 2 appears at frame 5 next to track 1's last box."""
    info = SeqInfo("toy", 10.0, 1000, 1000, 8)
    e_a = unit(np.r_[1.0, np.zeros(511)])
    e_b = unit(np.r_[0.0, 1.0, np.zeros(510)])
    box1 = (100.0, 100.0, 150.0, 200.0)
    far = (800.0, 800.0, 850.0, 900.0)
    dets, embs = {}, {}
    for f in range(1, 9):
        rows, es = [], []
        if f <= 3:
            rows.append([*box1, 0.9]); es.append(e_a)
        if f >= 5:
            rows.append([110.0, 100.0, 160.0, 200.0, 0.9]); es.append(e_a)     # same person, new track
            rows.append([*far, 0.9]); es.append(e_b)                           # someone else, far away
        dets[f] = np.asarray(rows, float).reshape(-1, 5)
        embs[f] = np.asarray(es, np.float16).reshape(-1, 512)
    frames, out_det = [], []
    for f in range(1, 9):
        outs, m = {}, {}
        if f <= 3:
            outs[1] = (box1, 0.9); m[1] = 0
        if f >= 5:
            outs[2] = (tuple(dets[f][0, :4]), 0.9); m[2] = 0
        frames.append(FrameTracks(f, outs, {}))
        out_det.append(m)
    return info, dets, embs, TrackerRun(frames, out_det)


def test_cues_values_and_gate():
    info, dets, embs, run = _toy()
    recs = build_records(info, run.frames)
    rows = [r for r in compute_cues(info, run, recs, dets, embs) if r["track"] == 1]
    by = {r["frame"]: r for r in rows}
    assert math.isclose(by[4]["dt"], 0.1) and math.isnan(by[4]["a_glob"])        # no candidates at frame 4
    assert by[5]["a_loc"] == pytest.approx(0.0, abs=1e-3)                         # same appearance nearby
    assert by[5]["a_glob"] == pytest.approx(0.0, abs=1e-3)
    assert by[5]["n_loc"] == 1 and by[5]["n_cand"] == 2                           # far box outside the gate


def test_cues_are_causal():
    info, dets, embs, run = _toy()
    recs = build_records(info, run.frames)
    base = {(r["track"], r["frame"]): r for r in compute_cues(info, run, recs, dets, embs)}
    dets2 = {f: d.copy() for f, d in dets.items()}
    embs2 = {f: e.copy() for f, e in embs.items()}
    dets2[8] = np.asarray([[105.0, 100.0, 155.0, 200.0, 0.99]])                   # change the future only
    embs2[8] = np.asarray([unit(np.r_[0, 0, 1.0, np.zeros(509)])], np.float16)
    new = {(r["track"], r["frame"]): r for r in compute_cues(info, run, recs, dets2, embs2)}
    for k, r in base.items():
        if k[1] < 8:
            assert r == new[k]


def test_stored_appearance_ignores_low_confidence():
    info, dets, embs, run = _toy()
    dets[3][0, 4] = 0.4
    embs[3] = np.asarray([unit(np.r_[0.0, 1.0, np.zeros(510)])], np.float16)
    s = dict(stored_appearance_series(run, dets, embs)[1])
    assert float(s[3] @ unit(np.r_[1.0, np.zeros(511)])) == pytest.approx(1.0, abs=1e-3)


def test_frame_labels_and_ambiguity():
    gt = {1: {7: ((0, 0, 10, 10), 1.0)}, 2: {7: ((0, 0, 10, 10), 0.2)}, 3: {7: ((0, 0, 10, 10), 1.0)}}
    matches = {1: {1: 7}, 2: {}, 3: {2: 7}}
    g = {1: 7, 2: 7}
    pur = purity(matches)
    assert frame_label(1, 2, g, pur, matches, gt) == "GAP"
    assert frame_label(1, 3, g, pur, matches, gt) == "TAKEOVER"
    assert frame_label(9, 2, g, pur, matches, gt) == "FALSE_TRACK"
    mixed = {1: {1: 7}, 2: {1: 8}, 3: {}}
    assert frame_label(1, 3, {1: 7}, purity(mixed), mixed, gt) == "AMBIGUOUS"


def test_incident_majority_rules():
    rows = [{"unit": "u", "track": 1, "frame": f, "label": lab}
            for f, lab in zip(range(1, 7), ["GAP", "GAP", "TAKEOVER", "TAKEOVER", "TAKEOVER", "TAKEOVER"])]
    rows += [{"unit": "u", "track": 1, "frame": 10, "label": "GONE"}]
    inc = incidents(rows)
    assert [i["label"] for i in inc] == ["TAKEOVER", "OTHER"] and inc[0]["length"] == 6
    rows2 = [{"unit": "u", "track": 2, "frame": f, "label": lab} for f, lab in zip(range(3), ["GAP", "TAKEOVER", "GAP"])]
    assert incidents(rows2)[0]["label"] == "GAP"
    rows3 = [{"unit": "u", "track": 2, "frame": f, "label": lab} for f, lab in zip(range(4), ["GAP", "TAKEOVER"] * 2)]
    assert incidents(rows3)[0]["label"] == "AMBIGUOUS"


def test_common_unit_states():
    gt = {f: {7: ((0, 0, 1, 1), 1.0)} for f in range(1, 7)}
    matches = {1: {1: 7}, 2: {}, 3: {1: 7}, 4: {}, 5: {2: 7}, 6: {2: 7}}
    c = common_unit(matches, gt)
    assert c == {"UNCOVERED": 2, "CONTINUED": 2, "SWITCHED": 1, "GAP_BRIDGED": 1, "GAP_FRAGMENTED": 1, "GAP_ENDED": 0}


def test_auc_ap_and_operating_point():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1.0, 300)
    s[:20] = np.round(s[:20])                                                    # ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert math.isnan(auc(np.zeros(5), np.arange(5)))
    op = operating_point(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.8, 0.2]), np.full(4, 0.5))
    assert op == {"gap_rejection": 0.5, "takeover_recall": 0.5, "takeover_retention": 0.5}


def test_bootstrap_is_seeded_and_zero_for_identical_scores():
    y = np.array([0, 1] * 20)
    s = np.linspace(0, 1, 40)
    cl = cluster_index(["u"] * 40, [i // 4 for i in range(40)])
    d1 = bootstrap_delta_auc(y, s, s, cl, np.random.default_rng(1), B=50)
    d2 = bootstrap_delta_auc(y, s * 2, s, cl, np.random.default_rng(1), B=50)
    assert np.allclose(d1, 0) and np.allclose(d2, 0)


def test_loso_never_trains_on_held_out_sequence():
    rows = [{"seq": q, "dt": 0.1 * (i % 7 + 1), "a_glob": 0.3, "a_loc": float("nan")} for q in "abc" for i in range(30)]
    y = np.array([i % 2 for i in range(90)])
    s, th = loso_scores(rows, y, ["a", "b", "c"], "T+A")
    assert np.isfinite(s).all() and np.isfinite(th).all()
    X = feature_matrix(rows[:2], ["a_loc", "a_loc_missing"])
    assert X.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_inference_modules_have_no_ground_truth_path():
    for mod in ("src/taorg/adapters/appearance.py",):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any("oracle" in n or "labels" in n for n in names), mod
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "gt.txt" not in node.value and "gt/" not in node.value, mod


@pytest.mark.skipif(not ZIP.exists(), reason="MOT17Labels.zip not available")
def test_v1_matches_observation_audit_run_and_indices():
    from taorg.adapters.appearance import run_bytetrack_v1
    from taorg.adapters.mot17 import load_detections, run_bytetrack
    info, dets = load_detections(ZIP, "02", "FRCNN")
    info = SeqInfo(info.name, info.fps, info.width, info.height, 120)
    run = run_bytetrack_v1(info, dets)
    ref = run_bytetrack(info, dets)
    assert [f.outputs for f in run.frames] == [f.outputs for f in ref]
    for fr, m in zip(run.frames, run.out_det):
        for t, j in m.items():
            assert np.allclose(dets[fr.frame][j, :4], fr.outputs[t][0])


FRAMES = Path("D:/TAORG_runs/appearance/MOT17Det/train/MOT17-02/img1")


@pytest.mark.skipif(not (ZIP.exists() and FRAMES.exists()), reason="MOT17 frames not available")
def test_botsort_variants_record_costs_and_appearance_changes_cost():
    pytest.importorskip("boxmot")
    from taorg.adapters.appearance import run_botsort
    from taorg.adapters.mot17 import load_detections
    info, dets = load_detections(ZIP, "02", "FRCNN")
    info = SeqInfo(info.name, info.fps, info.width, info.height, 40)
    same = unit(np.r_[1.0, np.zeros(511)])          # identical appearance: gated emb cost is 0 near a track
    embs = {f: np.tile(same, (len(d), 1)).astype(np.float16) for f, d in dets.items()}
    v2 = run_botsort(info, dets, FRAMES, None, use_embeddings=False)
    v3 = run_botsort(info, dets, FRAMES, embs, use_embeddings=True)
    assert np.allclose(v2.assoc_log[:, 6], v2.assoc_log[:, 4])                # appearance off: cost == geom
    assert np.isnan(v2.assoc_log[:, 5]).all()
    near = v3.assoc_log[:, 4] <= 0.5                                         # inside the proximity gate
    assert np.allclose(v3.assoc_log[near, 6], 0.0, atol=1e-5)                # appearance on changes the cost
    assert np.allclose(v3.assoc_log[~near, 6], v3.assoc_log[~near, 4])
    assert (v3.assoc_log[:, 6] <= v3.assoc_log[:, 4] + 1e-9).all()            # cost = min(geom, gated emb)
    for fr, m in zip(v3.frames, v3.out_det):
        for t, j in m.items():
            assert np.allclose(dets[fr.frame][j, :4], fr.outputs[t][0])


@pytest.mark.skipif(not (ZIP.exists() and FRAMES.exists()), reason="MOT17 frames not available")
def test_smooth_feat_mirror_includes_creation_detection():
    pytest.importorskip("boxmot")
    from taorg.adapters.appearance import run_botsort
    from taorg.adapters.mot17 import load_detections
    info, dets = load_detections(ZIP, "02", "FRCNN")
    info = SeqInfo(info.name, info.fps, info.width, info.height, 60)
    rng = np.random.default_rng(1)
    embs = {f: unit(rng.normal(size=(len(d), 512))).astype(np.float16) for f, d in dets.items()}
    run = run_botsort(info, dets, FRAMES, embs, use_embeddings=True, feat_snapshot_every=1)
    assert run.created                                                        # later tracks start unconfirmed
    ser = stored_appearance_series(run, dets, embs, include_creation=True)
    checked = 0
    for (f, tid), sf in run.smooth_feats.items():
        prev = [s for g, s in ser[tid] if g < f]
        assert float(unit(sf) @ prev[-1]) >= 0.999
        checked += 1
    assert checked > 0
