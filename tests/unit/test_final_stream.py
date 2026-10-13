"""Final-evaluation stream and genuine-event metrics (research/taorg_final_preregistration.md)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from taorg.adapters.mot17 import FrameTracks
from taorg.experiments.final.stream import assign, build_predicted_stream
from taorg.metrics.downstream import genuine_outcomes
from taorg.schemas.edges import EdgeKey
from taorg.schemas.tracks import VisibilityState

ROOT = Path(__file__).resolve().parents[2]


def _frames(n):
    out = []
    for f in range(1, n + 1):
        o = {1: ((0, 0, 10, 20), 0.9), 2: ((50, 0, 60, 20), 0.9)}
        if f > 6:
            o.pop(2)                                    # track 2 lost after frame 6
        out.append(FrameTracks(f, o, {}))
    return out


def _sgg(n, pred_until):
    boxes = np.array([[0, 0, 10, 20], [50, 0, 60, 20]], np.float32)
    out = {}
    for f in range(1, n + 1):
        sc = np.zeros((2, 2, 4), np.float32)
        if f <= pred_until:
            sc[0, 1, 3] = 0.5                           # (1 watch 2)
        out[f] = (boxes, np.array([0.9, 0.9], np.float32), sc)
    return out


def test_assign_is_one_to_one_with_iou_gate():
    outs = {7: ((0, 0, 10, 10), 0.9), 8: ((100, 100, 110, 110), 0.9)}
    q = assign(np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], np.float32), outs)
    assert list(q.values()).count(7) == 1 and 8 not in q.values() and 2 not in q


def test_predicted_stream_distinguishes_evaluated_negative_from_not_evaluated():
    frames = _frames(10)
    ps = build_predicted_stream("v", 30.0, frames, _sgg(10, 3), list(range(1, 11)), tau=0.1)
    k = EdgeKey(1, 2, 3)
    assert ps.predicted_keys == [k]
    rec = {g.frame_idx: {o.edge_key: o for o in g.edge_observations} for g in ps.graphs}
    assert rec[2][k].observed and rec[2][k].predicate_score == pytest.approx(0.5)
    assert not rec[5][k].observed and rec[5][k].object_visibility == VisibilityState.VISIBLE   # evaluated negative
    assert k not in rec[8]                                                                   # endpoint lost: no record


def test_predicted_stream_threshold_and_self_pairs():
    frames = _frames(4)
    ps = build_predicted_stream("v", 30.0, frames, _sgg(4, 4), [1, 2, 3, 4], tau=0.6)
    assert ps.predicted_keys == [] and all(not g.edge_observations for g in ps.graphs)


def test_genuine_outcomes_only_count_genuine_changes():
    n = 31
    alive = np.zeros(n, bool)
    alive[1:6] = True
    alive[11:16] = True
    alive[21:26] = True
    z = np.zeros(n, bool)
    ended = np.zeros(n, bool)
    ended[5] = True                                    # first end genuine, second end flicker
    started = np.zeros(n, bool)
    started[21] = True                                 # third episode genuine onset, second is flicker
    lab = {"k": {"alive": alive, "both_out": np.ones(n, bool), "ended_after": ended, "started_at": started,
                 "takeover": z, "gap": z}}
    out = alive.copy()
    out[6:8] = True                                    # held 2 steps after the genuine end
    out[21:23] = False                                 # genuine recurrence detected 2 steps late
    r = genuine_outcomes({"k": out}, lab, fps=10.0, min_dead_s=0.3, sustained_s=0.3)
    assert r["std_gen_n"] == 1 and r["std_gen_sum"] == pytest.approx(0.3)
    assert r["rdd_gen_n"] == 1 and r["rdd_gen_sum"] == pytest.approx(0.2)
    perfect = genuine_outcomes({"k": alive.copy()}, lab, fps=10.0, min_dead_s=0.3, sustained_s=0.3)
    assert perfect["std_gen_sum"] == pytest.approx(0.1) and perfect["rdd_gen_sum"] == 0.0
    none = genuine_outcomes({"k": z.copy()}, {"k": {**lab["k"], "alive": z}}, fps=10.0)
    assert none["std_gen_n"] == 0 and none["rdd_gen_n"] == 0


def test_inference_stream_has_no_annotation_access():
    tree = ast.parse((ROOT / "src/taorg/experiments/final/stream.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_predicted_stream")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert not names & {"active_relations", "gt_dict", "match", "identity_of", "frame_label", "VidorVideo"}
