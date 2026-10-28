"""Adapter contract + precomputed JSONL adapter."""

from __future__ import annotations

import json

import pytest

from taorg.adapters.base import PredictionAdapter, validate_video
from taorg.adapters.precomputed import PrecomputedJSONLAdapter
from taorg.schemas.edges import EdgeKey, FrameGraph
from taorg.schemas.tracks import VisibilityState
from taorg.temporal import METHODS


def _write(tmp_path, rows):
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _edge(f, s, o, p, score, vid="V1"):
    return {"type": "edge", "video_id": vid, "frame_idx": f, "timestamp": f / 10,
            "subject_track_id": s, "object_track_id": o, "predicate_id": p, "score": score}


def test_builds_contract_compliant_videos(tmp_path):
    rows = [_edge(0, 1, 2, 0, 0.9), _edge(0, 1, 3, 1, 0.8),
            {"type": "frame", "video_id": "V1", "frame_idx": 10, "timestamp": 1.0,
             "visibility": {"1": "visible", "2": "occluded_or_unreliable"}},
            _edge(20, 1, 3, 1, 0.7), _edge(20, 1, 2, 4, 0.6), _edge(30, 1, 2, 0, 0.05)]
    ad = PrecomputedJSONLAdapter(_write(tmp_path, rows), source_model="dummy", min_score=0.1)
    assert isinstance(ad, PredictionAdapter)
    (video,) = list(ad.videos())
    assert [f.frame_idx for f in video] == [0, 1, 2, 3]
    f1 = {o.edge_key: o for o in video[1].edge_observations}
    # pair (1,2) declared occluded -> absence record with OCC; pair (1,3) has no track 3 -> no record
    assert not f1[EdgeKey(1, 2, 0)].observed
    assert f1[EdgeKey(1, 2, 0)].object_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE
    assert EdgeKey(1, 3, 1) not in f1
    f2 = {o.edge_key: o for o in video[2].edge_observations}
    assert f2[EdgeKey(1, 2, 0)].object_visibility == VisibilityState.VISIBLE  # visible absence
    assert f2[EdgeKey(1, 2, 4)].observed
    # score below min_score dropped -> frame 3 has only absence records
    assert not any(o.observed for o in video[3].edge_observations)


def test_top_k(tmp_path):
    rows = [_edge(0, 1, 2, p, 0.1 * (p + 1)) for p in range(5)]
    (video,) = list(PrecomputedJSONLAdapter(_write(tmp_path, rows), "d", top_k=2).videos())
    assert {o.edge_key.predicate_id for o in video[0].edge_observations} == {3, 4}


def test_adapter_output_runs_through_temporal_methods(tmp_path):
    rows = [_edge(f, 1, 2, 0, 0.9) for f in range(0, 10) if f not in (4, 5)]
    (video,) = list(PrecomputedJSONLAdapter(_write(tmp_path, rows), "d").videos())
    for name in ("gated_ema", "vhped", "vhped_lazy"):
        m = METHODS[name]()
        m.reset_video("V1")
        for f in video:
            m.step(f)


def test_validate_video_rejects_bad_order():
    with pytest.raises(ValueError):
        validate_video([FrameGraph("v", 0, 0.1), FrameGraph("v", 1, 0.1)])


def test_unknown_record_type(tmp_path):
    p = _write(tmp_path, [{"type": "bogus", "video_id": "V", "frame_idx": 0, "timestamp": 0}])
    with pytest.raises(ValueError, match="bogus"):
        list(PrecomputedJSONLAdapter(p, "d").videos())
