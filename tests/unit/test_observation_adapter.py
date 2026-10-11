"""MOT17/ByteTrack adapter and oracle: heuristics, records, separation of inference and ground truth."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from taorg.adapters import mot17
from taorg.adapters.mot17 import FrameTracks, SeqInfo, build_records, cover, heuristic_reason, iou, outside_fraction
from taorg.experiments.observation import oracle
from taorg.schemas.observation import EvidenceSource, MissingReason
from taorg.schemas.tracks import TrackerState

W, H = 100, 100


def test_geometry_helpers():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert cover((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)
    assert outside_fraction((-10, 0, 10, 10), W, H) == pytest.approx(0.5)


def test_heuristic_precedence_and_age_rule():
    pred = (40, 40, 60, 60)
    fr = FrameTracks(10, {7: ((40, 40, 60, 60), 0.9)}, {})
    # Y=7 born AFTER x's last detection, IoU 1 -> identity suspect
    assert heuristic_reason(1, pred, 5, fr, {7: 8}, W, H) is MissingReason.IDENTITY_SUSPECT
    # same box but Y existed before x was lost -> occlusion evidence
    assert heuristic_reason(1, pred, 5, fr, {7: 2}, W, H) is MissingReason.OCCLUSION_EVIDENCE
    # off-image prediction dominates
    assert heuristic_reason(1, (90, 40, 130, 60), 5, fr, {7: 8}, W, H) is MissingReason.OUT_OF_VIEW_EVIDENCE
    assert heuristic_reason(1, pred, 5, FrameTracks(10, {}, {}), {}, W, H) is MissingReason.UNSPECIFIED


def test_build_records_states_reasons_and_time():
    info = SeqInfo("toy", 10.0, W, H, 8)
    box = (10.0, 10.0, 20.0, 20.0)
    frames = [FrameTracks(1, {3: (box, 0.9)}, {3: (box, 0)}),
              FrameTracks(2, {3: (box, 0.8)}, {3: (box, 0)}),
              FrameTracks(3, {}, {3: (box, 1)}),                 # lost, in buffer
              FrameTracks(4, {}, {}),                            # deleted by the tracker
              FrameTracks(5, {3: (box, 0.7)}, {3: (box, 0)}),
              FrameTracks(6, {}, {}), FrameTracks(7, {}, {}), FrameTracks(8, {}, {})]
    recs = build_records(info, frames, horizon_s=0.2)
    r = recs[3]
    assert [x.detected for x in r] == [True, True, False, False, True, False, False]
    assert r[2].tracker_state is TrackerState.LOST and r[2].missing_reason is MissingReason.TRACK_LOST
    assert r[3].tracker_state is TrackerState.REMOVED and r[3].missing_reason is MissingReason.TRACK_TERMINATED
    assert r[3].missing_reason_source is EvidenceSource.TRACKER
    assert r[3].time_since_detection_s == pytest.approx(0.2)
    assert all(x.association_confidence is None for x in r), "ByteTrack provides no association confidence"


def _code_strings(module) -> list[str]:
    """String constants in code, excluding docstrings."""
    import ast
    tree = ast.parse(inspect.getsource(module))
    docs = set()
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]:
        d = ast.get_docstring(node, clean=False)
        if d is not None:
            docs.add(d)
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value not in docs]


def test_inference_module_never_reads_ground_truth():
    assert not any("gt" in s.split("/") or "gt.txt" in s for s in _code_strings(mot17))
    assert any("gt.txt" in s for s in _code_strings(oracle))


def test_oracle_categories():
    gt = {5: {1: ((0, 0, 10, 10), 0.2), 2: ((50, 50, 60, 60), 0.9)}, 6: {2: ((50, 50, 60, 60), 0.9)}}
    matches = {5: {9: 2}, 6: {}}
    g = {3: 1, 4: 2, 8: 99}
    assert oracle.categorise(3, 5, g, matches, gt) == "OCCLUDED_GAP"
    assert oracle.categorise(4, 5, g, matches, gt) == "TAKEOVER"            # g=2 matched to track 9
    assert oracle.categorise(4, 6, g, matches, gt) == "VISIBLE_MISS"
    assert oracle.categorise(3, 6, g, matches, gt) == "GONE"
    assert oracle.categorise(5, 6, g, matches, gt) == "FALSE_TRACK"


@pytest.mark.skipif(not Path("D:/TAORG_runs/observation/MOT17Labels.zip").exists(), reason="MOT17Labels not present")
def test_real_sequence_runs_deterministically():
    z = Path("D:/TAORG_runs/observation/MOT17Labels.zip")
    a = build_records(*_run(z))
    b = build_records(*_run(z))
    assert {k: [(r.frame_idx, r.detected, r.missing_reason) for r in v] for k, v in a.items()} == \
           {k: [(r.frame_idx, r.detected, r.missing_reason) for r in v] for k, v in b.items()}


def _run(z):
    info, dets = mot17.load_detections(z, "09", "FRCNN")
    return info, mot17.run_bytetrack(info, dets)
