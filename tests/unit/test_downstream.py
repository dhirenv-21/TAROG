"""Downstream study tests (research/taorg_downstream_preregistration.md §8)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from taorg.adapters.mot17 import FrameTracks
from taorg.adapters.vidor import VidorVideo
from taorg.experiments.downstream.io import content_sha256, write_csv_gz
from taorg.experiments.downstream.observations import build_stream, scoring_arrays
from taorg.experiments.downstream.runner import VideoData, make_method, run_method
from taorg.metrics.downstream import video_outcomes
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.flag_policy import EndpointInvalidation
from taorg.temporal.gated_variants import GatedEMAHysteresis

ROOT = Path(__file__).resolve().parents[2]
B2P = {"alpha": 0.3, "decay_visible": 0.5, "decay_occluded": 0.95, "score_threshold": 0.3, "off_ratio": 0.5}


def graphs(obs_by_frame: dict[int, list[EdgeKey]], T: int) -> list[FrameGraph]:
    return [FrameGraph(video_id="v", frame_idx=f, timestamp=f / 30, track_observations=[],
                       edge_observations=[EdgeObservation(k, "v", f, f / 30, 1.0, True) for k in obs_by_frame.get(f, [])])
            for f in range(1, T + 1)]


def outputs(method, gs: list[FrameGraph]) -> list[set]:
    method.reset_video("v")
    return [{o.edge_key for o in method.step(g).edge_observations} for g in gs]


A, B, C = EdgeKey(1, 2, 0), EdgeKey(3, 2, 0), EdgeKey(4, 5, 0)


@pytest.mark.parametrize("inner", [lambda: FixedWindow(window_size=10), lambda: GatedEMAHysteresis(**B2P)])
def test_no_flags_is_identical_to_unwrapped(inner):
    gs = graphs({f: [A, C] for f in range(1, 6)}, 20)
    assert outputs(inner(), gs) == outputs(EndpointInvalidation(inner(), {}), gs)


@pytest.mark.parametrize("inner", [lambda: FixedWindow(window_size=10), lambda: GatedEMAHysteresis(**B2P)])
def test_invalidation_either_endpoint_and_unrelated_untouched(inner):
    gs = graphs({f: [A, B, C] for f in range(1, 4)}, 8)
    out = outputs(EndpointInvalidation(inner(), {5: frozenset({2})}), gs)
    base = outputs(inner(), gs)
    assert A in base[4] and B in base[4]                       # held without the flag
    assert A not in out[4] and B not in out[4]                 # track 2 is object of A and B: both deleted
    assert C in out[4] and [C in o for o in out] == [C in o for o in base]   # unrelated key untouched
    out_s = outputs(EndpointInvalidation(inner(), {5: frozenset({1})}), gs)
    assert A not in out_s[4] and B in out_s[4]                 # subject endpoint


@pytest.mark.parametrize("inner,onset", [(lambda: FixedWindow(window_size=10), 0),
                                         (lambda: GatedEMAHysteresis(**B2P), 0)])
def test_recovery_after_false_flag(inner, onset):
    obs = {f: [A] for f in list(range(1, 4)) + list(range(8, 12))}
    out = outputs(EndpointInvalidation(inner(), {5: frozenset({1})}), graphs(obs, 12))
    assert A not in out[5] and A in out[7 + onset]             # re-observed at frame 8 -> output again


def test_b2_onset_after_invalidation_follows_normal_dynamics():
    p = dict(B2P, alpha=0.2)                                     # needs 2 observations to reach 0.3
    obs = {f: [A] for f in list(range(1, 6)) + list(range(9, 13))}
    w = outputs(EndpointInvalidation(GatedEMAHysteresis(**p), {7: frozenset({2})}), graphs(obs, 12))
    assert A not in w[8] and A in w[9]                           # frames 9, 10: score 0.2 then 0.36


def toy_video():
    """Persons 7 and 8; relation (7, 8, 'watch') frames 0-29. Tracker: track 1=person 7 frames 1-10,
    missing 11-15 (gap), back 16-20; then person 7 taken over by track 3 from 21; track 2 = person 8."""
    T = 30
    boxes = [{7: (0, 0, 10, 20), 8: (50, 0, 60, 20)} for _ in range(T)]
    v = VidorVideo("v", "x.mp4", 30.0, 100, 100, T, boxes, [(7, 8, "watch", 0, 30)])
    frames = []
    for f in range(1, T + 1):
        o = {2: ((50, 0, 60, 20), 0.9)}
        if f <= 10 or 16 <= f <= 20:
            o[1] = ((0, 0, 10, 20), 0.9)
        if f >= 21:
            o[3] = ((0, 0, 10, 20), 0.9)
        frames.append(FrameTracks(f, o, {}))
    return v, frames


def test_stream_and_labels():
    v, frames = toy_video()
    s = build_stream(v, frames)
    k1, k3 = EdgeKey(1, 2, 0), EdgeKey(3, 2, 0)
    assert s.keys == [k1, k3]
    assert s.observed[k1][1:11].all() and not s.observed[k1][11:16].any() and s.observed[k1][16:21].all()
    g12 = s.graphs[11]                                           # frame 12: track 1 missing -> no record
    assert all(o.edge_key != k1 for o in g12.edge_observations)
    lab = scoring_arrays(v, frames, s)[k1]
    assert lab["gap"][11:16].all() and lab["alive"][11:16].all()  # genuine gap
    assert lab["takeover"][21:31].all() and not lab["alive"][21:31].any()
    assert lab["both_out"][1:11].all() and not lab["both_out"][21]


def test_stream_prefix_does_not_depend_on_future_annotations():
    v, frames = toy_video()
    s = build_stream(v, frames)
    boxes = [dict(b) for b in v.boxes]
    for f in range(20, 30):
        boxes[f] = {7: (80, 80, 90, 99)}
    v2 = VidorVideo("v", "x.mp4", 30.0, 100, 100, 30, boxes, [(7, 8, "watch", 0, 20), (8, 7, "watch", 25, 30)])
    s2 = build_stream(v2, frames)
    for f in range(1, 21):
        assert s.graphs[f - 1].edge_observations == s2.graphs[f - 1].edge_observations


def run_toy(flags):
    v, frames = toy_video()
    s = build_stream(v, frames)
    d = VideoData(v, frames, s, flags, scoring_arrays(v, frames, s), [])
    return d, {fam: video_outcomes(run_method(make_method(fam, p, flags or None), d), d.labels, 30.0)
               for fam, p in (("FB", {"window_size": 15}), ("B2", B2P))}


def test_flag_suppresses_takeover_ghost_and_costs_gap_persistence_when_false():
    d, base = run_toy({})
    _, flagged = run_toy({21: frozenset({1})})                  # true flag at the takeover
    for fam in ("FB", "B2"):
        assert base[fam]["igt_s"] > 0 and flagged[fam]["igt_s"] < base[fam]["igt_s"]
        assert flagged[fam]["pgg_num"] == base[fam]["pgg_num"]
    _, false = run_toy({12: frozenset({1})})                    # false flag during the genuine gap
    for fam in ("FB", "B2"):
        assert false[fam]["pgg_num"] < base[fam]["pgg_num"]


def test_metric_toys():
    n = 41
    alive = np.zeros(n, bool)
    alive[1:11] = True
    alive[21:31] = True                                          # recurrence at 21
    z = np.zeros(n, bool)
    both = np.ones(n, bool)
    lab = {"k": {"alive": alive, "takeover": z, "gap": z, "both_out": both}}
    perfect = video_outcomes({"k": alive.copy()}, lab, 10.0)
    assert perfect["J"] == pytest.approx(1.0) and perfect["igt_s"] == 0
    assert perfect["std_g_n"] == 2 and perfect["std_g_sum"] == pytest.approx(0.2)   # off one frame after each end
    assert perfect["rdd_n"] == 1 and perfect["rdd_sum"] == 0.0
    late = alive.copy()
    late[21:24] = False                                          # recurrence detected 0.3 s late
    assert video_outcomes({"k": late}, lab, 10.0)["rdd_sum"] == pytest.approx(0.3)
    hold = alive.copy()
    hold[11:16] = True                                           # held 5 frames past a visible termination
    assert video_outcomes({"k": hold}, lab, 10.0)["std_g_sum"] == pytest.approx(0.6 + 0.1)   # off from frame 16: (16-10)/10; second end 0.1
    fail = video_outcomes({"k": z.copy()}, lab, 10.0)
    assert fail["J"] == 0.0
    tk = np.zeros(n, bool)
    tk[31:36] = True
    gap = np.zeros(n, bool)
    gap[5:8] = True
    lab2 = {"k": {"alive": alive, "takeover": tk, "gap": gap, "both_out": both}}
    o = alive.copy()
    o[31:36] = True
    o[6] = False
    r = video_outcomes({"k": o}, lab2, 10.0)
    assert r["igt_s"] == pytest.approx(0.5) and r["pgg_num"] == 2 and r["pgg_den"] == 3
    assert r["ftt_s"] == pytest.approx(0.1)
    none = video_outcomes({"k": z.copy()}, {"k": {"alive": z, "takeover": z, "gap": z, "both_out": both}}, 10.0)
    assert none["pgg_den"] == 0 and none["std_g_n"] == 0 and none["rdd_n"] == 0


def test_invisible_termination_is_not_a_genuine_termination():
    n = 31
    alive = np.zeros(n, bool)
    alive[1:11] = True
    lab = {"k": {"alive": alive, "takeover": np.zeros(n, bool), "gap": np.zeros(n, bool),
                 "both_out": np.zeros(n, bool)}}
    assert video_outcomes({"k": alive.copy()}, lab, 10.0)["std_g_n"] == 0


def test_flag_requires_appearance():
    from taorg.adapters.vidor_pipeline import flag_rows

    class M:
        def predict_proba(self, X):
            return np.tile([0.0, 1.0], (len(X), 1))
    rows = [{"dt": 0.5, "a_glob": 0.1, "a_loc": float("nan")}, {"dt": 0.5, "a_glob": 0.1, "a_loc": 0.1}]
    out = flag_rows(rows, M(), 0.5)
    assert [r["flag"] for r in out] == [False, True] and [r["appearance_available"] for r in out] == [False, True]


def test_frozen_flag_artifact_loads_with_checksum():
    from taorg.adapters.vidor_pipeline import load_flag_model
    m, theta = load_flag_model(ROOT / "results/appearance/freeze/frozen_models.pkl")
    assert theta == pytest.approx(0.37567710275995964)
    assert m.predict_proba(np.zeros((1, 6))).shape == (1, 2)


def test_inference_modules_have_no_annotation_path():
    for mod in ("src/taorg/adapters/vidor_pipeline.py", "src/taorg/temporal/flag_policy.py"):
        tree = ast.parse((ROOT / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", "") or ""] + [a.name for a in node.names]
                assert not any(x in n for n in names for x in ("oracle", "labels", "observations", "vidor")
                               if not n.endswith("vidor_pipeline")), (mod, names)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "data/vidor" not in node.value and ".json" not in node.value, mod


def test_deterministic_gzip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    h1 = write_csv_gz(tmp_path / "1.csv.gz", rows)
    h2 = write_csv_gz(tmp_path / "2.csv.gz", rows)
    assert (tmp_path / "1.csv.gz").read_bytes() == (tmp_path / "2.csv.gz").read_bytes()
    assert h1 == h2 == content_sha256(tmp_path / "1.csv.gz")
    assert not math.isnan(len(h1))
