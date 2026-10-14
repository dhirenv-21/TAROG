"""Pipeline maintain/view stages on a synthetic work directory (no GPU, no stage environments)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from taorg.experiments.downstream.io import write_csv_gz
from taorg.pipeline.stages import StageError, maintain
from taorg.pipeline.viewer import build_viewer


def make_work(tmp: Path, n: int = 30, flag_frame: int | None = None) -> Path:
    (tmp / "perception").mkdir(parents=True)
    outs = []
    for f in range(1, n + 1):
        outs.append({"frame": f, "track": 1, "det": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 20, "conf": 0.9})
        if f <= 20:
            outs.append({"frame": f, "track": 2, "det": 1, "x1": 50, "y1": 0, "x2": 60, "y2": 20, "conf": 0.9})
    write_csv_gz(tmp / "perception" / "outputs.csv.gz", outs, list(outs[0]))
    miss = [{"track": 2, "frame": f, "last_frame": 20, "dt": (f - 20) / 30, "a_glob": 0.1, "a_loc": 0.1, "n_cand": 1,
             "n_loc": 1, "has_s": True, "p_takeover": 0.9, "appearance_available": True, "flag": f == flag_frame}
            for f in range(21, n + 1)]
    write_csv_gz(tmp / "perception" / "missing_records.csv.gz", miss, list(miss[0]))
    (tmp / "perception" / "meta.json").write_text(json.dumps({"video": "toy.mp4", "frames": n, "fps": 30.0,
                                                               "width": 100, "height": 100}))
    fr = list(range(1, n + 1, 3))
    boxes, offs, sc = [], [0], []
    for f in fr:
        b = np.array([[0, 0, 10, 20], [50, 0, 60, 20]], np.float32)
        s = np.zeros((2, 2, 4), np.float32)
        s[0, 1, 3] = 0.5 if f <= 12 else 0.0
        boxes.append(b)
        sc.append(s.reshape(-1, 4))
        offs.append(offs[-1] + 2)
    (tmp / "sgg").mkdir()
    np.savez_compressed(tmp / "sgg" / "relations.npz", frames=np.array(fr, np.int32), offsets=np.array(offs, np.int64),
                        boxes=np.concatenate(boxes), obj_scores=np.full(2 * len(fr), 0.9, np.float32),
                        classes=np.zeros(2 * len(fr), np.int16), pair_scores=np.concatenate(sc))
    return tmp


def test_maintain_and_view(tmp_path):
    work = make_work(tmp_path)
    r = maintain(work, "FB", {"window_size": 3}, tau=0.1, use_flag=False)
    g = json.loads((work / "graphs.json").read_text())
    assert r["steps"] == 10 and g["frames"][0]["edges"][0]["p"] == "watch"
    on = [bool(f["edges"]) for f in g["frames"]]
    assert on[:4] == [True] * 4 and not any(on[8:])                # observed early, dropped later
    assert (work / "stream.csv.gz").exists()
    html = build_viewer(work, with_frames=False).read_text(encoding="utf-8")
    assert "TAORG pipeline viewer" in html and '"method": "FB"' in html


def test_maintain_with_flag_clears_endpoint(tmp_path):
    work = make_work(tmp_path, flag_frame=21)
    maintain(work, "B2", {"alpha": 0.5, "decay_visible": 0.5, "decay_occluded": 0.99, "score_threshold": 0.3,
                          "off_ratio": 0.5}, tau=0.1, use_flag=True)
    g = json.loads((work / "graphs.json").read_text())
    step22 = next(f for f in g["frames"] if f["frame"] == 22)
    assert step22["flagged"] == [2] and step22["edges"] == []


def test_maintain_errors_are_actionable(tmp_path):
    with pytest.raises(StageError, match="run the perception stage first"):
        maintain(tmp_path, "FB", {"window_size": 3}, 0.1, False)
    work = make_work(tmp_path / "w")
    with pytest.raises(StageError, match="FB and B2 only"):
        maintain(work, "VH", {}, 0.1, True)


def test_export_score_keeps_tiny_raw_scores():
    from taorg.pipeline.stages import export_score
    assert export_score(6.6481234e-7) == 6.648e-7 and export_score(0.123456) == 0.1235 and export_score(0.0) == 0.0
