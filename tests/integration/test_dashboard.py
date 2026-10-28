"""Dashboard API: run discovery, summaries, input validation and replay fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from taorg.dashboard.app import create_app  # noqa: E402

REAL_RUNS = Path(__file__).resolve().parents[2] / "outputs" / "runs"
BASELINE = "20261027_202753_a7314fe5"


def _curve(thr: float, f1: float) -> dict:
    return {"threshold": thr, "frame_f1": f1, "persistence_f1": f1 / 2,
            "termination_delay_mean": 3.0, "false_persistence_rate": 0.1, "churn": 0.05}


def _metric_row(cond: str, method: str, f1: float) -> dict:
    keys = ["persistence_f1", "termination_delay_mean", "false_persistence_rate", "churn",
            "ece", "brier", "latency_ms", "selected_threshold"]
    return {"condition": cond, "method": method, "frame_f1": [f1, 0.01],
            **{k: [0.1, 0.01] for k in keys}}


@pytest.fixture()
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    run = root / "20260101_000000_aaaaaaaa"
    run.mkdir(parents=True)
    cfg = {"command": "scripts/run_ablation_matrix.py", "reference_method": "vhped_full_fit",
           "seeds": [1, 2], "splits": {"train": 1, "val": 1, "test": 2},
           "conditions": {"occlusion_flicker": {}},
           "methods": {"gated_ema": {"method": "gated_ema"},
                       "vhped_full_fit": {"method": "vhped", "fit": True}}}
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps(
        {"command": cfg["command"], "status": "completed", "seeds": [1, 2],
         "start_time": "2026-01-01T00:00:00+00:00", "timing": {"total_seconds": 60}}),
        encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({
        "kind": "SYNTHETIC ablation matrix",
        "table": [_metric_row("occlusion_flicker", "gated_ema", 0.9),
                  _metric_row("occlusion_flicker", "vhped_full_fit", 0.95)],
        "paired_vs_reference": [{"condition": "occlusion_flicker", "a": "vhped_full_fit",
                                 "b": "gated_ema", "frame_f1_video_ci95": [0.01, 0.02]}],
    }), encoding="utf-8")
    cells = [{"condition": "occlusion_flicker", "seed": s, "fitted": {},
              "methods": {m: {"test_curves": [_curve(0.2, f), _curve(0.6, f + 0.02 * s)]}
                          for m, f in (("gated_ema", 0.8), ("vhped_full_fit", 0.9))}}
             for s in (1, 2)]
    (run / "cells.json").write_text(json.dumps(cells), encoding="utf-8")

    dead = root / "20260102_000000_bbbbbbbb"
    dead.mkdir()
    (dead / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (dead / "INTERRUPTED.txt").write_text("killed before any cell finished", encoding="utf-8")

    cx = root / "complexity" / "20260103T000000Z"
    cx.mkdir(parents=True)
    (cx / "results.json").write_text(json.dumps({"args": {}, "rows": [
        {"speedup": 2.0, "decision_mismatches": 0}, {"speedup": 0.5, "decision_mismatches": 0}]}),
        encoding="utf-8")
    return root


@pytest.fixture()
def client(runs_root: Path) -> TestClient:
    return TestClient(create_app(runs_root))


def test_lists_runs_with_status(client: TestClient) -> None:
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert runs["20260101_000000_aaaaaaaa"]["status"] == "completed"
    assert runs["20260102_000000_bbbbbbbb"]["status"] == "interrupted"
    assert "killed" in runs["20260102_000000_bbbbbbbb"]["note"]
    assert runs["complexity/20260103T000000Z"]["kind"] == "complexity"
    assert list(runs)[0] == "complexity/20260103T000000Z"  # newest first


def test_matrix_summary_averages_curves_over_seeds(client: TestClient) -> None:
    m = client.get("/api/matrix/20260101_000000_aaaaaaaa").json()
    curve = m["curves"]["occlusion_flicker|vhped_full_fit"]
    assert curve["threshold"] == [0.2, 0.6]
    assert curve["frame_f1"] == pytest.approx([0.9, 0.9 + 0.03])
    assert m["reference_method"] == "vhped_full_fit"


def test_overview_reads_headline_and_complexity(client: TestClient) -> None:
    o = client.get("/api/overview").json()
    assert o["headline"]["run_id"] == "20260101_000000_aaaaaaaa"
    row = o["headline"]["rows"][0]
    assert row["reference"]["frame_f1"][0] == 0.95
    assert row["best_frame_f1"]["method"] == "vhped_full_fit"
    assert o["complexity"]["max_speedup"] == 2.0 and o["complexity"]["mismatches"] == 0
    assert o["counts"] == {"completed": 2, "interrupted": 1}


def test_unknown_or_incomplete_runs_are_rejected(client: TestClient) -> None:
    assert client.get("/api/matrix/nope").status_code == 404
    assert client.get("/api/matrix/20260102_000000_bbbbbbbb").status_code == 404
    assert client.get("/api/matrix/..%2F..%2Fetc").status_code == 404


def test_downloads_are_whitelisted(client: TestClient) -> None:
    ok = client.get("/api/files/20260101_000000_aaaaaaaa/metrics.json")
    assert ok.status_code == 200 and ok.json()["kind"] == "SYNTHETIC ablation matrix"
    assert client.get("/api/files/20260101_000000_aaaaaaaa/cells.json").status_code == 404
    assert client.get("/api/files/complexity/20260103T000000Z/results.json").status_code == 200


@pytest.mark.parametrize("patch, msg", [
    ({"condition": "bogus"}, "unknown condition"),
    ({"seed": 99}, "seed 99"),
    ({"video_index": 5}, "video_index"),
    ({"methods": []}, "at least one"),
    ({"methods": ["nope"]}, "unknown methods"),
])
def test_simulate_validates_input(client: TestClient, patch: dict, msg: str) -> None:
    body = {"run_id": "20260101_000000_aaaaaaaa", "condition": "occlusion_flicker", "seed": 1,
            "video_index": 0, "methods": ["gated_ema"], **patch}
    r = client.post("/api/explorer/simulate", json=body)
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_static_index_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "TAORG" in r.text
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.skipif(not (REAL_RUNS / BASELINE / "cells.json").exists(),
                    reason="saved baseline run not present")
def test_replay_matches_saved_per_video_results() -> None:
    """Replaying one saved test video reproduces the run's stored per-video frame F1."""
    client = TestClient(create_app(REAL_RUNS))
    body = {"run_id": BASELINE, "condition": "occlusion_flicker_idswitch", "seed": 42,
            "video_index": 0, "methods": ["vhped_full_fit", "gated_ema", "vhped_lazyexact_fit"]}
    sim = client.post("/api/explorer/simulate", json=body).json()
    assert sim["video_id"] == "te422000"
    assert all(r["matches_saved"] for r in sim["results"]), sim["results"]
    assert sim["id_switches"], "ID-switch condition should report its swaps"
    edge = sim["edges"][0]
    assert len(edge["gt"]) == sim["num_frames"]
    assert set(sim["scores"]) == set(body["methods"])
