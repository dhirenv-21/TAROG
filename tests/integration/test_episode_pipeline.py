"""Tiny end-to-end episode pipeline on SMOKE seeds (pipeline check only; never analysed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taorg.experiments.episodes import pipeline as P


def test_episode_pipeline_smoke_resume_and_completeness_gate(tmp_path):
    cfg = P.load_config(Path("configs/experiments/episodes.yaml"))
    for c in list(cfg["conditions"])[2:]:
        if c not in ("C6_id_switch", "S1_mixed_flagnoise"):
            del cfg["conditions"][c]
    out, run_dir = tmp_path / "res", tmp_path / "run"
    P.run_fit(cfg, out, smoke=True)
    P.run_calibration(cfg, out, run_dir, workers=2, smoke=True, max_samples=2)
    parts = sorted((run_dir / "calibration").glob("*.npz"))
    assert len(parts) == 3 * 2
    before = {p: p.stat().st_mtime_ns for p in parts}
    P.run_calibration(cfg, out, run_dir, workers=2, smoke=True, max_samples=2)        # resume: skips all
    assert {p: p.stat().st_mtime_ns for p in parts} == before
    meta = json.loads((out / "raw" / "stage_calibrate.json").read_text(encoding="utf-8"))
    assert meta["status"] == {"skipped": 6}
    frozen = P.run_selection(cfg, out, smoke=True)
    assert set(cfg["methods_order"]) <= set(frozen["selected"])
    P.run_evaluation(cfg, out, run_dir, workers=2, smoke=True)
    rows = P.read_csv_gz(out / "raw" / "evaluation.csv.gz")
    assert len(rows) == 4 * 2 * len(frozen["selected"]) and not any(r["error"] for r in rows)
    # identical manifests for every method on an instance (paired)
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], set()).add(r["manifest_sha256"])
    assert all(len(v) == 1 for v in by.values())
    P.run_latency(cfg, out, smoke=True)
    from taorg.experiments.episodes.analysis import run_analysis
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        run_analysis(cfg, out)                    # smoke data can never pass the completeness gate
