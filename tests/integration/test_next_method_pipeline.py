"""Tiny next-method pipeline on SMOKE seeds: prior from inputs, resume, pairing, completeness gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taorg.experiments.next_method import pipeline as P


def test_next_method_pipeline_smoke(tmp_path):
    cfg = P.load_config(Path("configs/experiments/next_method.yaml"))
    for c in list(cfg["conditions"]):
        if c not in ("C1_gaps", "C2_hidden_termination", "S2_mixed_unknown"):
            del cfg["conditions"][c]
    out, run = tmp_path / "res", tmp_path / "run"
    prior = P.run_fit(cfg, out, smoke=True)
    assert all(x.split("|s")[1] in ("9400", "9401") for x in prior["fitted_on"])
    assert not any(x.startswith("S") for x in prior["fitted_on"]), "transfer conditions never feed the prior"
    P.run_calibration(cfg, out, run, workers=2, smoke=True, max_samples=2)
    parts = sorted((run / "calibration").glob("*.npz"))
    assert len(parts) == 2 * 2
    P.run_calibration(cfg, out, run, workers=2, smoke=True, max_samples=2)
    assert json.loads((out / "raw" / "stage_calibrate.json").read_text())["status"] == {"skipped": 4}
    frozen = P.run_selection(cfg, out, smoke=True)
    assert {"gated_ema_hyst", "b2_hold_km", "b2_hold_buffer"} <= set(frozen["selected"])
    P.run_evaluation(cfg, out, run, workers=2, smoke=True)
    rows = P.read_csv_gz(out / "raw" / "evaluation.csv.gz")
    n_var = len(frozen["selected"]) + len(frozen["references"])
    assert len(rows) == 3 * 2 * n_var and not any(r["error"] for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["instance_id"], set()).add(r["manifest_sha256"])
    assert all(len(v) == 1 for v in by.values())
    P.run_latency(cfg, out, smoke=True)
    from taorg.experiments.next_method.analysis import run_analysis
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        run_analysis(cfg, out)
