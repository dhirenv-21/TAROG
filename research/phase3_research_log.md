# Phase 3 research log

Chronological, factual log of the Phase 3 pilot execution. Times are local (IST, UTC+05:30)
unless marked UTC. Protocol: `research/phase3_preregistration.md` (registered in `d8728b3`).

## 2026-09-28

- `d8728b3` — pre-registration, config (`configs/experiments/phase3_pilot.yaml`,
  SHA-256 `0dea058e…9ad52`) and H1 decision code committed before any run.
- `6b6bced` — Amendment A1 (Windows spawn workers exhausted the paging file; deferred imports,
  default ≤ 16 workers). Engineering only. Smoke run repeated; numbers not inspected.
- No calibration output was produced on 2026-09-28: `results/phase3/` held only the resolved config
  (written 09:08) and no Python process was running when work resumed on 2026-09-29. Calibration
  writes all artifacts only at the end of the stage, so an interrupted run leaves nothing on disk;
  no partial calibration data exists or was used.

## 2026-09-29

- Verified before starting calibration: config SHA-256 equals the registered value; the only code
  change since `d8728b3` is A1; 175/175 tests pass.
- `39bb238` — committed the reporting-only tables module (`report.py`, no decision logic), the paper
  skeleton and the evidence ledger, so the calibration stage records a clean commit.
- 05:28 — full calibration started (`--stage calibrate`, 16 workers, all 9 calibrated conditions ×
  6 gaps × calibration seeds 1000–1019 = 1080 instances × 6213 configurations). Log:
  `results/phase3/logs/calibrate.log`. Observed throughput ≈ 6.6 s per instance (≈ 2 h total).
- Free commit memory was ≈ 1.2 GB during calibration, so no parallel smoke run was started
  (A1 failure mode).
- **Exploratory Pareto-dominance definition added before any evaluation data exists**
  (`src/taorg/experiments/phase3/dominance.py`, tests in `tests/unit/test_phase3_dominance.py`,
  runner `scripts/phase3/dominance_analysis.py`). The pre-registration defines only the hypervolume
  frontier criterion; "Pareto-dominates" had no formal definition. The new definition (frontier set
  dominance inside the registered hypervolume box, with a bootstrap dominance probability, and an
  A/B/C classification) is exploratory, is not part of the H1 rule, and cannot change the H1
  decision. It was committed while calibration was still running, i.e. before selection, freezing
  and evaluation.
