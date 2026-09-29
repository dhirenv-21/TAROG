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
- 05:28–09:01 — the first calibration run stalled (200 instances at 05:48, 250 at 09:01) and was
  terminated; no data was written. **Amendment A2** (engineering only).
- 09:14–11:19 — calibration re-run from clean `929d082`, 8 workers, 1080/1080 instances, 7468 s
  (`results/phase3/raw/stage_calibrate.json`, run `phase3_calibrate_20260929_034436_e4842e`).
- 11:22 — `--stage select` (calibration artifacts only) → `configs/selected_params.json`,
  `configs/pareto_sets.json`.
- `scripts/phase3/freeze_params.py` → `results/phase3/configs/frozen_parameters.json` +
  `results/phase3/tables/frozen_parameters.md`. All 12 calibration checks passed: not smoke, full grid
  (6213 configs), 1080/1080 expected instances with no duplicates or missing ones, calibration
  seeds 1000–1019 only, no evaluation seed, every config saw every instance, no NaN F1/delay.
- Serial recomputation of the 11 C0 selected configurations on all 120 C0 calibration instances
  reproduced the saved means (max |Δ| = 1.8e-15).
- Observation from calibration (not evaluation): in C0 the selected Vis-Gated EMA has
  `decay_visible = 0` and calibration F1 = 1.000 at mean delay 1.0 frame for every budget, i.e. it
  degenerates to "drop on visible absence, hold on occluded absence". This is the oracle tautology
  declared in prereg §2, observed directly. 143–413 configurations tie at F1 = 1.0 (tie broken
  by the registered order).
- No evaluation stage had been run when the parameters were frozen and committed.
- `2ce7d1f` — **parameter-freeze commit** (selection, frozen parameters, calibration artifacts, A2).
- 11:23–11:26 — `--stage evaluate` from clean `2ce7d1f` (run `phase3_evaluate_20260929_055335_6ac64a`,
  8 workers, 177 s, 2340 instances, 156,240 rows); `--stage latency` (900 rows).
- `check_completeness.py`: COMPLETE — 156,240/156,240 runs, 0 missing/duplicate/failed/unexpected,
  seeds 2000–2029 only, no calibration seed, 2340/2340 manifests, 900/900 latency rows.
- `--stage analyze` → registered H1 rule output **PASS** (10/10/10 wins, 0 losses, frontier 5/4/5 of
  6); `--stage plots`, `--stage failures`, `dominance_analysis.py` (exploratory class B overall).
- `867b7be` — evaluation and analysis artifacts committed before any interpretation was written.
- Post-hoc observations (disclosed, not used to change anything): H1 wins are exactly the gap ≥ budget
  cells; C0-tuned GatedEMA is worse than Fixed Window/HMM on R1 and has ≈ 10× churn on R2
  (secondary analysis); d_z is degenerate (≈ 1e15 / inf) in zero-variance cells; 3 Holm-significant
  results favour the comparators (R1, budget 2, gap 1, ΔF1 −0.032).
- Full evaluation re-run into a scratch directory: 0 of 156,240 rows differ (excluding run ID and
  wall-clock runtime).
- Exit report `research/phase3_exit_report.md`; ledger, results notes and paper §§2, 6, 7, 9 updated.
  Phase 3 closed; waiting for approval before Phase 4.
