# Phase 4 research log

Chronological, factual log. Protocol: `research/phase4_preregistration.md`.

## 2026-09-29

- Phase 4 approved by the user with a strict method GO/NO-GO gate. Branch `phase4-vhped` created
  from `phase3-pilot` (`4dc57df`).
- Written before any V-HPED code: `research/vhped_formulation.md`,
  `research/phase4_preregistration.md`, `research/vhped_ablation_plan.md`,
  `configs/experiments/phase4.yaml` (SHA-256 `78b3dba3…6e752`), decision code
  `src/taorg/experiments/phase4/decision.py` + 11 tests. Test suite: 193 passed.
- Untracked Phase-3-era files classified as obsolete (B) and archived (not deleted) to
  `archive/pre_registration_pilot/` with a README.
- `bad0ef9` — **pre-registration commit** (no V-HPED code, no Phase 4 run).
- Infrastructure (after registration, before any run): `render_video` extracted from
  `generate_video` without behavioural change. Verified by re-building 156 Phase 3 evaluation
  instances (all manifest SHA-256s identical) and 52 stored Phase 3 metric rows (identical).
  New: bursty visibility and bbox-jitter injectors, predicate-succession world, exclusivity
  statistic, equal-budget sampler, visibility-aware HMM (M1), Phase 4 pipeline/CLI, freeze and
  completeness scripts; 20 new tests (202 total pass).
- Smoke run (seeds 9100–9101, 5 samples/method, scratch dir; numbers not inspected) exercised
  calibrate → select → evaluate → latency; about 11 ms CPU per configuration-instance run.
  It exposed a type bug in the freeze check (string vs int instance count), fixed before any real
  run.
- Observation about the frozen succession world (not a change): over the 20 calibration seeds,
  49 successions were added and 20 skipped because the partner key already existed in the video.
  The effective succession rate is ≈ 51% of terminations, not the configured 80%.
- 12:12 — baseline calibration started from clean `04941e2` (8 workers; 840 instances × 841
  configurations). Log: `results/phase4/logs/calibrate_baseline.log`.
- During the baseline calibration (new, uncommitted, unregistered files only; running workers had
  already imported the committed code): V-HPED eager reference `src/taorg/temporal/vhped.py` written
  from the frozen formulation, with 10 tests (`tests/unit/test_vhped.py`). Two numeric facts found
  while testing, both consistent with the formulation and recorded rather than hidden:
  (i) M2's "exact" flags use confusion 1e-6, so an OCC absence leaks O(1e-5) evidence;
  (ii) with no box jump the identity belief settles at u ≈ 0.9994 (not 1), so M5 ≈ M4 to within
  ≈ 0.004 in belief (formulation §7's "M5 = M4 outside switches" is approximate).
- Analysis, report, plot and failure modules written. **Failure-case selection rule fixed now,
  before any V-HPED evaluation**: `src/taorg/experiments/phase4/failures.py` docstring (per-seed
  regret of M5 vs the better of M0/M1 at gap 10, budget 5; worst and lower-median positive
  cases; M6 vs M5 on the H3 benefit conditions).
- Complexity and lazy-equivalence derivation: `research/derivations/vhped_complexity.md`.
- 12:12–12:25 — baseline calibration complete: run `phase3_calibrate_baseline_20260929_064253_a0796b`
  (the "phase3_" run-id prefix comes from the reused Phase 3 metadata helper; cosmetic), commit
  `04941e2`, clean, 840 instances, 11,774 config rows, 725 s.
- Baseline selection + `freeze_params.py --group baseline`: all 10 checks PASS →
  `results/phase4/configs/frozen_baseline.json`. Committed before the baseline evaluation.
