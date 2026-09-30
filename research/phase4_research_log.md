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
- `07b01c2` — **baseline parameter-freeze commit**.
- 12:33–12:39 — baseline evaluation from clean `07b01c2` (run `phase3_evaluate_baseline_20260929_070307_0e7494`,
  1980 instances, 351,990 rows, 385 s) + latency. Completeness (baseline group): 351,990/351,990,
  0 missing/duplicate/failed/unexpected, no calibration-seed leak.
- **Baseline-stage analysis, BEFORE any V-HPED calibration** (`results/phase4/baseline_stage/`):
  - A0 (VA-HMM − GatedEMA, frontier HV): within ±0.02 on all 14 calibrated conditions except the
    identity frontier (F5, F8), where VA-HMM is worse (≈ −0.04, CI < 0). Using visibility
    probabilistically adds nothing over deterministic gating once both are tuned.
  - With condition-matched tuning, the Phase 3 failures A and B largely disappear for GatedEMA.
    On F1 (termination during occlusion) and F2 (all UNKNOWN), every method, including
    visibility-unaware EMA, reaches nearly the same frontier (e.g. F2 gap 10: GatedEMA 0.898,
    EMA 0.899). This matches the a-priori expectation (prereg §9) that these conditions are
    information-limited.
  - Headroom observed now and recorded before V-HPED is run: GatedEMA's gap-10 HV is 0.957 (F3),
    0.953 (F4), 0.930 (F7), 0.919/0.511 (F8 default/identity), 0.982/0.510 (F5). N0/T3 = 1.000.
- `0c2c4ab` — baseline evaluation + baseline-stage analysis committed.
- V-HPED registered in `taorg.temporal.METHODS` (rungs `vhped_m2`–`vhped_m6`). The Phase 2
  `run_synthetic_eval` default method list was made explicit (the five Phase 1 baselines),
  because it tried to build every registered method without parameters. No Phase 2/3 result
  depends on it. Smoke run of the proposed group plus the full analysis, plots and failure
  extraction on scratch data (numbers not inspected): pipeline complete. 215 tests pass.
- `095101b` — V-HPED implementation commit (before any V-HPED calibration).
- 12:44–13:04 — proposed-group calibration from clean `095101b`
  (run `phase3_calibrate_proposed_20260929_071404_3d738e`, 840 instances, 12,000 config rows, 1227 s).
- Proposed selection (including the pre-registered half-sample tuning control for M5) and
  `freeze_params.py --group proposed`: all 10 checks PASS. Committed before the proposed evaluation.
- `490b28c` — V-HPED freeze commit; 13:05–13:10 proposed evaluation (126,630/126,630 rows) + latency;
  completeness COMPLETE for both groups, with identical manifests across groups.
- 13:10 — pre-registered analysis: **H2 = MIXED** (6/21, wins F3 g20, F5 ×3, F8 g10/20; 0 losses),
  tuning control MIXED (8 wins), **H3 = MIXED** (3/6, T1 only), **H4 = FAIL** (2 wins, 12 losses).
  Ablation retention: M3 yes, M4 no, M5 yes.
- Post-results scrutiny (disclosed): M5's best F1-condition frontier point had F1 0.975 at delay
  3.1 but false persistence 0.131. Diagnosis: M3+ filters **relapse** after the first
  post-termination drop, which the frozen first-drop delay metric cannot see. **Post-hoc
  exploratory diagnostics** were added (`scripts/phase4/exploratory_diagnostics.py`): a relapse
  rate and an F1-vs-false-persistence frontier. On the latter, M5 is worse than both baselines on
  F1/F7/F8, and only F3/N1 gap 20 (+0.01–0.02) and F5 (identity axis) survive. The
  pre-registered decisions were NOT changed; the diagnostics inform interpretation only.
- The pre-registered failure-regret rule shares the delay blind spot (it reports no F1 failure);
  reported as such.
- Exit report `research/phase4_exit_report.md`: METHOD CONTRIBUTION = NO; recommended pivot to an
  evaluation/failure-analysis paper. Phase 4 closed; awaiting approval.
- Correction to the baseline-stage entry above: A0 is within ±0.02 everywhere except the identity
  frontier (−0.04) **and F7 gap 20, where VA-HMM is +0.020 (CI > 0)**.
