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
