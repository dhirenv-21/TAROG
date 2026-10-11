# TAORG relationship-episode study — research log

Branch `codex/taorg-episodes`, created 2026-10-11 from `phase5h-hawkes-feasibility` @ `b210619`
(the Phase 5H-1 registered-results commit; that branch has no PR and was 5 commits ahead of its
remote when work began). The PR is stacked on `phase5h-hawkes-feasibility`.

Unrelated working-tree changes found at start and **left untouched**:
- `demo/output/demo_data.json` and `demo/output/taorg_demo.html` (modified);
- `outputs/` and `scripts/phase5h1/visual_demo.py` (untracked).

## Checkpoint commits (full hashes)

| # | Checkpoint | Commit |
|---|---|---|
| 1 | Formulation, related work, pre-registration, decision rules (before comparisons) | `42b61ea92583be8ef54cda97bf077e79da500e63` |
| 2 | Tested implementation and experiment infrastructure | `e7e7804c0833de56bfa8f3f47e5bb514040dc85b` |
| 3 | Calibrated parameters and selection rationale (before evaluation) | `d0276343ec39e6a0a413a189904cf302636cca06` |
| 4 | Evaluation summaries and analysis | `c312fb951b21de79d065dbdd16875a0719faf335` |
| 5 | Exit report, evidence ledger, documentation | the commit that adds this row (its hash is reported in the PR description) |

## Log

- 2026-10-11:
  - Read AGENTS.md, Memory.md, Rules.md, Phases.md, the Phase 4 prereg and exit report, the
    evidence ledger and the metric definitions.
  - Verified that no episode formulation, pre-registration, code or experiment existed.
  - Checked the closest literature (`research/taorg_episode_related_work.md`).
  - Wrote the formulation, pre-registration, config and decision code; 8 decision-rule tests pass.
  - No method code existed at checkpoint 1.
- 2026-10-11 (checkpoint 2):
  - Implemented the episode world, injectors, identity module, B2/B4/B5, the episode ladder P1–P4,
    episode metrics, the checkpointed pipeline, analysis, figures and failure extraction.
  - Tests: estimator 12, baselines 7, metrics 6, world/injectors/sampler/fitting separation 10,
    pipeline smoke 1, decision 8. Full suite 302 passed.
  - **Implementation issue found on smoke data and fixed in the test, not the injector.** A world
    test assumed every record inside a hidden-termination block is an OCC absence. The registered
    order applies background false detections *after* hidden termination, so a registered false
    detection can land inside a block (seed 9300, key (1,6,0), frame 186). The injector follows the
    pre-registration; the test now exempts exactly the records listed in the false-detection
    manifest.
  - **Smoke run** (seeds 9300–9301, 3 samples per method; never analysed): every stage completed.
    One worker died at import with "paging file is too small": C: has about 1 GB free and only
    about 3 GB of RAM was free. The pool respawned the worker and no task was lost, but the full
    runs will use ≤ 6 workers.
- 2026-10-11 (checkpoint 3, before any evaluation run):
  - `fit` → `calibrate` → `select` with 6 workers. Calibration covered 140/140 instances × 1801
    configurations, status `{'done': 140}`, 0 failed, 1083 s, code at `e7e7804`.
    - `git_dirty = True` refers only to the unrelated demo files and the uncommitted
      `results/episodes/README.md`; no code was modified.
    - The heavy parts (53 MB) are in `D:/TAORG_runs/episodes/calibration/`, with checksums in
      `results/episodes/raw/calibration_parts_sha256.json`.
  - Selection rationale: `results/episodes/configs/frozen_parameters.md`.
  - **Observations at calibration (not decisions):**
    - Pooled J is highest for the episode ladder (P1 0.9286, P2 0.9291, P3 0.9303, P4 0.9302).
      The ladder steps are ≤ 0.002 at calibration.
    - The next are B2 0.9029, B4 0.8925, B3 0.8915, B1 0.8778 and B5 0.8575.
    - Adding identity to GatedEMA lowered its pooled calibration J.
    - P2–P4 selected the same base index (#0101, a common-random-numbers draw); P1 selected #0176.
  - The decisions rest only on the registered evaluation analysis.
- 2026-10-11 (checkpoint 4):
  - `evaluate` (550 instances, 11,000 rows, 45 s, code `d027634`), `latency`, `analyze`, `figures`
    and `failures` completed.
  - Completeness: 11,000/11,000 rows; 0 missing, 0 duplicate, 0 failed; manifests consistent;
    calibration 140/140 instances; no calibration-seed leak (`aggregated/completeness.json`).
  - **Registered decision: STOP.** All three STOP clauses hold:
    - 0 benefit-condition wins (≤ 1);
    - 6 losses vs 1 win over the 11 conditions;
    - GatedEMA + hysteresis is never beaten in C2/C3/C5.
  - The half-sample control also gives STOP.
  - Ladder: identity not retained (J loss in C6); context retained vs P2 (C3 ΔJ +0.024); history
    not retained (pooled ΔJ +0.00004). T = P1.
  - Q1 NO, Q2 NO, Q3 NO, Q4 NO, Q5 "no in-distribution advantage to transfer", Q6 NO.
  - The details go to the exit report. No post-hoc analysis has been run yet.
- 2026-10-11 (checkpoint 5):
  - **Post-hoc, exploratory** diagnostics, defined after the registered STOP
    (`scripts/episodes/posthoc_diagnostics.py`, `results/episodes/posthoc/`):
    - E1, attribution: with the base configuration held fixed, the identity mechanism changes C6
      ghost time by +0.03 s. The ladder's identity effect comes from base-parameter selection.
    - E2, ghost timing: P1's ghosts mostly linger after swaps end.
  - Wrote the exit report, ledger L43–L50, results notes, Memory.md and registry rows.
  - Full suite: 302 passed.
