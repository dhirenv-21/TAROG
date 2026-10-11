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
| 2 | Tested implementation and experiment infrastructure | |
| 3 | Calibrated parameters and selection rationale (before evaluation) | |
| 4 | Evaluation summaries and analysis | |
| 5 | Exit report, evidence ledger, documentation | |

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
