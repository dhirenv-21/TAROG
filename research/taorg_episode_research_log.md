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
| 1 | Formulation, related work, pre-registration, decision rules (before comparisons) | (recorded after commit) |
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
