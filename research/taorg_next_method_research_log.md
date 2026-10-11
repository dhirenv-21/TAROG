# TAORG next-method study — research log

Branch `codex/taorg-next-method`, created 2026-10-11 from `codex/taorg-episodes` @ `ecb3859`
(PR #3). The new PR is stacked on `codex/taorg-episodes`.

Unrelated working-tree items were left untouched and are never staged:
- `demo/output/demo_data.json`, `demo/output/taorg_demo.html` (modified);
- `outputs/`, `scripts/phase5h1/visual_demo.py` (untracked).

## Checkpoint commits (full hashes)

| # | Checkpoint | Commit |
|---|---|---|
| 1 | Failure audit, formulation, pre-registration | `fe084a6cce6d5500b2b10b8c390bdfc0c4eb9df1` |
| 2 | Tested implementation | |
| 3 | Frozen parameters | |
| 4 | Evaluation and analysis | |
| 5 | Exit report and documentation | |

## Log

- 2026-10-11 (relocation):
  - The repository was copied by the user from C: to `D:\ADSA project`.
  - The editable install was re-pointed to D:.
  - All episode checkpoint checksums were re-verified (0 mismatches); 302 tests passed.
  - Free RAM is about 1.6 GB, so studies use ≤ 4 workers.
- 2026-10-11 (audit):
  - Read the calibration artifacts.
  - Ran `scripts/next_method/audit_dev.py` on dev seeds 7000–7019 (220 instances × 205 runs).
  - Ran two development pilot iterations (`scripts/next_method/dev_pilot.py`, dev seeds
    7000–7009):
    - iteration 1 negative (re-detection ≠ persistence);
    - iteration 2 positive vs B2, but static ≈ online.
  - Literature check.
  - Selected K1 over K2 and K3 (audit §3).
  - Wrote the formulation, pre-registration, config and decision code (2 tests).
  - The final eager reference `src/taorg/temporal/cure.py` exists but is **not** part of
    checkpoint 1. The pilot prototype snapshot is committed for provenance.
- 2026-10-11 (checkpoint 2):
  - Implemented the eager reference `B2SurvivalHold` (modes km/static/pooled/geometric/buffer),
    registered as `b2_hold_*`, plus the pipeline (prior → calibrate → select → evaluate →
    latency), analysis, figures and failure rule.
  - 11 unit tests in `test_cure.py`: B2 equivalence without non-visible absences; hold vs the
    visible rule; P1; P2 (static == buffer); P3 by brute force on 200 random cure models; online
    KM counts vs brute force; online learning moves the horizon; recurrence; irregular timestamps;
    numerics, determinism and causality; prior uses inputs only.
  - Plus 2 decision tests and 1 pipeline integration test. Full suite 316 passed.
  - **Two test-design errors were fixed in the tests (no model change):**
    - a flat prior hazard cannot show learning, so the test now uses a prior that says
      "return within 5 bins or never";
    - a malformed leftover assertion was removed.
  - Smoke run (seeds 9400–9401, 3 samples): all stages run; the completeness gate refuses smoke
    data.
