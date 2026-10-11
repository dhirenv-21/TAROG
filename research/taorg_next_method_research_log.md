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
| 2 | Tested implementation | `ed9a021ba16e3a91191d882f93316ca74e4b68d9` |
| 3 | Frozen parameters | `8a7c916568bc4fd6eb95d599fb2b392107a80448` |
| 4 | Evaluation and analysis | `320bc2e984f31053e83500857ec0528483349f19` |
| 5 | Exit report and documentation | the commit adding this row (hash in the PR description) |

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
- 2026-10-11 (checkpoint 3, before any evaluation run):
  - fit → calibrate → select: 140/140 instances × 1200 configurations, 0 failed, 1324 s, code
    `ed9a021`, 4 workers.
  - The prior used 2958 OCC runs and 254 UNKNOWN runs from the calibration input streams.
  - B2 re-selected #0091 (identical to the episode study).
  - **Calibration observation:** all prior-based modes selected τ = 0.376 and κ = 838. κ swamps
    the per-video counts, so online adaptation is effectively off at the selected point.
    Calibration J: K1 0.9233 = static 0.9233; BUF 0.9235; pooled 0.9249; GEO 0.9224;
    B2 0.9029.
  - The half-sample control picked the same points.
  - Rationale: `results/next_method/configs/frozen_parameters.md`.
- 2026-10-11 (checkpoint 4):
  - evaluate (550 instances, 8250 rows, 77 s, code `8a7c916`), latency, analyze, figures and
    failures. Complete: 0 missing, failed or duplicate rows.
  - **Registered: STOP, contribution UNSUPPORTED.** All gates fail.
- 2026-10-11 (checkpoint 5):
  - **Post-hoc, exploratory, dev seeds only** (`scripts/next_method/posthoc_kappa_dev.py`): a
    weak prior lets K1 adapt in S2 (J 0.573 -> 0.826), but it stays below BUF (0.885) and gains
    C6 ghosts. No variant was re-evaluated on evaluation seeds.
  - **Implementation detail documented** (exit report §1): decisions use the current frame's
    flag stratum, while learning uses the run onset stratum. These are identical for runs with
    constant flags; the registered result is unchanged.
  - Wrote the exit report, supervisor update, ledger L51-L56, results notes and Memory.
