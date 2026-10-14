# TAORG project completion

User direction, 2026-10-13: finish the full original project scope, including model
integration and the ADSA expiry heap, before further research additions.

## Verified starting point

The D: checkout is on `codex/taorg-downstream` at `6636a4e`. Existing demo output
changes, `outputs/`, and `scripts/phase5h1/visual_demo.py` are unrelated and preserved.
Completed study decisions remain unchanged. No new claim of algorithm novelty follows
from finishing engineering components.

## Completion order and acceptance gates

1. **Phase 5: state maintenance.** Add versioned scheduling, then connect lazy
   materialization to the eager filter only where its assumptions hold. Require
   randomized output/score equivalence, visibility-transition tests, recurrence,
   identity and cross-predicate fallback checks, and operation-count benchmarks.
   State explicitly the cost of producing every active edge in every frame.
2. **Phase 6: one real relationship predictor.** Verify compatibility, licence,
   checkpoint provenance, dataset and resource requirements before selecting it.
   Implement one adapter, document preprocessing and identities, and compare its
   raw outputs with the source implementation. A file loader alone is not complete
   model integration. Further substantial downloads require user approval.
3. **Runnable project.** Provide a documented entry point from model predictions
   through canonical records and temporal maintenance to saved graphs and a minimal
   demo. Label annotation-driven and synthetic examples explicitly.
4. **Final evaluation and submission.** Define the final protocol before new
   comparative results, preserve historical results, run the complete matrix, and
   produce reproducibility instructions, figures, ADSA derivations and paper draft.
   Earlier STOP results do not authorize another estimator search.

## Phase 5 mathematical boundary

The existing `VHPED` implementation is the four-state filter described in
`research/vhped_formulation.md`, not the initially proposed exponential Hawkes state.
That document's section 10 already identifies nonmonotone score trajectories and
time-varying identity evidence. Dropping a key permanently at its first threshold
crossing is therefore unsafe. A heap deadline must represent a justified next
decision change or reevaluation, and must retain the posterior needed for recurrence.
The scheduler itself does not derive deadlines or establish filter equivalence.

## Implemented foundation

`src/taorg/temporal/expiry_queue.py` supplies finite deadlines, replacement,
cancellation, globally unique generations, stale-entry checks, bounded compaction,
and counters. Three tests pass, including 2,000 seeded operations against a
dictionary oracle. Full Phase 5 integration and scaling benchmarks remain open.

## Phase 5 status: COMPLETE (2026-10-13)

**Exit gate met.**
- Optimised decisions match the eager reference: 28/28 benchmark runs with 0 decision mismatches
  and maximum |Δscore| 5.6e-16, plus 25 randomized/property tests.
- Complexity and latency are reproducible: `scripts/phase5/benchmark.py`, `results/phase5/`.

**Documents.**
- Derivation and report: `research/derivations/lazy_maintenance.md`. It covers the Birkhoff
  certificate (no monotonicity assumed), exact materialisation, the termination-delay bound and the
  per-frame complexity.

**Results.**
- Key updates fall about 4×. Wall-clock speed-up is 1.6–3.1× in typical settings.
- Lazy is **slower** when most keys are alive (0.79×) and under frequent frame-interval changes
  (0.47×).
- M5 identity is an eager fallback.
- Output enumeration Ω(E_t) remains; there is no sublinear claim.

## Phase 6 status: integration COMPLETE; predictor-level evaluation STOP

**Integration and verification.**
- EGTR is integrated through the official code (`research/taorg_egtr_integration.md`).
- The adapter is bit-identical to the official evaluation path on 200 VG test images. Subset recall
  agrees with the authors' within sampling variation.
- Licence, provenance and deviations are documented.

**Registered final evaluation** (`research/taorg_final_preregistration.md`): the development gate
failed, so Q1 and Q2 are **STOP** (`research/taorg_final_exit_report.md`).
- **Cause:** the registered τ grid ignored EGTR's tiny score scale (a pre-registration error), and
  the predictor's person–person signal on VidOR is weak.
- **What still passed:** E1 (lazy = eager) holds exactly on 30 real streams.

## Runnable project: COMPLETE

- **Entry point:** `python -m taorg.pipeline run --video V --work W` (`docs/PIPELINE.md`,
  `docs/SETUP.md`, `environments/`).
- **Smoke run:** recorded in `results/pipeline_smoke/`.
- **Tests:** 3 pipeline tests plus 5 final-stream tests.

## Paper: working draft COMPLETE

`paper/taorg_paper_draft.md` cites ledger entries L1–L72. Its `[VERIFY]` references still need
checking against the primary sources before submission.

## Remaining incomplete original requirements

- **Predictor-level evaluation on real VidSGG outputs** (L6, L72 UNTESTED). It needs a new
  registration with rank-normalised scores, or a video SGG model.
- **Action Genome standard evaluation; base models STTran, DSG-DETR and TEMPURA:** not done (scope).
- **Full ablation matrix on real data (Phase 8):** not done. The synthetic Phase 3/4 matrices
  exist.
- **Citation verification for the paper:** still to do.
