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
