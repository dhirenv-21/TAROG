# TAORG Research — Claims Evidence Ledger

> **Purpose:** Track the evidential status of every paper claim.
> Each claim must eventually map to specific experiments, tables, or
> figures with traceable run IDs.
>
> Do NOT mark a claim as "supported" without linking to actual experiment artifacts.
> Negative or neutral results must also be recorded here.

## Claims Registry

| ID | Claim | Status | Evidence | Run IDs | Notes |
|----|-------|--------|----------|---------|-------|
| H1 | Visibility gating improves persistence/termination trade-off vs visibility-unaware methods | UNTESTED | — | — | Central hypothesis. Tested in Phase 3 pilot. |
| H2 | Cross-predicate Hawkes interactions provide measurable benefit beyond gated EMA | UNTESTED | — | — | May fail; if so, scope claims accordingly. |
| H3 | Learned temporal parameters outperform hand-set parameters | UNTESTED | — | — | Requires Phase 4+ V-HPED implementation. |
| H4 | Lazy decay + expiry scheduling matches eager reference within tolerance | UNTESTED | — | — | ADSA contribution. Phase 5. |
| C1 | Lazy/heap achieves O((k_obs+k_vis+k_exp) log A) vs O(A) eager | UNTESTED | — | — | Must be derived and empirically verified. |

## Negative / Neutral Results

| Date | Result | Experiment | Impact |
|------|--------|------------|--------|
| — | — | — | No experiments run yet. |

---

## Update Protocol
1. Add new claims when they emerge from experiments.
2. Update status: UNTESTED → TESTING → SUPPORTED / REFUTED / INCONCLUSIVE.
3. Link to run IDs and artifact paths.
4. Never delete refuted claims — mark them and record the evidence.
