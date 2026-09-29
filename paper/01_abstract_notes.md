# 1. Abstract — NOTES ONLY (no abstract may be written yet)

A responsible abstract requires: (a) the final proposed method (V-HPED or a narrower
alternative), (b) real-dataset experiments on base-model predictions, (c) main quantitative
results traceable to run artifacts. **None of these exist yet.**

## Elements the abstract will likely need (conditional)

- Problem: relationship outputs of VidSGG models flicker under occlusion / tracking failure;
  bridging gaps trades off against timely termination.
- Positioning: model-agnostic post-processing layer; no retraining; explicit
  persistence/termination trade-off. Must acknowledge that temporal/object-permanence
  modelling already exists inside VidSGG / world-scene-graph models.
- Evidence: [TBD — depends on Phase 3 outcome and later phases; see
  research/paper_evidence_ledger.md]. Only SUPPORTED claims may appear.
- Evaluation protocol (C3) as a possible contribution — contingent on a literature check that
  no equivalent protocol exists for VidSGG.

## Phrases that must NOT appear

"first to handle occlusion", "existing methods cannot handle occlusion", "state of the art",
any number not present in a generated results table.
