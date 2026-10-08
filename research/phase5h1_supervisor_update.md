# Phase 5H-1 — Supervisor update (8 Oct 2026): registered comparison complete

> **Result: HAWKES TRACK = NARROW** under the pre-registered rule. Evidence class:
> **EXPLORATORY, annotation-only** (human VidOR relation intervals; no detector or tracker).
> All numbers come from `results/phase5h1/full/` and are checked by
> `scripts/phase5h1/verify_results.py`. The full account is in
> `research/phase5h1_exit_report.md`.

## Question

Does the recent history of *other* relations on the same object pair (e.g. "release starts")
help predict when a relation starts or ends, **beyond** pair heterogeneity, relation duration
and the current relation state?

## What was run (registered order, nothing retuned)

- Fit on 5600 TRAIN videos, ridge selected on 700 SELECT videos.
- **Parameters frozen and committed (`289319e`) before evaluation data was unlocked.**
- Held-out evaluation on 1535 videos (84,715 events).
- 38 null replicates (19 N-circ + 19 N-jit), 5-fold stability with 5 fold nulls, and 4
  hypothesis tests.
- The full matrix completed with 0 missing runs and 0 failures.

## Positive findings

| Finding | Value |
|---|---|
| Cross-event history beyond heterogeneity + duration (K1) | **+0.145** nats/event, above every null (max +0.031) |
| History beyond the simple transition model (K2) | **+0.083** [0.076, 0.090], above every null (max +0.017); ≈ 1.09× likelihood per event |
| Practical size (K3) | passes |
| Both evaluation halves agree | yes (EVAL-A and EVAL-B CIs > 0) |
| Families passing the registered test | SPATIAL, MOTION |
| Mechanism | "another relation starts → this one ends" (+0.048) at lags ≤ 2 s; inhibition matters |
| Stage 0 transitions replicated out of sample | hold→grab (RR 4.93), release→hold (RR 44.5) |

## Negative findings (same weight)

| Finding | Value |
|---|---|
| Individual interactions reproducible across folds (K4) | **No.** 0 discoveries; the minimum FDR is 0.16. Real 5,924 sign-consistent cells vs null 5,589–5,857 |
| Positive-only (classical) Hawkes | **worse** than signed history: −0.031 nats/event |
| Contact / vehicle / attention families | fail (contact misses the practical bar; vehicle CI includes 0) |
| Annotation artefact risk | 17.0% of events tie exactly with another relation's event on the same pair: synchronised annotation boundaries |
| Null quality | N-circ adds +16% events, N-jit loses −9.3% to −9.8%; neither is a clean jitter test |
| K4 statistic | inflated by ridge shrinkage in sparse cells (post-hoc); a design flaw of our registration |

Inter-fold weight agreement does beat the nulls (Spearman 0.299 vs max 0.270). So the pattern
of weights is reproducible, but no single interaction is.

## Interpretation

- Cross-predicate timing carries real predictive information about **annotated relation
  boundaries**, mostly spatial and motion relabelling within 2 s.
- This is predictive association in annotation timing. **It is not causal and not shown on
  real video.**
- It does **not** justify a method contribution. The model is a standard point-process GLM,
  and classical Hawkes is the worse variant.

## Decision requested

1. Accept NARROW. Keep the **robustness/evaluation paper** (Phases 3–4) as the main track, and
   report this as an exploratory analysis section (draft:
   `research/paper_notes/phase5h1_exploratory_section.md`).
2. Optionally approve a **confirmatory protocol** on an independent dataset (ImageNet-VidVRD or
   PVSG). It would use repaired nulls (reflection jitter, boundary-tie control) and a
   support-aware stability statistic, and would be pre-registered before any data access
   (exit report §15). **Not started.**

## Materials

- Figures (`results/phase5h1/full/`):
  - `fig_main_comparison` — main comparison with CIs;
  - `fig_null_comparison` — nulls;
  - `fig_ablation_stability` — ablations and family test;
  - `fig_k4_stability` — negative K4 result.
- Deck: `results/phase5h1/showcase/phase5h1_supervisor_update.pptx` (updated). Script:
  `research/phase5h1_presentation_script.md`.
- **Separate from the research results:** the synchronised video demo (`demo/`) uses a
  *synthetic* scenario with hand-picked settings. It illustrates the problem; it is not evidence
  and contains no Phase 5H-1 numbers.
