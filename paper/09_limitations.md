# 9. Limitations — DRAFT (grows with each phase)

## 9.1 Declared before the Phase 3 pilot (research/phase3_preregistration.md §10)

1. **Synthetic data only.** The pilot uses a parametric synthetic world with exactly known
   relationship lifetimes. No real VidSGG predictions, detector, or tracker are involved.
   Nothing measured there may be stated as a property of Action Genome or any benchmark.
2. **Oracle visibility is tautological.** When the occlusion injector itself sets the
   visibility flag, a visibility-gated method is advantaged by construction (see §5 P7). Only
   the degraded-visibility and flicker conditions test robustness, and even those use a
   simplified i.i.d. per-frame error model; real visibility errors are temporally correlated
   and correlated with termination events.
3. **Frame-rate dependence.** All Phase 1 methods are per-frame recursions; results hold at
   30 fps only (§4.2).
4. **Metric blind spots.** Persistence F1 ignores frames outside occlusion intervals; the
   churn implementation averages in never-GT edges with churn 0; termination metrics only use
   each edge's final termination. Diagnostics partially compensate but are not frozen.
5. **Tuning freedom.** Vis-Gated EMA has one more parameter than EMA and hence a larger grid;
   selection uses calibration seeds only, so this cannot leak into evaluation, but the
   treatment receives more tuning freedom.
6. **Weak HMM baseline.** The HMM is a hand-parameterised scaffold (grid-searched, not fitted
   by Baum–Welch).
7. **Single simple world.** One fixed edge-count, lifetime distribution and score model;
   no dense scenes, no correlated multi-edge occlusion (an occluded object should affect all
   its edges; the injector corrupts edges independently).

## 9.2 Added after the Phase 3 results

[TBD — filled from research/paper_results_notes.md after the pilot.]
