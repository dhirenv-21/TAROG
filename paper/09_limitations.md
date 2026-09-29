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

8. **The primary (oracle) condition is circular in practice, not only in principle.**
   Calibration chose `decay_visible = 0`; the method then reads the answer from the flag
   (F1 = 1.0, delay 1 frame, zero variance). Only R1/R2 and the sweeps are informative.
9. **Win counts are fixed by the design grid.** The winning cells are exactly those with
   gap ≥ budget, a direct consequence of the bridging–delay coupling. Effect sizes and
   dose–response carry the evidence, not the count.
10. **Tuning dependence.** The advantage requires tuning `decay_visible` to the reliability of the
    visibility signal and of the detector. Oracle-tuned parameters are worse than Fixed Window/HMM
    under 20% flag noise and cause ≈ 10× churn under flicker.
11. **Structural failure modes.** Termination during occlusion (delay ≈ 27 vs 5 frames),
    UNKNOWN visibility (≈ 38 frames), ID-switch ghosts (≈ 2–4× ghost edge-frames) and a finite
    hold horizon (≈ 38 frames at the selected decay) are not addressed by per-frame gating.
12. **No visibility-aware probabilistic comparator.** Only GatedEMA uses visibility, so the
    results cannot show that this *form* of gating matters, only that using the flag does.
13. **Degenerate effect sizes.** Where both methods are deterministic across videos (C0), paired
    SDs are ≈ 0 and Cohen's d_z is undefined; such cells are reported but not interpreted.
