# Paper results notes — Phase 3 (controlled synthetic evidence only)

> Every number below is copied from a generated artifact under `results/phase3/` (paths given).
> Phase 3 is a **synthetic** pilot: no real VidSGG model, tracker or dataset is involved.
> V-HPED is **not implemented**; nothing here is evidence about V-HPED.
> Evaluation run `phase3_evaluate_20260929_055335_6ac64a` (commit `2ce7d1f`, clean tree), seeds
> 2000–2029 (n = 30 videos). Operating points below are budget = 5 frames unless stated.

## Pre-registered outcome

**H1 = PASS** under the rule in `research/phase3_preregistration.md` §8
(`results/phase3/aggregated/h1_decision.json`): 10/18 cell wins and 0 losses in each of C0, R1 and
R2; frontier criterion in 5/6 (C0), 4/6 (R1), 5/6 (R2) gaps.

## Strongest result

**Visibility-gated persistence survives both kinds of signal degradation, at unchanged
termination cost.** With a 20%-noisy visibility flag (R1), GatedEMA keeps Persistence F1 at
0.953 (gap 10) and 0.808 (gap 20). The visibility-unaware methods reach 0.550–0.571 and
0.315–0.333, with mean termination delay 4.36–4.44 vs 4.82–5.00 frames. With 10% base-model
flicker (R2), GatedEMA holds F1 = 1.000 at every gap, with delay 4.33–4.38 vs 4.67–4.90 frames
(`tables/results_R1_visnoise_0.2.md`, `tables/results_R2_flicker_0.1.md`). Frontier hypervolume
gains are +0.083 (R1) and +0.106 (R2) at gap 10 against the strongest comparator, with bootstrap
CIs excluding 0 (`tables/hypervolume.md`). The flicker advantage persists up to q = 0.3
(F1 ≥ 0.993 at every gap vs ≤ 0.873 for all baselines at gap ≥ 5, `tables/sweeps.md`).

## Weakest result

**The oracle condition C0, which is the H1 *primary* condition, is close to a tautology.** In C0,
calibration selected `decay_visible = 0` for every budget. GatedEMA therefore reduces to "drop on
a VISIBLE absence, hold on an OCCLUDED absence". It reaches F1 = 1.000 at delay 1.0 frame with
zero variance across seeds, and its hypervolume is exactly 1.000 at every gap. This was predicted
in prereg §2 and is not evidence of a useful mechanism. The paired statistics there are
degenerate: Cohen's d_z is printed as ~10^15 or ±inf because the SD of the paired differences is
floating-point noise (`tables/effect_sizes.md`). C0 should be presented as a sanity check, not as
the headline result.

## Structural caveat on the win count (observed after results; must be disclosed)

In every H1 condition the 10 winning cells are **exactly the cells with gap ≥ budget**:
budget 2 → gaps 2–20, budget 5 → gaps 5–20, budget 10 → gaps 10–20. When gap < budget, every
visibility-unaware smoother bridges the gap within its delay budget, so the cells are neutral.
When gap ≥ budget, no visibility-unaware method can bridge the gap without exceeding the budget.
This follows from the bridging/termination coupling (ledger L4; empirical check below). The
decision rule was fixed before the data, so the PASS is valid. However, the *number* of wins is
largely determined by the grid of (gap, budget) pairs, not by a property of GatedEMA beyond
"uses an informative visibility flag". The informative content of the PASS is the size of the
margin in R1/R2 and the sweeps, not the win count.

Empirical check of the coupling (C0, evaluation frontier points,
`aggregated/frontier_points.csv`): the minimum mean delay among visibility-unaware configurations
that fully bridge gap g is Fixed Window g+1 exactly, EMA ≥ g+1 (2.0, 3.46, 4.42, 8.16, 12.83,
26.45), and HMM 1.988, 2.957, 3.949, 5.993, 12.83 (none bridges g = 20). HMM is *slightly below*
g+1 on the mean at g ≤ 5, so the "delay ≥ g+1" statement is not exact for score-dependent
methods on the mean. It needs a proof with correct conditions (per edge, equal state at gap start
and at termination).

## Unexpected results

1. **Tuning mismatch reverses the advantage** (secondary analysis, prereg §5.4,
   `tables/tuning_mismatch.md`). C0-tuned GatedEMA (`decay_visible = 0`) evaluated on R1 is
   **worse than Fixed Window and HMM**: F1 0.864 / 0.703 / 0.511 at gaps 1 / 5 / 10 vs
   1.000 / 0.889 / 0.571. Evaluated on R2 it has churn 0.154–0.178 vs 0.015 for the baselines
   (≈ 10–12×) and clean-period recall 0.90 vs 0.999. The benefit depends on tuning
   `decay_visible` to the reliability of the visibility signal and of the detector.
2. **ID switches cut both ways** (`tables/stress.md`, S2). GatedEMA keeps the true key alive
   through a swap: swap recall 0.864 [0.795, 0.933] vs 0.248–0.262 at d = 20 frames. It also keeps
   the ghost key alive: 251.1 [209.1, 293.1] ghost edge-frames vs 110.8–112.0. Both follow from
   one code path: an edge key with **no record** in a frame is treated as UNKNOWN, and
   `decay_unknown` defaults to `decay_occluded` (`src/taorg/temporal/gated_ema.py`).
3. **A small but statistically significant loss at the shortest gap.** In R1 at budget 2, gap 1,
   GatedEMA has ΔF1 = −0.032 [−0.048, −0.018] against all three comparators (Holm p = 0.033).
   This is below the −0.05 loss threshold, so it is not a rule-defined loss, but it is a real
   degradation caused by noisy flags at 1-frame gaps.

## Negative results

- **No advantage when the visibility flag is uninformative.** At ρ = 0.5, GatedEMA F1 is
  0.886 / 0.582 / 0.341 at gaps 5 / 10 / 20 vs Fixed Window 0.889 / 0.571 / 0.333
  (`tables/sweeps.md`). The advantage shrinks monotonically with ρ: gap 10 gives 0.998 (ρ 0.1),
  0.953 (ρ 0.2, R1), 0.888 (ρ 0.3), 0.582 (ρ 0.5). This is the expected negative control and
  supports the claim that the gain comes from the visibility signal and nothing else.
- **Formal Pareto dominance does NOT hold overall** (exploratory definition committed before the
  evaluation data, `tables/dominance.md`). The classification is **B (condition-specific)**: A in
  C0, B in R1 (no dominance at gaps 1–3) and B in R2 (no dominance over EMA/Fixed Window at gaps 1
  and 20, p_dom 0.37). GatedEMA's frontier is never dominated by any comparator (reverse p = 0
  everywhere); the frontiers cross.
- **Stress conditions, C0-tuned, budget 5** (`tables/stress.md`): termination under occlusion (S1)
  gives delay 27.4 [26.2, 28.7] frames vs 4.8–5.0 and false persistence 0.182 vs 0.026–0.028.
  All-UNKNOWN visibility (S4) gives delay 38.3 [37.9, 38.7] and false persistence 0.259. Long
  absence of 60 frames (S5) gives F1 0.750, because the hold horizon is finite (≈ 38 frames for
  `decay_occluded = 0.93`, threshold 0.05).

## Important failure case

**Termination under occlusion** (`failures/termination_under_occlusion__worst.png`, seed 2015,
edge (2, 6, 3)). The relationship ends and the object becomes occluded in the same frame.
GatedEMA keeps outputting the edge for 41 frames; Fixed Window, EMA and HMM drop it after 5–6.
- OBSERVATION: 144/144 terminating edges in S1 have positive delay; the mean is 27.5 frames vs 5.0.
- HYPOTHESIZED EXPLANATION: an occluded absence carries no evidence about termination, so a
  method whose only termination evidence is *visible* absence cannot end a relationship while the
  object is hidden. This limitation is inherent to any visibility-gated rule. It is not a tuning
  artifact, but this is not separately tested.

## Likely reviewer criticisms

1. "C0 is circular": the injector writes the answer into the visibility flag. (Agreed; see
   weakest result.)
2. "The win count is set by the grid": see the structural caveat.
3. "i.i.d. visibility noise is unrealistic": real visibility errors are bursty and correlated
   with termination events (e.g. a person puts a cup down and walks behind a shelf).
4. "Condition-matched tuning assumes you know the noise regime": the mismatch analysis shows
   this matters.
5. "The baselines are weak": the HMM is hand-parameterised, and there is no learned temporal
   model and no visibility-aware HMM (an HMM whose emissions depend on visibility would be the
   natural fair comparator).
6. "One synthetic world, 8 edges, independent per-edge corruption": no dense or correlated
   occlusion.
7. "Frozen metrics are blind to ghosts": Persistence F1 cannot see the ID-switch ghost
   penalty; only the non-decisive diagnostics show it.

## Experiments still required

- A visibility-aware HMM / Bayesian filter baseline (same information as GatedEMA) before any
  claim that a *particular* gating form matters.
- Temporally correlated (bursty, Markov) visibility-flag errors and termination-correlated
  occlusion, at realistic rates estimated from a real tracker.
- Tuning-transfer study: choose one configuration robust across regimes (calibrated on a
  mixture of conditions) and report its cost.
- An explicit "no record" visibility policy (`decay_unknown` ≠ `decay_occluded`) as an ablation,
  given the S2/S4 failures.
- Frame-rate variation (15/30/60 fps) — every Phase 1 method is per-frame.
- Real data: Action Genome with ≥ 2 base models and tracker-derived visibility (Phase 6–7).
- Phase 4 question: does V-HPED improve on GatedEMA at all?
