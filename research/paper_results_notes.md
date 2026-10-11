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

---

# Phase 4 results notes (controlled synthetic evidence only)

> Source: `results/phase4/` (`Q/`). Pre-registration `bad0ef9`; baseline freeze `07b01c2`; V-HPED
> code `095101b`; V-HPED freeze `490b28c`; evaluation seeds 4000–4029 (n = 30). No real data.
> Pre-registered decisions: **H2 = MIXED, H3 = MIXED, H4 = FAIL**.

## Strongest result
**Identity-reliability tempering suppresses ID-switch ghosts.** On F5, at the budget-5 operating
point, M5 has 220 ghost edge-frames per video against 902 (GatedEMA) and 575 (VA-HMM). F1, delay,
false persistence and swap recall are equal. Identity-frontier HV is 0.709 [0.634, 0.779] vs
0.510 / 0.470, and the gain survives the half-sample tuning control (0.634). This is the only H2
gain that survives every post-hoc check.

## Weakest result
**The default (F1, termination-delay) frontier over-credits V-HPED.** The joint (Z,V) filter with
uncertain flags dips below threshold at occlusion onset and later climbs back (relapse). The frozen
delay metric records only the first drop. On F1 (termination during occlusion), M5 appears to
reach F1 0.95 at "delay" 4 frames, but its false persistence is 0.14 vs 0.03. On an
F1-vs-false-persistence frontier (post-hoc, exploratory), M5 is worse than both baselines on
F1, F7 and F8.

## Unexpected results
1. **Matched tuning removes most Phase 3 GatedEMA failures** (baseline stage, before V-HPED ran).
   On F1/F2 every method, including EMA, reaches nearly the same frontier. Phase 3 failures A/B
   were largely failures of oracle *tuning*, not of gating.
2. **VA-HMM (probabilistic flags) ≈ GatedEMA** everywhere except the identity frontier, where it is
   worse. "Using visibility probabilistically" is not what helps.
3. **Continuous time hurts at 15 fps.** Evidence-rate tempering doubles the weight of each missed
   frame, so churn goes 0.001 → 0.08. The frame-rate-invariance argument fails for per-frame
   detector misses.
4. **Cross-predicate evidence (M6) helps only when visibility is uninformative** (T1: gap-10 F1
   0.898 vs 0.568, lower delay and FPR), not under flag noise (T2).

## Negative results
- H4 = FAIL: M5 tuned on low noise is *less* robust to shift than GatedEMA (ρ = 0.5 FPR 0.37 vs
  0.05; flicker churn 10–20×).
- M4 (occlusion-onset hazard) earns nothing on its target conditions (0/9), as predicted a priori.
- M2 ≈ M1 at 30 fps, as predicted a priori.
- The identity mechanism (A4) costs up to −0.17 HV on non-ID failure conditions because of
  box-jitter false alarms.

## Important failure case
`Q/failures/F3_visnoise__worst.png` (seed 4012): terminated edges are held about 30 extra frames
through false-OCC flags, and one edge relapses after being dropped.
- OBSERVATION: in F3, 37% of M5's terminations relapse.
- HYPOTHESIZED EXPLANATION: posterior mass in (Z=0, V=1) is penalised by each OCC flag, so the
  normalised Z=1 share grows again (explaining away). This is inherent to exact joint filtering
  with confusable flags; not separately tested.

## Likely reviewer criticisms
1. The headline "win" (F5) rests on a synthetic identity cue (static boxes + Gaussian jitter).
2. The registered default frontier was gameable; the decisive post-hoc check is exploratory.
3. The H3 world is constructed and E is learned from the same world.
4. There is no learned temporal baseline; everything is hand-parameterised and randomly searched.

## Experiments still required
- Re-define termination with relapse (a "last-drop" delay or relapse-penalised delay) and repeat
  Phase 4 as a *new* pre-registered experiment.
- A monotone-termination variant (e.g. absorbing termination or hysteresis) and online
  flag-reliability estimation, before any shift claim.
- Real tracker identity cues (appearance/motion) before the ghost-suppression claim.

---

# Phase 5H-1 results notes (EXPLORATORY; VidOR annotations only)

> Source: `results/phase5h1/full/tables.md` / `analysis.json` (freeze `289319e`). Full account:
> `research/phase5h1_exit_report.md`. Numbers checked by `scripts/phase5h1/verify_results.py`.
> Every annotated VidOR video was already used in Stage 0, so nothing here is confirmatory.

## Pre-registered outcome

**HAWKES TRACK = NARROW (MIXED)**: K1, K2, K3 pass; K4 fails; family test passes for SPATIAL and
MOTION.

## Strongest result

Cross-predicate event history adds **+0.083 [0.076, 0.090] nats/event** of held-out
log-likelihood beyond the state-dependent transition model (G4 − G3), and +0.145 beyond
heterogeneity + duration (G4x − G2). Both are about 5× the largest of 38 null replicates and are
consistent across EVAL-A and EVAL-B (T3, T4).

## Weakest result

No individual interaction is reproducible across training folds (K4: 0 discoveries, minimum
FDR 0.16), and the registered fold statistic is shrinkage-inflated (T7, post-hoc). The
interaction *structure* cannot be named with confidence; only the aggregate predictive gain is
robust.

## Must-disclose caveats

1. Annotation timing: 17.0% of events tie exactly with another predicate's event on the same
   pair. The short-lag structure may be annotation convention (`posthoc_boundary.json`).
2. Nulls are imperfect: N-circ +16% events (registered Stage 0 function shifts censored ends),
   N-jit −9.3% to −9.8% (window clipping).
3. Positive-only Hawkes is worse than signed history (−0.031); "Hawkes" is not the right
   description of the useful model.
4. 7,058 EVAL onsets at the pair-window start are excluded by the registered event
   construction.
5. A4 changed only the optimiser iteration limit; the selected ridges are identical with and
   without it.

---

# Relationship-episode study (2026-10-11; controlled SYNTHETIC evidence; `results/episodes/`)

> Pre-registered (`42b61ea`), parameters frozen (`d027634`) before evaluation; n = 50 evaluation
> seeds per condition; one pooled operating point per method. Report:
> `research/taorg_episode_exit_report.md`. Every number below is from `results/episodes/`.

## Pre-registered outcome

**STOP** (`aggregated/analysis.json`). All three STOP clauses hold:
- 0 benefit-condition wins;
- 6 losses vs 1 win;
- GatedEMA + hysteresis is never beaten in C2/C3/C5.

The half-sample control also gives STOP. The selected estimator was T = P1, because identity was
not retained. Q1–Q6 are all negative.

## Strongest result

**A two-threshold GatedEMA (B2) is as good a relapse control as an explicit episode estimator,
and better under boundary flicker and termination under occlusion.**
- C5: J 0.886 vs 0.823; FRR 0.142 vs 0.283.
- C2: sustained termination delay 0.665 vs 0.922 s.

At the same time, the episode estimator has the highest mean J in 6/7 calibrated conditions and
does not delay recurrence (RDD 0.034 s in C4). Its J advantage over the VA-HMM and HSMM fails the
registered churn guard (churn 0.014 vs 0.007).

## Weakest result

Identity ghosts: the episode estimator is **worse** than every GatedEMA variant (8.23 vs
2.2–2.3 s per video).
- Post-hoc E2 (exploratory): the ghosts linger *after* the swap ends.
- Post-hoc E1 (exploratory): the identity rung's apparent ghost reduction comes from
  base-parameter selection, not from the identity cue.

## Must-disclose caveats

1. One pooled operating point per method, selected on a registered composite (J). Condition-matched
   tuning was not tested.
2. The decisive churn guard is 10% of a very small comparator churn.
3. Ladder contrasts compare selected operating points (selection and mechanism are confounded;
   see E1).
4. Per-edge visibility (it matters for the transition-context rung); a synthetic identity cue.
5. The run records show `git_dirty` because of unrelated demo files only.

---

# Next-method study (2026-10-11; controlled SYNTHETIC evidence; `results/next_method/`)

> Failure audit on development seeds; pre-registered (`fe084a6`); parameters frozen (`8a7c916`)
> before evaluation; evaluation seeds 8000-8049, n = 50 per condition. Report:
> `research/taorg_next_method_exit_report.md`.

## Pre-registered outcome

**STOP**, and the algorithm contribution is **UNSUPPORTED**. All gates G1-G7 fail:
- K1 loses to B2 in S2 (-0.156 J);
- 4 losses vs B2;
- no gain over a tuned fixed buffer, the static prior or a geometric learner;
- latency 3.71x B2.

## Strongest result

Holding non-visible absences for a calibrated duration fixes B2's gap failure: C1 J 0.844 -> 0.974
with K1, and 0.944 with the established fixed buffer. This is an established rule.

## Weakest result

Online label-free recalibration contributes nothing at the selected prior strength (kappa = 838).
UNKNOWN, learned from calibration as "never returns", collapses S2 (J 0.568).

## Must-disclose caveats

1. One pooled operating point per method; in-distribution tuning disfavours adaptation.
2. The "no record = UNKNOWN" input convention.
3. Synthetic world.
4. A documented per-frame vs run-stratum implementation detail (exit report §1).
