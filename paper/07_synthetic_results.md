# 7. Controlled Synthetic Evaluation (Phase 3) — DRAFT

> **Scope label (must be kept in the final paper): controlled synthetic evidence.** All results
> in this section come from a parametric synthetic world with known relationship lifetimes (§6.3).
> No real VidSGG model, detector, tracker or dataset is involved. The method evaluated is the
> **Visibility-Gated EMA baseline**; V-HPED is not implemented and is not evaluated here. Numbers
> are taken from generated tables in `results/phase3/tables/`; do not edit by hand.

## 7.1 Question and decision rule

We test H1: *does visibility-aware gating improve the persistence–termination trade-off
compared with visibility-unaware smoothing (EMA, Fixed Window, HMM) under controlled observation
corruption?* The decision rule, conditions, seeds, grids, metrics and statistical tests were
registered before any calibration or evaluation run (commit `d8728b3`). Parameters were
selected on disjoint calibration videos and frozen in commit `2ce7d1f` before evaluation. The
evaluation covered all 156,240 planned (configuration, instance) runs, with none missing or
failed.

## 7.2 Main result (pre-registered)

The registered rule returns **PASS**:

| Condition | Cell wins / 18 | Cell losses | Frontier criterion (gaps) | Required for PASS |
|---|---|---|---|---|
| C0 oracle visibility (primary) | 10 | 0 | 5 / 6 | ≥ 9 wins, 0 losses, ≥ 4 / 6 |
| R1 visibility-flag noise ρ = 0.2 | 10 | 0 | 4 / 6 | ≥ 6 wins, ≤ 2 losses, ≥ 3 / 6 |
| R2 base-model flicker q = 0.1 | 10 | 0 | 5 / 6 | ≥ 6 wins, ≤ 2 losses, ≥ 3 / 6 |

A win at a (condition, gap, budget) cell requires ΔF1 ≥ 0.05 against each of the three
comparators, a bootstrap CI above 0, Holm-adjusted p < 0.05 over the 162-test family, and
non-inferiority in termination delay (+1 frame), false persistence (+0.02) and churn.

**How to read this result.** The win count is less informative than it looks. In every
condition the winning cells are exactly those with gap length ≥ budget. A visibility-unaware
method cannot bridge a g-frame gap without also delaying termination by about g+1 frames (§5;
checked empirically on the evaluation frontiers, where it holds exactly for Fixed Window and to
within 0.05 frames on the mean for HMM — a proof with correct conditions is still owed). So whenever the budget is shorter than the gap, it must drop
the edge; whenever the budget is longer, it bridges the gap and the cell is neutral. The
informative quantities are therefore the size of the gains under degraded signals (§7.3) and how
the advantage behaves as the signal degrades (§7.4).

In the oracle condition C0 the result is **circular by construction**, as declared in the
pre-registration. Calibration selects `decay_visible = 0`, which reduces the method to "drop on a
visible absence, hold on an occluded absence". It reaches F1 = 1.0 at a delay of one frame with
no variance across videos. We report C0 as a sanity check only.

## 7.3 Persistence and termination under degraded signals (budget 5 frames)

| Condition | Gap | Vis-Gated EMA F1 | best unaware F1 | Vis-Gated delay | unaware delay | Vis-Gated churn | unaware churn |
|---|---|---|---|---|---|---|---|
| R1 (ρ = 0.2) | 10 | 0.953 | 0.571 | 4.36 | 4.82–5.00 | 0.004 | 0.014 |
| R1 (ρ = 0.2) | 20 | 0.808 | 0.333 | 4.44 | 4.82–5.00 | 0.011 | 0.014 |
| R2 (q = 0.1) | 10 | 1.000 | 0.556 | 4.38 | 4.73–4.90 | 0.001 | 0.015 |
| R2 (q = 0.1) | 20 | 1.000 | 0.326 | 4.38 | 4.71–4.90 | 0.000 | 0.015 |

(Means over 30 evaluation videos; CIs in `results/phase3/tables/results_R1_visnoise_0.2.md` and
`results_R2_flicker_0.1.md`.) Under both kinds of degradation, gating retains most of its
persistence advantage at **equal or lower** termination delay, false persistence and churn.
Frontier hypervolume gains against the strongest comparator at gap 10 are +0.083 (R1) and
+0.106 (R2), with bootstrap CIs excluding zero.

**Figures.** Fig. A (F1 vs gap), Fig. B (false persistence vs gap), Fig. C (delay vs gap),
Fig. D (churn vs gap), Fig. E (persistence–termination frontiers, gap 10):
`results/phase3/plots/`.

## 7.4 Dose–response: when the visibility signal degrades

As the probability ρ that an absence carries the wrong visibility flag grows, the advantage
shrinks monotonically. At gap 10 (budget 5), GatedEMA F1 is 0.998 at ρ = 0.1, 0.953 at 0.2,
0.888 at 0.3 and 0.582 at 0.5, against 0.571 for Fixed Window. At ρ = 0.5, where the flag carries
no information about absences, there is no advantage. This negative control is consistent with
the gain coming from the visibility signal and not from extra tuning freedom. Under base-model
flicker, GatedEMA stays at F1 ≥ 0.993 up to q = 0.3.

## 7.5 Pareto analysis

With a formal definition of frontier set dominance (exploratory; defined and committed before
evaluation data existed), GatedEMA's frontier dominates all simpler methods at every gap only in
C0. In R1 it does not dominate at gaps 1–3; in R2 it does not dominate EMA and Fixed Window at
gaps 1 and 20. It is never dominated by any comparator. **Classification: condition-specific
improvement (B), not Pareto dominance.**

## 7.6 Failure analysis (stress conditions with oracle-tuned parameters, budget 5)

| Stress | Observation (GatedEMA vs visibility-unaware) | Hypothesised explanation |
|---|---|---|
| Termination under occlusion (S1) | delay 27.4 [26.2, 28.7] vs 4.8–5.0 frames; false persistence 0.18 vs 0.03 | Occluded absence carries no termination evidence; inherent to visibility gating |
| All visibility UNKNOWN (S4) | delay 38.3 vs ≈ 5; false persistence 0.26 | UNKNOWN is decayed like OCCLUDED (`decay_unknown` default) |
| ID switch, 20 frames (S2) | true-key recall 0.86 vs 0.25–0.26, but ghost edge-frames 251 vs ≈ 111 | Keys with no record are treated as UNKNOWN and held, both true and ghost keys |
| Very long absence, 60 frames (S5) | F1 0.75 (vs 0.12 unaware) | Finite hold horizon (≈ 38 frames at the selected decay) |
| Parameters tuned on oracle, used under noise (R1) | F1 0.70 vs 0.89 at gap 5 — **worse** than Fixed Window / HMM | `decay_visible = 0` turns every false VISIBLE flag into an immediate drop |
| Parameters tuned on oracle, used under flicker (R2) | churn ≈ 0.17 vs 0.015 | Every flicker frame (a visible miss) causes a drop and re-appearance |

Short, long and repeated occlusions with a correct flag produced no GatedEMA failure
(0 / 240 units with positive severity). Representative timelines are in `results/phase3/failures/`.

## 7.7 What this section does not show

It does not show that the mechanism works on real VidSGG predictions, that it is model-agnostic,
that a continuous-time model (V-HPED) adds anything beyond per-frame gating, or that the
i.i.d. corruption models represent real tracker behaviour.
