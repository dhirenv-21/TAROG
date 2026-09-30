# Phase 4 — Method-selection experiment — Pre-registration

> **Status:** REGISTERED before V-HPED is implemented and before any Phase 4 calibration or
> evaluation run.
> **Registered:** 2026-09-29. Commit hash recorded in `research/phase4_research_log.md`.
> **Config:** `configs/experiments/phase4.yaml` (its SHA-256 is recorded in every stage's metadata).
> **Formulation:** `research/vhped_formulation.md`. **Ablation plan:** `research/vhped_ablation_plan.md`.
> **Decision code:** `src/taorg/experiments/phase4/decision.py`, committed together with this file.
>
> Nothing in Phase 3 is changed: Phase 3 seeds, metrics, criteria and artifacts are untouched,
> and Phase 4 uses disjoint seeds.

## 1. Question

Can a principled temporal state estimator use *uncertain* visibility evidence to improve the
persistence–termination trade-off **specifically under the conditions where simple visibility
gating failed in Phase 3**, by enough to justify its complexity?

This is a method-selection experiment. Phase 3 already showed that an informative visibility
flag helps. Phase 4 does not re-test that.

## 2. Hypotheses

- **H2 (PRIMARY).** A probabilistic, visibility-aware, continuous-time estimator (V-HPED rung
  **M5**) improves the persistence–termination Pareto frontier over **both** GatedEMA (M0) and a
  visibility-aware HMM (M1) on the Phase-3-derived failure conditions.
- **H3 (secondary).** Cross-predicate interaction (M6) gives a measurable benefit over the
  otherwise identical single-edge model (M5).
- **H4 (secondary).** M5 is more robust to visibility-distribution shift than M0 and M1 when
  all are tuned once, on a low-noise source regime, and never retuned.

If H2 fails, the method-paper direction is in serious danger regardless of H3/H4.

## 3. Conditions (frozen; derived from the Phase 3 failures)

All conditions use the Phase 3 synthetic world (300 frames, 30 fps, 8 edges), one occlusion gap
per edge with gaps {3, 10, 20} frames unless stated, and bounding-box centre jitter of 0.15 box
diagonals on every track in every condition (the identity cue is never an oracle). Full
definitions are in the config.

| Condition | Role | Phase 3 origin | Corruption |
|---|---|---|---|
| F1_term_occlusion | H2 failure | A (S1) | every termination followed by 15–40 occluded frames |
| F2_unknown | H2 failure; H3 NI control | B (S4) | all absence records UNKNOWN |
| F3_visnoise | H2 failure | E (R1/SWV) | i.i.d. flag noise ρ = 0.3 |
| F4_bursty_visibility | H2 failure | "visibility flicker" | Markov flag errors, stationary 0.2, mean burst 10 frames (new injector) |
| F5_id_switch | H2 failure | C (S2) | two 20-frame same-class ID swaps |
| F7_mix_occlusion_visnoise | H2 failure | mixed | 50% terminations occluded + ρ = 0.2 + 20% UNKNOWN |
| F8_mix_occlusion_idswitch | H2 failure | mixed | 50% terminations occluded + two 20-frame ID swaps |
| N0_clean, N1_visnoise_0.2, N2_flicker_0.1 | H2 non-inferiority | Phase 3 C0/R1/R2 | as Phase 3 |
| T1_succession_unknown, T2_succession_visnoise | H3 benefit | new world | 80% of terminations followed by the partner predicate on the same pair; + all-UNKNOWN / ρ = 0.3 |
| T3_succession_clean | H3 non-inferiority | new world | succession, oracle visibility |
| SC_low_noise | H4 source (tuned) | — | ρ = 0.1, flicker 0.05 |
| SH1–SH8 | H4 shift (never tuned) | D, E | ρ 0.3 / 0.5; flicker 0.2 / 0.3; 50% UNKNOWN; bursty flags; 15 fps; 60 fps |

The user-listed condition "parameter transfer / distribution shift" is the H4 block. The
numbering F1–F8 skips F6 for that reason. No condition may be added, removed or altered after
V-HPED results are seen.

## 4. Methods

| Id | Method | Search |
|---|---|---|
| — | No Memory, Fixed Window (windows 1–40), EMA, HMM | references |
| M0 | GatedEMA (Phase 1; `decay_unknown` is now **tuned**, not defaulted) | 200 random samples |
| M1 | Visibility-aware HMM (**new mandatory baseline**; same flags as V-HPED) | 200 |
| M2–M6 | V-HPED ladder (`vhped_formulation.md`) | 200 each |

**Equal tuning budget.** Every tunable method receives exactly 200 configurations drawn by a
seeded sampler from the distributions in the config. Fixed Window receives its whole 40-point
space and No Memory has 1. Higher-dimensional methods get the *same* number of samples, which is
conservative against complexity. M6 runs only on T1–T3 and F2. The exclusivity statistic E used
by M6 is estimated from the **calibration** ground truth of each condition (formulation §9) and
saved as an artifact.

**Baseline-first order.** The baselines (No Memory … M1) are calibrated, frozen, committed and
evaluated **before** any V-HPED configuration is calibrated. V-HPED is implemented after this
registration, strictly following the frozen formulation. Any deviation is an amendment.

## 5. Tuning protocol

1. Seeds: calibration 3000–3019 (20 videos), evaluation 4000–4029 (30 videos), smoke
   9100–9101 (never analysed). Bootstrap seed 4242, sampler seed 777.
2. **Condition-matched tuning** for every calibrated condition and every method, with the same
   procedure. GatedEMA's Phase 3 S1/S4 failures came partly from oracle tuning; tuning every
   method on the regime it is evaluated in removes that explanation.
3. **Operating points:** for each condition × method × budget B ∈ {5, 10} frames, choose the
   configuration with the highest mean calibration Persistence F1 among those with mean delay
   ≤ B (Phase 3 rule and tie-breaks). The primary budget is 5.
4. **Frontiers:** each method's calibration Pareto set evaluated on evaluation seeds.
   - Default frontier: quality = Persistence F1, cost = termination delay, HV box
     F1 ∈ [0,1] × delay ∈ [1, 40] frames (Phase 3).
   - Identity frontier (F5, F8 only): quality = swap-interval recall of true keys, cost = ghost
     edge-frames per video, box [0,1] × [0, 300].
5. H4: every method is tuned once, on SC_low_noise. Its budget-5 operating point is evaluated on
   SH1–SH8 with **no retuning**.
6. Nothing is tuned on evaluation instances.

## 6. Metrics

Frozen Phase 3 metrics (research/metric_definitions.md, unchanged): Persistence P/R/F1,
termination delay, false persistence rate, churn, latency.

**Pre-registered Phase 4 promotions** (already implemented as Phase 3 diagnostics; now primary
only where stated):
- swap-interval recall (`diag_swap_recall`) and ghost edge-frames (`diag_ghost_edge_frames`):
  primary on the identity frontier of F5/F8, because the frozen metrics cannot see identity
  corruption;
- delay in seconds (= frames / fps) for H4, because SH7/SH8 change the frame rate. This is a unit
  conversion, not a new metric.

No other metric may be added for decisions.

## 7. Statistics

- Unit: evaluation video (n = 30), paired across methods on identical corrupted instances.
- Descriptive: N, mean, SD, 95% t-interval.
- Paired: difference per seed; 95% percentile bootstrap CI (5000 resamples); Cohen's d_z,
  **reported as "undefined" when the SD of the differences is < 1e-9** (fixes the Phase 3
  reporting defect for Phase 4 only); two-sided Wilcoxon signed-rank.
- Frontier: normalised hypervolume with paired bootstrap over seeds (2000 resamples).
- Multiplicity: Holm over the H4 F1 family (24 cells × 2 comparators = 48 tests). H2 and H3 are
  decided by CI-based frontier criteria with a practical margin, not by p-values. Everything else
  is exploratory and labelled so.

## 8. Decision rules (exact; implemented in `decision.py`)

### H2 (primary): M5 vs M0 and M1

Cell = (failure condition, gap): 7 × 3 = 21 failure cells, plus 3 × 3 = 9 non-inferiority cells
(N0–N2).

- **Win vs m:** ΔHV(M5 − m) ≥ 0.02 on the condition's frontier with bootstrap CI lower bound > 0,
  **and** at the budget-5 operating point the churn difference has CI upper ≤ max(0.005,
  0.10 × m's mean churn). For F5/F8 the default-frontier ΔHV CI lower bound must also be
  > −0.02 (identity gains must not be bought with persistence/termination losses).
- **Loss vs m:** ΔHV ≤ −0.02 with CI upper < 0.
- **Cell win** = win vs M0 **and** M1. **Cell loss** = loss vs M0 **or** M1.

| Decision | Rule |
|---|---|
| **PASS** | ≥ 11/21 failure-cell wins, spanning ≥ 4 of 7 failure conditions; ≤ 2 failure-cell losses; 0 losses in the 9 NI cells |
| **FAIL** | ≤ 4 failure-cell wins, **or** wins span ≤ 1 condition, **or** total losses (failure + NI) ≥ total wins |
| **MIXED** | anything else (improvement confined to specific regimes) |

*Tuning-capacity control (secondary, reported, input to the contribution gate):* re-select M5
from a seeded random half (100) of its calibration samples and recompute the H2 cell outcomes.

### H3: M6 vs M5

Benefit cells: T1, T2 × 3 gaps (6); NI cells: T3, F2 × 3 gaps (6); default frontier.
Win / loss as for H2 (single comparator M5).
**PASS:** ≥ 4/6 wins and 0 NI losses. **FAIL:** ≤ 1 win or ≥ 2 NI losses. **MIXED:** otherwise.
If H3 is not PASS, the cross-predicate mechanism is removed from the proposed method.

### H4: robustness under shift (M5 vs M0 and M1)

Cells: SH1–SH8 × 3 gaps = 24, budget-5 operating points selected on SC. Delay in seconds.

- **Persistence win vs m:** ΔF1 ≥ 0.05, CI lower > 0, Holm p < 0.05 (48-test family); and
  non-inferiority: Δdelay CI upper ≤ 1/30 s, ΔFPR CI upper ≤ 0.02, Δchurn CI upper ≤
  max(0.005, 0.10 × m's churn).
- **Termination/churn win vs m:** ΔF1 CI lower ≥ −0.02, and either (Δdelay ≤ −2/30 s with CI
  upper < 0) or (Δchurn ≤ −max(0.005, 0.5 × m's churn) with CI upper < 0).
- **Loss** = the mirror image of either win (roles swapped).
- Cell win = win vs both; cell loss = loss vs either.

**PASS:** ≥ 12/24 wins and ≤ 2 losses. **FAIL:** ≤ 4 wins or losses ≥ wins. **MIXED:** otherwise.

### Mechanism retention (ablation, `vhped_ablation_plan.md`)

A rung Mk (k = 3, 4, 5) earns its place if ΔHV(Mk − M(k−1)) ≥ 0.02 with CI lower > 0 in at least
one third of the cells of its target conditions, and it loses (ΔHV ≤ −0.02, CI upper < 0) in no
N0–N2 cell. M2 versus M1 is reported descriptively (formulation §8 predicts near-equivalence at
30 fps; its only a-priori advantage is H4 SH7/SH8).

### Method-contribution gate (answered in the exit report)

NEW METHOD = YES only if **all** hold:
1. H2 = PASS, or H2 = MIXED with wins in failure conditions judged important (F1, F2, F3, F5);
2. the H2 wins survive the tuning-capacity control;
3. the ablation identifies which rung(s) cause the gain;
4. latency stays practical (< 1 ms per frame for 8 edges on the reference machine, and within
   20× GatedEMA);
5. the formulation is distinct from standard filtering/Hawkes approaches (literature check;
   formulation §9 already concludes the Hawkes label is not warranted).

Otherwise: NO. In that case the recommended pivot is to a robustness-evaluation and
failure-analysis paper.

## 9. A-priori expectations (declared so they cannot be retrofitted)

- **F1 (termination during occlusion):** no estimator can separate terminations from gaps before
  their durations diverge (formulation §3). We expect M0, M1 and M2–M5 to reach nearly the same
  frontier, i.e. **no H2 win on F1**.
- **F2 (all UNKNOWN):** with `decay_unknown` / θ_UNK tuned, M0 and M1 should largely remove the
  Phase 3 failure; M5 gains are expected to be small.
- **F3/F4 (noisy/bursty flags):** the most plausible H2 wins for M3+ (flags as noisy sensors with
  persistence). Bursty errors may *hurt* the V-persistence prior.
- **F5/F8 (ID switches):** M5 may gain on the identity frontier; bounding-box jitter makes this
  uncertain, and M5 pays for false alarms elsewhere.
- **M2 vs M1:** near-equivalent at 30 fps.
- **H3:** gains plausible on T1 (UNKNOWN + succession), smaller on T2; the succession world is
  constructed, so an H3 PASS is weak evidence for real data.
- **H4:** M3–M5 have *tuned*, not online-estimated, flag reliabilities, so they are also
  misspecified under shift. H4 may well fail. SH7/SH8 are the only conditions where continuous
  time is expected to matter.
- **Overall:** H2 = FAIL or MIXED is a realistic outcome, and it would be reported as such.

## 10. Declared limitations (before results)

1. Synthetic world only; per-edge (not per-object) occlusion; i.i.d. scores; no real detector
   or tracker.
2. The succession world (H3) is constructed; E is learned from calibration ground truth of the
   same world.
3. Boxes are static apart from jitter; the identity cue is a simplification of real
   motion/appearance consistency.
4. Equal sample counts do not equalise search difficulty: higher-dimensional methods are
   *disadvantaged*.
5. The tempered (power) likelihood is a modelling assumption.
6. H4's frame-rate conditions change per-frame noise rates (flicker probability is per frame),
   so per-second corruption rates differ across frame rates.

## 11. Amendments

None at registration. Any change must be dated, justified, and must state whether it was made
before or after viewing any Phase 4 evaluation result.

## 12. Untracked Phase-3-era files (decision recorded at registration)

`scripts/run_pilot.py` and `tests/fixtures/pilot_fixtures.py` (2026-09-28 01:36–01:38) predate
the Phase 3 pre-registration. They are referenced by no code or test, no output of them exists
(`outputs/pilot/` absent), and their ad-hoc H1 assessment differs from the registered rule.
Classification: **B (obsolete)**. To preserve the audit trail they are **archived, not
deleted**: moved to `archive/pre_registration_pilot/` with a README stating that they were never
used for any reported result.
