# Phase 4 Exit Report — Method-selection experiment (V-HPED vs simple visibility-aware baselines)

> **Evidence class: controlled synthetic evidence.** No real VidSGG model, tracker or dataset.
> All numbers come from generated artifacts under `results/phase4/` (`Q/`). Date: 2026-09-29.
> Branch `phase4-vhped`.

## Summary of decisions

| Hypothesis | Pre-registered decision | One-line reading |
|---|---|---|
| **H2 (primary)** M5 vs GatedEMA & VA-HMM on failure conditions | **MIXED** (6/21 wins, 0 losses) | Real gain only on ID-switch ghosts (F5); a modest gain under i.i.d. flag noise at long gaps; the rest of the apparent frontier gain is a termination-metric blind spot (relapse) |
| H3 cross-predicate (M6 vs M5) | **MIXED** (3/6 wins, 0 NI losses) | Helps only when visibility is uninformative (T1), in a constructed world |
| H4 robustness under shift | **FAIL** (2 wins, 12 losses) | V-HPED is *less* robust than GatedEMA |
| **METHOD CONTRIBUTION** | **NO** | see §26 |

## 1. Pre-registration commit

**`bad0ef9`**: `research/phase4_preregistration.md`, `research/vhped_formulation.md`,
`research/vhped_ablation_plan.md`, `configs/experiments/phase4.yaml` (SHA-256
`78b3dba3f8db…6e752`, identical in every stage's metadata) and the decision code
`src/taorg/experiments/phase4/decision.py` with 11 tests. No V-HPED code and no Phase 4 run existed
at this commit. There were no amendments. Deviations recorded, none of which changes the protocol:
- the effective succession rate is ≈ 51%, not 80%, because of key collisions (a property of the
  frozen world);
- run IDs carry a "phase3_" prefix from the reused metadata helper (cosmetic);
- two formulation statements hold only approximately and are recorded in the log: M2's OCC
  absence leaks O(1e-5) evidence, and M5 ≈ M4 to within 0.004 without switches.

Order of work, as registered:

| Step | Commit |
|---|---|
| Pre-registration | `bad0ef9` |
| Infrastructure + VA-HMM | `04941e2` |
| Baseline freeze | `07b01c2` |
| Baseline evaluation + baseline-stage analysis | `0c2c4ab` |
| V-HPED implementation | `095101b` |
| V-HPED freeze | `490b28c` |
| Evaluation artifacts | `dfd98c6` |

## 2. Mathematical formulation

`research/vhped_formulation.md` (frozen). Per edge, an exact filter over the joint latent
(Z = relationship exists, V = observable):

- a continuous-time Markov prior with rates in s⁻¹; absence of evidence relaxes the belief to the
  prior π = ν/(μ+ν), not to FALSE;
- the absence likelihood ratio (q·m + (1−q)(1−ε))/(1−f), which formalises "no observation ≠
  negative evidence" and "occluded ≠ terminated";
- flags treated as noisy sensors of V (M3);
- an occlusion-onset termination coupling (M4);
- identity-tempered mixture likelihoods (M5);
- cross-predicate exclusivity evidence learned from calibration ground truth (M6);
- evidence-rate (power) tempering for frame-rate invariance.

Complexity and lazy/eager equivalence: `research/derivations/vhped_complexity.md`.
**Conclusion of §9 of the formulation: this is a continuous-time factorial HMM filter, not a
Hawkes process.**

## 3. Visibility-aware HMM baseline (M1)

`src/taorg/temporal/va_hmm.py`: a discrete-time two-state HMM whose absence likelihood depends on
the flag (θ_VIS, θ_OCC, θ_UNK tuned). It receives exactly the same flag / no-record information
as V-HPED and got the same 200-sample tuning budget.

Baseline-stage result, obtained before any V-HPED configuration was run
(`Q/baseline_stage/`): **VA-HMM ≈ GatedEMA** (A0 ΔHV within ±0.02 on every default frontier except F7 gap 20 (+0.020 for VA-HMM);
worse on the identity frontier, about −0.04). Using the flag probabilistically adds nothing by
itself.

## 4. Proposed method variants

M2–M6 in `src/taorg/temporal/vhped.py` (eager reference; 10 tests against the formulation's
closed forms: prior relaxation, hold horizon, frame-rate invariance, lazy = eager to rtol 1e-9,
identity hold, cross-predicate inhibition).

## 5. Ablations

`Q/tables/ablation.md`, `Q/plots/fig6_ablation.png`. Cells are the calibrated conditions × 3
gaps; "+"/"−" means ΔHV ≥ 0.02 (resp. ≤ −0.02) with a CI excluding 0.

| Step | Comparison | Result | Pre-registered retention |
|---|---|---|---|
| A0 | M1 − M0 | no gains; −0.04 on the identity frontier (F5, F8) | reported |
| A1 | M2 − M1 | ≈ 0 except F2/T1 at gap 20 (as predicted) | descriptive |
| A2 | M3 − M2 | + on F1, F3, F7, T2 (gaps 10/20), F4/N1 (gap 20) | **retained** (6/12 target cells), **but** these gains coincide with relapse (§20) |
| A3 | M4 − M3 | 0/9 on targets (F1, F7, F8); + only on F2/T1 | **not retained** (as predicted) |
| A4 | M5 − M4 | +++ on F5 and F8 (identity); −− on F1, F2, F7, T1 (up to −0.17) | **retained** (6/6); pays elsewhere |
| A5 | M6 − M5 | +++ on T1; 0 elsewhere | = H3 (MIXED) |

## 6. Frozen parameters

`Q/configs/frozen_baseline.json` and `Q/configs/frozen_proposed.json` (with per-selection
feasibility and ties; tables `Q/tables/frozen_*.md`). Both groups passed all 10 calibration checks
(no smoke, full sample budget, config hash, exact instance set 840/840, calibration seeds only,
equal budget, every config saw every instance, no NaN). Examples at budget 5:
- M5 on F5: μ 0.23 s⁻¹, miss 0.002, σ_id 0.36, r_id 12.5 s⁻¹, τ 0.10.
- M5 on F3: r_ov 0.12, r_vo 0.03, τ 0.41. The same sample was also selected on SC.
- GatedEMA on F2: `decay_unknown` 0.45 (tuned). This is why the Phase 3 failure B disappears.

## 7. Evaluation conditions

As registered (§3 of the pre-registration): 7 failure conditions (F1–F5, F7, F8), 3
non-inferiority conditions (N0–N2), 3 H3 conditions (T1–T3), the H4 source SC and 8 shift
conditions (SH1–SH8). Gaps {3, 10, 20} (fps-scaled for SH7/SH8); bbox jitter 0.15 everywhere.

## 8. Run completeness

`Q/tables/completeness.md`: **COMPLETE**; manifests are identical across the two groups.

| Group | Expected | Actual | Missing | Duplicate | Failed |
|---|---|---|---|---|---|
| baseline (run `…evaluate_baseline_20260929_070307_0e7494`, commit `07b01c2`) | 351,990 | 351,990 | 0 | 0 | 0 |
| proposed (run `…evaluate_proposed_20260929_073456_78e309`, commit `490b28c`) | 126,630 | 126,630 | 0 | 0 | 0 |

Calibration: baseline 840 instances × 841 configurations; proposed 840 × 800 + 240 × 200 (M6). No
calibration-seed leak. Latency rows complete.

## 9. Statistical analysis

As registered:
- n = 30 paired videos; descriptive mean/SD/95% t-CI (`Q/aggregated/descriptive.csv`);
- frontier HV with a 2000-resample paired bootstrap (`Q/aggregated/hypervolume.csv`,
  `hv_differences.csv`);
- operating-point paired differences with a 5000-resample bootstrap, Wilcoxon, and Holm over the
  48-test H4 family (`Q/aggregated/paired_operating_points.csv`);
- d_z reported as undefined when SD < 1e-9.

**Post-hoc, exploratory, labelled so, never decisive:** a relapse diagnostic and an
F1-vs-false-persistence frontier (`scripts/phase4/exploratory_diagnostics.py`,
`Q/tables/exploratory_diagnostics.md`), defined *after* the results revealed the relapse
behaviour.

## 10. Persistence results

Budget 5, gap 20 (`Q/tables/operating_points.md`, Fig. 1):

| Condition | GatedEMA | VA-HMM | M5 |
|---|---|---|---|
| F3 (ρ 0.3) | 0.680 | 0.612 | **0.968** |
| N1 (ρ 0.2) | 0.876 | 0.861 | **0.989** |
| F7 (mixed) | 0.411 | 0.407 | 0.678 |
| F1 (term.+occl.) | 0.320 | 0.333 | 0.951* |
| F2 (UNKNOWN) | 0.333 | 0.333 | 0.334 |
| F4 (bursty) | 0.776 | 0.873 | 0.786 |

\* Bought with 5× false persistence and relapse (§12, §20). At gap 3, M5 is *lower* on F3
(0.949 vs 0.994) and F7 (0.834 vs 0.943–0.983). A single configuration per condition trades
short gaps against long ones.

## 11. Termination results

Mean termination delay at budget 5 is ≤ 5 frames for all selected points by construction. M5's
delays are often lower (e.g. F1 3.9–4.1 vs 4.9–5.0 frames). **This metric records only the first
post-termination drop and is therefore unreliable for M3+ (§20).**

## 12. False persistence

Budget 5. M5 vs GatedEMA:

| Condition | M5 | GatedEMA | Ratio |
|---|---|---|---|
| F1 | 0.140–0.148 | 0.028 | ≈ 5× |
| F8 | 0.079–0.083 | 0.025–0.027 | ≈ 3× |
| F7 | 0.048–0.051 | 0.022–0.024 | ≈ 2× |
| F3 | 0.032–0.045 | 0.027–0.029 | slightly higher |
| F5 | 0.019 | 0.019–0.020 | equal |
| N0–N2 | ≈ equal | | |
| T3 | 0.007 | 0.000 | |

## 13. Churn

M5 churn is at or below the baselines on F3/F4/F5 (0.001–0.006). It is higher on F1 (0.021 vs
0.014), F7 (0.024–0.033 vs 0.012–0.014) and F2 (0.020 vs 0.014). The churn non-inferiority check
is what prevented the relapse-inflated F1/F7 frontier gains from counting as H2 wins.

## 14. ID-switch robustness

F5, budget 5 (all gaps):
- ghost edge-frames: **M5 ≈ 217–220**, VA-HMM ≈ 567–576, GatedEMA ≈ 894–902;
- swap recall: 0.810–0.817 for all three;
- F1, delay and FPR: equal.

Identity-frontier HV at gap 10: M5 **0.709 [0.634, 0.779]** vs GatedEMA 0.510 and VA-HMM 0.470;
half-sample control 0.634.

F8 (plus occluded terminations): the identity frontier also favours M5 (+0.19 to +0.25), but
M5's selected operating point has swap recall 0.23 vs 0.77 (GatedEMA) and 3× FPR. So F8 is a
frontier win, not an operating-point win.

## 15. Visibility-noise robustness

- **F3 (ρ = 0.3):** H2 win only at gap 20 (ΔHV +0.057 / +0.066). Neutral at gaps 3/10. On the
  exploratory FPR frontier, +0.010 / +0.022 at gap 20.
- **N1 (ρ = 0.2):** gap 20 +0.034 / +0.038 (a non-inferiority cell, so a "win" is not required).
- **F4 (bursty flags):** no gain (M5 ≈ GatedEMA). VA-HMM relapses on 93% of terminations there
  (FPR 0.166).

## 16. Distribution-shift results (H4)

`Q/tables/h4.md`, `Q/tables/h4_detail.md`, Fig. 5. All methods were tuned once on SC (ρ 0.1,
flicker 0.05).

| Shift | M5 outcome |
|---|---|
| SH1 (ρ 0.3) | win at gap 20 |
| SH2 (ρ 0.5) | loss at gap 3; FPR 0.36–0.37 vs 0.05 (M5 keeps trusting OCC flags) |
| SH3 / SH4 (flicker 0.2 / 0.3) | losses; churn 0.05–0.11 vs 0.005–0.018 (Phase 3 failure D, worse) |
| SH5 (50% UNKNOWN) | losses (termination/churn) |
| SH6 (bursty) | 1 loss |
| SH7 (15 fps) | 3 losses; churn 0.08 vs 0.001 (tempering doubles per-miss evidence) |
| SH8 (60 fps) | win at gap 40 |

**H4 = FAIL.**

## 17. Pareto analysis

Fig. 4 (`Q/plots/fig4_frontiers_gap10.png`). The M5 frontier is shifted up-left on F1, F3, F7 and
N1 on the *registered* (F1, delay) axes and dominates on the F5/F8 identity axes. After the
post-hoc false-persistence check, the only regime where M5 clearly shifts the frontier is **ID-switch
ghost suppression**, plus a small gain under i.i.d. flag noise at long gaps. There is no universal
dominance, and it was not required.

## 18. Complexity

`research/derivations/vhped_complexity.md`:
- **Time:** O(R_t + K_t + A_t·(S² + P)) per frame, S = 4, with expm cached per distinct Δ.
- **Space:** O(A·S + K + P²).
- **Pruning:** exact only in the limit (a stated δ-bound).
- **Lazy materialisation:** exact for constant-evidence runs (proof; tested to rtol 1e-9); valid
  for M2–M4, and not for M5 while u < 1.
- **Heap expiry:** needs a monotonicity certificate. Relapse *demonstrates* that the M3+
  posterior is non-monotone, so naive expiry scheduling would be wrong for exactly the rungs that
  help.
- The lazy/heap mechanism itself was **not** implemented (Phase 5).

## 19. Latency

`Q/tables/latency.md`, mean ms/frame on N0, 8 edges, one machine:

| Method | ms/frame |
|---|---|
| No Memory | 0.0013 |
| Fixed Window | 0.0069 |
| EMA | 0.0108 |
| GatedEMA | 0.0114 |
| HMM | 0.0128 |
| VA-HMM | 0.0147 |
| M2 / M3 / M4 | 0.020 / 0.020 / 0.019 |
| **M5** | **0.0301** (2.6× GatedEMA) |

This is practical at synthetic scale; not measured on real graph sizes.

## 20. Failure cases

`Q/failures/` (selection rule fixed before evaluation) and `Q/tables/exploratory_diagnostics.md`.

1. **Relapse / dip-and-return (critical).**
   - OBSERVATION: M3/M5 terminated edges are output again after the first post-termination drop:
     F1 98–100% of terminations (17–25 extra frames), F3 37–39%, F7 32–41%, F8 44–47%, T2
     36–46%. GatedEMA, VA-HMM and M2 are ≈ 0 (except VA-HMM on F4, 93%).
   - HYPOTHESIS: with confusable flags, OCC evidence progressively penalises (Z=0, V=1). The
     normalised Z=1 share then rises again (explaining away), so the exact filter is non-monotone.
   - The pre-registered failure-regret rule uses (F1, delay) and therefore reports "no failure
     observed" on F1. That rule shares the metric's blind spot; the relapse diagnostic shows the
     failure.
2. **F3 worst case** (seed 4012, `F3_visnoise__worst.png`). OBSERVATION: M5 holds terminated edges
   about 30 frames through false-OCC flags, and one edge relapses. HYPOTHESIS: as above, plus a
   tuned r_vo that trusts OCC flags.
3. **F8** (28/30 seeds with positive identity regret at the operating point). OBSERVATION: swap
   recall 0.23 vs 0.77. HYPOTHESIS: selection by F1 under a delay budget ignores identity, so the
   chosen M5 configuration drops true keys during suspected switches.
4. **Identity false alarms.** OBSERVATION: A4 costs up to −0.17 HV on F1/F2/F7/T1. HYPOTHESIS:
   jitter (σ = 0.15 diag) triggers switch posteriors that temper genuine evidence.
5. **Shift failures (§16).** OBSERVATION: see §16. HYPOTHESIS: flag reliabilities are tuned, not
   estimated online; the power-likelihood assumption over-weights per-frame misses at low fps.
6. **T2** (M6 regret positive on 24/30 seeds; small). OBSERVATION: exclusivity evidence does not
   help under flag noise. HYPOTHESIS: relapse from the M3 component dominates.

## 21. H2 decision

`Q/tables/h2.md`, `Q/aggregated/decisions.json`. Registered rule: PASS needs ≥ 11/21 wins in
≥ 4 conditions, ≤ 2 failure losses, 0 NI losses. FAIL if ≤ 4 wins, or ≤ 1 condition, or losses
≥ wins.

| Criterion | Threshold | Observed | Satisfied? |
|---|---|---|---|
| Failure-cell wins | ≥ 11 / 21 | 6 (F3 g20; F5 g3/10/20; F8 g10/20) | No |
| Conditions with a win | ≥ 4 / 7 | 3 (F3, F5, F8) | No |
| Failure-cell losses | ≤ 2 | 0 | Yes |
| NI-cell losses | 0 | 0 | Yes |
| FAIL: wins ≤ 4 | — | 6 | not triggered |
| FAIL: ≤ 1 condition | — | 3 | not triggered |
| FAIL: losses ≥ wins | — | 0 vs 6 | not triggered |

# **H2 = MIXED**

Tuning-capacity control (M5 re-selected from 100 samples): MIXED, 8 wins (F1 g10/20, F3 g20,
F5 ×3, F8 g10/20), 1 loss (F1 g3). The F5 wins survive. The *additional* F1 wins are the relapse
artifact, which passes the churn check with that configuration.

## 22. H3 decision

| Criterion | Threshold | Observed | Satisfied? |
|---|---|---|---|
| Benefit-cell wins | ≥ 4 / 6 | 3 (T1 ×3; T2 none) | No |
| NI losses (T3, F2) | 0 | 0 | Yes |
| FAIL: wins ≤ 1 or ≥ 2 NI losses | — | 3 wins, 0 losses | not triggered |

# **H3 = MIXED**

As pre-registered, cross-predicate interaction is therefore **not** retained in the proposed method.

## 23. H4 decision

| Criterion | Threshold | Observed | Satisfied? |
|---|---|---|---|
| Wins | ≥ 12 / 24 | 2 (SH1 g20, SH8 g40) | No |
| Losses | ≤ 2 | 12 | No |
| FAIL: wins ≤ 4 or losses ≥ wins | — | 2 wins, 12 losses | **triggered** |

# **H4 = FAIL**

## 24. Evidence-ledger changes

`research/paper_evidence_ledger.md`:

| Claim | Change |
|---|---|
| L7 (V-HPED > GatedEMA) | UNTESTED → **PARTIAL** (ghost suppression; small gain under i.i.d. flag noise at long gaps) |
| L15 (frame-rate invariance) | UNTESTED → **NOT SUPPORTED** |
| L21 (need beyond a visibility-aware probabilistic baseline) | UNTESTED → **SUPPORTED** (negative: probabilistic flags alone ≈ gating) |
| L22 (matched tuning removes Phase 3 failures A/B) | new → **SUPPORTED** |
| L23 (clean termination) | new → **NOT SUPPORTED** (relapse) |
| L24 (ghost suppression) | new → **PARTIAL** |
| L25 (cross-predicate) | new → **PARTIAL** |
| L26 (shift robustness) | new → **NOT SUPPORTED** |
| L27 (Hawkes terminology) | new → **INVALIDATED** |
| L28 (practical cost) | new → **PARTIAL** |

## 25. Reviewer attack

**Reviewer A (dynamic scene graphs).**
- *Strongest:* the stress-condition design derived from observed failures, the mandatory
  visibility-aware baseline, and a negative result on shift.
- *Weakest:* the only surviving win (ghosts) depends on a toy identity cue: static boxes with
  Gaussian jitter. Real ID switches come with appearance change, crowding and re-identification;
  none are modelled. Per-edge (not per-object) occlusion remains.
- *Unsupported if claimed:* "V-HPED handles ID switches", "model-agnostic".
- *Missing:* real tracker cues (ByteTrack/OC-SORT confidence, re-ID similarity), multi-object
  occlusion, a real VidSGG base model.
- *Convinced by H2?* No, beyond "an identity-reliability term can suppress ghosts in simulation".

**Reviewer B (temporal modelling).**
- *Strongest:* an honest formulation (§9 rejects the Hawkes label); M2 ≈ M1 predicted and
  confirmed; closed-form hold horizon; lazy/eager proof.
- *Weakest:* the exact filter is non-monotone with confusable flags, so it *relapses*. That is a
  known consequence of explaining away, and the method provides no termination semantics
  (absorbing state, hysteresis, or a smoothed decision). The evidence-rate power likelihood is
  unjustified and breaks at 15 fps. Flag reliabilities are tuned rather than learned online, which
  explains the H4 failure.
- *Confound:* V-HPED's apparent frontier gains on the registered delay axis are partly produced
  by the metric.
- *Missing:* EM/online estimation of r_ov, r_vo, m; a decision rule over the smoothed or
  fixed-lag posterior; a comparison with IMM/particle filters or switching Kalman models.
- *Convinced?* No: the complexity is not earned except for the identity term, which is a
  one-line mixture likelihood that could be bolted onto GatedEMA.

**Reviewer C (statistics / methodology).**
- *Strongest:* pre-registration before implementation, baseline-first freeze, equal sample
  budgets, complete matrices, a half-sample tuning control, bit-identical Phase 3 regression
  checks, and negative results reported.
- *Weakest:* the primary frontier's cost (first-drop delay) was gameable. The decisive correction
  (false-persistence frontier, relapse) is post-hoc and therefore only exploratory, so the paper
  cannot claim it as confirmatory.
- *Confounds:* identical M5 samples selected on F3 and SC (a shared sample pool across
  conditions; fine, but it reduces the effective independence of those two results); the
  succession world plus E learned from the same world (H3); n = 30 synthetic videos with
  near-deterministic baselines.
- *Missing:* a re-registered replication with a relapse-aware termination metric; sensitivity to
  the 0.02 HV margin; an equal-dimension control (e.g. a GatedEMA extended with an identity term).
- *Convinced by H2 MIXED?* The process is sound and the decision traceable. The substantive
  evidence for a new method is weak.

**Responses / future experiments:**
1. Pre-register a Phase 4b with a relapse-aware termination metric (last-drop delay or delay +
   relapse frames), as a *new* experiment. Phase 4 is not reinterpreted.
2. Test "GatedEMA + identity-tempered updates", the simplest carrier of the only surviving
   mechanism.
3. Add online flag-reliability estimation, and remove evidence-rate tempering or make it
   frame-rate aware for detector misses.
4. Use real tracker identity cues.

## 26. METHOD CONTRIBUTION = **NO**

| Gate criterion | Required | Observed | Met? |
|---|---|---|---|
| 1. H2 PASS, or MIXED with wins in important failure conditions (F1, F2, F3, F5) | yes | MIXED; wins in F3 (1 cell) and F5 (3 cells) | Formally met, but the F3 win is small (+0.01–0.02 on the FPR frontier) |
| 2. Not solely extra tuning capacity | wins survive the half-sample control | F5 wins survive (0.634 vs 0.510) | Met for F5 |
| 3. Ablation identifies the causal rung | yes | A4 (identity) causes the F5 gain; A2 gains coincide with relapse | Partly (A4 yes; A2 confounded) |
| 4. Computationally practical | < 1 ms/frame, < 20× GatedEMA | 0.030 ms, 2.6× | Met |
| 5. Formulation clearly distinct from standard filtering / Hawkes | yes | Own analysis: a standard continuous-time factorial HMM filter with a mixture likelihood; the Hawkes label is invalidated; not literature-checked | **Not met** |

Criterion 5 fails. In addition, H4 FAILS, termination relapses, and the only robust gain comes
from a single likelihood term that does not require the V-HPED machinery. **A new-method
contribution cannot be legitimately claimed.**

**Recommended pivot:** *"Robustness Evaluation and Failure Analysis of Temporal Relationship
Persistence under Observation Corruption."* Phases 3–4 already provide its core:
- a pre-registered, reproducible corruption protocol with matched-termination-cost evaluation;
- the finding that informative visibility breaks the bridging–delay coupling, and its
  dose–response;
- the finding that matched tuning removes the headline failures of gating, while tuning
  mismatch reverses gains;
- identity corruption as the regime where a simple reliability term helps;
- **a metric result:** first-drop termination delay is gameable by non-monotone estimators, so
  relapse-aware termination metrics are needed. This is a concrete, publishable methodological
  point.

## 27. Recommendation for real-data validation

Do **not** take V-HPED to Action Genome as "the method". If the user approves, real-data work
should evaluate the robustness protocol and the simple estimators (No Memory, EMA, GatedEMA,
VA-HMM, and GatedEMA plus an identity-reliability term) on ≥ 2 base VidSGG models with a real
tracker. The aim is to measure whether the synthetic failure regimes (identity switches, UNKNOWN
visibility, flicker) occur in real outputs and how large they are. A relapse-aware termination
metric should be pre-registered first. The model-agnostic claim remains untested.

## 28. Current journal-paper viability

- **As a method paper (V-HPED):** weak; not currently viable.
- **As an evaluation/analysis paper:** plausible for a workshop or a methodology track, *if* it is
  validated on real outputs from at least two base models and positioned against existing VidSGG
  robustness and temporal-consistency work (a literature check is still owed).
- **Biggest risk:** the synthetic findings may not match the magnitude of real failure modes;
  identity switches in particular may dominate or be negligible depending on the tracker.

## 29. Git commits (Phase 4, branch `phase4-vhped`)

| Commit | Content |
|---|---|
| `bad0ef9` | pre-registration, formulation, decision rules; obsolete files archived |
| `04941e2` | infrastructure + VA-HMM |
| `07b01c2` | baseline freeze |
| `0c2c4ab` | baseline evaluation + baseline-stage analysis |
| `095101b` | V-HPED implementation |
| `490b28c` | V-HPED freeze |
| `dfd98c6` | evaluation + analysis artifacts |
| (this report) | exit report, ledger, results notes, Memory.md (see `git log`) |

## 30. Test count

**215 passed, 0 failed** (`python -m pytest -q`): 182 from Phase 3, 11 decision, 9
infrastructure, 3 selection and 10 V-HPED tests.

---

**STOP.** Phase 4 is complete. Action Genome has not been started, no base model has been
integrated, Phase 5 has not been started, no abstract has been written, and publication readiness
is not claimed. Awaiting explicit approval.
