# TAORG relationship-episode study — Exit report

> **Evidence class: controlled synthetic evidence.** There is no real VidSGG model, tracker or
> dataset. Every number comes from a generated artifact under `results/episodes/` (`R/`).
> Date 2026-10-11, branch `codex/taorg-episodes`.
>
> Earlier conclusions are unchanged and not reinterpreted:
> - Phase 4: METHOD CONTRIBUTION = NO.
> - Stage 0: WEAK-MIXED.
> - Phase 5H-1: HAWKES TRACK = NARROW (exploratory, annotation-only).

## Summary

| Item | Registered outcome |
|---|---|
| Question | Can explicit relationship episodes prevent false reactivation and identity ghosts without harming persistence or genuine recurrence detection? |
| Selected estimator T (ladder rule) | **P1 (episode estimator)**. Identity reliability was not retained, so P2–P4 cannot be T |
| **Study decision** | **STOP**. All three STOP clauses hold; the half-sample control also gives STOP |
| Q1 Does episode handling reduce relapse? | **NO** (registered); see §8 for what was observed |
| Q2 Does it delay genuine recurrence? | **NO** (no ΔRDD loss against any baseline) |
| Q3 Does identity help beyond the identical addition to GatedEMA? | **NO** |
| Q4 Does event history beat simple context? | **NO** (pooled ΔJ +0.00004) |
| Q5 Does the result survive parameter transfer? | **No in-distribution advantage to transfer** |
| Q6 Does the full estimator justify its complexity? | **NO** |
| Recommendation | **Simplify.** GatedEMA with hysteresis (B2) matches or beats the episode estimator on the relapse/termination conditions. Use it as the default termination-aware baseline; keep the episode metrics |

## 1. Provenance

| Step | Commit (full hash) |
|---|---|
| 1. Formulation, related work, pre-registration, decision rules (before any method code) | `42b61ea92583be8ef54cda97bf077e79da500e63` |
| 2. Tested implementation and infrastructure (302 tests) | `e7e7804c0833de56bfa8f3f47e5bb514040dc85b` |
| 3. Parameter freeze (calibration only; before any evaluation run) | `d0276343ec39e6a0a413a189904cf302636cca06` |
| 4. Evaluation summaries and registered analysis | `c312fb951b21de79d065dbdd16875a0719faf335` |
| 5. This report, post-hoc diagnostics, ledger, notes, Memory | the commit containing this file (see the research log) |

**Config.** `configs/experiments/episodes.yaml`, SHA-256 `R/configs/config_sha256.txt`, identical in
every `R/raw/stage_*.json`.

**Stage records:**
- Calibration was run from `e7e7804`, evaluation from `d027634`.
- Both report `git_dirty = True`. The dirty files were the unrelated demo files
  (`demo/output/*`), plus the then-uncommitted results README at calibration. No code was modified.

**Branch.** The branch was created from `phase5h-hawkes-feasibility` @ `b210619`. The PR is
stacked on that branch.

**Amendments: none.** Implementation deviations, all recorded in the research log:
1. **A world test assumption was wrong, not the injector.** Background false detections are
   injected after hidden termination in the registered order, so they can fall inside a hidden
   block. The test now exempts exactly the manifest-listed false detections. No model or
   injector code changed.
2. **A smoke-run worker died at import** with "paging file too small" (the system drive is nearly
   full). The pool respawned it and no task was lost. The registered runs used 6 workers, and
   their logs record no errors.
3. **The half-sample control's latency** uses the primary operating point's latency for the same
   method. Latency depends on the method, not the configuration, at this scale.

## 2. Method (frozen formulation)

See `research/taorg_episode_formulation.md`. The study keeps five quantities separate:
1. **Existence** Z: a latent belief.
2. **Observability**: the absence flag, treated as a covariate. Every absence has a likelihood
   ratio ≤ 1.
3. **Identity reliability** w: from constant-velocity box residuals.
4. **Termination**: an absorbing within-episode death filter crossing θ_end.
5. **Birth of a subsequent episode**: a CUSUM on detections, with a refractory term after
   termination.

Lemma 1 (no relapse without detections) holds by construction and is tested. Incorrect termination
is recovered only by a new birth, which is counted as fragmentation.

The baselines were B1 GatedEMA, B2 GatedEMA + hysteresis, B3 VA-HMM, B4 explicit-duration HSMM and
B5 GatedEMA + the identical identity module. The ladder rungs were P1, P2 (+ identity),
P3 (+ transition context) and P4 (+ event history).

## 3. Frozen parameters

`R/configs/frozen_parameters.md` (generated before evaluation) holds the selection rationale.
One operating point per method was selected by the maximum pooled calibration J over C1–C7 ×
20 seeds:

| Method | Cal. J | Cal. churn | Key parameters |
|---|---|---|---|
| B1 GatedEMA | 0.878 | 0.027 | decay_occ 0.9989, decay_unk 0.070, decay_vis 0.454 |
| B2 GatedEMA+hyst | 0.903 | 0.017 | θ_on 0.431, θ_off = 0.47·θ_on, decay_occ 0.959 |
| B3 VA-HMM | 0.892 | 0.011 | θ_occ 0.885, p_birth 0.0006 |
| B4 HSMM | 0.893 | 0.013 | ON Weibull (1.74, 2.15 s), OFF (1.52, 17.2 s) |
| B5 GatedEMA+ID | 0.858 | 0.017 | σ_id 0.81 diag (permissive) |
| P1 Episode | 0.929 | 0.016 | m 0.011, f 0.010, a_occ 0.85, a_unk 0.82, μ 0.020 s⁻¹, θ_end 0.034, h 6.1, h_r 4.5, τ_r 0.38 s |
| P2 +identity | 0.929 | 0.015 | base #0101: a_unk 0.41, θ_end 0.185, h_r 11.5, τ_r 1.5 s; σ_id 0.73 |
| P3 / P4 | 0.930 / 0.930 | 0.016 | base #0101; κ_c 24.4; κ_h 11.3, τ_h 0.65 s |

The transition statistic E was fitted on calibration ground truth only. It is in
`R/configs/transition.json`; the partner entries are 0.27–0.35.

## 4. Completeness and tests

**Completeness** (`R/aggregated/completeness.json`):
- Evaluation: 11,000 expected = 11,000 actual (11 conditions × 50 seeds × 20 variants).
- 0 missing, 0 duplicate, 0 failed. Every instance has a single manifest SHA across methods.
- Calibration: 140/140 instances × 1801 configurations, not smoke, full sample budget.
- No calibration-seed leak.

**Tests:** 302 passed at checkpoint 2, and the final count is given in §17. The episode-study
tests cover:
- temporary gaps; hidden termination (monotone belief, eventual termination); misleading flags
  after termination;
- genuine recurrence as a new episode; recovery from false termination; isolated false
  detections; the refractory term;
- irregular timestamps (death-filter partition invariance; HSMM sub-steps; identity null
  widening);
- numerical extremes; determinism; causality;
- identity swap vs ordinary and curving motion;
- metric toy cases (perfect, always-off, always-on, relapse, censoring, bridging, ghosts,
  undefined cases);
- injector ground-truth checks; common random numbers; the transition statistic fitted on
  calibration seeds only;
- the pipeline smoke run with resume, and the completeness gate refusing smoke data.

## 5. Results by question

Values are evaluation means over n = 50 seeds per condition (`R/tables/descriptive.md`). Paired
differences are P1 − comparator with 95% bootstrap CIs (`R/aggregated/paired.csv`).

### 5.1 Overall quality J

| | C1 gaps | C2 hidden term. | C3 misleading | C4 recurrence | C5 bnd. flicker | C6 ID swap | C7 motion |
|---|---|---|---|---|---|---|---|
| B1 GatedEMA | 0.970 | 0.896 | 0.855 | 0.964 | 0.635 | 0.818 | 0.957 |
| B2 GatedEMA+hyst | 0.844 | 0.926 | 0.888 | 0.969 | **0.886** | 0.825 | 0.969 |
| B3 VA-HMM | 0.958 | 0.905 | 0.861 | 0.964 | 0.754 | 0.858 | 0.964 |
| B4 HSMM | 0.964 | 0.909 | 0.882 | 0.965 | 0.742 | 0.826 | 0.966 |
| P1 Episode | **0.987** | **0.933** | **0.899** | **0.990** | 0.823 | **0.881** | **0.991** |

**Observations:**
- P1 has the highest mean J in 6 of 7 calibrated conditions.
- B2 is better under boundary flicker (ΔJ −0.062 [−0.070, −0.054]) and in three of four transfer
  conditions (S0 −0.024, S1 −0.038, S3 −0.025).
- B2's pooled operating point is poor under heavy gaps (C1 J 0.844, gap recall 0.869).

**Why the higher J does not count as registered wins:**
- Against B3 and B4, P1's higher J (+0.017 to +0.081 in C1–C7) **fails the registered churn
  non-inferiority guard**. P1's churn is ≈ 0.014 vs ≈ 0.007, and the margin is
  max(0.005, 10% × 0.007).
- Against B2 in C2 and C3, the J difference is not significant: C2 +0.006 [−0.002, 0.015], C3
  +0.011 [−0.005, 0.027]. Holm p = 0.33 for both.

### 5.2 Relapse: false reactivation and false persistence (Q1)

**Observation:** relapse was rare outside flicker. FRR is ≤ 0.009 for every baseline in C1–C4
and C7; it matters only with boundary flicker and in the mixed conditions.

| FRR | C5 | S0 | S1 | S2 | S3 |
|---|---|---|---|---|---|
| B1 | 0.594 | 0.314 | 0.390 | 0.218 | 0.328 |
| B2 | **0.142** | 0.040 | 0.046 | 0.031 | 0.042 |
| B3 / B4 | 0.382 / 0.382 | 0.141 / 0.207 | 0.234 / 0.274 | 0.149 / 0.270 | 0.181 / 0.285 |
| P1 | 0.283 | 0.133 | 0.156 | 0.093 | 0.165 |
| P2 (descriptive) | 0.046 | 0.014 | 0.016 | 0.020 | 0.024 |

- P1 reduces FRR relative to B1 (−0.311), B3 (−0.099) and B4 (−0.100) in C5, all with CI
  excluding 0.
- But P1 has **twice B2's FRR** (+0.140 [0.116, 0.166]), which is a registered loss.
- **False persistence under misleading flags (C3)** is ≈ 14–19 s per 12-s video for every method.
  B2 is lowest (14.1 s); P1 is at 17.3 s (+3.2 s vs B2).
- **Answer Q1 = NO.** P1 does not beat all of B1–B4 in C3 or C5, because B2 is not beaten.

### 5.3 Termination under occlusion (C2)

**Observations:**
- Sustained termination delay: B2 0.665 s; P1 0.922 s; B1/B3/B4 0.935–0.962 s.
- P1 vs B2: +0.257 s [0.236, 0.278], a 39% increase. That is a registered loss on the target
  harm.
- No method terminates quickly when termination coincides with an OCC flag. This matches the
  formulation's information limit (formulation §3; Phase 4 §9).

### 5.4 Recurrence detection (Q2)

Recurrence-detection delay was **not** worse for P1:
- in C4: P1 0.034 s, B1 0.034 s, B2 0.086 s, B3 0.066 s, B4 0.045 s;
- in C5: P1 0.085 s vs B2 0.151 s.

There is no ΔRDD loss in C4 or C5. **Q2 = NO (does not delay recurrence).**

Episode merging (bridge rate) was ≤ 0.005 for P1 in C1–C7. B2 bridged 14% of recurrences in C3,
and B5 52–66% in C2 and C3.

### 5.5 Persistence through temporary gaps

- Gap recall: P1 0.997–1.000 in C1–C4 and C7, comparable to B1, B3 and B4.
- B2 0.869 in C1, the cost of its high on-threshold. B5 0.956 in C1.
- Under 50% UNKNOWN flags (S2), GatedEMA-family recall collapses (B1 0.06, B2 0.18), while P1
  stays at 0.981. This is because P1 treats UNKNOWN absences as weak evidence (a_unk 0.82).

### 5.6 Identity ghosts (C6) and identity false alarms (C7) — Q3

| C6 | B1 | B2 | B3 | B4 | B5 | P1 | P2 |
|---|---|---|---|---|---|---|---|
| ghost time (s/video) | 2.34 | 2.28 | 7.74 | 3.43 | **2.19** | 8.23 | 3.72 |
| swap recall | 0.11 | 0.17 | 0.77 | 0.30 | 0.18 | 0.76 | 0.46 |
| J | 0.818 | 0.825 | 0.858 | 0.826 | 0.823 | **0.881** | 0.844 |

**Observations:**
- **The episode estimator produces more ghost output than every GatedEMA variant:** P1 vs B1
  +5.9 s [4.9, 6.9].
- It keeps the true keys during swaps (swap recall 0.76).
- Identity (P2 − P1) cuts ghost time by 4.5 s [−5.3, −3.7], far more than the identical addition
  to GatedEMA (B5 − B1, −0.15 s). The interaction CI is [−5.14, −3.61].
- But P2 still has more ghost time than B5 (3.72 vs 2.19 s), so **P2 does not beat B5 in C6.
  Q3 = NO.**
- C7 (fast motion, no swaps): no identity false-alarm loss. P2 − P1 ΔJ = −0.003; P1 ≥ 0.988 J.

### 5.7 Parameter transfer (Q5)

Parameters were tuned on C1–C7 and never retuned:
- **S2 (50% UNKNOWN):** P1 wins against all four comparators (J 0.846 vs ≤ 0.767).
- **S0, S1 and S3:** P1 loses to B2 on J by 0.024–0.038.
- There is no in-distribution advantage to transfer (0 BEN wins). The S2 win is the flip side of
  P1's weak-UNKNOWN assumption, and that assumption is also why its FPT in S2 is high: 12.2 s vs
  0.5–0.7 s for B1, B2 and B5.

### 5.8 Latency (computational cost)

Mean ms per frame on C1, one machine (`R/tables/latency.md`):

| Method | ms/frame | × GatedEMA |
|---|---|---|
| B1 GatedEMA | 0.0077 | 1.0 |
| B2 GatedEMA+hyst | 0.013 | 1.7 |
| B3 VA-HMM | 0.011 | 1.5 |
| B4 HSMM | 0.112 | 14.5 |
| B5 GatedEMA+ID | 0.020 | 2.6 |
| P1 | 0.016 | 2.1 |
| P2 | 0.027 | 3.6 |
| P3 | 0.030 | 3.9 |
| P4 | 0.033 | 4.3 |

All are practical at synthetic scale (≈ 8–13 keys).

## 6. Ablation ladder (`R/tables/ladder.md`, Fig. 5)

| Step | Registered rule | Result | Retained |
|---|---|---|---|
| P2 − P1 identity | C6 ghost ↓ ≥ 25% with CI; no J loss in C1–C7 | Ghost −4.51 s [−5.30, −3.72], but C6 ΔJ −0.037 [−0.048, −0.026] (a loss) | **No** |
| P3 − P2 transition context | Gain in C2 or C3; no J loss | C3 ΔJ +0.024 [0.016, 0.032]; C2 ΔSTD −0.053 s; no losses | Yes (vs P2), but P2 itself was not retained |
| P4 − P3 event history | Pooled ΔJ ≥ 0.01 with CI > 0, and a C2/C3 win | Pooled ΔJ +0.00004 [−0.0002, 0.0003]; no target win | **No** |

So T = P1. The transition context improved hidden and misleading termination slightly, but by the
consecutive rule it cannot enter T. **Event history adds nothing over simple context in this
world**, which was the pre-registered expectation.

## 7. Representative failures (predefined rule, `R/failures/`)

The rule: J regret of T against the comparator with the best mean J in the condition.

| Condition | Best comparator | Seeds with positive regret | Worst / median seed (regret) |
|---|---|---|---|
| C2 | B2 | 26/50 | 6040 (0.060) / 6026 (0.011) |
| C3 | B2 | 22/50 | 6016 (0.129) / 6015 (0.031) |
| C5 | B2 | 50/50 | 6015 (0.121) / 6039 (0.062) |
| C6 | B3 | 5/50 | 6038 (0.003) / 6032 (0.001) |
| C4 | B2 | 0/50 | No failure observed |

**C5 worst (seed 6015).**
- *Observation:* P1 splits true episodes when misses cluster near boundaries, e.g. key (2,6,0)
  is output as four runs. B2 keeps each episode whole. P1's failure is mostly **fragmentation
  (false termination plus delayed re-birth)**, not reactivation.
- *Explanation (hypothesis):* the selected m = 0.011 makes a visible miss strong termination
  evidence, and the refractory term delays recovery.

**C3 worst (seed 6016).**
- *Observation:* both methods hold through the misleading post-termination OCC flags. P1 holds to
  the end of the flag window; B2 drops earlier.
- *Explanation (hypothesis):* with a_occ 0.85 and μ 0.02 s⁻¹, P1's hold horizon under OCC is
  several seconds, while B2's decay_occluded 0.959 decays within ≈ 1 s.

## 8. Post-hoc, exploratory diagnostics (defined after the registered STOP; never decisive)

`scripts/episodes/posthoc_diagnostics.py`, `R/posthoc/`.

**E1 — attribution.** P1 and P2 selected different base configurations (#0176 vs #0101).
Re-running P1 with P2's base (#0101, identity off) gives:
- *Identity mechanism, same base (P2#0101 − P1#0101):*
  - C6 ghost +0.03 s [−0.01, +0.07] and ΔJ −0.003;
  - C5 FRR −0.046 [−0.060, −0.033].
  - **The identity cue contributes almost nothing to ghost suppression.** The selected identity
    detectors were permissive (σ_id ≈ 0.7–0.8 box diagonals), because false alarms cost J on the
    pooled objective.
- *Base-parameter choice (P1#0101 − P1#0176):*
  - C6 ghost −4.54 s [−5.34, −3.77];
  - C5 FRR −0.191 [−0.217, −0.167].
  - **The ladder's identity contrast mostly reflects a different base configuration**:
    no-record evidence (a_unk 0.41 vs 0.82) and a stronger refractory term (h_r 11.5 vs 4.5).
- *P1#0101 vs B2:*
  - C5 FRR −0.051 [−0.079, −0.023] but ΔJ −0.030 [−0.041, −0.020];
  - C2 ΔSTD +0.270 s.
  - Even the stronger-refractory base does not beat hysteresis on the registered quality metric.

**E2 — ghost timing in C6.**
- *Observation:* B1 has 70.1 ghost frames per video during swaps and 0.0 outside them. P1 has
  77.3 during and 169.7 after; B3 83.3 and 149.0.
- *Explanation:* consistent with the hypothesis that ghost keys born during a swap **linger after
  it ends**. The swapped key has no records afterwards, and "no record" is treated as weak UNKNOWN
  evidence by a long-hold estimator.
- An episode cannot terminate on evidence it never receives. **Absorbing termination does not
  help when the false episode simply stops being observed.**

These diagnostics explain the registered result. They do not change it.

## 9. Criterion-by-criterion decision (`R/aggregated/analysis.json`, `R/tables/decision.md`)

T = P1; the comparator sets are B1–B4, plus B5 in C6 and C7.

| Condition | Role | Cell | Reason |
|---|---|---|---|
| C1 | NI | neutral | wins vs B2; churn guard fails vs B3 and B4; ΔJ +0.017 vs B1 (< 0.02) |
| C2 | benefit | **loss** | STD +39% vs B2 |
| C3 | benefit | neutral | not beaten: B2 (ΔJ n.s.), B3 and B4 (churn guard) |
| C4 | NI | neutral | churn guard vs B3 |
| C5 | benefit | **loss** | ΔJ −0.062 and FRR +99% vs B2 |
| C6 | benefit | **loss** | ghost time +250% vs B1, B2, B5; +140% vs B4 |
| C7 | NI | neutral | churn guard vs B3 and B4 |
| S0, S1, S3 | transfer | **loss** | ΔJ −0.024 / −0.038 / −0.025 vs B2 |
| S2 | transfer | **win** | ΔJ +0.079 to +0.291 against all four comparators |

| Rule | Threshold | Observed | Met? |
|---|---|---|---|
| GO (i) BEN wins | ≥ 3 of 4, ≥ 2 of {C2, C3, C5} | 0 | No |
| GO (ii) calibrated losses | 0 | 3 (C2, C5, C6) | No |
| GO (iii) transfer losses | ≤ 1 | 3 | No |
| GO (iv) latency | ≤ 1 ms and ≤ 20× | 0.016 ms, 2.1× | Yes |
| GO (v) half-sample control not STOP | — | STOP | No |
| STOP (a) BEN wins ≤ 1 | — | 0 | **triggered** |
| STOP (b) losses ≥ wins | — | 6 ≥ 1 | **triggered** |
| STOP (c) B2 never beaten in C2, C3, C5 | — | not beaten in any | **triggered** |

# **Decision: STOP**

**The simple baseline matches the episode estimator where it was supposed to help; on C2 and C5
it beats it.**

**Half-sample control:**
- The decision is STOP, T = P1.
- Cells: one win (S2) and one loss (C6), with the rest neutral.

**Holm (reported, not decisive):** Wilcoxon on ΔJ for T vs the comparators, over the 46-test
family. All adjusted p-values are < 0.01 except C2 and C3 vs B2 (p = 0.33).

## 10. Novelty assessment

Every mechanism is established (`research/taorg_episode_related_work.md`):
- existence probability with deletion (IPDA/JIPDA, Bernoulli filters);
- absorbing termination and relabelled rebirth (labelled RFS, MHT, SORT/ByteTrack lifecycles);
- CUSUM births (Page 1954);
- explicit-duration HSMMs (Yu 2010);
- hysteresis (Canny 1986);
- identity-tempered evidence (Phase 4 L24).

The combination is not claimed as novel, and the experiment did not justify it as a method.

**What remains potentially useful:**
1. **Episode-level evaluation.** The registered J composite, FRR, sustained termination delay,
   recurrence delay with bridging, fragmentation and ghost time. These metrics exposed failures
   that frame-level metrics hide: B2's gap losses, P1's fragmentation and lingering ghosts, B5's
   episode merging.
2. **The negative finding.** On these conditions, a two-threshold GatedEMA is as good a relapse
   control as an explicit episode estimator.

## 11. Limitations

1. Synthetic world; per-edge visibility; a constant-velocity identity cue on synthetic boxes;
   confident false detections with true-like scores.
2. **One pooled operating point per method.** P1 ranks first on mean J in most conditions, yet
   condition-matched tuning (Phases 3–4) might reorder the methods. That was not tested.
3. The decisive guards (churn non-inferiority, harm margins) are strict and were registered.
   P1's J advantage over B3/B4 was disqualified by churn (0.014 vs 0.007). A different but
   equally defensible margin could have produced wins there; it would not change the B2 losses.
4. J is a registered composite; other weightings could rank differently. Components are reported.
5. Ladder contrasts compare **selected** operating points, so mechanism and base-parameter
   selection are confounded. E1 shows this matters for the identity rung.
6. The world parameters (recurrence gaps, flicker windows, succession timing) are design choices;
   all conclusions are conditional on them.
7. The calibration and evaluation stage records are `git_dirty` because of unrelated demo files
   only.

## 12. Three critical reviewer assessments

**Reviewer A (tracking / VidSGG).**
- *Strongest:* the study adopts track-management semantics (confirmation, deletion, rebirth as a
  new label), tests them against the matching simple baselines, and reports that they lose.
- *Weakest:* the identity cue is a toy, and the selected σ_id made it nearly inert (E1). No real
  tracker cues (re-ID, track confidence) are used. Ghost suppression in real VidSGG would depend
  on the tracker.
- *Missing:* real tracker outputs; per-object occlusion.
- *Convinced?* By the negative result, yes, within this world. By any positive claim, no.

**Reviewer B (temporal modelling).**
- *Strongest:* Lemma 1 removes the Phase 4 explaining-away relapse by construction; the
  formulation is clean; the HSMM baseline is included.
- *Weakest:* the estimator's real failure modes come from the *observation model*:
  - a strong visible-miss likelihood fragments episodes;
  - weak UNKNOWN/no-record evidence keeps ghost keys alive.
  The episode structure does not address either. Absorbing termination does not help when
  termination evidence never arrives.
- *Missing:* learning m, f and a_F online; a "no record" category distinct from UNKNOWN;
  fixed-lag smoothing for the termination decision.
- *Convinced?* No. The structure is principled, but the gains belong to the evidence model, not
  the episodes.

**Reviewer C (statistics / methodology).**
- *Strongest:*
  - pre-registration before any method code;
  - common random numbers;
  - a complete matrix;
  - a completeness gate that refuses smoke data;
  - checkpoint commits in order;
  - registered STOP reported with every clause shown;
  - post-hoc diagnostics clearly separated.
- *Weakest:*
  - pooled single-point tuning makes results depend on the composite objective;
  - the churn guard (10% of a very small baseline churn) decides several cells;
  - ladder contrasts confound selection with mechanism.
- *Confounds:* P1 was selected on pooled J, which rewards persistence. That raises false
  persistence under misleading flags (C3) and lowers termination speed (C2).
- *Convinced by STOP?* Yes: three independent STOP clauses hold, the half-sample control agrees,
  and the B2 losses in C2 and C5 have tight CIs.

## 13. Recommended next research step (not started)

1. **Simplify.**
   - Use GatedEMA + hysteresis (B2) as the termination-aware baseline in the robustness-evaluation
     paper. Do not propose the episode estimator as a method.
   - Report the episode metrics (FRR, sustained termination delay, recurrence delay with
     bridging, fragmentation, ghost time, J) as the methodological contribution, with this study
     as the evidence that they separate methods which frame metrics do not.
2. If the user wants one more synthetic check before real data, the only question worth
   pre-registering is the evidence-model issue from E1/E2: separate "no record" from UNKNOWN, and
   examine condition-matched tuning. **Not started.**
3. Real-data validation (base VidSGG models plus a real tracker) remains the gate for any claim
   about real video. **Not started; it needs user approval.**

## 14. Evidence-ledger changes

`research/paper_evidence_ledger.md`: added L43–L49 (see the change log there). None was upgraded
on p-values.

## 15. Figures

All in `R/plots/`:
- `fig1_persistence`: gap recall and J;
- `fig2_false_reactivation`: FRR and FPT;
- `fig3_recurrence_delay`: RDD, bridge rate and STD;
- `fig4_identity_ghosts`;
- `fig5_ablation_costs`: ladder ΔJ and latency;
- `fig6_decision_matrix`.

## 16. Artifacts

- `results/episodes/README.md` explains the layout and reproduction.
- Heavy per-instance checkpoints (53 MB calibration plus evaluation JSON) are in
  `D:/TAORG_runs/episodes/`, with SHA-256 in `R/raw/*_parts_sha256.json`. They are not needed to
  reproduce any table, because everything downstream reads committed artifacts.

## 17. Test count

**302 passed, 0 failed** (`python -m pytest -q`, run before the final commit). Of these, 44 are
episode-study tests: decision 8, estimator 12, baselines 7, metrics 6, world/injectors/sampler/fitting 10,
pipeline 1.

---

**STOP.** The episode study is complete. Not started, as instructed: an independent dataset,
detector integration, heap optimisation, or another research phase.
