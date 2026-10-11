# TAORG next-method study — Pre-registration

> **Status:** REGISTERED before any registered calibration or comparison of the compared methods
> (2026-10-11, branch `codex/taorg-next-method`). The commit hash is in
> `research/taorg_next_method_research_log.md`.
>
> **Documents and code:**
> - Config: `configs/experiments/next_method.yaml` (its SHA-256 is recorded in every stage record).
> - Formulation: `research/taorg_next_method_formulation.md`.
> - Audit and candidate selection: `research/taorg_next_method_failure_audit.md`.
> - Decision code: `src/taorg/experiments/next_method/decision.py` (tested).
>
> **Frozen and not reinterpreted:** episode study STOP, Phase 5H-1 NARROW, Phase 4 NO, Stage 0
> WEAK-MIXED. The metric definitions are those of the episode pre-registration §6, **unchanged**,
> and the historical metrics are reported alongside. No new metric is introduced.

## 1. Question and hypothesis

> Can a small, principled extension of the strongest practical baseline, GatedEMA + hysteresis
> (B2), improve its audited failure regime without sacrificing its advantages elsewhere?

**Failure regime.** Non-visible-absence calibration mismatch: B2's fixed pooled parameters hold
OCC- or UNKNOWN-flagged absences for the wrong duration when the stream's mix of gaps and
terminations differs from the tuning mix. Audit evidence:
- matched-tuning headroom of 0.134 J in C1 and 0.150 J in S2 on development seeds;
- the dominant errors are missed gap frames.

**H-K1.** B2 + an online, label-free, mixture-cure (Kaplan–Meier) hold for non-visible absences
(K1) improves J over B2 in the primary regime, with no regressions. Its gain is attributable to
the online re-detection-time learning: it beats a tuned fixed per-flag coasting buffer, its own
static prior, and a geometric adaptive detection-profile learner with the same information.

**A-priori expectation, declared from the development pilot** (audit §4):
- K1 probably beats B2 in the primary regime.
- **The static rule, equivalently a tuned per-flag buffer (formulation P2), probably captures most
  of that gain.** G3, G4 and G5 are therefore expected to fail, which gives NARROW with an
  UNSUPPORTED algorithm contribution, or STOP.
- That outcome would be reported as such.

## 2. Methods (all through `reset_video / step / finalize_video`; same input stream)

| Id | Method | Information used | Search |
|---|---|---|---|
| B2 | GatedEMA + hysteresis | records and flags | 200 samples (identical draws and sampler seed to the episode study) |
| BUF | B2 + fixed per-flag coasting buffer (N_OCC, N_UNK) | same | 200 (B2 draws + buffer draws) |
| GEO | B2 + geometric adaptive detection-profile hold (online p_F, π_F) | same + calibration prior | 200 (B2 draws + (τ, κ) draws) |
| K1-static | B2 + survival hold with the prior hazards only | same + calibration prior | the same 200 draws as K1 |
| K1-pooled | K1 with one OCC+UNK stratum | same + calibration prior | the same 200 draws as K1 |
| **K1** | B2 + online shrunk-KM mixture-cure hold | same + calibration prior | 200 (B2 draws + (τ, κ) draws) |

**References (descriptive only, never decisive).** The episode study's frozen operating points
for HSMM, P1 and VA-HMM, evaluated on the new evaluation seeds without re-tuning.

**Equal information:**
- Every method sees exactly the same input records and flags.
- None uses boxes, ground truth or manifests.
- The prior used by K1, K1-static, K1-pooled and GEO is fitted on **unlabeled calibration input
  streams** (C1–C7 × seeds 5000–5019).
- BUF is tuned on the same calibration instances with the same labeled objective as everything
  else.

**Equal budget.** 200 samples per method. All B2-derived methods share B2's base draws (common
random numbers). K1, K1-static, K1-pooled and GEO share the same (τ, κ) draws. H = 3 s is fixed.

## 3. Data

**World, corruptions and conditions.** Exactly those of the episode study (copied verbatim into
the config): C1–C7 calibrated, S0–S3 transfer (never tuned).

**Seeds:**
- calibration (tuning, and the prior fit): 5000–5019, the same as the episode-study calibration;
- **evaluation: 8000–8049 (new; never used before)**;
- development seeds 7000–7019 (audit and pilot) are excluded;
- smoke 9400–9401 (never analysed);
- sampler 2026; bootstrap 4344; half-sample control 2028.

**Fitting vs evaluation.** The prior and all parameter selection read only calibration seeds.
Evaluation ground truth is used only by the metrics.

## 4. Selection

**Rule.** For each method, the configuration with the maximum pooled mean J over the 140
calibration instances (C1–C7 × 20). Ties: lower churn, then lower index. One operating point per
method is used in all 11 conditions.

**Tuning-capacity control.** Re-select every method from the same seeded random half of the
sample indices (seed 2028), and recompute G1 and G3.

**Freeze.** The selected parameters and the fitted prior are committed **before** any evaluation
run.

## 5. Metrics (unchanged definitions; episode prereg §6)

- **Primary:** J = ½ frame-F1 + ½ episode-F1.
- **Harm metrics:**
  - false-persistence time (FPT);
  - false-reactivation rate (FRR);
  - sustained termination delay (STD);
  - ghost time;
  - recurrence-detection delay (RDD).
- **Also:** churn (frozen definition); gap recall; bridge rate; fragmentation; missed rate;
  onset delay; latency (frozen instrumentation; ms per frame on 10 C1 evaluation instances).

## 6. Statistics

- **Unit:** evaluation video, n = 50 per condition, paired across methods on identical instances.
- **Paired difference:** mean, with a 95% percentile bootstrap CI (5000 resamples, seed 4344) and
  a two-sided Wilcoxon test.
- **Multiplicity:** Holm over the registered family of K1 ΔJ tests: 11 conditions × 4 comparators
  (B2, BUF, GEO, K1-static) = 44 tests. Reported alongside; the decisions use CIs and practical
  margins.

## 7. Conditions by role

| Set | Conditions | Why |
|---|---|---|
| PRIMARY | C1, S2 | The largest audited headroom |
| ADAPT | C1, S1, S2, S3 | Where online recalibration *could* matter: gap-heavy, or visibility shift |
| Negative controls (should change little) | C2, C3, C4, C5, C6, C7, S0 | Little or no non-visible-absence mismatch; C6 is also a ghost-harm check (the UNK hold applies to ghost keys) |

## 8. Decision rules (exact; `decision.py`)

**Pairwise result of K1 vs Y in condition c.**

*Win* requires all of:
- ΔJ ≥ 0.02 with CI_lo > 0;
- churn CI_hi ≤ max(0.005, 0.10 × Y's churn);
- no loss.

*Loss* if any of:
- ΔJ ≤ −0.02 with CI_hi < 0;
- a harm metric h ∈ {FPT, FRR, STD, ghost, RDD} rises by ≥ max(floor_h, 25% × Y's mean) with
  CI_lo > 0. The floors are FPT 0.25 s, FRR 0.02, STD 1/30 s, ghost 0.25 s and RDD 1/30 s. Harm
  metrics are used only with ≥ 10 paired seeds;
- churn rises by ≥ max(0.005, 25% × Y's churn) with CI_lo > 0.

Loss takes precedence.

**Algorithm gate:**

| Gate | Requirement |
|---|---|
| G1 | K1 wins vs B2 in **both** PRIMARY conditions |
| G2 | K1 has **0 losses** vs B2 over all 11 conditions |
| G3 (beyond the established fixed rule) | K1 wins vs BUF in ≥ 2 ADAPT conditions, with 0 losses vs BUF |
| G4 (online adaptation causes the gain) | K1 wins vs K1-static in ≥ 1 ADAPT condition, with 0 losses vs K1-static |
| G5 (beyond the established adaptive learner) | K1 wins vs GEO in ≥ 1 ADAPT condition, with 0 losses vs GEO |
| G6 (not extra tuning) | Under the half-sample selection, G1 holds, and K1 wins vs BUF in ≥ 2 ADAPT conditions |
| G7 (cost) | K1 latency ≤ 3 × B2 (ms per frame) |

**Decision:**

| Decision | Rule |
|---|---|
| **STOP** | K1 wins vs B2 in no PRIMARY condition, **or** has ≥ 2 losses vs B2 |
| **GO** | All of G1–G7 |
| **NARROW** | Otherwise |

**Algorithm contribution:**
- **SUPPORTED** if GO;
- **PARTIAL** if NARROW and (G3 or G5) holds;
- **UNSUPPORTED** otherwise.

A gain over B2 that a tuned fixed buffer matches is a useful *engineering* finding, but it is not
an algorithm contribution.

**Reported, not gated:** K1-pooled vs K1 (stratification); BUF vs B2 (does the established rule
already fix B2?); the references.

## 9. Completeness, failures and figures

- **Completeness, checked before analysis:**
  - evaluation: expected 11 × 50 × (6 methods × 2 selections + 3 references), compared with actual,
    missing, duplicate and failed runs;
  - calibration: 140 instances × all configurations;
  - consistent manifests;
  - no calibration-seed leak.
- **Representative failures, predefined rule:**
  - In each PRIMARY condition and in C6, take the evaluation seed with the largest J regret of K1
    vs B2, plus the lower-median positive-regret seed.
  - Each is saved as a timeline.
  - The result is "No failure observed" if no seed has positive regret.
- **Figures:**
  1. ΔJ (K1 − comparator) with CIs per condition;
  2. gap recall and FPT;
  3. ablation: K1 vs static, pooled and GEO, plus latency;
  4. held hold horizons L*_F (static) vs online, per condition;
  5. decision matrix.

## 10. Compute and artifacts

- Local CPU only, **≤ 4 workers** (≈ 1.6 GB of RAM free).
- Calibration and evaluation are checkpointed per instance, with SHA-256 sidecars and resume.
- Heavy parts go to `D:/TAORG_runs/next_method/`; compact artifacts go to `results/next_method/`.

## 11. Declared limitations

1. Synthetic world (the episode-study world, per-edge visibility).
2. Adaptation is per video (tens of runs).
3. The prior is fitted on calibration streams from the same world.
4. The pilot suggests a static rule captures most of the effect.
5. One pooled operating point per method.
6. H = 3 s is fixed.
7. Evaluation is at 30 fps only.

## 12. Amendments

None at registration. Any change must be dated and justified, and must say whether it was made
before or after any registered calibration or evaluation result was seen.
