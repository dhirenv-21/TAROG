# TAORG next method — Exit report

> **Evidence class: controlled synthetic evidence** (the episode-study world). There is no real
> model, tracker or dataset.
>
> All numbers come from `results/next_method/` (`R/`). Development-data results are in `R/audit/`
> and are labelled; post-hoc results are in `R/posthoc/` and are labelled.
>
> Date 2026-10-11, branch `codex/taorg-next-method`.
>
> Earlier decisions are unchanged: episode study STOP, Phase 5H-1 NARROW, Phase 4 NO, Stage 0
> WEAK-MIXED.

## Summary

| Item | Outcome |
|---|---|
| Starting point | GatedEMA + hysteresis (B2), the strongest practical baseline |
| Audited failure | Non-visible-absence calibration mismatch: B2's pooled hold of OCC/UNK absences is wrong for gap-heavy and UNKNOWN-heavy streams (matched-tuning headroom on development seeds: C1 0.134, S2 0.150 J) |
| Candidate (one of three) | **K1:** B2 + an online, label-free, mixture-cure (Kaplan–Meier) hold for non-visible absences |
| **Registered decision** | **STOP**: G1 fails (K1 loses in S2) and there are 4 losses vs B2 (≥ 2) |
| **Algorithm contribution** | **UNSUPPORTED**: no gain over a tuned fixed per-flag buffer, over its own static prior, or over a geometric adaptive learner; latency 3.7× B2 |
| What the evidence supports | (i) Holding non-visible absences for a calibrated duration fixes B2's gap failure (C1 +0.13 J), but it is an established coasting rule and it costs false persistence after hidden terminations. (ii) The repository's convention "no record = UNKNOWN" conflates two phenomena with opposite optimal holds: identity-swap ghosts vs unknown-visibility gaps |
| Recommendation | Do **not** build a paper around K1. Next decision: (a) fix the input representation (separate "no record" from "UNKNOWN") and re-audit, as a **new pre-registered study**; or (b) consolidate the robustness-evaluation paper with B2 (+ fixed buffer) as the practical method |

## 1. Provenance

| Checkpoint | Commit |
|---|---|
| 1. Failure audit, candidate selection, formulation, pre-registration (before any registered comparison) | `fe084a6cce6d5500b2b10b8c390bdfc0c4eb9df1` |
| 2. Tested implementation (316 tests) | `ed9a021ba16e3a91191d882f93316ca74e4b68d9` |
| 3. Frozen parameters and prior (before any evaluation run) | `8a7c916568bc4fd6eb95d599fb2b392107a80448` |
| 4. Evaluation and registered analysis | `320bc2e984f31053e83500857ec0528483349f19` |
| 5. This report and documentation | the commit containing this file (see the research log) |

**Branch.** Created from `codex/taorg-episodes` @ `ecb3859` (PR #3). The new PR is stacked on it.

**Code commits per stage.** Calibration was run from `ed9a021` and evaluation from `8a7c916`. The
stage records show `git_dirty` only because of the unrelated demo files.

**Amendments: none.**

**Deviations recorded in the log:**

1. **Two test-design errors fixed in tests.** A flat prior cannot show learning; there was also a
   malformed leftover assertion. No model change.
2. **Implementation detail found while writing this report.** The decision on a non-visible
   absence frame uses the posterior of the *current frame's* flag stratum with the run clock.
   Formulation §2/§4 describe the stratum as the run's *first* non-visible flag. The two coincide
   whenever a run's flags do not change:
   - every C1–C7 run;
   - S0/S3 OCC runs.

   They can differ in S1 and S2, where flags flip within a run. The registered result stands as
   run. Under either reading, an UNKNOWN-onset run uses q_UNK, which is 0 at the frozen point (§6),
   so the S2 failure is not explained by this detail.

## 2. Failure audit and candidate selection

See `research/taorg_next_method_failure_audit.md`.

**Key observations:**
- **Calibration** (`results/episodes/raw/calibration_summary.csv.gz`): B2's pooled point has a
  per-condition regret of 0.141 J in C1, and the minimax regret of a single B2 configuration is
  0.141.
- **Development seeds** (`R/audit/a2_headroom.csv`): matched B2 tuning gains 0.134 J in C1 and
  0.150 in S2, and 0 in C2/C3/S0/S3. The dominant errors are missed gap frames: 82 per video in
  C1 and 172 in S2 (`R/audit/a3_errors.csv`).
- **Development seeds** (`R/audit/a4_runs.csv`): a video contains tens of observable absence runs
  that end in re-detection.

**Candidates:**
- **K1** (selected): survival-calibrated hold, learned online.
- **K2:** identity-swap memory remapping. Its headroom is 0.014 J, and the synthetic swap
  structure would flatter it.
- **K3:** a risk-calibrated two-sided threshold. It overlaps the episode estimator's CUSUM.

**Development pilot** (declared before registration; `R/audit/dev_pilot*.csv`):
- Iteration 1 failed: re-detection does not imply persistence, so holding visible absences
  bridges terminations.
- Iteration 2, restricted to non-visible absences, was positive vs B2. But static ≈ online, so
  the registered expectation was "NARROW or STOP likely".

## 3. Related-work comparison

| Mechanism | Closest established work | Status here |
|---|---|---|
| Hold an unseen object for a bounded time | Tracker coasting buffers (SORT `max_age`, ByteTrack `track_buffer`; SoDA lost thresholds) | **Comparator BUF**; it matches K1 or beats it |
| Online learning of detection reliability | Mahler, Vo & Vo 2011 (unknown detection profile, CPHD); online EM for HMMs (Cappé & Moulines 2009; Cappé 2011) | **Comparator GEO**; K1 is identical to it at the selected point |
| Survival with a never-returning fraction | Mixture cure models (Farewell 1982; Maller & Zhou 1996); Kaplan–Meier 1958 | Used, not claimed |
| Age-dependent survival in trackers | Labelled-RFS visual MOT (arXiv 1611.06011) | Related; hand-designed rather than learned |
| Temporal scene graphs | STTran (ICCV 2021), TEMPURA (CVPR 2023), VidVRD (MM 2017) | Learned temporal context; not post-processing persistence rules |

**Potential distinction (not demonstrated):** label-free, nonparametric, per-visibility-stratum
online recalibration of the termination hold. **The experiment did not support it.**

## 4. Formulation and implementation

See `research/taorg_next_method_formulation.md` and `src/taorg/temporal/cure.py`
(`B2SurvivalHold`). B2's decisions are unchanged except on non-visible absence frames of ON keys.
There, the key is held while

    q_F(l) = 1 − Ŝ_F(H)/Ŝ_F(l) ≥ τ

where Ŝ_F is the Kaplan–Meier survival of re-detection times, learned online per video and shrunk
to a prior fitted on unlabeled calibration input streams.

**Ablations and comparators** (one class, five modes): static, pooled, geometric, buffer.

**Sampling.** Run length is in bins of 1/30 s. With irregular timestamps the elapsed bins are
counted. Frame-rate invariance is not claimed.

## 5. Experiment and completeness

**Methods:** B2, BUF, GEO, K1-static, K1-pooled and K1, at 200 samples each with common random
numbers. The episode-study HSMM, P1 and VA-HMM are included as descriptive references.

**Selection:** pooled calibration J over C1–C7 × seeds 5000–5019 (140 instances × 1200
configurations; 0 failures; 1324 s).

**Evaluation:** seeds 8000–8049 (new), 11 conditions.

**Completeness** (`R/aggregated/completeness.json`):
- 8250/8250 rows;
- 0 missing, duplicate or failed;
- consistent manifests;
- no calibration, evaluation or development seed leak.

**Frozen points** (`R/configs/frozen_parameters.md`):
- Every extension kept B2's base #0091.
- K1, static, pooled and GEO all selected τ = 0.376, **κ = 838**.
- BUF selected N_OCC 40, N_UNK 46.
- The half-sample control selected identical points.

## 6. Results

Mean J over n = 50 (`R/tables/descriptive.md`). ΔJ is K1 − comparator, with 95% bootstrap CIs
(`R/tables/decision.md`, Fig. 1).

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 | S0 | S1 | S2 | S3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 0.844 | 0.925 | 0.876 | 0.965 | 0.870 | 0.820 | 0.967 | 0.888 | 0.814 | 0.724 | 0.850 |
| BUF | 0.944 | 0.929 | 0.871 | 0.977 | 0.890 | 0.848 | 0.982 | 0.888 | 0.820 | **0.864** | 0.857 |
| K1 | **0.974** | 0.921 | 0.856 | 0.977 | 0.890 | 0.827 | 0.982 | 0.886 | 0.820 | 0.568 | 0.855 |
| K1-static | 0.974 | 0.921 | 0.856 | 0.977 | 0.890 | 0.827 | 0.982 | 0.886 | 0.820 | 0.563 | 0.855 |
| K1-pooled | 0.956 | 0.926 | 0.868 | 0.977 | 0.890 | **0.850** | 0.982 | 0.888 | 0.820 | 0.864 | 0.856 |
| GEO | 0.974 | 0.918 | 0.854 | 0.977 | 0.890 | 0.827 | 0.982 | 0.885 | 0.820 | 0.563 | 0.855 |
| ref P1 episode | 0.988 | 0.934 | 0.902 | 0.990 | 0.808 | 0.873 | 0.991 | 0.866 | 0.776 | 0.848 | 0.828 |

### K1 vs B2 (Q: does the extension improve the failure regime without regressions?)

**Wins:**
- C1: ΔJ +0.129 [0.117, 0.142]; gap recall 0.984 vs 0.866.
- C5: +0.021 [0.014, 0.028].

**Losses:**
- **S2:** ΔJ −0.156 [−0.166, −0.146]; gap recall 0.06 vs 0.17; churn +0.025.
- **C3:** ΔJ −0.021 [−0.031, −0.010].
- **C2:** FPT +3.92 s [3.56, 4.28] on a 10.2 s base; sustained termination delay +0.244 s.
- **S0:** FPT +3.06 s; STD +0.171 s.

**Neutral:** C4, C6, C7, S1, S3. The gains there are below the 0.02 margin, e.g. C7 +0.015.

### K1 vs its comparators (Q: is the gain attributable to the proposed mechanism?)

- **K1 vs K1-static (online learning):**
  - ΔJ within ±0.005 in every condition; 0 wins.
  - The learned hold horizon moved by at most 1 bin (`R/aggregated/hold_horizons.csv`, Fig. 4):
    OCC L* = 50 bins (1.67 s) and UNK L* = 0 everywhere.
- **K1 vs BUF (the established fixed rule):**
  - one win, C1 +0.030 (P2 implies this is a different selected hold length — static L*_OCC = 50
    vs N_OCC = 40 — not a different mechanism);
  - losses in **S2 (−0.297)** and **C6 (−0.021)**.
- **K1 vs GEO (the established adaptive learner):** identical (|ΔJ| ≤ 0.005).
- **K1 vs pooled:**
  - C1 +0.018;
  - S2 −0.297 and C6 −0.023, where pooling makes UNK inherit the OCC hold.

### BUF vs B2 (does the established rule already fix B2?)

- Wins in C1 (+0.100) and C5 (+0.021).
- J +0.140 in S2, but registered harm losses there (FPT/STD).
- Harm losses in C2 and S0 (FPT/STD) and C6 (ghost time 8.0 vs 2.3 s).

### Holm

Over the 44 registered ΔJ tests, adjusted p > 0.05 in 25 cells, all with |ΔJ| ≤ 0.008.
The decisions use CIs and margins.

### Latency (`R/tables/latency.md`)

| Method | ms/frame | × B2 |
|---|---|---|
| B2 | 0.017 | 1.0 |
| BUF | 0.029 | 1.7 |
| GEO | 0.034 | 2.0 |
| K1-static | 0.055 | 3.3 |
| **K1** | **0.063** | **3.7** (above the 3× gate) |

The K1 cost comes from the O(H) posterior evaluation, which is uncached across frames.

### Observations vs explanations

**O1.** At the selected κ = 838, online learning is inert.
- *Explanation:* the pooled in-distribution objective cannot reward adaptation, because the
  benefit appears only under shift, while per-video estimates add noise. The tuning therefore
  chose the strongest prior available.

**O2.** S2 collapse with UNK L* = 0.
- *Explanation:* in the calibration conditions, an UNKNOWN or no-record run almost always comes
  from a key that stopped being observed after an identity swap, and it never returns. The prior
  therefore learned "UNKNOWN means terminated". Under S2, UNKNOWN flags mark gaps.
- The same assumption gives K1 the **lowest** C6 ghost time (2.19 s vs 8.02 for BUF): one flag
  stands for two phenomena.

**O3.** C2, C3 and S0 harms.
- *Explanation:* any longer OCC hold trades gap recall against false persistence after hidden
  terminations. This is the information limit found in Phase 4 and the episode study. K1's C1 gain
  is largely a move along that frontier.

## 7. Representative failures (predefined rule, `R/failures/`)

| Condition | Seeds with positive regret | Worst / median seed |
|---|---|---|
| C1 | 0/50 | No failure observed |
| S2 | 50/50 | worst 8003 (regret 0.246), median 8011 (0.152) |
| C6 | 20/50 | worst 8037 (0.020), median 8020 (≈ 0) |

In S2, K1 drops gap episodes at the first UNKNOWN-flagged frame. In C6, the regrets are small.

## 8. Post-hoc, exploratory (after STOP; development seeds; never decisive)

`scripts/next_method/posthoc_kappa_dev.py`, `R/posthoc/posthoc.md`. At the frozen τ, lowering
κ (838 → 2):
- raises S2 J from 0.573 to 0.826 (gap recall 0.06 → 0.73);
- leaves C1 (0.975) and C2 (≈ 0.921) unchanged;
- raises C6 ghost time from 2.6 to 5.8 s;
- **stays below BUF in S2** (0.885).

So the online mechanism can adapt when the prior is weak. But even at its best it is slower than
a tuned fixed buffer, and it learns to hold ghosts. This explains the registered result; it does
not reverse it. **No variant was re-evaluated on the evaluation seeds.**

## 9. Mathematical claims actually established

- **P1** (structural): q is non-increasing within a run. Tested.
- **P2** (equivalence): the static rule equals a per-flag buffer with L* = max{l : q(l) ≥ τ}.
  Tested. It explains why K1-static and BUF differ only by the selected hold lengths.
- **P3** (optimality): under the mixture-cure model with S_G(H) = 0, independent runs, an additive
  per-bin loss and known hazards, the posterior-threshold hold minimises the expected per-run
  frame cost among all non-anticipating policies. The proof is in formulation §9; it is verified
  by brute force on 200 random models (`tests/unit/test_cure.py`).
  - This is a standard optimal-stopping argument, **not claimed as new**.
  - Its assumptions fail in the regimes that mattered: recurrence, flag mixing, J's episode term,
    and the UNKNOWN/no-record conflation.

**Not established:** any guarantee on J; frame-rate invariance; robustness to non-stationarity.

## 10. Gate (criterion by criterion; `R/aggregated/analysis.json`)

| Gate | Requirement | Observed | Met |
|---|---|---|---|
| G1 | K1 wins vs B2 in C1 **and** S2 | C1 win; S2 loss (−0.156) | No |
| G2 | 0 losses vs B2 | 4 (C2, C3, S0, S2) | No |
| G3 | ≥ 2 ADAPT wins vs BUF, 0 losses | 1 win (C1), 2 losses (S2, C6) | No |
| G4 | ≥ 1 ADAPT win vs K1-static, 0 losses | 0 wins | No |
| G5 | ≥ 1 ADAPT win vs GEO, 0 losses | 0 wins | No |
| G6 | Half-sample: G1, plus ≥ 2 ADAPT wins vs BUF | Identical selection, so same as G1/G3 | No |
| G7 | Latency ≤ 3× B2 | 3.71× | No |
| STOP rule | No PRIMARY win, **or** ≥ 2 losses vs B2 | 4 losses | **triggered** |

# **Decision: STOP. Algorithm contribution: UNSUPPORTED.**

A statistically clear gain (C1 +0.129) exists, but it is explained by an established fixed rule
(BUF, formulation P2). It is paid for by regressions elsewhere.

## 11. Three critical reviewer assessments

**Reviewer A (tracking / VidSGG).**
- *Strongest:* the study compared against the actual tracker mechanism (a coasting buffer) and
  found it equivalent or better.
- *Weakest:* the identity-swap world makes "UNKNOWN" mean "no record after a swap". Real trackers
  separate "lost track" from "unknown visibility", so the S2 failure could be partly an artefact of
  the repository's input convention.
- *Missing:* real tracker states (tracked / lost / removed) as distinct inputs.
- *Convinced?* By the negative result, yes.

**Reviewer B (statistics / sequential analysis).**
- *Strongest:* a clean cure-model posterior; P3 verified numerically; ablations designed to
  isolate adaptation.
- *Weakest:* per-video online estimation with tens of runs cannot beat a prior tuned on 140
  videos, unless the deployment shift is large *and* the estimate pools across videos. The study
  reset the statistics per video by design.
- *Missing:* cross-video (session-level) adaptation; hierarchical shrinkage with a learned κ;
  explicit detection of regime change.
- *Convinced?* No algorithmic contribution here.

**Reviewer C (methodology).**
- *Strongest:*
  - the audit used only development data;
  - the pilot was declared before registration, together with the expectation of failure;
  - the freeze was committed before evaluation;
  - completeness was verified;
  - all gates were reported;
  - the post-hoc analysis was confined to development seeds.
- *Weakest:*
  - pooled in-distribution tuning structurally disfavours adaptive components (O1). A
    transfer-aware selection objective would have been a fairer test of adaptation; that is a
    design choice for a *new* pre-registration, not a reinterpretation;
  - the per-frame vs run-stratum detail (§1) is documented but untested.
- *Convinced by STOP?* Yes. Four independent regressions vs B2, no attribution, and a failed cost
  gate.

## 12. Exact remaining requirements for an algorithm paper

There is currently **no supported algorithm contribution**:
- Phase 4: NO.
- Episode study: STOP.
- This study: STOP / UNSUPPORTED.

A credible algorithm paper would need all of the following.

1. **A mechanism that beats B2 and its tuned established analogue** (buffer, adaptive detection
   profile, HSMM) with **no regressions**, attributed by ablation, under a **pre-registered**
   protocol. No candidate so far meets this.
2. **The most promising untested direction from the evidence**, which is a representation change
   rather than a new estimator: distinguish "no record / track lost" from "visibility UNKNOWN",
   and use real tracker states. This must be a **new research decision and pre-registration**. It
   is not started.
3. **Real-data validation before any effectiveness claim.** Outputs from one real VidSGG model
   (e.g. STTran or TEMPURA on Action Genome, or a VidVRD-style model on VidOR) with a real tracker.
   Standard task accuracy (Recall@K / mR@K) must be reported alongside persistence, false
   persistence, FRR, recurrence delay, ghosts and latency.
4. **A second, sufficiently different base model** before any model-agnostic claim.
5. **Strong published comparators under matched information and tuning:** B2 + buffer, HSMM, and
   the base models' own temporal modules.
6. **Compute for steps 3–4 (estimate, not approved).** Inference of 2 pretrained models on one
   dataset split (roughly a GPU-day each), plus CPU post-processing comparable to this study
   (≈ 1–2 CPU-hours per configuration sweep). Dataset download and licensing checks are needed.
   **Not started.**

**A realistic alternative outcome:** an evaluation/methodology paper. Contents would include the
relapse-aware episode metrics, the information-limit findings, B2 (+ buffer) as the practical
recommendation, and the negative results of three method attempts.

## 13. Figures (`R/plots/`)

1. ΔJ of K1 vs each comparator;
2. gap recall and FPT;
3. ablation and latency;
4. hold horizons (static vs online) per condition;
5. decision matrix.

## 14. Tests

**316 passed, 0 failed** (`python -m pytest -q`) at checkpoint 2, and re-run before the final
commit (see the research log). Of these, 14 are new: 11 for `test_cure`, 2 decision tests and
1 pipeline test.

---

**STOP.** The focused study is complete. Not started: dataset downloads, detector or model
integration, heap optimisation, or another research phase.
