# Stage 3 — Diagnosis, Candidate Comparison, and Selection

## 1. Diagnosing the ID-switch failure (two falsification probes)

Condition: synthetic occlusion + flicker + 2 ID switches/video (5 seeds, 20/10/20 videos, val-selected threshold). All runs from the frozen config of the Stage 1 baseline.

| Method | Frame F1 | Pers. F1 | Delay | False pers. | Churn | ECE | Selected τ | Run |
|---|---|---|---|---|---|---|---|---|
| Vis-Gated EMA | 0.891 | 0.878 | 3.74 | 0.061 | 0.050 | 0.192 | 0.20 | `20261028_044818_696b2840` |
| V-HPED hand-set | 0.909 | 0.876 | 3.74 | 0.060 | 0.029 | 0.028 | 0.56 | 〃 |
| V-HPED full fit (baseline) | 0.875 | 0.895 | 2.12 | 0.038 | 0.059 | 0.211 | 0.90 | 〃 |
| **H-ID:** fit with `π₀` pinned 0.5 | 0.882 | 0.894 | 2.11 | 0.025 | 0.067 | 0.199 | 0.90 | 〃 |
| **H-ID2:** fit with `ρ_unk` pinned 0.5 | **0.919** | **0.920** | 3.42 | 0.065 | 0.030 | 0.051 | 0.63 | `20261028_045023_e10fb99f` |
| fit with `ρ_unk` pinned 1.0 | 0.914 | 0.899 | 3.42 | 0.064 | 0.034 | 0.053 | 0.67 | 〃 |

* **H-ID (over-confident births) is largely falsified**: +0.007 F1, ECE unchanged, τ still at the grid edge.
* **Error localisation** (seeds 42, 123; test): fitted V-HPED's extra errors are mostly *outside* swap windows (FP_other 2444 vs 1847 hand-set; ECE outside swaps 0.21 vs 0.02).
* **H-ID2 is supported**: pinning `ρ_unk` recovers frame F1 (0.919, above hand-set and gated EMA) and calibration (ECE 0.051).

**Mechanism.** An ID switch between objects A and B (labels 100 ↔ 101) moves the records
of the real pair (1,100) to a *phantom* pair (1,101) and back. During the swap the real pair
has no records (UNK) while still alive; after the swap the phantom pair has no records while
never having existed. One global `ρ_unk` must explain both: maximum likelihood drives it to
≈0 (protecting the real pair), which censors the phantom pair's absence forever, so phantom
edges persist and calibration collapses globally. **Identity-induced absence is confounded
with visibility-induced absence.** Pinning `ρ_unk` is a diagnostic, not a method: it simply
gives up censoring for missing pairs.

Note on encoding: with the frozen adapter contract (absence records for present tracks), the
same switch would instead produce *visible* absences on the real pair (spurious termination
evidence). The confounding then moves from `ρ_unk` to `ρ_vis`-driven termination. Both
encodings must be handled.

## 2. Candidate directions

| Criterion | A: identity-aware updates | B: learnable observation reliability | C: uncertainty-aware termination policy | D: stronger lazy theory |
|---|---|---|---|---|
| Research gap (Stage 2) | **G1** — no post-hoc relationship filter separating identity from visibility found | Overlaps occupancy-model detection covariates (established) | Filter is already Bayes-optimal for its model; a new rule is a decision-theory variant | **G3**, modest |
| Evidence of need | **Measured failure + diagnosed confounding** (§1) | The confounding in §1 *is* a reliability-identifiability problem; but more free reliability parameters would re-create it | Churn issue real but small (0.038 vs 0.040 gated EMA in A) | Lazy slower at f=1; conditions only sufficient so far |
| Mathematical depth | Joint (identity, alive) filter; exact one-step posterior; contamination analysis | Identifiability analysis | Bayes risk / hysteresis | Necessity counterexamples, extended exactness |
| Feasibility | Moderate: needs an inference-time identity-continuity score | Moderate | Easy | Easy |
| Testability | Swap frequency × score informativeness sweep, incl. uninformative score | Parameter recovery | Threshold trade-offs | Equivalence harness, adversarial workloads |
| Risk | Requires tracker confidence; synthetic score must not leak GT | Unidentifiable without labels | Arbitrary thresholds | Low |

## 3. Decision

* **Primary: Direction A — identity-aware V-HPED (IA-VHPED).** It targets the only measured
  failure, has a clear gap (G1), a falsifiable mechanism (§1), and a principled model:
  probabilistic-data-association-style evidence weighting (established for object state;
  Bar-Shalom & Tse 1975) applied to relationship existence, with a joint one-step posterior
  over (identity-consistent, alive). It reduces exactly to V-HPED when identity confidence is 1.
* **Supporting: Direction D (restricted)** — (i) prove the identity extension preserves the
  lazy-exactness guarantee, (ii) give counterexamples showing L1 and L2 are not merely
  convenient (exact elapsed-time closed forms fail without them), (iii) add memory
  measurement and an adversarial workload. No change to the existing lazy algorithm.
* **Not selected:** B (adding free reliability parameters reproduces the confounding found in
  §1; parameter-recovery experiments are still run as an identifiability check), C (no
  defensible gap; the forward filter already gives the Bayes posterior).

## 4. Falsifiable hypotheses for the selected extension

* **HA1:** With an informative identity score, IA-VHPED (fitted) has higher frame F1 and lower
  ECE than fitted V-HPED under ID switches. *Falsifier:* paired CI of frame F1 difference
  includes 0 or is negative at 2 and 4 swaps/video.
* **HA2 (no harm):** Without ID switches, IA-VHPED matches V-HPED (|ΔF1| CI within ±0.005).
* **HA3 (honest dependence):** With an *uninformative* identity score, IA-VHPED gives no
  significant gain over V-HPED. If it does gain, the gain is not due to identity information
  and must be explained.
* **HA4 (contamination):** IA-VHPED's fitted `ρ_unk` under ID switches stays far from 0
  (> 0.1) and its parameters move less between the no-switch and switch conditions than
  V-HPED's.
