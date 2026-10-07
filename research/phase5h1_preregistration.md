# Phase 5H-1 — Heterogeneity-controlled temporal dependence in VidOR — Pre-registration

> **Status:** REGISTERED before any Phase 5H-1 model comparison. Committed with
> `configs/experiments/phase5h1.yaml` and the decision code `src/taorg/hawkes/decision1.py` (with
> tests). Literature basis: `research/phase5h1_literature.md`. Fitting-split audit:
> `results/phase5h1/audit/`. Stage 0 (`research/phase5h_stage0_report.md`, HAWKES SIGNAL =
> WEAK-MIXED) is frozen and is not reinterpreted here.
>
> **Evidence class of every Phase 5H-1 comparison: EXPLORATORY** (see §2). No untouched
> annotated VidOR data exist.

## 1. Research questions

- **RQ-H1:** After controlling for (i) event base rates, (ii) pair-level interaction propensity
  (heterogeneity), (iii) relation duration dependence (semi-Markov sojourn hazards) and (iv) the
  current relation configuration (state-dependent transitions), does the *recent history of
  other relations' onsets/offsets on the same pair* improve held-out prediction of relation
  onsets/offsets?
- **RQ-H2:** Are the remaining effects excitatory or inhibitory? Which of the four classes
  ONSET→ONSET, ONSET→OFFSET, OFFSET→ONSET, OFFSET→OFFSET carries them? At which lag scale
  (short ≤ 0.5 s, medium 0.5–2 s, long 2–10 s)?
- **RQ-H3:** Are the effects reproducible across video-level folds and distinguishable from
  matched nulls?

## 2. Data-use audit and split policy

| Data | Prior use (Stage 0) | Inspected? | Phase 5H-1 role |
|---|---|---|---|
| Stage 0 FIT (5600 training videos) | model fitting; generation of the post-hoc transition hypotheses; Phase 5H-1 audit | yes, fully | **TRAIN** (fitting, folds, smoke development) |
| Stage 0 DEV (1400 training videos) | hyperparameter selection by *total* log-likelihood of Stage 0 models | aggregate only | split in two by seed 6101 → **SELECT** (700, λ selection) and **EVAL-A** (700) |
| Stage 0 TEST = VidOR validation (835) | held-out *aggregate* log-likelihood of Stage 0 models | aggregate only | **EVAL-B** |
| VidOR test split | annotations not public | — | unavailable |

**Consequences:**
- Model comparisons are evaluated on EVAL = EVAL-A ∪ EVAL-B (1535 videos). These videos were not
  used for fitting or for generating cell-level hypotheses, but they *were* used in aggregate in
  Stage 0. **All Phase 5H-1 results are therefore labelled EXPLORATORY.**
- The four Stage-0-generated transition hypotheses (§8) are tested on EVAL, i.e. on data not used
  to generate them. This is an **out-of-sample exploratory replication**, not a confirmatory test.
- Confirmatory evidence would require an independent annotated dataset with relation start/end
  times (candidate: ImageNet-VidVRD or PVSG) under a new pre-registration. That is not part of
  Phase 5H-1.
- **Smoke/development data:** DEV-SMOKE = 300 TRAIN videos (seed 6103; 200 fit / 100 check). It is
  used only to test the pipeline. Smoke outputs are never reported as evidence.
- Leakage guard: SELECT/EVAL videos are loaded only by the registered evaluation entry point. The
  smoke entry point refuses to read them (tested).

## 3. Event representation (unchanged from Stage 0, §3 of its pre-registration)

ONSET(r) and OFFSET(r) per ordered pair; strict-history ties; same censoring rules; time in
seconds. PAIR_BIRTH is replaced by the elapsed-time heterogeneity bins (§4).

## 4. Model family and ladder

**Family:** a point-process GLM / multiplicative-intensity relational event model (Truccolo et al.
2005; Perry & Wolfe 2013) with an exp link and **piecewise-constant** covariates:

  λ_{p,k}(t) = R_{p,k}(t) · exp(η_k + w_k · x_{p,k}(t)),

where R is the risk-set indicator. The intensity is positive for any sign of w (valid inhibition),
and the compensator is an exact sum over constant segments. Each target type k is an independent
Poisson regression on (segment, exposure, count) with a ridge penalty λ‖w‖² (intercept η_k
unpenalised).

- **Targets:** event types with ≥ 200 events in TRAIN (audit: 34 ONSET, 35 OFFSET types).
- **Sources** for history covariates: types with ≥ 50 TRAIN events.

| Covariate block | Content | Role |
|---|---|---|
| HET | subject-category and object-category one-hot (ridge ⇒ Gaussian random intercepts, MAP); elapsed time since the pair appeared {≤2, 2–10, >10 s}; events so far on the pair {0, 1, 2, 3–4, 5–8, ≥9}; relations currently on {0, 1, 2, ≥3} | pair-level propensity / frailty-posterior proxies (online, causal) |
| DUR | own sojourn time. ONSET(r): time since r's last offset {≤1, 1–4, 4–16, >16 s, never-on}. OFFSET(r): age since onset {≤1, 1–4, 4–16, >16 s, unknown (left-censored)} | semi-Markov duration dependence (**simple duration control**) |
| STATE | indicator "r′ currently on" for every other predicate r′ | state-dependent transition model (**simple transition control**) |
| HIST | indicator "an event of source type c (other predicate) occurred in lag window m before t", with m ∈ {(0, 0.5], (0.5, 2], (2, 10]} s | cross-predicate event history (the Hawkes-type term) |

| Rung | Blocks | Answers |
|---|---|---|
| G0 | intercept | base rates |
| G1 | + HET | heterogeneity |
| G2 | + HET + DUR | duration dependence (simple duration model) |
| G3 | + HET + DUR + STATE | simple transition model |
| G4x | + HET + DUR + HIST | cross-history beyond heterogeneity and duration (RQ-H1) |
| G4 | + HET + DUR + STATE + HIST (signed) | timing-specific history beyond the transition model |
| G4+ | as G4 with HIST weights ≥ 0 | positive-only vs excitation + inhibition |
| G3+HIST_class / G3+HIST_lag | G3 plus HIST restricted to one interaction class (4) or one lag window (3) | RQ-H2 (descriptive decomposition) |

- **Ridge λ:** chosen per rung (G1–G4+) from {0.3, 1, 3, 10} by total SELECT log-likelihood, with
  models fitted on TRAIN. Class/lag variants reuse G4's λ.
- **Not used (recorded):** a joint gamma-frailty marginal likelihood (a sensitivity analysis still
  owed); exponential-decay kernels.

## 5. Metrics and statistics

- **Primary:** held-out log-likelihood per EVAL event over modelled target types (nats/event).
  95% CI from a percentile bootstrap over EVAL videos (2000 resamples, seed 6500), computed on
  the paired per-video differences.
- **Secondary (descriptive):**
  - next-event-type log-loss and top-1 accuracy among at-risk modelled types at each EVAL event;
  - the same per-event ΔLL for EVAL-A and EVAL-B separately (consistency).

## 6. Nulls (R = 19 each; applied to TRAIN and EVAL; refit at the selected λ)

- **N-circ** (Stage 0): circular shift of each (pair, predicate) interval set mod T. Seeds 6200–6218.
- **N-jit** (new): rigid shift of each (pair, predicate) interval set by U[−5, +5] s, clipped to
  the pair window. It preserves slow (≳ 5 s) co-activity, counts, durations, pair identity and
  pair activity, and destroys sub-5-s relative timing. Seeds 6300–6318.
- Limitations (literature §3): the jitter cannot test effects slower than 5 s; clipping distorts
  edge intervals; shifting can create co-occurrences of normally exclusive predicates.

## 7. Reproducibility of interactions (RQ-H3)

- G4 is fitted on 5 disjoint TRAIN folds (seed 6102).
- For each HIST cell (target, source, lag), report support (TRAIN source events with the target at
  risk), the mean weight, the SD across folds, sign agreement and t = mean / (SD/√5).
- **Discovery:** 5/5 sign agreement and |t| ≥ τ. τ is the smallest value at which the estimated
  FDR = (mean discoveries over 5 N-jit fold-fit replicates, seeds 6400–6404) / (real discoveries)
  ≤ 0.10.
- Inter-fold Spearman correlation of HIST weights: real vs N-jit.

## 8. Stage-0-generated hypotheses (EXPLORATORY replication on EVAL)

H-a: ONSET(hold) → OFFSET(grab); H-b: ONSET(release) → OFFSET(hold); H-c: ONSET(get_off) →
OFFSET(ride); H-d: ONSET(beneath) → OFFSET(above).

- **Statistic (model-free):** target-offset hazard during the target's on-time within (0, 2] s
  after a source onset on the same pair, divided by the target-offset hazard during the remaining
  on-time.
- Inference: log rate ratio with a 95% video-bootstrap CI, and an N-jit permutation p-value with
  199 replicates (seed 6600). Holm correction across the four hypotheses at α = 0.05.
- **Underpowered** if fewer than 20 target-offset events fall in the post-source window on EVAL.
- Audit note: TRAIN support for H-c and H-d is only 15 and 11 events within 10 s, so both are
  expected to be underpowered.

## 9. Decision (exact; `src/taorg/hawkes/decision1.py`)

Let Δ(a, b) be the per-event EVAL log-likelihood difference with its bootstrap CI, and let
null_max / null_med be over **both** nulls combined (38 replicates).

| Criterion | Definition |
|---|---|
| **K1** cross-history beyond heterogeneity + duration | CI_lo Δ(G4x, G2) > 0 and Δ(G4x, G2) > null_max of Δ(G4x, G2) |
| **K2** beyond the simple transition model | CI_lo Δ(G4, G3) > 0 and Δ(G4, G3) > null_max of Δ(G4, G3) |
| **K3** practical size | Δ(G4, G3) − null_med ≥ 0.10 · Δ(G2, G1) (10% of the duration-dependence gain, same scale-free convention as Stage 0; a disclosed judgement) |
| **K4** reproducibility | ≥ 1 discovery at FDR ≤ 0.10 (§7) and inter-fold Spearman (real) > max over N-jit |
| **Family test (for NARROW)** | per family F ∈ {SPATIAL, MOTION, ATTENTION_COMMUNICATION, VEHICLE_SUPPORT, CONTACT_MANIPULATION}: Δ_F(G4, G3) over F's target events has CI_lo > 0 in **both** EVAL-A and EVAL-B, and exceeds null_max_F, and Δ_F − null_med_F ≥ 0.10 · Δ_F(G2, G1) |

| Label | Rule |
|---|---|
| **HAWKES TRACK = GO** | K1 ∧ K2 ∧ K3 ∧ K4 |
| **HAWKES TRACK = NARROW (MIXED)** | not GO, K1 holds, and ≥ 1 family passes the family test |
| **HAWKES TRACK = STOP (NO-GO)** | otherwise |

Even GO only licenses a confirmatory replication on independent data (§2). It is not a paper claim.

## 10. Seeds

| Use | Seed(s) |
|---|---|
| SELECT/EVAL-A split | 6101 |
| Folds | 6102 |
| DEV-SMOKE | 6103 |
| N-circ | 6200–6218 |
| N-jit | 6300–6318 |
| Fold nulls | 6400–6404 |
| Bootstrap | 6500 |
| Hypothesis null | 6600 |

## 11. Amendments

None at registration.
