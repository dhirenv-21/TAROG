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

Amendments A1–A3 were written on 2026-10-08, **before any registered fit**, i.e. before any
SELECT or EVAL quantity was computed. Only the 300-video TRAIN smoke run and a 1000-video TRAIN
benchmark (both development data) had been run. None of them changes a model family, the search
budget, a null, a metric or a decision criterion.

### A1 — Execution infrastructure (technical)
- **Exact row aggregation.** The target log-likelihood Σ_i [y_i z_i − d_i e^{z_i}] depends on the
  rows only through (Σy, Σd) over rows with an identical covariate vector. Rows with identical
  (group, duration bin, covariates) are merged before fitting. The objective, gradient and MLE are
  unchanged up to floating-point summation order. Groups are TRAIN folds (so fold fits remain
  possible) or EVAL videos (so per-video held-out log-likelihoods remain exact). Verified on 1000
  TRAIN videos: max |Δw| = 1e-10 and |ΔLL| = 2e-10 against unaggregated fits
  (`tests/unit/test_phase5h1_run.py`, benchmark JSON).
- **Chunked design construction** (400 videos per chunk, then exact merge) to bound memory.
- **Bounded parallel execution** (`src/taorg/hawkes/runner.py`). Each job runs in one fresh
  spawned process at below-normal priority. Each task (one fit) has an id equal to the hash of
  its full spec (data set, target, rung, ridge, fold, seeds via the data-set name, config hash,
  layout hash, code hash). Checkpoints are written atomically and reused only on an identical
  spec. Every attempt is logged append-only; failed jobs are recorded and their dependents
  blocked. Completeness is checked against the expected task matrix.
- **Serial/parallel agreement** was verified on synthetic data (identical task ids; parameters
  within 1e-6; identical ridge selection). Scheduling cannot change seeds, tuning or splits,
  because none of them depend on execution order.
- Heavy run state lives outside the repository (`D:/TAORG_runs/phase5h1/main`; the C: drive had
  6.8 GB free). Compact artifacts are copied to `results/phase5h1/full/`.
- **Structural leakage guard.** EVAL-A/EVAL-B records are created only by `evaluate_prepare`,
  which refuses to run unless `results/phase5h1/full/frozen/{frozen.json, params.npz}` are
  committed and unchanged. Workers refuse to read EVAL records without an unlock token tied to
  the committed freeze hash.
- **Optimiser diagnostics** are stored per fit: convergence flag, iterations, projected-gradient
  norm, objective, and the number of evaluations in which the linear predictor exceeded the
  overflow cap of 50.

### A2 — Null construction details (not specified at registration)
- **N-jit** (`src/taorg/hawkes/nulls1.py`): offset d ~ U[−5, 5] s per (pair, predicate),
  rounded to frames; the pair window is unchanged and intervals are clipped to it.
  **Censored endpoints (begin frame 0, end frame T) are pinned, not shifted.** Shifting them
  would manufacture onset/offset events that the annotation does not contain, which violates the
  registered event rules (§3). Intervals that leave the window are dropped, and pinned endpoints
  can merge intervals; both are counted per replicate. Randomness is drawn per video from
  `Random(f"{seed}|{part}|{video_id}")`, so it is order- and chunk-independent. TRAIN and EVAL
  use the part labels "TRAIN" / "EVAL" under the same registered seed.
- **This is a shift-jitter variant, not the interval-jitter test of Amarasingham et al. (2012).**
  No exactness property of interval jitter is claimed for it.
- **N-circ** is the registered Stage 0 function, applied unchanged; EVAL uses seed + 100000, as in
  Stage 0. **Disclosed property:** it shifts left-censored intervals too, so it can create onset
  events at the shift point, and it re-derives the pair window as a hull, which enlarges some
  risk sets. Event counts of every null replicate are recorded and reported. N-circ is not
  modified, because it is registered.
- EVAL for nulls = EVAL-A followed by EVAL-B (each sorted by video id). Null Δ = per-event
  difference on the null EVAL data with its own event count.
- Hypothesis null (§8): replicate i ∈ {0..198} uses the N-jit transform with seed string
  "6600:i" on EVAL.

### A3 — Analysis details left open at registration (fixed before any result)
- **Bootstrap (§5):** the per-event Δ is the ratio Σ_v ΔLL_v / Σ_v n_v, and the percentile CI
  comes from 2000 video resamples. One index matrix is drawn per video set (EVAL, EVAL-A,
  EVAL-B) from `numpy.random.default_rng(6500)` and shared by all comparisons on that set
  (paired).
- **Families:** a target belongs to the family of its predicate (`families.py`). Δ_F is computed
  over F's target types and normalised by F's EVAL events. Δ_F(G2, G1) is the point estimate on
  EVAL. null_max_F and null_med_F are over the 38 replicates.
- **K4:** for every HIST cell of G4 (target, other-predicate source, lag) a discovery requires
  5/5 strictly agreeing signs and |t| ≥ τ. Candidate τ values are the |t| of real sign-agreeing
  cells, scanned in ascending order; τ is the first value with mean N-jit fold-null discoveries /
  real discoveries ≤ 0.10. The inter-fold Spearman is the mean over the 10 fold pairs, computed
  over all G4 HIST cells; it is compared with each of the 5 fold-null replicates. Support follows
  the registered definition (TRAIN source events with the target at risk just after the event).
  At-risk exposure and target events with the cell active are reported as well.
- **Hypotheses (§8):** two-sided permutation p = (1 + #{|log RR_null| ≥ |log RR_obs|}) / 200,
  with Holm correction over the four hypotheses. The CI is a video bootstrap from
  default_rng(6500) on EVAL. A hypothesis is underpowered if fewer than 20 target offsets fall in
  the window; its p-value is then reported but not interpreted.
- **Next-event metric (§5):** the competing set is the modelled target types at risk on the same
  pair at the event time; p = λ_k / Σ λ. Log-loss and top-1 accuracy per event; descriptive only.
- The decision is computed by `decide_phase5h1` exactly as registered. These details do not
  create an alternative rule.

