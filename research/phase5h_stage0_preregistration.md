# Phase 5H Stage 0 — Ground-truth Hawkes signal probe (VidOR) — Pre-registration

> **Status:** REGISTERED before any model is fitted to VidOR data, on any split. Committed
> together with `configs/experiments/phase5h_stage0.yaml` and the decision code
> `src/taorg/hawkes/decision.py` (with tests). Design context: `research/phase5h_hawkes_design.md`
> (gate = MODIFY, modifications M-1…M-7). Output: exactly one of **HAWKES SIGNAL = STRONG /
> WEAK-MIXED / ABSENT**, then STOP.

## 0. What was done before registration (disclosure)

- The VidOR **annotation** zips were downloaded (training 150,728,902 B, MD5 `85f39fdd…219a2`;
  validation 17,212,872 B, MD5 `96c68709…1c0cb5`; both match the published values) from the
  HuggingFace mirror `shangxd/vidor` (card licence: CC-BY-SA-4.0). They are stored in `data/vidor/`,
  which is gitignored and not redistributed. **No video file was downloaded.**
- **Audit: video files are not technically required.** Every annotation JSON contains `fps`,
  `frame_count`, per-frame trajectories and `relation_instances` with `begin_fid` (inclusive) and
  `end_fid` (exclusive). Stage 0 uses annotations only.
- Structural audit of the **training** split only (no model fitted): 7000 videos; 267,210
  relation instances; 50 predicates; 59,390 left-censored (`begin_fid = 0`) and 31,310
  right-censored (`end_fid ≥ frame_count`) instances; no overlapping or touching intervals of the
  same (pair, predicate); median video length 28.6 s; fps mostly 29.97. The validation split was
  only counted (835 files). Its contents were not inspected.

## 1. Question

**RQ-H′ (Stage 0).** In VidOR *ground-truth* annotations, do relationship onset/offset events on
an ordered object pair depend on the recent history of **other** relationships on the same pair
(cross-relation dependence)? The test asks whether that dependence exceeds what within-relation
dynamics already explain, and whether it survives a null that preserves each relation's own
temporal structure while destroying cross-relation alignment.

Stage 0 involves no visibility, no identity handling and no detector.

## 2. Data, splits and seeds

- **Training** split (7000 videos): video ids sorted, then shuffled with seed 5101.
  - **DEV**: the first 1400 videos, used for hyperparameter selection only.
  - **FIT**: the remaining 5600, used for fitting.
  - **FIT folds**: FIT split into 5 disjoint folds (shuffle seed 5102), used for stability only.
- **TEST**: the official VidOR **validation** split (835 videos). It is untouched until the final
  evaluation, and is evaluated once per model with hyperparameters frozen on DEV.
- Seeds: null shifts 5200–5218 (C1) and 5300–5304 (C4); bootstrap 5400.

## 3. Event extraction (frozen rules)

- **Time:** t = frame index / fps (seconds).
- **Pair:** an ordered pair (u, v), u ≠ v, of track ids in a video. The history scope is the same
  ordered pair only; reverse and neighbouring pairs are excluded in Stage 0.
- **Pair window:** W_p = [w₀, w₁], the hull of (i) the co-existence span of the two tracks (each
  track's first to last annotated frame) and (ii) all relation intervals of the pair. Pairs whose
  tracks never co-exist and that have no relation are dropped.
- **Types:** for each of the 50 predicates r, ONSET(r) and OFFSET(r), i.e. K = 100 modelled event
  types. There is one extra exogenous covariate event, **PAIR_BIRTH**, at w₀ if w₀ > 0. It can
  excite relation events in HK1 and HK2 (it controls for the common cause "relations start
  because the pair appeared"), but it is not itself modelled.
- **Relation interval** [b, e) from `begin_fid` / `end_fid`:
  - ONSET(r) at b / fps iff b > 0; otherwise left-censored (on at the start, no onset event).
  - OFFSET(r) at e / fps iff e < frame_count; otherwise right-censored (no offset event).
- **Risk sets:** ONSET(r) is at risk on W_p minus the on-intervals of (p, r); OFFSET(r) is at risk
  on the on-intervals of (p, r).
- **Ties:** history is strictly earlier (t_j < t). Events at the same timestamp are not in each
  other's history.

## 4. Models (all fitted by penalised maximum likelihood on FIT)

For modelled type k on pair p: λ_{p,k}(t) = R_{p,k}(t) · (μ_k + Σ_{k′} α_{k,k′} Σ_{j: t_j<t, type k′,
pair p} e^{−β(t − t_j)}), with μ_k > 0, α ≥ 0 (linear Hawkes; no inhibition in Stage 0) and a single
shared β.

| Model | α support | Role |
|---|---|---|
| **HK0** | none (μ only) | base rate: Poisson onsets, exponential durations |
| **DUR** | none; OFFSET(r) hazard piecewise-constant in time since onset (bins 0, 0.5, 1, 2, 4, 8, 16, ∞ s); left-censored intervals use a separate "unknown-age" rate; ONSET as in HK0 | independent duration/hazard control, no history |
| **HK1** | within-relation history: α_{k,k′} only for k′ ∈ {ONSET(r), OFFSET(r)} of the **same** predicate, plus PAIR_BIRTH | self/within-relation Hawkes; the mandatory "exponential forgetting" control with the same kernel |
| **HK2** | all 100 types + PAIR_BIRTH | **genuine multivariate exponential Hawkes** |

- The log-likelihood decouples by type k. Each row is a concave problem in (μ_k, α_k) and is
  solved with L-BFGS-B under bounds μ_k ≥ 1e-10, α ≥ 0.
- Penalty: λ_pen·‖α_k‖²₂.
- Hyperparameters, selected **separately for HK1 and HK2** by total DEV log-likelihood of models
  fitted on FIT: β ∈ {0.1, 0.3, 1, 3, 10} s⁻¹ and λ_pen ∈ {0, 0.1, 1, 10}.
- HK0 and DUR have no hyperparameters.
- **Reference implementation:** an eager O(N²) likelihood must match the O(N) recursive/closed-form
  one (unit tests), and a simulation-recovery test must pass, before the VidOR run.

## 5. Primary quantity

Held-out **log-likelihood per event** on TEST: LL = Σ_events log λ(t_e) − Σ ∫ λ over risk sets,
divided by the number of TEST events (identical for all models). It is reported in nats/event;
exp(ΔLL) is the per-event predictive-density ratio. The 95% CI comes from a percentile bootstrap
over TEST **videos** (2000 resamples, seed 5400).

Derived quantities:
- G_cross = ΔLL(HK2 − HK1)
- G_self = ΔLL(HK1 − HK0)
- G_dur = ΔLL(HK2 − DUR)
- G_base = ΔLL(HK2 − HK0)

## 6. Null control (destroys cross-relation alignment, keeps within-relation structure)

**Circular time shift.** For each (video, pair, predicate), all intervals of that predicate on
that pair are shifted by one common random offset δ ~ U[0, T) modulo the video duration T. An
interval that wraps is split into two. Censoring and events are then re-derived by §3 (a piece
touching 0 or T becomes censored). Counts, durations and inter-interval gaps of each (pair,
predicate) are preserved up to wrapping. The relative timing between different predicates on a
pair, and the timing relative to PAIR_BIRTH, are destroyed.

- **C1 null:** R = 19 shifted copies of FIT+TEST (seeds 5200–5218). HK1 and HK2 are refitted on
  shifted FIT **at the real-data DEV-selected hyperparameters** and evaluated on shifted TEST,
  giving G_cross^null,i.
- **C4 null:** 5 shifted copies of FIT (seeds 5300–5304), each fitted per fold as in §7.

## 7. Cross-split interaction stability

HK2 is fitted separately on each of the 5 FIT folds, at the DEV-selected hyperparameters.

- **Cross cells:** (k, k′) with k′ a different predicate's ONSET/OFFSET. Same-predicate cells and
  PAIR_BIRTH are excluded; 100 × 98 cells in total.
- A cell is **present** in a fold if its branching ratio α_{k,k′}/β ≥ 0.01, i.e. at least one
  expected triggered event per 100 parent events. This threshold is a detectability convention
  and is calibrated by the null, which uses the same threshold (§8, C4).
- **S** = number of cross cells present in ≥ 4 of the 5 folds.
- **ρ̄** = mean pairwise Spearman correlation of the cross-cell α vectors across the 10 fold
  pairs.

## 8. Decision rule (exact; `src/taorg/hawkes/decision.py`)

| Criterion | Definition |
|---|---|
| **C1 statistical signal** | CI_lower(G_cross) > 0 **and** G_cross > max_i G_cross^null,i (permutation p ≤ 1/20 = 0.05) |
| **C2 beats independent controls** | CI_lower(G_dur) > 0 **and** CI_lower(G_base) > 0 |
| **C3 practically non-negligible** | E ≥ 0.10 · G_self, with excess E = G_cross − median_i G_cross^null,i. If G_self ≤ 0: E > 0 |
| **C4 stable interactions** | S_real > max_j S_null,j **and** ρ̄_real > max_j ρ̄_null,j |

| Label | Rule |
|---|---|
| **HAWKES SIGNAL = STRONG** | C1 ∧ C2 ∧ C3 ∧ C4 |
| **HAWKES SIGNAL = ABSENT** | ¬C1 |
| **HAWKES SIGNAL = WEAK-MIXED** | C1, but at least one of C2, C3, C4 fails |

**Justification of the thresholds (no absolute nats threshold is used):**
- C1 is calibrated entirely by the data: the bootstrap and a permutation null that keeps each
  relation's own dynamics.
- C3 is scale-free. It asks whether the dependence on *other* relations is at least one tenth as
  large, in held-out likelihood, as the dependence on the relation's *own* history. An absolute
  nats/event threshold (the 0.05 in the design draft) would depend on event density and
  censoring, so it is **not used**. The 10% ratio is a judgement, disclosed as such: below it,
  cross-relation structure would be a second-order refinement that does not justify a
  multivariate model in later stages. Ratios of 0.05 and 0.20 are reported **descriptively only**
  and cannot change the label.
- C4 replaces the draft's "≥ 5 effects" with a null-calibrated count and a rank correlation, so
  the number of effects that counts as stable is set by what shifted data produce.

## 9. Pre-registered descriptive analyses (training/FIT only, never decisive)

- Counts per type; interval-duration distributions per predicate; censoring fractions.
- Peri-event onset rates: for the 20 most frequent ordered predicate pairs (r′ → r) on the same
  pair, the empirical ONSET(r) rate in lag bins after OFFSET(r′)/ONSET(r′), vs the unconditional
  ONSET(r) rate.
- Learned HK2 A-matrix (FIT fit): heat map; top-20 cross cells by branching ratio, with fold
  presence counts.

## 10. Declared limitations and threats

1. **Annotation artefacts.** VidOR boundaries come from human (tool-assisted) segmentation;
   correlated boundaries may reflect annotation practice rather than scene dynamics. A STRONG
   signal on ground truth is not evidence of perceptual usefulness.
2. **Predicate imbalance.** Spatial predicates (`next_to`, `in_front_of`, `behind`, `away`,
   `towards`) dominate; any signal may be concentrated among spatial predicates.
3. **Limited history scope.** Only the same ordered pair; no subject-level or scene-level
   excitation.
4. **Model restrictions.** One shared β; no inhibition; linear link.
5. **Track spans.** First-to-last-frame spans ignore within-track gaps.
6. **Null wrap-around.** The circular shift creates artificial censored pieces at the video
   boundary; this affects HK1 and HK2 equally.
7. **Multiplicity.** One registered primary label; everything else is descriptive.

## 11. Amendments

None at registration.

- **A1 — 2026-09-30, after the TEST evaluation had run once, before any null or stability result
  existed (engineering only; no protocol change).** The first full run crashed after DEV selection
  and the TEST evaluation. The config key `null:` is parsed by YAML as a null key, so
  `cfg["null"]` raised KeyError at the C1 null stage. Fix: the key was renamed to `null_control`
  (same seeds 5200–5218 and 5300–5304) in the config and the pipeline. Nothing else changed.
  - The TEST numbers printed by the crashed run were seen before the null control. The decision
    rule (§8) was frozen at registration and is not altered.
  - The rerun is fully deterministic, so the TEST numbers must reproduce exactly; this is
    verified in the Stage 0 report. The crashed log is kept as
    `results/phase5h/stage0/run_attempt1_crashed.log`.
  - Disclosure: after registration and before the first run, a timing check fitted HK2 on a
    10% subset of *training* videos (no likelihood evaluated or inspected).
