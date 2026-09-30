# Phase 5H — Hawkes feasibility: design and novelty gate (DESIGN ONLY — nothing implemented)

> Track B of the paper strategy. Separate from Phase 5A (Track A). Evidence base:
> `research/phase5h_hawkes_literature.md`. The earlier "V-HPED" filter is **not** a Hawkes process
> (Phase 4 formulation §9, ledger L27) and is not renamed or reused as one here.

## 0. Gate decision

# HAWKES TRACK = **MODIFY**

**Not GO as posed**, for four reasons:

1. **The mathematics is not novel.** Every candidate ingredient already exists:
   - multivariate exponential Hawkes;
   - valid inhibition through a positive-part or softplus link;
   - inference with missing, censored, thinned or ghost events;
   - mark-dependent (weighted) excitation.

   The proposed "identity-weighted history" λ = μ + Σ w_j α e^{−β(t−t_j)} is exactly a marked
   Hawkes process with mark-dependent impact (ETAS-style ρ(m_j) = w_j).
2. **Action Genome cannot support it.** About 35 annotated frames per video (5 per action) means
   the ground-truth event stream is interval-censored by design.
3. **Persistence/robustness is the wrong target for Hawkes.** Phase 4 showed persistence under
   occlusion is near an information limit, which simple exponential filters already reach. A
   self-exciting model of one edge's own detections is exponential filtering in different words.
4. **The vision application gap is empirical, not theoretical.** Whether relationship events show
   cross-relation history dependence is simply unknown.

**Not NO-GO**, for three reasons:

1. No verified prior work applies a genuine Hawkes/TPP intensity to visual-relationship events in
   video. The closest works are TKG Hawkes models (GHNN, GHT, Know-Evolve) on text/news facts,
   action-level point processes (APP-VAE), and scene-graph forecasting with Neural ODE/SDE
   (SceneSayer, FDSG). This search was not exhaustive.
2. Datasets with explicit relation onset/offset times exist: VidOR, ImageNet-VidVRD, PVSG.
3. Whether cross-relation dependence exists can be falsified cheaply, on ground truth alone,
   before any perception pipeline is involved.

**Required modifications (M-1 … M-7):**

| # | Modification |
|---|---|
| M-1 | Events are relation **onsets and offsets** (state transitions), **not frame-level evidence** (§2) |
| M-2 | Primary question: **transition / next-event dynamics** (does r′'s history change r's onset/offset intensity?). Persistence under occlusion is secondary |
| M-3 | Data: **VidOR or PVSG** (dense temporal extents), not Action Genome |
| M-4 | Missing/censored/noisy observation handled with **existing** theory (Shelton et al. AAAI 2018; Mei et al. ICML 2019; thinning/censoring/ghost framework), cited, not claimed |
| M-5 | Inhibition only through a valid link (positive part, as in Bonnet et al. 2021/2023, or softplus) |
| M-6 | Identity reliability only as a **marked-Hawkes ablation** (mark-dependent impact), with standard terminology |
| M-7 | A **ground-truth-only Stage 0 probe**, separately pre-registered, gates everything else; if it fails, Track B stops |

## 1. Research question (refined)

**RQ-H′.** On densely annotated video relationship data, do relationship onset/offset events
exhibit cross-relation history dependence? That is, do multivariate Hawkes intensities give a
measurably better held-out model of *when* and *which* relationship transitions occur than
(a) independent per-relation duration/hazard models with the same exponential kernel family and
(b) self-excitation only? Only if yes: does censoring-aware inference driven by visibility recover
that dependence under occlusion, flicker and identity noise?

## 2. Event representation (M-1), justified

| Candidate | Verdict | Reason |
|---|---|---|
| RELATION_EVIDENCE (every frame a detection fires) | **rejected** as the primary event | At 30 fps an alive relation emits an event every frame, so "self-excitation" just encodes persistence. That is equivalent to per-edge exponential filtering, the regime Phase 4 exhausted; α_{r,r} > 0 would be an artefact of frame rate |
| RELATION_ONSET(u, v, r), RELATION_OFFSET(u, v, r) | **primary** | Sparse, well-defined transitions; directly available as start/end frames in VidOR/VidVRD/PVSG; cross-type excitation ONSET(r) ← OFFSET(r′) is a transition |
| RELATION_TRANSITION(r_i → r_j) | **derived**, not a separate type | Represented by α_{on r_j, off r_i} on the same pair; a separate type would double-count |
| OBJECT_INTERACTION_EVENT | deferred | Needs a definition independent of relation labels; revisit after Stage 0 |

**State constraint (risk set).** ONSET(p, r) is possible only while (p, r) is off, and OFFSET only
while it is on. As in relational event models (Butts 2008), the intensity is multiplied by a
risk-set indicator. This is standard and keeps the process well-defined.

## 3. Model family (to be literature-checked again at registration)

For pair p = (u, v) and event type k ∈ {on_r, off_r : r = 1..P}:

  λ_{p,k}(t) = R_{p,k}(t) · φ( μ_k + Σ_{k′} Σ_{j ∈ H_p(t), type k′} ρ(m_j) α_{k,k′} e^{−β_{k,k′}(t − t_j)} ),

- R_{p,k}(t) ∈ {0,1}: risk-set indicator (§2).
- φ = identity with α ≥ 0 (linear Hawkes), or φ(x) = x⁺ / softplus when inhibition is allowed
  (M-5).
- H_p(t): past events on the same pair. A later variant may add pairs sharing an entity (graph
  neighbourhood, as in GHNN).
- ρ(m_j) = 1 by default; = w_j (identity/detection confidence) only in the marked ablation (M-6).
- Parameters are shared across pairs (object-class-conditioned variants are later work).
- Exponential kernels: the likelihood is computable in O(N·K) per pair by the standard recursion.
  An eager O(N²) reference implementation is required for testing (project rule).

**Mandatory control (§5):** the same model with α_{k,k′} = 0 for k′ ≠ k, and an independent
duration model (onset: Poisson; offset: constant or Weibull hazard since onset). Both use the same
exponential forgetting. Any Hawkes gain must come from **cross-event history**, not from
exponential forgetting.

## 4. Observation model for perception (Stage 2 only; theory cited, not claimed)

Detected events relate to true events by existing mechanisms, instantiated with tracker signals:

- **censoring**: pair not visible → events in that window unobserved (Xu et al. 2017 for window
  censoring; Shelton et al. 2018 / Mei et al. 2019 for latent events inside the window);
- **thinning**: a detection probability that depends on visibility confidence;
- **ghost superposition**: false onsets/offsets from flicker and ID switches (Poisson noise
  superposition, cf. Bonnet et al. 2025; ABC framework arXiv 2006.09015).

The contribution, *if any*, would be an empirical demonstration that a visibility-conditioned
observation model recovers relationship dynamics from real VidSGG outputs, compared with Track A
filters under relapse-aware termination metrics. **Not** a new inference theory.

## 5. Comparisons and ablations (renamed HK0–HK4, to avoid a clash with hypotheses H1–H4)

Stage 1 (ground truth, fully observed) and Stage 2 (corrupted or perceived) compare:

| Model | Role |
|---|---|
| Independent EMA / Phase 1 baselines | Track A reference (Stage 2 only) |
| Continuous-time exponential filter (Phase 4 M2-type) | "simple exponential model" control |
| GatedEMA; visibility-aware HMM | Track A strongest simple baselines (Stage 2) |
| Independent duration / hazard model (per relation type) | **mandatory point-process control** |
| **HK0** base intensity only (Poisson onsets, constant-hazard offsets) | |
| **HK1** + self-excitation (α_{k,k} only) | |
| **HK2** + cross-relation excitation (full A, α ≥ 0) | the scientific core |
| HK2i: + inhibition through φ = (·)⁺ | only if HK2 residuals show suppression |
| **HK3** + censoring/visibility observation model | Stage 2, only if HK2 earns its place |
| **HK4** + marked identity/detection confidence ρ(m) | Stage 2 ablation |

Each rung must improve a pre-registered metric by a pre-registered margin, with a CI excluding 0
on held-out data, or it is dropped.

## 6. Metrics (proposed; frozen only at registration)

- **Stage 1:** held-out log-likelihood per event (nats/event); next-event time MAE and type
  accuracy / MRR; calibration of the time-rescaled residuals (Kolmogorov–Smirnov on the
  compensator-transformed inter-event times — a standard goodness-of-fit test); stability of the
  learned A across splits.
- **Stage 2:** the Stage 1 metrics on held-out ground truth given corrupted observations, plus the
  Track A persistence metrics with a **relapse-aware termination metric** (Phase 4 lesson).

## 7. Stage 0 — ground-truth feasibility probe (draft pre-registration outline; NOT registered)

| Item | Draft |
|---|---|
| Data | VidOR training annotations (relation start/end frames); fallback PVSG. **Download needs explicit user approval** (licence and size to be checked). |
| Split | video-level train/validation/test, frozen before fitting; seeds recorded |
| Events | ONSET/OFFSET per (pair, predicate), time in seconds |
| Models | independent duration model; HK0; HK1; HK2 (α ≥ 0), with exponential kernels and a small shared β grid chosen on validation |
| Primary test | held-out Δ log-likelihood per event, HK2 − HK1 and HK2 − duration model, with a video-level bootstrap CI |
| **Decision (proposed values, to freeze at registration)** | **PROCEED** if HK2 beats both controls by ≥ 0.05 nats/event with CI lower > 0, **and** ≥ 5 off-diagonal entries of A are non-zero with a split-stable sign (bootstrap). **STOP Track B** if HK2 − HK1 CI includes 0. **MODIFY** otherwise (e.g. dependence confined to a few predicate families) |
| Deliverables | A-matrix visualisation with CIs; list of transitions with the strongest α; residual diagnostics |
| Implementation rule | numpy/scipy only (no new dependency); eager O(N²) reference likelihood tested against the O(N) recursion |

Stage 0 does not touch visibility, identity or any base model. If there is no cross-relation
dependence in *ground truth*, no perception-level Hawkes model can be justified.

## 8. Paper-rule checklist (current status)

| Rule | Status |
|---|---|
| 1. Genuine temporal point process | Design satisfies it (§3); not implemented |
| 2. Defensible literature gap | **Empirical/application gap only**; the mathematics is covered (literature §3–§5) |
| 3. Multivariate history beats independent exponential filtering | UNTESTED (Stage 0/1) |
| 4. Survives simple baselines | UNTESTED |
| 5. Valid missing-observation handling | Available from existing theory (M-4); UNTESTED here |
| 6. Generalises beyond the synthetic world | Stage 0 uses real annotations by design |
| 7. Real-data support | UNTESTED |

Until rules 3–7 are met, Hawkes remains an **investigated direction**, not a contribution.

## 9. Relationship to Track A

A combined paper is coherent **only if** Stage 2 shows that the Track A failure regimes
(censoring by occlusion, thinning by flicker, ghost superposition from ID switches) are exactly
the observation-process corruptions that the Hawkes literature models, *and* that modelling them
recovers cross-relation dynamics that simple filters cannot. Otherwise the tracks stay separate.
