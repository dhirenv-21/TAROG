# 3. Related Work — WORKING DRAFT

> **Citation policy.** Every reference below is tagged `[VERIFY]` until its title, authors,
> venue and the specific characterization given here have been checked against the primary
> source. Where the characterization itself is uncertain it is marked **characterization
> pending** and must not be used in the manuscript until verified. Nothing here was added
> because a search "found nothing" — absence of evidence is not evidence of novelty.

## 3.1 Video / dynamic scene graph generation

- **Action Genome** (Ji et al., CVPR 2020) `[VERIFY]` — video benchmark with frame-level
  human–object relationship annotations; the intended real-data benchmark for Phase 6–7.
- **STTran** (Cong et al., ICCV 2021) `[VERIFY]` — spatial-temporal transformer for dynamic
  scene graph generation; temporal context is learned inside the model.
- **DSG-DETR** (Feng et al., WACV 2023) `[VERIFY]` — exploits longer-term temporal
  dependencies via object tracking/association within the model.
- **TEMPURA** (Nag et al., CVPR 2023) `[VERIFY]` — targets unbiased (long-tail-aware) video
  scene graph generation with temporal consistency mechanisms.
- **OED** (Wang et al., CVPR 2024) `[VERIFY]` — one-stage end-to-end dynamic scene graph
  generation with temporal context aggregation.
- **PWG, MWAE, GHNN** — identified in the pre-Phase-3 literature review as relevant temporal /
  graph-based prior work. **Characterization pending**: exact titles, venues and mechanisms
  must be taken from the review notes and primary papers before any sentence about them
  enters the manuscript.

**Distinction to be argued (not yet evidenced):** these methods learn temporal behaviour
*inside* the scene-graph model. TAORG studies an *external* post-processor applied to the
outputs of any such model without retraining it. Whether this external layer adds value on
top of already-temporal base models is an open empirical question (Phase 6–8), not an
assumption.

## 3.2 World scene graphs and object permanence

- **4DST / World Scene Graph Generation** `[VERIFY — exact title/authors/venue]` — reported
  in the literature review as explicitly maintaining relationships through temporary object
  disappearance / occlusion. **This is the closest competitor** to TAORG's persistence goal.
  The manuscript must state plainly that relationship persistence through occlusion is
  addressed there, and position TAORG's difference as: model-agnostic post-processing, no
  retraining, explicit interpretable continuous-time dynamics, an explicit
  persistence/termination trade-off, formal temporal/complexity analysis, and a reproducible
  corruption protocol. Each of these differences must be substantiated, and if 4DST can be
  reproduced it should be a comparison point (Phase 8).

## 3.3 Temporal state estimation

- Hidden Markov models and forward filtering (Rabiner, Proc. IEEE 1989 `[VERIFY]`) — the
  2-state HMM baseline is an instance.
- Exponential smoothing / EMA and fixed-window memory — standard signal-processing baselines.
- Recursive Bayesian filtering (Kalman, 1960 `[VERIFY]`) and track management in
  multi-object tracking (e.g. track "coasting"/lost states; ByteTrack, Zhang et al.,
  ECCV 2022 `[VERIFY]`) — the tracker-state signals from which TAORG derives visibility.
- Visibility- or occlusion-conditioned state persistence is common engineering practice in
  tracking; TAORG does not claim the idea of "hold state while occluded" as novel. The Phase 3
  question is whether it yields a *measurable trade-off advantage* for relationship edges.

## 3.4 Hawkes processes and continuous-time models

- Self- and mutually-exciting point processes (Hawkes, Biometrika 1971 `[VERIFY]`).
- Neural Hawkes process (Mei & Eisner, NeurIPS 2017 `[VERIFY]`) and later neural
  temporal point processes.
- TAORG must distinguish standard Hawkes machinery (intensity, excitation kernels,
  likelihood) from its own contribution (visibility-dependent regimes, explicit
  negative-evidence handling, alive-probability mapping, expiry analysis). Rules.md §1.8.
- Visible absence is not a standard Hawkes event; how negative evidence enters the model is an
  open design decision (Memory.md open decision 1).

## 3.5 Complex event processing (engineering context only)

- SASE (Wu, Diao & Rizvi, SIGMOD 2006 `[VERIFY]`), Cayuga (Demers et al., CIDR 2007
  `[VERIFY]`), TESLA / T-Rex (Cugola & Margara `[VERIFY]`) — compiled temporal event
  matching. These cover the formerly proposed event-query contribution, which is
  **INVALIDATED** as a research novelty; any query layer is engineering only.

## 3.6 Robustness evaluation

- Corruption benchmarks for image classifiers (ImageNet-C, Hendrycks & Dietterich, ICLR 2019
  `[VERIFY]`) established controlled, parameterised corruption as an evaluation methodology.
- TAORG's C3 contribution is a *temporal-relationship* analogue: deterministic occlusion,
  flicker, visibility-signal noise and ID switches with saved manifests, evaluated with
  persistence, churn, termination and false-persistence metrics at matched termination cost.
  Whether such a protocol already exists for VidSGG must be checked in the literature review
  before C3 is claimed. `[OPEN — literature check required]`
