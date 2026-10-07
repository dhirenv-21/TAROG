# Phase 5H-1 — Literature check (heterogeneity, inhibition, nulls, simple alternatives)

> Verification level per source: **(abstract)** = authors/venue/main formulation confirmed from
> the abstract or publisher landing page in a 2026-09-30 / 2026-10-07 web check; **(secondary)** =
> from a search summary only. Full-text checks are still owed before any of this is cited in a
> manuscript. Nothing here is claimed as a project contribution; these are the standard tools the
> experiment must use or be compared against.

## 1. Heterogeneity controls

| Source | Formulation | Assumptions | Implication for Phase 5H-1 |
|---|---|---|---|
| Andersen & Gill (1982) counting-process Cox model; shared frailty reviews, e.g. **Gorfine & Zucker, "Shared frailty methods for complex survival data: a review of recent advances"** (arXiv 2205.05322; Annual Review of Statistics) *(abstract)* | intensity λ_i(t) = Z_i · λ₀(t) · exp(β·x_i(t)); frailty Z_i a cluster/individual random effect acting **multiplicatively**; gamma frailty is the classical choice (closed-form marginal likelihood) | frailty independent of covariates; one frailty per cluster; proportional effect on all event types it multiplies | **Fits** the "pair-level propensity" problem: a pair is a cluster of recurrent relation events. Under gamma frailty with an otherwise Poisson baseline, the posterior mean of Z given the pair's history is (θ + N_p(t⁻))/(θ + Λ_p(t)). Prediction for an unseen pair can therefore only use the pair's *own past*. A random effect cannot be "looked up" for new held-out pairs |
| **DuBois, Butts, McFarland & Smyth, "Hierarchical models for relational event sequences"**, J. Math. Psych. 57 (2013) *(abstract)* | relational event model with parameters varying across sequences; hierarchical (random-effect) sharing across sequences | exchangeable sequences; per-sequence parameters drawn from a population distribution | **Fits** heterogeneity across *videos*/categories. Ridge-penalised category intercepts are the MAP form of Gaussian random intercepts, which is the standard regularised "random effect" when groups are sparse |
| **Perry & Wolfe, "Point process modelling for directed interaction networks"**, JRSS-B 75(5) (2013) *(abstract)* | Cox multiplicative intensity for dyadic events with covariates computed from the **history of the process** (e.g. recent send/receive counts in time windows); partial-likelihood estimation; consistency and asymptotic normality | proportional (log-linear) intensity; piecewise-constant history statistics | **Closest template** for our model: a dyadic (subject → object) event process with log-linear intensity and windowed history statistics |
| Mixed / random-effect Hawkes (search summaries, e.g. arXiv 2605.22038; "Hawkes Models and Their Applications", Annual Review of Statistics 2024) *(secondary)* | random effects in baseline and/or excitation | — | They warn directly about our Stage 0 finding: **unmodelled heterogeneity biases excitation estimates upward** (apparent contagion) |

**Which fit our data.** A per-pair *free* intercept would memorise sparse pairs, and it cannot be
evaluated on held-out pairs. We therefore use (a) ridge-penalised subject- and object-category
intercepts per target (Gaussian random intercepts, MAP), and (b) the frailty-posterior-motivated
history covariates "events so far on this pair", "elapsed time since the pair appeared" and
"number of relations currently on". These are online, causal statistics of the pair's own past.
**Not used:** a joint gamma-frailty marginal likelihood across all event types. It couples all
100 event-type rows and breaks the decoupled fitting. It is recorded as a sensitivity analysis
still owed, not done.

## 2. Valid inhibition

| Source | Formulation | Assumptions | Implication |
|---|---|---|---|
| **Truccolo, Eden, Fellows, Donoghue & Brown, "A point process framework for relating neural spiking activity to spiking history, neural ensemble, and extrinsic covariate effects"**, J. Neurophysiol. 93 (2005) *(abstract)* | point-process GLM: log λ(t \| H_t) = linear function of own history, other units' history and extrinsic covariates; exact likelihood; goodness of fit by time rescaling | conditional intensity with exponential link; history entered through basis functions | **Fits.** An exp link gives a positive intensity for any sign of weight, so excitation (w > 0) and inhibition (w < 0) are both valid. With **piecewise-constant history bases** the compensator is an exact finite sum (no quadrature) |
| Nonlinear Hawkes / point-process GLM stability (e.g. Gerhard, Deger & Truccolo, PubMed 28234899) *(abstract/title)*; Bonnet, Martinez Herrera & Sangnier, Stat. Probab. Lett. 2021 and Stat. Comput. 2023 *(abstract)* | λ = φ(μ + Σ h(t − t_j)) with φ = exp, or φ = (·)⁺ for exponential kernels with an exact MLE | stability conditions differ from the linear case | The (·)⁺ link needs the zero-crossing times of the intensity in the compensator, which is awkward with many event types. The exp link with step bases is simpler and exact. **Chosen: exp link.** |
| Mei & Eisner, "The Neural Hawkes Process", NeurIPS 2017 *(abstract)* | softplus link on a continuous-time LSTM; Monte Carlo compensator | — | Not chosen: needs sampling-based integrals and is far heavier than needed for a feasibility question |

**Positive-only comparison.** The same GLM with cross-history weights constrained ≥ 0. Note
that with an exp link, "positive-only" means multiplicative excitation. It is a different
family from the Stage 0 linear Hawkes model, so Phase 5H-1 results are not directly comparable to
Stage 0's numbers and will not be presented as such.

## 3. Null design

| Null | What it preserves | What it destroys | Source / status |
|---|---|---|---|
| **Circular shift** of each (pair, predicate) interval set by one offset mod T (Stage 0) | per-(pair, predicate) counts, durations, gaps; pair identity; relation set; video length | all alignment between different predicates on a pair, and alignment with pair appearance; also the *co-occurrence* (state) structure | Stage 0, used as is. Too destructive for a "beyond heterogeneity" question: it also removes slow co-modulation, which is not the hypothesis |
| **Interval / shift jitter** (± J seconds) of each (pair, predicate) interval set, clipped to the pair window | everything the circular shift preserves, **plus** coarse (≳ J) timing and co-activity of predicates on the pair | fine (< J) relative timing and ordering between predicates | Conceptual basis: **Amarasingham, Harrison, Hatsopoulos & Geman, "Conditional modeling and the jitter method of spike resampling"**, J. Neurophysiol. 107 (2012) *(abstract)*. Interval jitter tests hypotheses about *local* placement and gives an exact test for processes without fine temporal structure. Caveat: Platkiewicz et al., "Spike-centered jitter can mistake temporal structure" (2017) *(title)*. Our rigid per-set shift is a variant, not the published interval jitter; its exact properties are stated, not borrowed |

**Limitations recorded:**
1. The jitter null keeps slow co-modulation, so effects on time scales > J cannot be tested by it.
2. Clipping at the window edges distorts edge intervals slightly.
3. Shifting a whole interval set preserves its internal structure but can create overlaps
   between predicates that never co-occur. That can create "impossible" states for mutually
   exclusive predicates, a conservative bias *against* finding inhibition.

## 4. Simple alternatives (non-Hawkes)

| Source | Formulation | Implication |
|---|---|---|
| **Asanjarani, Liquet & Nazarathy, "Estimation of semi-Markov multi-state models: a comparison of the sojourn times and transition intensities approaches"**, Int. J. Biostatistics (arXiv 2005.14462) *(abstract)* | semi-Markov multi-state model: transition intensities depend on the time since entering the current state; the intensity-based parameterisation splits the likelihood into two-state (survival) problems | **Fits** our per-relation on/off process directly: ONSET = off→on transition, OFFSET = on→off transition, each with a hazard depending on the sojourn time ("duration hazard"). This is the registered *simple duration control* |
| Piecewise-exponential (Poisson-regression) survival models (standard; e.g. "piecewise exponential additive models" in the search summaries) *(secondary)* | hazards constant within time intervals; likelihood ∝ Poisson likelihood of event counts with log-exposure offsets | Gives the **estimation engine** for all models: every model in the ladder is a Poisson regression on (segment, target) exposures |
| State-dependent Markov transition model | onset/offset intensity of r depends on *which other relations are currently on* (the current configuration), not on when they changed | The registered **simple transition control**. If cross-predicate structure is mutual exclusivity of the current state, this model captures it without any event-history (Hawkes) term |

## 5. Consequences for the design (carried into the pre-registration)

1. One model family for every rung: a **point-process GLM with exp link and piecewise-constant
   bases**, i.e. a Cox/Perry–Wolfe-type multiplicative-intensity relational event model. Rungs
   differ only in which covariate blocks are present. This makes the heterogeneity → duration →
   state → cross-history ladder an exact nested comparison.
2. Heterogeneity is controlled by standard tools only (category random intercepts, frailty-posterior
   history statistics). **No novelty is claimed for it.**
3. Inhibition is allowed through the exp link. The positive-only variant is a constrained fit of
   the same model.
4. Two nulls: the Stage 0 circular shift (retained) and a ±J shift jitter (new, matched to the
   "beyond heterogeneity" question).
5. The decisive simple controls are the semi-Markov duration model and the state-dependent
   transition model. Event-history (Hawkes-type) terms must beat both.
