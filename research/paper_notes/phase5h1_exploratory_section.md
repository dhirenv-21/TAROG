# Draft section — Exploratory analysis: does cross-relation timing carry information in annotated relation boundaries?

> DRAFT for the paper (exploratory-results section). Not a final abstract. No novelty claim.
> Numbers are copied from `results/phase5h1/full/tables.md` (frozen commit 289319e) and checked
> by `scripts/phase5h1/verify_results.py`. Citations below are the ones verified in
> `research/phase5h1_literature.md`; others need checking before submission.

## X.1 Setting and question

Video scene-graph relations change over time. A natural modelling question is whether the
timing of one relation change on an object pair predicts other changes on the same pair, as in
multivariate Hawkes or relational event models. Such dependence can easily be confounded by
pair heterogeneity (busy pairs have more events of every kind), by duration dependence and by
the current relation configuration. A preliminary probe (Stage 0) found that about half of an
apparent cross-excitation gain was reproduced by a timing-destroying null. We therefore ran a
pre-registered, heterogeneity-controlled comparison.

**Data and evidence class.** We use the human relation annotations of VidOR (no detector,
no tracker). Relation onsets and offsets on each ordered pair are the events. All VidOR
annotations had been used in our earlier probe, so this analysis is **exploratory**:
- fitting on 5600 training videos;
- penalty selection on 700 held-out training videos;
- evaluation on 1535 videos (84,715 events) that were not used for fitting or selection.

Parameters were frozen and committed (289319e) before the evaluation data was loaded.

## X.2 Models

Each event type k on pair p has intensity λ_{p,k}(t) = R_{p,k}(t) exp(η_k + w_k·x_{p,k}(t)). This
is a multiplicative-intensity point-process GLM (Truccolo et al., 2005; Perry & Wolfe, 2013)
with piecewise-constant covariates and an exact likelihood. The exp link keeps intensities
valid when weights are negative, so inhibition is allowed. The nested ladder adds:
- pair heterogeneity proxies: category random intercepts, elapsed time, event and
  active-relation counts (G1);
- semi-Markov duration bins (G2);
- current-state indicators of the other relations (G3);
- lag-window indicators of other relations' onsets/offsets at ≤ 0.5, 0.5–2 and 2–10 s
  (history; G4x = G2 + history, G4 = G3 + history);
- a variant with history weights constrained non-negative (G4+, "classical" excitation only).

## X.3 Results

**Held-out gain.** Cross-relation history improves held-out log-likelihood by +0.145 nats/event
[0.135, 0.155] beyond heterogeneity and duration (G4x − G2). It improves by +0.083
[0.076, 0.090] beyond the state-dependent transition model (G4 − G3), about 9% higher
likelihood per event. Both evaluation halves agree, including the official VidOR validation
split.

**Nulls.** Two timing-destroying nulls are refitted at the frozen penalty: a circular shift and
a ±5 s shift jitter of each relation's interval set. They reach at most +0.031 and +0.017 for
the two gains. Both nulls are imperfect: the circular shift adds +16% spurious events by moving
censored interval ends, and the jitter loses −9.3% to −9.8% of events at window edges.

**Where the gain comes from.**
- Adding one history block at a time, the largest contribution is "another relation starts →
  this relation ends" (+0.048) at lags ≤ 0.5 s (+0.052) and 0.5–2 s (+0.057). Slow history
  adds little.
- Positive-only history is clearly worse than signed history (−0.031), so short-lag
  suppression is an essential part of the structure.
- By family, the gain passes the pre-registered family test for spatial and motion relations
  only.

**Reproducibility of individual interactions (negative).**
- Across five disjoint training folds, the overall weight pattern is more consistent than
  under fold-level nulls.
- However, no individual interaction reaches the pre-registered discovery criterion: the
  minimum estimated FDR is 0.16.
- Our fold t-statistic is inflated by ridge shrinkage in sparse cells. This criterion needs
  redesign before any confirmatory study.

**Annotation timing.** 17.0% of events coincide exactly with another relation's event on the
same pair, and the dominant short-lag patterns are spatial relabelling (e.g. *in front of* ends
as *next to* starts). The history signal is therefore at least partly a signature of how
annotators segment relation instances.

## X.4 What this does and does not show

- **Shows (exploratory):** in VidOR annotations, cross-relation event timing within about 2 s
  predicts relation boundaries beyond heterogeneity, duration and current state. This holds
  mainly for spatial and motion relations and mainly as "start of one relation → end of
  another", with inhibition.
- **Does not show:**
  - causation;
  - that the structure reflects physical scene dynamics rather than annotation convention;
  - that any specific interaction is reproducible;
  - any benefit for scene-graph prediction from real video;
  - that a Hawkes (positive-excitation) model is appropriate. The signed GLM is better, and
    it is a standard model class.
- **Required next step:** an independent, pre-registered replication on a dataset annotated by
  a different process, with repaired nulls and a boundary-tie control.
