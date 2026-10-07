# Phase 5H-1 — Supervisor update (7 Oct 2026)

> Evidence classes: **synthetic** = simulated worlds (Phases 3–4, demo). **Real ground truth** =
> human VidOR annotations, no detector (Phase 5H). **Detector outputs** = not used anywhere in
> Phase 5H. Every number below is checked against saved artifacts by
> `scripts/phase5h1/verify_showcase.py` (all 22 checks pass).

## Research problem

Video scene-graph models output relationships (e.g. *person–holds–cup*) frame by frame. We ask
whether **relationship events depend on each other in time**: does "release starts" make "hold
ends" more likely soon after? Specifically, is a temporal point-process (Hawkes-type) model of
these dependencies justified *beyond simpler explanations*, namely how busy an object pair is,
how long relations last, and which relations are currently on?

## Completed work (each experiment pre-registered before evaluation)

| Stage | Data | Outcome |
|---|---|---|
| Phase 3 | synthetic | visibility-aware gating helps under controlled corruption: **H1 = PASS** |
| Phase 4 | synthetic | the complex "V-HPED" filter does not beat simple baselines: **method contribution = NO** |
| Phase 5H design | literature | Hawkes theory for missing/noisy/inhibitory/marked events already exists; any gap is empirical: **MODIFY** |
| Stage 0 | VidOR GT annotations | **HAWKES SIGNAL = WEAK-MIXED** |
| Phase 5H-1 (this session) | VidOR GT, fitting split only | literature check, audit, pre-registration, tested model engine |

## What Stage 0 established

Source: `results/phase5h/stage0/summary.json`; 835 test videos, 50,509 events.

- Adding the history of *other* relations improved held-out prediction by **+0.074 nats/event
  (95% CI 0.066–0.083)**, more than all 19 shuffled-timing nulls (0.039–0.043).
- But the shuffled-timing null already reaches a median of **0.040**, about **54%** of the observed
  gain. The timing-specific excess (**0.034**) is only **7%** of the gain from a relation's own
  history; the pre-registered bar was 10%.
- Interaction structure was not more stable than the null (404 stable cells vs a null maximum of 422).

Figure: `results/phase5h1/showcase/fig_stage0_gain_vs_null.png`.

## Why WEAK-MIXED motivates heterogeneity controls

The null keeps *which* relations each object pair has and destroys *when* they happen, yet it
recovers more than half of the gain. The simplest explanation (a hypothesis, not yet tested) is
**pair heterogeneity**: busy pairs produce more events of every kind, and a model that only
"remembers activity" looks like it learned triggering. The literature says the same: unmodelled
heterogeneity inflates apparent excitation. We also saw short-lag *suppression* (e.g. 0.0001 vs a
baseline of 0.0087 onsets within 0.5 s), which a positive-only Hawkes model cannot express. The
question must therefore be re-asked with heterogeneity, duration and the current state controlled,
and with inhibition allowed.

## New progress this session (commits on `phase5h-hawkes-feasibility`, pushed)

1. **Literature check** (`research/phase5h1_literature.md`), each source with formulation,
   assumptions and fit to our data:
   - frailty / hierarchical relational event models (heterogeneity);
   - point-process GLMs (valid inhibition);
   - interval jitter vs circular shift (nulls);
   - semi-Markov / transition models (simple alternatives).
2. **Fitting-split audit** (Stage 0 FIT only; `results/phase5h1/audit/`):
   - 5,600 videos, 356,040 events;
   - the top 6 predicates are 86% of events, and spatial relations alone 63.5%;
   - 34 onset / 35 offset target types meet the support rule;
   - predicate families are fixed by a semantic rule;
   - two of the four Stage 0 "hypotheses" have tiny support (11–15 events).
3. **Pre-registration** committed before any comparison (`research/phase5h1_preregistration.md`,
   commit `07bf369`): a nested model ladder G0 → G4 (heterogeneity → duration → current state →
   cross-event history with excitation *and* inhibition), two nulls, data-use audit and
   GO / NARROW / STOP rule.
   - **Key integrity point:** every annotated VidOR video was already used in Stage 0, so all
     Phase 5H-1 comparisons are **exploratory**. Confirmation requires an independent dataset.
4. **Model engine** (`src/taorg/hawkes/glm.py`, commit `4d810ce`):
   - exp-link point-process GLM with an exact likelihood;
   - tests show it matches an independent brute-force reference, stays valid under extreme
     inhibition, honours positive-only constraints, reproduces exactly, and blocks evaluation data
     from development code;
   - 234 tests pass.
5. **Smoke run** on 300 training videos (G0–G3): finite, reproducible, leakage-free.
   **A pipeline check, not evidence.**
6. **Showcase:** 6-slide deck, two figures, a demo, and a 90-second script (links below).

## Limitations

- No untouched VidOR evaluation data remain (exploratory label).
- Heterogeneity is controlled by category random intercepts and pair-history proxies; a joint
  frailty fit is still owed.
- The jitter null cannot test effects slower than 5 s.
- The full registered run (ridge selection, 2 × 19 nulls, 5-fold stability) needs parallel
  fitting: estimated 30–60 single-thread CPU-hours.
- Ground-truth annotations only; VidOR boundaries may reflect annotation convention.

## Next decision

1. **Approve compute** for the full registered run (and target-level parallelisation).
2. **Choose an independent dataset** with relation start/end annotations (candidates:
   ImageNet-VidVRD, PVSG) for a confirmatory replication, *if* the exploratory run reaches GO or
   NARROW.
3. If the result is STOP, Hawkes is removed from the contribution list, and the
   robustness/evaluation paper (Phases 3–4) remains the main track.

## Materials

- Deck: [`results/phase5h1/showcase/phase5h1_supervisor_update.pptx`](../results/phase5h1/showcase/phase5h1_supervisor_update.pptx)
- Figures: [`fig_stage0_gain_vs_null.png`](../results/phase5h1/showcase/fig_stage0_gain_vs_null.png),
  [`fig_fit_support.png`](../results/phase5h1/showcase/fig_fit_support.png) (PDF versions alongside)
- Script: [`research/phase5h1_presentation_script.md`](phase5h1_presentation_script.md)
- Demo (synthetic data, about 5 s). It shows that the model recovers a planted dependency and
  shows nothing when none exists:

```bash
python scripts/phase5h1/demo.py
```

- Number check: `python scripts/phase5h1/verify_showcase.py`
