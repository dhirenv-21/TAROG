# Phase 5H-1 — presentation script (about 2 minutes, 9 slides)

*(About 330 words. Slides 1–5 are context; slides 6–9 are the registered results. Numbers match
`results/phase5h1/full/tables.md`.)*

**Slide 1.** The question: when one relationship in a video changes, for example someone
*releases* a cup, does that make another change likely soon after, like *holding* ending? And
does a temporal point-process model add anything beyond simpler explanations?

**Slide 2.** Every experiment was pre-registered. The synthetic phases showed that visibility
helps, but our complex filter wasn't justified, so we moved to real VidOR annotations.

**Slide 3.** Stage 0 found a real but small timing signal, and more than half of it survived
shuffling the timing.

**Slide 4.** The suspicion was that busy object pairs simply produce more events. So we needed
heterogeneity controls, inhibition, and honesty that all VidOR data had already been seen.

**Slide 5.** We pre-registered a nested model ladder with two nulls and a GO / NARROW / STOP
rule. The parameters were frozen and committed before any evaluation data was opened.

**Slide 6.** The result is NARROW. Cross-relation history adds 0.145 nats per event beyond
heterogeneity and duration, and 0.083 beyond the transition model. Both evaluation halves agree.
K1 to K3 pass. K4, reproducible individual interactions, fails.

**Slide 7.** Both gains are about five times the largest of 38 nulls. But no single interaction
replicates across folds. Our nulls are imperfect: one adds 16% events, the other loses about
9.5%. Our fold statistic also turned out to be inflated by shrinkage. These failures are reported
as prominently as the gains.

**Slide 8.** The gain is "another relation starts, so this one ends", within 2 seconds, and
inhibition matters: positive-only Hawkes is worse. But 17% of events tie exactly with another
relation's boundary. This may be how annotators segment relations, not scene dynamics. It is
predictive association in annotations, not causation, and not shown on real video.

**Slide 9.** So there is no method contribution. The evaluation paper stays the main track, with
this as an exploratory section. If you agree, the next step is a pre-registered confirmation on
an independent dataset, which has not been started. The video demo is a synthetic illustration,
separate from these results.
