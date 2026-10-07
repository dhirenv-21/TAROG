# Phase 5H-1 — 90-second presentation script

*(About 230 words; roughly 15 seconds per slide.)*

**Slide 1.** The question is simple: when one relationship in a video changes, for example
someone *releases* a cup, does that make another change likely soon after, like *holding*
ending? And does a temporal point-process model add anything beyond simpler explanations?

**Slide 2.** We've built this step by step, and pre-registered every experiment before running it.
The synthetic phases showed visibility helps, but our complex filter wasn't justified. So we moved
to real VidOR annotations.

**Slide 3.** That first real test, Stage 0, found a genuine signal: plus 0.074 nats per event,
above every shuffled-timing control. But the shuffled controls already got 0.040, more than half.
What's left is only 7% of what a relation's own history explains, under our 10% bar. So the
verdict is weak-mixed.

**Slide 4.** Most likely, busy object pairs simply produce more events of every kind, so we may
be measuring activity rather than triggering. We also saw short-lag suppression the old model
can't express. And every annotated video has already been used, so anything new is exploratory.

**Slide 5.** This session I checked the literature, audited the training data, pre-registered a
controlled comparison, and built and tested the model engine. The smoke test passed, but that's a
pipeline check, not a result.

**Slide 6.** Next is the full comparison. If cross-event timing survives these controls, we confirm
it on a separate dataset; if not, we drop Hawkes and keep the evaluation paper. I need approval
for the compute and the dataset.
