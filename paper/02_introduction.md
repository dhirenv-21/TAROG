# 2. Introduction — WORKING DRAFT

> Status: motivation draft only. Contains **no results** and **no novelty claims** beyond
> what `research/paper_evidence_ledger.md` marks SUPPORTED. Citations tagged `[VERIFY]`
> must be checked against the primary sources before submission.

## 2.1 Motivation (draft)

Video scene graph generation (VidSGG) models describe a video as a sequence of
subject–predicate–object relationships between tracked entities, e.g.
*person–holding–cup* [Action Genome, Ji et al., CVPR 2020 — VERIFY]. Because these models
operate on imperfect perception — detectors miss objects, trackers lose or swap identities,
objects are occluded — a relationship that is physically continuous is frequently
*observed* intermittently. Downstream consumers of the graph (activity understanding,
robotics, video question answering) then see relationships flicker on and off.

Temporal reasoning in VidSGG is an active research area. Transformer-based models aggregate
information across frames [STTran, DSG-DETR, OED — VERIFY], debiasing approaches target the
long-tailed predicate distribution in the temporal setting [TEMPURA — VERIFY], and recent
work on world / 4D scene graphs explicitly models object permanence so that relationships can
be maintained while an object is temporarily out of view [4DST / World Scene Graph Generation —
VERIFY]. Temporal consistency and robustness to temporary disappearance are therefore
**not** unaddressed problems.

However, in these approaches the temporal mechanism is typically part of the model
architecture and is learned jointly with it. Using such a mechanism with a different base
model, detector or tracker generally requires retraining, and the persistence behaviour it
induces is implicit in learned weights rather than exposed as an explicit, tunable
trade-off. In particular, a system that bridges observation gaps must also decide when a
relationship has *genuinely ended*; persistence and termination are two sides of the same
decision, and improving one can silently degrade the other.

This motivates the question studied here: **can an external, model-agnostic temporal state
estimator — attached after any frame-wise relationship predictor, without retraining it —
improve robustness to temporary observation loss while keeping relationship termination
controlled and timely?** A natural signal for such an estimator is *visibility*: when the
objects involved are occluded or unreliably tracked, a missing relationship is weak
evidence; when both objects are clearly visible and the relationship is absent, that absence
is evidence of termination.

## 2.2 What this paper will (and will not) claim — to be finalised

Working contributions (hypotheses until supported; see the evidence ledger):

- **C1 (method):** a model-agnostic, visibility-aware temporal post-processing layer for
  relationship outputs; the advanced formulation (V-HPED) is *not yet implemented*.
- **C2 (analysis):** continuous-time edge dynamics with closed-form decay, expiry
  scheduling, lazy/eager equivalence and complexity analysis (the heap itself is standard
  and is not claimed as novel).
- **C3 (evaluation):** a reproducible robustness protocol for temporal relationship methods
  (deterministic corruption with manifests; persistence, churn, termination delay, false
  persistence, latency; matched-termination-cost comparison).

Explicit non-claims:

- We do not claim to be the first to maintain relationships through occlusion.
- We do not claim that existing VidSGG methods cannot handle occlusion.
- We do not claim state-of-the-art accuracy on any benchmark.

## 2.3 Evidence status of the introduction's premises

| Premise | Status |
|---|---|
| Visibility gating improves the persistence/termination trade-off over visibility-unaware smoothing | See ledger H1 (Phase 3 pilot, synthetic only) |
| The improvement transfers to real VidSGG predictions | UNTESTED (Phase 6–7) |
| Continuous-time formulation adds value over per-frame gating | UNTESTED (Phase 4) |

[TBD — contribution list and paper organisation after Phase 4–8 evidence exists.]
