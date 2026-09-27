# TAORG / V-HPED — Project Requirements Document

## 0. Document status
- Project: **TAORG — Temporal Adaptive Object Relationship Graph**
- Core method: **V-HPED — Visibility-aware Hawkes-Process Edge Dynamics**
- Project type: ADSA research project + conference-paper-oriented research codebase
- Primary objective: test whether visibility-aware temporal edge dynamics can distinguish observation failure caused by occlusion/tracker noise from genuine relationship termination in video scene graphs.
- This repository is **research infrastructure first**. A polished demo/UI is explicitly secondary and must not displace reproducibility, baselines, ablations, proofs, or evaluation.

## 1. Problem statement
Video scene graph (VidSGG) models produce subject–predicate–object relationships over time. Tracker failures, occlusion, missed detections, and identity switches can cause an otherwise continuous relationship to disappear for a few frames. Naive frame-wise prediction therefore creates relationship churn such as:

`person-holding-bag -> missing -> person-holding-bag -> missing`

A temporal memory can bridge gaps, but conventional fixed windows or EMA-like decay generally do not distinguish two semantically different cases:
1. **Occluded / unreliable observation:** absence should be treated as censored or weak evidence.
2. **Visible objects but missing relationship:** absence is negative evidence that the relationship ended.

TAORG adds a model-agnostic temporal post-processing layer, V-HPED, after an existing VidSGG model.

## 2. Research question
**When a relationship edge stops being observed, can a visibility-aware temporal point-process model separate observation failure from true relationship termination more effectively than simpler temporal-memory methods?**

The project must answer this empirically. It must not assume V-HPED wins.

## 3. Research hypotheses
- **H1 — Visibility gating:** visibility-aware temporal memory improves the persistence/termination trade-off relative to visibility-unaware memory.
- **H2 — Hawkes interactions:** learned cross-predicate excitation/inhibition provides measurable benefit beyond visibility-gated EMA.
- **H3 — Learned dynamics:** likelihood-trained parameters outperform hand-set temporal parameters when evaluated on held-out videos.
- **H4 — Efficient maintenance:** lazy decay + expiry scheduling produces the same logical keep/drop decisions as eager decay, subject to explicitly documented numerical tolerances and event ordering, while reducing temporal-state maintenance cost when few edges are touched.

If H2 fails, claims must shrink accordingly. If H1 fails in the pilot, pivot to a tracker-noise robustness benchmark rather than forcing the proposed model narrative.

## 4. Target users
### Primary
- The 5-person student research team implementing, evaluating, and writing the paper.
- Faculty/advisors reviewing the ADSA and research contribution.

### Secondary
- VidSGG researchers who may want to attach the temporal module to an existing model.
- Paper reviewers/reproducers who need exact configs, seeds, environment details, and metric definitions.

## 5. In scope
1. Adapter interface for frame-wise scene-graph outputs from existing models.
2. Visibility estimation/gating from track state and available detector/tracker signals.
3. Temporal baselines: no memory, fixed window, EMA, visibility-gated EMA, and a 2-state/HMM-style baseline.
4. V-HPED state model with predicate interactions and visibility-aware dynamics.
5. Training/fitting of temporal parameters on training data only.
6. Synthetic occlusion injection with deterministic seeds.
7. Synthetic ID-switch injection with deterministic seeds.
8. Real tracker-noise evaluation when tracker outputs are available.
9. Standard VidSGG metrics plus persistence, churn, termination delay, calibration, and latency.
10. Lazy temporal-state evaluation and expiry min-heap implementation.
11. Unit tests, invariant tests, equivalence tests, and complexity/latency benchmarks.
12. Experiment manifests and paper-ready machine-readable result exports.
13. Reproduction/adapters for selected base models: STTran, DSG-DETR, TEMPURA, subject to dependency and dataset availability.
14. Datasets targeted: Action Genome; ActionGenome4D if legally/technically obtainable; VidOR or PVSG for multi-person/ID-switch experiments.

## 6. Explicitly out of scope
- Training a new object detector.
- Improving YOLO or another detector.
- Building a new tracker.
- Surveillance, defence, weapon, or IoT applications.
- A dashboard as a core deliverable.
- Claiming invention of Hawkes processes, scene graphs, detectors, trackers, or generic exponential decay.
- Hiding negative results or selecting only favorable seeds.

## 7. Core functional requirements
### FR-1: Canonical edge/event representation
Every adapter must map predictions into a common schema containing at minimum:
- video_id, frame_idx, timestamp
- subject track ID and object track ID
- subject/object class IDs
- predicate ID and score
- subject/object boxes and confidence when available
- visibility/track state inputs when available
- source model name and checkpoint/config identity

### FR-2: Visibility state
The temporal layer must support at least:
- `VISIBLE`
- `OCCLUDED_OR_UNRELIABLE`
- `UNKNOWN`

The exact mapping from tracker/detector signals to these states must be configurable and logged. `UNKNOWN` must not silently become `VISIBLE`.

### FR-3: Temporal baselines
All baselines must consume the same canonical edge stream and noise realization. They must be evaluated through the same metric implementation.

### FR-4: V-HPED state
For edge `(i,j)` and predicate `p`, support a state conceptually based on:

`z_ij^p(t) = mu_p + sum_q sum_{t_k<t} alpha[q,p] * exp(-beta_p * (t-t_k))`

with a positive output intensity such as `softplus(z)` when a Hawkes likelihood requires positivity.

Important: implementation must distinguish **event intensity**, **edge-alive probability/score**, and **drop threshold**. Do not conflate them. If `P(alive)=1-exp(-lambda*dt)` or a calibrated sigmoid is used, the chosen mapping must be validated and documented as a modeling choice.

### FR-5: Visibility-aware absence handling
- observed edge: positive event/update
- low-visibility missing edge: censored/weak-evidence behavior; use occlusion regime
- visible missing edge: negative termination evidence / fast decay behavior

The precise mathematical treatment of negative evidence must be explicit. Standard Hawkes likelihood models positive event arrivals; visible absence cannot be described as a standard positive Hawkes event without defining an observation/termination model. The code and paper must not overclaim this point.

### FR-6: Parameter learning
Support train/validation/test separation. Fit temporal parameters only on training tracks; choose thresholds/calibration/hyperparameters using validation data; report final metrics on held-out test data.

If using `softplus(z)` intensity, do **not** assume the integral term has the same closed form as a linear exponential Hawkes intensity. Use a mathematically valid numerical/analytic treatment and test it.

### FR-7: Lazy decay + expiry heap
Maintain active temporal edges without eagerly decaying every active edge every frame.
- store last materialized state and timestamp
- analytically materialize exponential decay on access where valid
- schedule next threshold-crossing time when it is mathematically defined
- maintain min-heap of expirations
- handle stale heap entries using version numbers/generation counters, or use an indexed heap with key updates
- recompute expiry when state jumps or visibility regime changes

### FR-8: Expiry correctness
The optimized implementation must be checked against a simple eager reference implementation on randomized traces. Decisions should match within documented tolerance and event-order semantics.

### FR-9: Noise injection
Synthetic corruption must support:
- box/detection removal for specified durations, including 1/2/3/5/10/20 frames
- controlled ID swaps for specified durations
- reproducible random selection via seed
- corruption manifest saved to disk so every method sees exactly the same corrupted input

### FR-10: Metrics
Implement and unit-test:
- Recall@K / mean Recall@K where supported by the base benchmark
- Edge Persistence Precision/Recall/F1 with a written definition of what counts as a positive persistence instance
- Relationship Churn Rate with a written denominator
- Termination Delay in frames and optionally seconds
- false persistence after true termination
- ECE or another stated calibration metric for alive probability
- per-frame temporal-module latency and end-to-end latency separately

Metric definitions must be frozen before final experiments.

### FR-11: Experiment runner
Every experiment run must save:
- unique run ID
- git commit hash
- command/config
- dataset split/version
- base-model checkpoint identity
- seed(s)
- environment/package snapshot
- metrics JSON/CSV
- runtime statistics
- corruption manifest if applicable
- warnings/failures

### FR-12: Paper outputs
Provide scripts that aggregate raw run outputs into tables/CSV suitable for paper figures. Paper numbers must be traceable back to run IDs; no manually typed headline numbers as the sole source of truth.

## 8. Non-functional requirements
- Python-first, Linux-friendly, CUDA optional for temporal-only tests.
- Deterministic where practical; nondeterministic CUDA operations must be documented.
- Config-driven experiments; avoid editing source code to change experimental conditions.
- Modular adapters; temporal code must not depend directly on one base model's internals.
- Clear error messages for missing datasets/checkpoints.
- No dataset or checkpoint redistribution unless licensing permits it.
- Unit tests should run without downloading full datasets.

## 9. Success criteria
### Two-week go/no-go pilot
Using ground-truth boxes or a simplified canonical stream on Action Genome with synthetic occlusions:
- implement no-memory, EMA, and visibility-gated EMA
- evaluate persistence versus true-termination behavior
- demonstrate whether visibility gating creates a meaningful Pareto/trade-off improvement across thresholds rather than cherry-picking one threshold

**Go:** visibility gating consistently improves the persistence/termination trade-off across multiple seeds/occlusion lengths.

**No-go/pivot:** no stable advantage. Reframe the project around a tracker-noise robustness benchmark and analysis; do not fabricate a V-HPED advantage.

### Final success
A reproducible repository containing baselines, V-HPED, optimized state maintenance, formal complexity argument, controlled noise protocol, ablations, and paper-ready results.

## 10. Required ablations
- V-HPED without visibility
- V-HPED without cross-predicate alpha
- V-HPED with hand-set parameters/no MLE
- visibility-gated EMA
- full V-HPED

Where practical, also separate excitation-only from excitation+inhibition.

## 11. Required comparisons
- no temporal memory
- fixed window
- EMA
- visibility-gated EMA
- HMM/2-state filter
- closest reproducible recent WSGG/occlusion-aware method, after related-work verification
- base models without and with temporal post-processing

## 12. Paper-integrity requirements
- Never claim a method/dataset property without a citable source or direct verification.
- Never label a result SOTA unless the exact evaluation protocol is comparable and verified.
- Report negative/neutral ablations.
- Report mean ± std over predeclared seeds for stochastic experiments.
- Keep test set untouched during tuning.
- Maintain a `research/claims.md` evidence ledger later in development: claim -> experiment/table/run IDs -> status.
- Preserve raw result files.
- Treat the current equations and proposed complexity/bounds as hypotheses/specifications to validate, not automatically proven facts.

## 13. Open research decisions to resolve before final implementation
1. Exact mathematical observation model for visible negative evidence.
2. Whether `beta_vis`/`beta_occ` are learned, constrained, or partly hand-set.
3. Exact alive-score mapping and calibration method.
4. Threshold definition used by the expiry equation: threshold in `z`, `lambda`, or calibrated alive probability.
5. Correct expiry formula under multivariate contributions, inhibition, baseline, softplus, and visibility-regime changes.
6. Dataset availability/licensing and exact ActionGenome4D annotations relevant to occlusion.
7. Exact recent WSGG baseline and reproducibility status.

These must be resolved with derivation and/or literature verification before they become paper claims.
