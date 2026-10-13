# TAORG downstream study — Pre-registration

> Written 2026-10-12 on branch `codex/taorg-downstream`, **before any relationship observation,
> temporal-method output, label or comparison was computed**.
>
> Done before this document:
> - the data-suitability report (`research/taorg_downstream_data_suitability.md`);
> - the user-approved downloads: 30 development videos and the detector weights;
> - one label-free engineering check: all 30 development videos decode to exactly the annotated
>   frame count and size.
>
> Frozen and unchanged:
> - appearance audit **NARROW**;
> - motion-only observation audit **NOT USABLE**;
> - episode study and K1 **STOP**;
> - Phase 5H-1 **NARROW**.
>
> **Evidence class:** a **controlled downstream test on real annotations**. The relation
> predictor is an oracle, while the identities, gaps and takeovers come from a real detector and
> tracker. This is **not** end-to-end scene-graph validation.

## 1. Question

> Does the **frozen** appearance-based takeover flag reduce **identity-related relationship
> ghosts** without unacceptable false termination, beyond strong baselines that receive
> **identical** information?

**A gain shared by both flagged baselines is evidence for the information source**, not for
TAORG temporal inference. Novelty is not presumed.

## 2. Data, sampling, splits and prior exposure

**Dataset.** VidOR v1.0 annotations (`data/vidor/`, the Phase 5H download; MD5s in
`research/phase5h_stage0_report.md`). The videos come from the HuggingFace mirror `shangxd/vidor`
(range-fetched; per-video SHA-256 manifests).

**Scope.** Persons only (`adult`, `child`, `baby`) and **person–person** relation instances
only.

**Sampling rule** (`taorg.adapters.vidor.sample`; annotation-only):
- pool: videos with ≥ 3 annotated persons, ≥ 1 person–person relation and duration ≤ 60 s;
- the pool is listed in sorted file order, then `random.Random(seed).sample(pool, k)`;
- **development:** official *training* split, k = 30, seed 20261013 (pool 2,368);
- **evaluation:** official *validation* split, k = 60, seed 20261014 (pool 263). Its videos are
  fetched **only after the freeze commit**.

**Prior exposure.** All VidOR annotations, including both splits, were used in Phase 5H Stage 0
and 5H-1, at the level of relation-event statistics. The evaluation split is **held back for this
study, not untouched**. No VidOR video, detection or tracker output was processed before this
study.

**Alignment rule.**
- A video is analysed only if its decoded frame count and size equal the annotation's.
- Mismatches are excluded and reported; the rule was fixed before any evaluation video was
  fetched.
- Development check: 30/30 equal.

## 3. Perception pipeline (identical for every method; `src/taorg/adapters/vidor_pipeline.py`)

1. **Detector.** torchvision 0.29.0 `fasterrcnn_resnet50_fpn_v2`, COCO_V1 weights
   (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, SHA-256 `dd69338a…706bf`, 175,221,657 B).
   - It runs on every frame at native resolution in eval mode on CUDA.
   - **COCO class 1 (person) only**, score ≥ 0.1 (BoT-SORT's low threshold), with torchvision's
     default NMS.
   - Scores are not probabilities (they are softmax class scores); they are used only as tracker
     inputs.
2. **Embeddings.** The appearance audit's BoxMOT `ReID` with OSNet x1.0 MSMT17 (SHA-256
   `b7d73dc6…`), with default preprocessing and a float16 cache.
3. **Tracker.** The appearance audit's **V3**: BoxMOT 25.0.0 `BotSort`, `use_embeddings=True`,
   `BOTSORT_SETTINGS` unchanged, `frame_rate = round(fps)`.
4. **Records and cues.** Missing records come from `taorg.adapters.mot17.build_records` (horizon
   3 s), and the cues `dt`, `a_glob`, `a_loc` from `taorg.adapters.appearance.compute_cues`, both
   unchanged.

**Frozen takeover flag (transferred exactly):**
- **Artifact:** `results/appearance/freeze/frozen_models.pkl`, SHA-256
  `98d7e46a6ea2cc44a4e989165a8783618ae16683e93074b3644a085995427f5a`, entry **`("V3", "T+A")`**,
  with θ = **0.37567710275995964** (also in `frozen_params.json`). The SHA-256 is checked at load.
- **flag(X, t) = 1** iff all three conditions hold:
  - X is a confirmed track with a **missing record** at frame t (not output; within 3 s of its
    last output);
  - **`a_loc` is defined** (appearance evidence is available);
  - the frozen model gives P(TAKEOVER) ≥ θ.
- **Appearance unavailable** (`a_loc` undefined) → **no flag** at that frame. The frozen model
  could still score elapsed time and the missingness indicator, but that information is already
  available to the baselines and is not the appearance signal under test.
- **No retuning** of the model, θ, cues or tracker on VidOR, development or evaluation.

Frames are indexed 1..T internally (as in the MOT adapters); annotation frame f maps to
internal f + 1.

## 4. Relationship observations (the only relational input to the methods)

**The oracle predictor.** Matching is computed per frame from **the current frame only**:
Hungarian assignment at IoU ≥ 0.5 between the **detection boxes of the tracker's outputs** and
the annotated person boxes. This is `taorg.experiments.observation.oracle.match`, unchanged.

**Observed edges.** At frame t, for each ordered pair of tracker outputs (X, Y), X ≠ Y, matched
to annotated persons (i, j): for each predicate p with (i, j, p) active at t, emit
**`EdgeObservation(EdgeKey(X, Y, p), observed=True, score=1.0, VISIBLE, VISIBLE)`**. Predicate
ids index the sorted list of the 50 VidOR predicates.

**Absence records.** For every key observed earlier in the video and not observed at t:
- **if X and Y are both output at t:** a *visible-absence* record (`observed=False`, score 0,
  VISIBLE, VISIBLE). The predictor evaluated the pair and returned nothing;
- **otherwise: no record.** The legacy convention applies: methods treat it as UNKNOWN.

**What this is.** A perfect relation classifier applied to tracker outputs. It uses the
annotations only to decide, at the current frame, whether a relation holds between two currently
tracked boxes.
- Methods receive **no** ground-truth identity, no future frame, no relation truth for untracked
  people and no gap outcome.
- Track identities in the keys are **tracker IDs**.

**Leakage tests (§8)** check two things:
- changing future annotations or frames leaves the stream at t unchanged;
- the inference modules contain no ground-truth path.

## 5. Methods (exact reuse) and the single flag policy

| Id | Method | Implementation (unchanged) |
|---|---|---|
| **FB** | Fixed buffer: a key is output iff observed within the last W frames | `taorg.temporal.fixed_window.FixedWindow(window_size=W, score_threshold=0)` |
| **FB+F** | FB with the flag policy | wrapper below |
| **B2** | GatedEMA + hysteresis | `taorg.temporal.gated_variants.GatedEMAHysteresis` |
| **B2+F** | B2 with the flag policy | wrapper below |

### Flag policy: endpoint invalidation (one policy, fixed now)

The policy is implemented in `taorg.temporal.flag_policy.EndpointInvalidation`.
- **Which state.** Before the method's `step(frame t)`: for each track X with flag(X, t) = 1,
  every key in the method's memory with X as subject **or** object (either endpoint) is
  **deleted**:
  - FB: its window history;
  - B2: its GatedEMA score and hysteresis ON state.
- **Clears memory.** The key is then off, exactly as if never seen.
- **Missing appearance.** No flag, so no change.
- **Re-observation.** A deleted key returns only through a new observation. That requires X to be
  output again with the relation present. It then follows the method's normal onset dynamics:
  - FB: immediately;
  - B2: when its score reaches θ_on.
- **Unrelated keys.** Keys not involving a flagged endpoint are untouched.
- **Repeated flags.** Flags on later frames of the same gap are harmless (nothing is left to
  delete).

The wrapper reads and deletes the methods' internal dictionaries. This is a declared internal
use, covered by tests.

**Identical information.** FB, FB+F, B2 and B2+F receive the **same** `FrameGraph` stream.
The +F variants additionally receive the same per-frame flag set. Flags are computed once per
video and shared.

## 6. Scoring (oracle; evaluation only; `src/taorg/metrics/downstream.py`)

### Endpoint state of track X at frame t

| State | Meaning |
|---|---|
| OUT | X is output at t; its current identity m(X, t) is its matched person, or none |
| GAP / TAKEOVER / GONE / FALSE_TRACK / AMBIGUOUS | X is not output at t; labelled as in the appearance audit (`frame_label`: majority identity g(X), purity ≥ 2/3). GAP = g(X) annotated at t and not matched to any output. TAKEOVER = g(X) matched to another output |

### Key truth

alive\*(k, t) for a tracker key k = (X, Y, p) is 1 iff the relation (id(X, t), id(Y, t), p) is
annotated as active at t. Here id(X, t) is:
- m(X, t) if X is OUT and matched;
- g(X) if X is in GAP;
- undefined otherwise, in which case alive\* = 0.

During a takeover, the relationship lives under another key. Under VidOR's convention a relation
is not annotated while a person is unannotated (verified in Phase 5H: 0 relations outside their
pair's co-existence span).

Scored keys are those observed at least once in the video; this set is the same for all methods.

### Primary outcomes (per video; pooled over videos as stated)

| Outcome | Definition |
|---|---|
| **O1. Identity ghost time (IGT)** | Seconds of (k, t) output while an endpoint of k is in TAKEOVER. Pooled as Σ_v IGT |
| **O2. Persistence during genuine gaps (PGG)** | Among frames where alive\* = 1 and k is not observed ("genuine-gap frames": some endpoint is in GAP), the share output. Pooled as a ratio of sums. Its complement in seconds is the **false-termination time (FTT)** |
| **O3. Stable termination delay at genuine terminations (STD-g)** | For each alive\* episode of k ending at b, where at b + 1 **both endpoints are OUT** (the relation visibly ended) and the following not-alive span is ≥ 10 frames: seconds from b to the first frame that starts ≥ 15 consecutive off frames (or the end of the span; censored at the span length). It is relapse-aware: reactivation postpones it. This is the episode-study STD definition, restricted to visible terminations. **FRR-g** = the share of those terminations with any re-output after the first off |
| **O4. Recurrence detection delay (RDD)** | For each alive\* episode of k after its first, not bridged by continuous output: seconds from its start to the first output. This is `taorg.metrics.episodes.episode_metrics` `rdd_mean_s`, unchanged, computed with gt = alive\* runs |

**Secondary outcomes:**
- frame and episode F1 and J (`episode_metrics` on the alive\* runs);
- FTT;
- onset delay;
- the number of flagged endpoint-frames;
- latency.

**Flag behaviour on the new domain** (missing records of V3 on VidOR, descriptive):
- coverage (`a_loc` defined);
- label composition;
- flag rate on GAP and on TAKEOVER frames, i.e. false-positive and true-positive rates;
- set against the appearance audit's MOT17 figures.

The earlier threshold-transfer weakness is reported, not corrected.

## 7. Baseline calibration (development only; unflagged; equal treatment of flags)

**Grids:**
- **FB:** W ∈ {5, 10, 15, 30, 45, 60, 90, 120} frames (8 configurations).
- **B2:** α ∈ {0.1, 0.2, 0.3, 0.5} × decay_visible ∈ {0.3, 0.5, 0.7, 0.9} ×
  decay_occluded ∈ {0.8, 0.9, 0.95, 0.98, 0.99} (decay_unknown = decay_occluded, the default) ×
  θ_on ∈ {0.2, 0.3, 0.5}, with off_ratio = 0.5 (240 configurations).
  - The grids differ in size because FB has one parameter. This is disclosed.

**Selection.** Per family, take the configuration with the highest **mean J** over the 30
development videos, computed **without the flag**. Ties go to the first configuration in grid
order.

**The +F variants use the selected parameters unchanged.** The flag is never used in selection.

**What is not inspected before the freeze:** no flagged method is scored, and no comparison is
computed.

## 8. Implementation tests (before any comparison)

The tests cover:
1. equal inputs across the four methods;
2. endpoint-to-edge invalidation, either endpoint, with unrelated keys untouched;
3. no access to future observations or ground truth: a stream-prefix invariance test and an AST
   test;
4. missing appearance gives no flag;
5. recovery after a false flag: the key returns with normal onset;
6. genuine recurrence;
7. ghost suppression without deleting unrelated keys;
8. agreement between saved outputs and the metrics, with toy tests for every new metric (perfect,
   failure, no positive, gap, termination, recurrence, takeover ghost);
9. deterministic artifacts: gzip with mtime 0, and **content** SHA-256 values of the uncompressed
   bytes.

## 9. Development feasibility gate (decided before the freeze; no flagged comparison is used)

| Rule | Criterion |
|---|---|
| **F1** | All §8 tests pass; alignment and completeness checks pass on development |
| **F2 (exposure)** | With the selected **unflagged** B2, across development: ≥ **20** identity-takeover incidents on scored keys, in ≥ **5** videos, and total IGT ≥ **10 s**. An incident is a maximal run of frames in which a scored key has an endpoint in TAKEOVER |
| **F3 (flag computability)** | Flag coverage (`a_loc` defined) ≥ **0.50** of V3 missing records on development. This is label-free |

**PASS** requires F1, F2 and F3. **FAIL → STOP:**
- no evaluation videos are downloaded;
- the exit report is written;
- there is no redesign.

## 10. Freeze → evaluation

**The freeze commit records:**
- the selected FB and B2 parameters;
- the flag artifact checksum and θ;
- the pipeline settings;
- the analysis code commit;
- the evaluation sample list.

**After the freeze:**
- fetch the 60 evaluation videos;
- run the pipeline on each and the 4 methods per video;
- check completeness: expected 60 pipeline runs and 240 method-runs, with actual, missing,
  duplicate and failed counts;
- compute the registered analysis without retuning.

Development flagged results may be computed after the freeze, labelled as development and
descriptive only.

## 11. Statistics and decision

**Unit of resampling.** The **video**: a paired cluster bootstrap over the 60 evaluation videos,
2,000 resamples, seed 20261015 (via `taorg.utils.seed.set_seed`), percentile intervals.

**What the intervals capture.** Between-video variation in the per-video totals, with the
pipeline fixed.

**What they do not capture.**
- Detector, tracker or flag-model fitting variance: these are frozen artifacts.
- Variation over annotators.
- Generalisation beyond VidOR's user-video domain.

**Pooled outcomes.** Ratios of sums or means over pooled events, recomputed per resample.
Adjacent frames are never treated as independent.

**Primary contrasts (two flag contrasts):** C1 = FB+F vs FB; C2 = B2+F vs B2.

**Multiplicity.** All primary intervals are two-sided **97.5%** percentile intervals (Bonferroni
over C1 and C2). The harm checks are non-inferiority tests within each contrast; they form an
intersection–union, so no further correction is needed.

**Per contrast c:**

| Check | Criterion |
|---|---|
| **Benefit** | Ghost reduction 1 − ΣIGT(+F) / ΣIGT(base) ≥ **0.25**, and its 97.5% lower bound > 0 |
| **Harm margins** (all required) | ΔPGG = PGG(+F) − PGG(base): lower bound ≥ **−0.05** |
| | ΔSTD-g: upper bound ≤ **+0.20 s** |
| | ΔRDD: upper bound ≤ **+0.20 s** |

c **passes** iff the benefit and all harm margins hold. Non-estimable quantities (no events)
fail.

**Decision:**

| Decision | Rule |
|---|---|
| **GO** | C1 and C2 both pass |
| **NARROW** | Exactly one passes |
| **STOP** | Neither passes, or the development gate failed |

**Information vs temporal method** (descriptive; 95% intervals):
- C3 = B2+F vs FB+F on IGT, PGG, STD-g, RDD and J;
- the interaction (B2+F − B2) − (FB+F − FB) on IGT and J.

A TAORG-method claim would need B2+F to beat the equally informed FB+F, and the flag's effect to
depend on the temporal method. **A shared gain supports only the information source.**

## 12. Examples (predefined; timelines only)

VidOR video redistribution terms are unclear, so **no frames are redistributed**.

From the evaluation split, comparing B2 with B2+F:
- **(a) ghost suppressed:** among takeover incidents where B2+F has fewer ghost frames than B2,
  the incident with the median reduction;
- **(b) false termination:** among genuine-gap runs where B2+F outputs fewer frames than B2, the
  run of median length;
- **(c) delayed recurrence:** among recurrences where B2+F detects later than B2, the median
  delay.

Ties go to the first in (video, key, start) order. An empty set is reported as empty.

## 13. Commits

| # | Content |
|---|---|
| 1 | Data-suitability report, this pre-registration, the sampling and fetch code |
| 2 | Tested implementation |
| 3 | Freeze (only if the gate passes) |
| 4 | Results and exit report |

Amendments are appended to §14, each with a statement of whether it could affect the results.

## 14. Amendments

**A1 (2026-10-13; operational; could not affect results).** The session ended during evaluation
perception after 32/60 videos.
- The partially processed video's working directory was deleted, and the pipeline resumed from
  its checkpoint at the same frozen commit `0942146` (clean).
- The untracked `scripts/downstream/figures.py` was moved out of the tree during the resume, so
  the provenance check stayed clean.
- Logs: `eval_pipeline_part1_interrupted.log` and `eval_pipeline_part2.log` on D:.

**A2 (2026-10-13; disclosure).** FFmpeg `swscaler` "slice parameters" warnings appeared while
decoding at least one evaluation video. All 60 videos still met the alignment rule.

**A3 (2026-10-13; post-hoc, labelled).** Two scripts were added *after* the registered decision
was computed; neither changes it:
- `scripts/downstream/posthoc.py`: relative harm, and the genuine-vs-flicker event audit;
- `scripts/downstream/figures.py`.
