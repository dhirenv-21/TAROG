# TAORG appearance-based tracking audit — Pre-registration

> Written 2026-10-12 on branch `codex/taorg-appearance-audit`, **before any comparative analysis**
> and before any label, cue value or tracker-vs-oracle comparison was computed. Engineering
> checks run before this document (§2.6) touched no labels.
>
> Frozen and unchanged:
> - observation audit: **NOT USABLE** (motion-only ByteTrack);
> - next-method K1: STOP;
> - episode estimator: STOP;
> - Phase 5H-1: NARROW.
>
> This is a **tracker-information audit**. No scene-graph predictor is run, and no
> relationship-persistence result will be reported. A positive result would not establish a TAORG
> algorithm. A negative result stops this line.

## 1. Primary question

> Does appearance evidence add practically useful information **beyond time since last
> detection** for distinguishing **genuine gaps** from **identity takeovers**?

Secondary questions (each answered descriptively unless stated):
- S1. Is the evidence available early enough to help **during** a gap?
- S2. Does any gain survive both detector sources and the individual sequences?
- S3. What happens when appearance evidence is unavailable or ambiguous?
- S4. Is any improvement caused by better **tracking** (fewer takeovers), better **diagnosis** of
  missing records, or both?

## 2. Source and version audit

### 2.1 Correction to the observation exit report

The observation exit report (§9) said BoT-SORT with re-ID was "available in `trackers` as
`BoTSORTTracker`". **That is not correct for `trackers` 2.6.1.**
- Its `BoTSORTTracker` docstring reads "IoU association + optional CMC".
- A source grep finds no re-ID, appearance or embedding path.
- It needs frames only for camera-motion compensation.

The report's NOT USABLE decision is unaffected. This correction is recorded here and in the
research log, and the old report is left unchanged.

### 2.2 Selected tracker: BoxMOT `BotSort`

| Item | Value |
|---|---|
| Package | `boxmot` **25.0.0** (PyPI wheel SHA-256 `ec926008b6d3eaba82f2d3c11cb6befe03001516d3d4479f9b1ee9743adda90d`) |
| Licence | AGPL-3.0. Used unmodified as a library in an isolated environment; not vendored or redistributed |
| Environment | venv `D:/TAORG_runs/appearance/venv` (`--system-site-packages`; outside git). C: has only ~2 GB free |
| Pinned extras (venv) | `lapx` 0.5.12, `gdown` 5.2.0, `timm` 1.0.15, `huggingface-hub` 1.7.1, `opencv-python` 4.11.0.86 (BoxMOT requires < 5), `safetensors` 0.8.0, `beautifulsoup4` 4.15.0. `filterpy` (declared by BoxMOT) did not build and is not imported by `BotSort`. From the system: `torch` 2.14.0+cu132, `numpy` 2.2.6, `scipy` 1.15.3, `scikit-learn` 1.7.2 |
| Class | `boxmot.trackers.box.botsort.tracker.BotSort` |

**Why BoxMOT.** It is a maintained implementation of BoT-SORT with an explicit
`use_embeddings: bool` switch. It accepts precomputed embeddings through the public
`Detections.embeddings` field, and it returns a `detection_indices` field for every output.

**How appearance enters association** (source read, `tracker.py` `_first_association_cost`).
In the **first (high-confidence) association** the cost is:

```
geom = 1 − IoU(predicted track box, detection box)
emb  = cosine distance(track.smooth_feat, detection feature)
emb[emb > appearance_thresh] = 1
emb[geom > proximity_thresh] = 1
cost = min(geom, emb)
```

The same rule applies to the unconfirmed-track association, with `emb / unconfirmed_emb_scale`.
The **second (low-confidence) association** is IoU-only in both variants.
- With `use_embeddings=False`, the first-association cost is exactly `geom`.
- `smooth_feat` is an EMA (α = 0.9) of the L2-normalised features of the high-confidence
  detections matched to the track, renormalised after each update.

**Costs are not probabilities.**
- `geom` ∈ [0, 1] is an IoU distance.
- `emb` ∈ [0, 2] is a cosine distance between unit vectors.
- Neither is calibrated, and none is called a "confidence" in this study.

**Internal APIs used (declared).** These are read only, never written:
- `tracker.lost_stracks`, plus each `STrack`'s `.id`, `.mean` (Kalman state), `.end_frame` and
  `.smooth_feat`;
- `tracker._first_association_cost`, wrapped to **record** the geometry, raw embedding and final
  cost matrices without changing the returned value.

### 2.3 Tracker settings (identical for both BoT-SORT variants)

**The BoxMOT `BotSort` constructor defaults:**
- `track_high_thresh` 0.5, `track_low_thresh` 0.1, `new_track_thresh` 0.6;
- `track_buffer` 30 (scaled by `frame_rate / 30`), `match_thresh` 0.8;
- `proximity_thresh` 0.5, `appearance_thresh` 0.25;
- `second_match_thresh` 0.5, `unconfirmed_match_thresh` 0.7, `unconfirmed_emb_scale` 2.0;
- `use_cmc` True with `cmc_method` "ecc", `fuse_first_associate` False;
- `frame_rate` = `int(seqinfo frameRate)`.

**Variants:**
- **V2:** `use_embeddings=False`, no embeddings supplied.
- **V3:** `use_embeddings=True`, with the cached embeddings of §2.4 supplied through
  `Detections.embeddings`. These are float16-rounded values cast back to float32, the same
  values used for the cues.

**The only intended difference between V2 and V3** is the appearance term in the first and
unconfirmed association costs, and the resulting feature bookkeeping. Detections, frames,
frame rate, Kalman filter, buffers, CMC and thresholds are identical.

**Why not the tuned settings.** BoxMOT's `configs/trackers/botsort.yaml` defaults equal its
`botsort-mot17-ablation` preset, which was tuned on MOT17 train data, and our evaluation
sequences are MOT17 train sequences. Those tuned values are **not** used. Disclosed limitation:
the published BoT-SORT / ByteTrack defaults were themselves chosen by their authors on MOT17
data. No setting is tuned in this study.

**Variant V1** is the existing ByteTrack audit (`trackers` 2.6.1, library defaults; adapter
`src/taorg/adapters/mot17.py`), re-run unchanged (it is deterministic, as tested).
- It differs from V2 in tracker family: it has no CMC, and its thresholds are high 0.6,
  activation 0.7, IoU 0.1.
- V1 vs V2 differences are therefore **not** attributed to appearance.

### 2.4 Selected re-identification model

| Item | Value |
|---|---|
| Model | OSNet x1.0, torchreid model zoo "same-domain" MSMT17 model (softmax loss, input 256×128) |
| Reported accuracy | MSMT17 rank-1 74.9, mAP 43.8 (torchreid MODEL_ZOO.md) |
| Source | Google Drive file `112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M`. The ID is identical in torchreid's MODEL_ZOO.md and BoxMOT's `reid/core/catalog.py` (`osnet_x1_0_msmt17.pt`) |
| File | 10,994,685 bytes, SHA-256 `b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2`; stored at `D:/TAORG_runs/appearance/weights/`, not committed |
| Safe load | `torch.load(weights_only=True)` succeeds: an `OrderedDict` of 567 tensors with no executable objects |
| Training identities | `classifier.weight` has shape (1041, 512). **1041 = the number of MSMT17 training identities**, consistent with the stated provenance. BoxMOT loads 565/565 required tensors (the classifier is unused) |
| Training data | MSMT17: campus cameras at a Chinese university. The model zoo implies ImageNet initialisation but does not state it |
| **Leakage** | **None known.** MSMT17 contains no MOT17 footage, and the weights were not trained on MOT17, MOT16, MOT20 or CrowdHuman. Weights fine-tuned on MOT17 (e.g. BoT-SORT's `mot17_sbs_S50`) were deliberately **not** used, because they would leak the evaluation sequences' identities. DukeMTMC-trained weights (dataset withdrawn) were avoided |
| Licence | torchreid code is MIT; no separate licence is attached to the weights; MSMT17 is distributed for non-commercial research. This use is non-commercial research, and the weights are not redistributed |
| Preprocessing | BoxMOT `ReID` runtime defaults: box crop, resize to 256×128 (`resize`), ImageNet normalisation, L2-normalised 512-d output |
| Hardware | RTX 4060 Laptop GPU (8 GB); smoke test ≈ 20–27 ms per 1080p frame, about 233 MB peak GPU memory |

Only this one ReID model is used.

### 2.5 Data

**Labels and public detections.** These are the same as in the observation audit:
`MOT17Labels.zip`, SHA-256 `0aa79322…82b2b9`, at `D:/TAORG_runs/observation/`.

**Frames.** The official `MOT17Det.zip` (https://motchallenge.net/data/MOT17Det.zip):
- 1,952,547,113 bytes; licence CC BY-NC-SA 3.0, the same as the labels;
- fetched **selectively** with HTTP range requests (`scripts/appearance/fetch_mot17_frames.py`),
  transferring only the central directory and the members of the requested sequences;
- **development 02/09/11 only** (≈ 345 MB) at this stage;
- 04/05/10/13 (≈ 545 MB) will be fetched **only after the freeze** (§9), and only if the
  development gate passes.

Per-file SHA-256 manifests are written next to the frames and copied into
`results/appearance/manifests/`.

**Frame alignment** (checked before analysis; a failure aborts the run). For each sequence:
- the `img1` image count and size must equal `seqLength`, `imWidth` and `imHeight` in the
  `seqinfo.ini` of MOT17Labels;
- frame *f* ↔ `img1/{f:06d}.jpg` ↔ det/gt frame *f*.

**Detectors and split.** Unchanged from the observation audit:
- **FRCNN** and **SDP** public detections. DPM is excluded (its scores are not in [0, 1], which
  BoxMOT rejects);
- development {02, 09, 11};
- held-back evaluation {04, 05, 10, 13}, untouched (no frames downloaded, no tracker run).

**No personal videos** are used.

### 2.6 Engineering checks already run (no labels, no cues, no oracle)

1. BoxMOT imports in the venv with CUDA available. The ReID runtime embeds three test boxes on
   frame 1 of MOT17-02: unit norm, identical boxes give cosine 1.0.
2. A 120-frame timing smoke test on MOT17-02 FRCNN with V2 and V3 settings: about 16 ms/frame
   tracker time, and 20–27 ms/frame ReID time.
3. `trackers.eval` (2.6.1) states TrackEval parity, including MOT17 distractor classes
   (2, 7, 8, 12).

## 3. Units, information sets and labels

### 3.1 Records and prediction time

**A missing record.** For tracker T and confirmed track X, a missing record is a frame *t*
where X has no output and *t* is within **3.0 s** of X's last output (the observation audit's
horizon).

**The prediction is made at frame t**, using only information available at t (causal):
- detections up to and including t, and their embeddings;
- the outputs of T up to and including t;
- X's own history (its outputs, its stored appearance and its last observed box).

**Ground truth (identities, visibility) is used only to label and evaluate.** An AST test
enforces that the inference modules have no ground-truth path (as in the observation audit).

### 3.2 Frame-level labels (oracle; unchanged from the observation audit §5, plus one rule)

Each missing-record frame of X gets one category, computed with
`taorg.experiments.observation.oracle` (Hungarian matching at IoU ≥ 0.5 on the **detection
box** of each output; g(X) = X's majority matched ground-truth identity):

| Category | Meaning |
|---|---|
| GAP | OCCLUDED_GAP ∪ VISIBLE_MISS: g(X) is present and unmatched. **Genuine gap: hold** |
| TAKEOVER | g(X) is matched to another output Y ≠ X. **Identity takeover: drop** |
| GONE, FALSE_TRACK | Excluded from the primary task; counted. FALSE_TRACK = the *unmatched* case (X never matched to any ground-truth identity) |
| AMBIGUOUS (new rule) | X's majority identity covers < 2/3 of X's matched frames (X is itself an identity-mixed track). All of X's missing frames are labelled AMBIGUOUS, excluded from the primary task, and counted |

**Primary task:** binary, TAKEOVER (positive) vs GAP (negative).

### 3.3 Incident-level labels

**An incident** is a maximal run of consecutive missing-record frames of X. Among its
GAP/TAKEOVER frames:
- ≥ 2/3 TAKEOVER → TAKEOVER incident;
- ≥ 2/3 GAP → GAP incident;
- otherwise → AMBIGUOUS incident (excluded; counted).

Incidents with < 50% of their frames in {GAP, TAKEOVER} are OTHER (excluded; counted).

### 3.4 Common ground-truth-aligned unit (cross-tracker comparison)

Because trackers produce different missing intervals, trackers are compared on **ground-truth
identity-frames** (g, t): every considered pedestrian box, after g's first match by that tracker.

Each is classified per tracker as one of:
- **CONTINUED**: matched to the same track as g's previous match;
- **SWITCHED**: matched to a different track. This is the ground-truth-side image of a takeover;
- **UNCOVERED**: unmatched.

**A ground-truth gap** is a maximal UNCOVERED run that ends with g being matched again. It is
resolved as either:
- **BRIDGED** (re-matched to the same track), or
- **FRAGMENTED** (re-matched to a different track).

Runs that never end in a match are **ENDED** and are counted separately.

**Each tracker's own missing-record distribution** (§3.2 categories) is reported separately.

## 4. Cues (inference time; tracker-independent; computed for V1, V2 and V3 records)

**Embeddings.** Every public detection gets one OSNet embedding, computed once and cached
(float16, SHA-256 manifest). The **same cache** feeds V3 and all cues.

**Stored appearance `s_X`.** The EMA (α = 0.9, renormalised: the BoT-SORT `smooth_feat` rule) of
the embeddings of X's output detections with confidence ≥ 0.5 (= `track_high_thresh`; only those
carry features in BoT-SORT), up to X's last output.
- If X never had such a detection, `s_X` is undefined.
- **Implementation check:** for V3, while X is in the lost buffer, `s_X` must equal the tracker's
  `smooth_feat` (cosine ≥ 0.999).

**Candidates.** The public detections at frame *t* with confidence ≥ 0.5.

**The registered cues:**

| Cue | Definition | Available when |
|---|---|---|
| `dt` | Elapsed time since X's last output (s); also `log(dt)` | always |
| `a_glob` | min over candidates of the cosine distance `1 − s_X·e` | `s_X` defined and ≥ 1 candidate |
| `a_loc` | the same, restricted to candidates whose box centre lies within R = h·(1 + 2·dt) of the centre of X's last output box, where h is that box's height. Body-height units; it allows about 2 heights/s of motion. It uses the **last observed** position, not the drifting Kalman prediction (observation audit §6.4) | `s_X` defined and ≥ 1 candidate within R |

**Missing cues.** A missing `a_glob` or `a_loc` is imputed as 1.0 and flagged with a missingness
indicator (`a_loc_missing`, `a_glob_missing`).

**"Appearance evidence available"** at a record frame means that `a_loc` is defined. A missing
track has no current crop of its own: it is compared with *other* detections in the current
frame through its *stored* appearance. When nothing is detected near its last position, no
appearance evidence exists, and this is measured (coverage).

**Retrospective cue (labelled RETROSPECTIVE; not decision-relevant).** `a_loc_retro`: the minimum
of `a_loc` over the **whole incident**, which uses frames after the prediction time. It is
reported only at incident level, as an upper reference, and is **not a solution for decisions
during the gap**.

**V3 tracker-internal costs.** For each first association, these are recorded:
- `geom`, the raw `emb` and the final cost for every (track, detection) pair;
- the resulting matches.

They are used to (i) verify that appearance changes the association computation (§6) and (ii)
check that the raw `emb` row of a lost track equals `a_glob`'s per-candidate distances (an
equality check, not a separate cue).

## 5. Predictors

All predictors are logistic regressions (scikit-learn, `StandardScaler` then `LogisticRegression`,
L2, C = 1.0, lbfgs, max_iter 1000, no class weighting), on these feature sets:

| Predictor | Features |
|---|---|
| **T** (elapsed time only) | `dt`, `log(dt)` |
| **A** (appearance only) | `a_glob`, `a_loc`, `a_loc_missing`, `a_glob_missing` |
| **T+A** (combined) | the union |

No other model families are tried. The output is a fitted P(TAKEOVER), calibrated only in the
logistic-fit sense, so the **Brier score** is reported.

**Development cross-validation.** Leave-one-sequence-out (3 folds: 02 | 09 | 11).
- Both detectors of the held-out sequence are held out together.
- Each fold's model is fitted on the frames of the other two sequences, pooled over detectors.
- All development metrics use **out-of-fold** scores.

**Registered operating point.** In each training fold, θ is the 90th percentile of the
training-fold GAP frames' scores (≤ 10% of real gaps rejected; this matches the observation
audit's 10% U2 bound). It is applied to the held-out fold. At the frozen stage (§9), θ is
computed on all development frames.

## 6. Development feasibility checks (reported; any failure aborts before analysis)

1. **Frame alignment** (§2.5).
2. **Feature extraction:** embedding count = detection count per frame, unit norm, no NaNs.
3. **Appearance changes the association:**
   - in V3, the final first-association cost ≠ `geom` for a non-zero share of track–detection
     pairs;
   - V2 and V3 outputs differ on at least one frame.
   - Both shares are reported.
4. **`s_X` / `smooth_feat` equality and raw-`emb` equality** (§4), reported as the share of
   checked pairs passing.
5. **Latency and memory:**
   - ReID ms/frame;
   - tracker ms/frame (V2, V3);
   - peak GPU memory, process peak RSS.

## 7. Development analysis and gate

**Primary record set:** the missing records of **V3** (the appearance-enabled tracker), primary
task (§3.2).

**Primary comparison:** the out-of-fold frame-level ROC-AUC of **T+A minus T**, computed
**per detector** (FRCNN, SDP).

**Uncertainty.**
- The cluster bootstrap resamples **tracks X** with replacement within each
  (sequence, detector) unit, with 2,000 resamples (seed 20261012, via `taorg.utils.seed`) and
  percentile intervals.
- Out-of-fold scores are held fixed, so fitting variance is not included (disclosed).
- **Multiplicity:** two primary tests (one per detector), so **Bonferroni** gives two-sided
  **97.5%** intervals. All other intervals are 95% and descriptive.

**Also reported** (descriptive): for T, A and T+A, on V3 records (primary) and on V1 and V2
records (secondary; same procedure):
- ROC-AUC and average precision (PR-AUC);
- the Brier score;
- at θ: the real-gap rejection rate (gaps incorrectly rejected) and the takeover retention rate
  (takeovers incorrectly retained, = 1 − recall);
- coverage (the share of frames with `a_loc` available), and performance split by coverage.
  Abstention = frames without appearance evidence; there T+A relies on `dt` and the indicators;
- incident-level ROC-AUC at elapsed **τ ∈ {0.2, 0.5, 1.0} s**: each incident's score is taken
  at its first frame with `dt` ≥ τ, and incidents that ended earlier are excluded at that τ
  (counts reported). **Delay** is reported as the smallest τ at which A reaches incident AUC
  ≥ 0.70;
- RETROSPECTIVE incident AUC of `a_loc_retro`;
- per-sequence ΔAUC (3 values per detector);
- the coverage figure: `a_loc` availability vs `dt` bin, per tracker;
- tracking accuracy per tracker and unit: HOTA, IDF1, MOTA and IDSW (`trackers.eval`,
  considered pedestrians, MOT17 distractors);
- the common-unit distribution (CONTINUED / SWITCHED / UNCOVERED; BRIDGED / FRAGMENTED /
  ENDED).

**Development gate** (decided on development data only):

| Rule | Criterion |
|---|---|
| **G1** | For **both** detectors: ΔAUC(T+A − T) ≥ **0.03**, and the lower bound of its 97.5% cluster-bootstrap interval is > 0 |
| **G2** | Pooled `a_loc` coverage of the V3 primary-task frames ≥ **0.50** |

**Gate PASS** requires G1 **and** G2. **Gate FAIL → STOP:**
- no held-back frames are downloaded and no held-back sequence is processed;
- the exit report is written;
- the cue is **not** redesigned.

## 8. Final decision (only if the gate passes; held-back sequences, frozen everything)

**Applied on held-back data:** the models fitted on all development frames (V3 records), with θ
from development, applied to {04, 05, 10, 13} × {FRCNN, SDP}.

| Decision | Criteria |
|---|---|
| **GO** | For **both** detectors: (a) ΔAUC(T+A − T) ≥ **0.05** with the 97.5% lower bound > 0; (b) at θ, takeover recall ≥ **0.50** *and* real-gap rejection ≤ **0.15**; (c) incident-level A AUC at τ = 0.5 s ≥ **0.70** (usable during the gap); (d) `a_loc` coverage ≥ **0.80** |
| **NARROW** | ΔAUC ≥ 0.03 with the 97.5% lower bound > 0 for both detectors, but some GO criterion fails |
| **STOP** | Otherwise |

If the development gate fails, the decision is **STOP**.

## 9. Freeze (only if the gate passes)

Before any held-back access, one commit freezes:
- the tracker settings;
- the embedding model and preprocessing;
- the cue definitions;
- the fitted coefficients and scaler;
- θ;
- the analysis code.

After that commit:
- held-back frames are fetched;
- every expected run (4 sequences × 2 detectors × 3 trackers) is executed;
- completeness is checked (expected / actual / missing / duplicate / failed);
- the registered comparisons are computed without retuning.

## 10. Examples (predefined rule)

From the **V3 FRCNN development** incidents of ≥ 10 frames, scored by the T+A out-of-fold model
at τ = 0.5 s with θ, one incident per cell:

| Cell | Meaning |
|---|---|
| TP | takeover flagged |
| TN | gap held |
| FN | takeover missed |
| FP | gap rejected |

**Selection.** Within each cell, pick the incident of median length. Ties go to the first in
(sequence, track, start) order. An empty cell is reported as empty.

**Annotated frames** (the last-observed frame and the decision frame, with X's last box, the
`a_loc` gate and the arg-min candidate) are redistributed under CC BY-NC-SA 3.0, with
attribution to MOTChallenge. They are cropped to the relevant region.

## 11. Separation of exploratory work

Any analysis not listed above is labelled **EXPLORATORY**, kept in a separate file and section,
and cannot change the decision.

## 12. Commits

| # | Content |
|---|---|
| 1 | This pre-registration, the source/version audit and the fetch script |
| 2 | Tested implementation |
| 3 | Parameter freeze (only if the gate passes) |
| 4 | Results and the exit report |

## 13. Amendments (each made before any label, cue value or comparison was computed)

**A1 (2026-10-12).** BoT-SORT's high-confidence split is **strict**: `conf > track_high_thresh`
(BoxMOT `split_by_confidence`). Everywhere this document says "confidence ≥ 0.5" (`s_X` and the
candidates in §4), it means **conf > 0.5**, so that `s_X` mirrors the tracker's `smooth_feat`
exactly. The public detection scores are continuous, so the change affects at most measure-zero
ties.

**A2 (2026-10-12; after the label-free run stage, before any label or comparison).**

*What failed.* Feasibility check 4 failed on the first run:
- `s_X` equalled `smooth_feat` in only 58–83% of the checked pairs (minimum cosine 0.74);
- the raw-`emb` equality held for only 58–65% of rows.

*Diagnosis* (scratch instrumentation of `STrack._after_update` / `_after_reactivate`):
- the feature-update sequence equals the output sequence for every track;
- tracks confirmed on frame 1 match exactly (cosine 1.0);
- every later track's `smooth_feat` is **seeded at creation** with the embedding of the
  detection that spawned it, one frame **before** its first output. That pre-confirmation
  detection is not part of any tracker's public output;
- its influence decays as 0.9ⁿ, which explains both failures.

*Resolution.*
- The **registered cue definition is unchanged**: `s_X` is the EMA over X's **output**
  detections. This keeps it tracker-independent and identical for V1, V2 and V3.
- The **implementation check** now compares `smooth_feat` with a mirror that also prepends the
  recorded creation detection (`TrackerRun.created`, captured from BoxMOT's unconfirmed tracks).
  The pass criterion is unchanged (cosine ≥ 0.999; raw-`emb` absolute error < 1e-4).
- For V3, `a_glob`/`a_loc` therefore differ from the tracker's own internal costs on young tracks
  by the creation term. This is disclosed, not corrected.
- The failed run's log is kept at `D:/TAORG_runs/appearance/dev_run_check4_failed.log`, and its
  `run_records.json` at `results/appearance/dev/run_records_check4_failed.json`.

*Earlier operational failures* (no outputs used): two runs aborted on out-of-memory in OpenCV,
because C: is full, so the pagefile cannot grow and the commit limit binds. A third run aborted
on a JSON float32 serialisation bug.
- The OOM fix: the association log was made compact, and one tracker run is kept in memory at a
  time.
- No result was produced by these runs. Their logs are kept on D:.
