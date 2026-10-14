# Maintaining Relationships Under Imperfect Tracking: Certified Lazy State Maintenance and an Evidence-Audited Evaluation of Temporal Relationship Filters

> **Status: complete working draft, 2026-10-14.**
> - Every number cites a results artifact or an evidence-ledger entry (`research/paper_evidence_ledger.md`, L1–L72).
> - `[VERIFY]` marks references whose bibliographic details have not yet been checked against the primary source.
> - This draft supersedes the section notes in `paper/01–10`, which remain as history.
> - **No novel temporal estimator is claimed.**

## Abstract

Relationship outputs of scene-graph models flicker when objects are occluded, missed or re-identified
under a new track ID. A temporal post-processing layer must decide whether a relationship whose
evidence has disappeared is still alive.

We report a pre-registered series of studies of such a layer, together with a data-structure result.

**First, certified lazy maintenance.** We give a lazy maintenance scheme for a four-state
visibility-aware relationship filter. A Birkhoff–Hilbert contraction certificate proves when an
unobserved relationship can never be emitted again, without assuming monotone score trajectories.
Frozen posteriors are materialised exactly in closed form, and certification deadlines live in a
versioned min-heap. The lazy filter reproduces the eager reference exactly:
- 0 decision mismatches in 28 synthetic benchmark runs;
- 0 mismatches on 30 real-video streams.

It cuts per-frame state updates by about 4× and wall-clock time by 1.6–3.1× in typical settings. It
is slower under variable frame rates.

**Second, an evidence-audited evaluation.** The proposed estimators (V-HPED, an episode estimator, a
survival-calibrated hold) do not beat visibility-gated EMA with hysteresis or a fixed buffer on
synthetic benchmarks.

On real MOT17 tracker outputs, about 20% of missing track records are identity takeovers rather than
observation gaps.
- Motion-only signals cannot tell the two apart.
- Appearance evidence can: held-back ΔAUC over elapsed time is +0.19 / +0.11.
- Feeding a frozen appearance-based takeover flag to two baselines removes about three quarters of
  identity-related relationship ghosts. This holds in a controlled test with an oracle relation
  predictor, and the gain is shared by both baselines, so it belongs to the information source.

**A real scene-graph model.** EGTR is integrated with source-exact outputs. Its frame-wise person–person
relation scores on VidOR are weak (AUC 0.53–0.69), and a pre-registered predictor-level evaluation
stopped at its development gate.

## 1. Introduction

Video scene-graph generation (VidSGG) predicts ⟨subject, predicate, object⟩ triplets per frame or
per clip. When a person is occluded, missed by the detector, or re-identified under a new tracker ID,
a relationship either disappears from the output or reappears under a different key.

A post-processing temporal layer can bridge gaps, but bridging trades off against timely
termination. This project asked three questions:
1. Can a *visibility-aware* layer improve this trade-off?
2. Can its state be maintained efficiently?
3. Which *information* (visibility, track state, appearance) actually distinguishes a genuine gap
   from a relationship that should end?

**Contributions (each supported only to the stated extent):**
1. **Certified lazy maintenance (§5).** An exact lazy-evaluation scheme for a continuous-time
   (Z, V) relationship filter, with a contraction-based certificate and a versioned deadline heap.
   It is equivalent to the eager filter (L8, L71). The heap is an implementation mechanism; the
   certificate argument is the contribution.
2. **Missing-record information audit (§7.2).** On real tracker outputs, missing records conflate
   observation gaps and identity takeovers (L57, L58). Motion-only signals do not separate them
   (L59); appearance does, and early enough to act during the gap (L61, L62).
3. **Negative and narrow results (§7).** Pre-registered comparisons show:
   - no algorithm contribution from the proposed estimators (L7, L43, L52, L54);
   - the benefit of the appearance flag belongs to the information source, not to temporal
     inference (L65, L66);
   - an off-the-shelf image SGG model gives weak person–person evidence on VidOR (L70).

## 2. Related work

**Video and dynamic scene graphs.** Action Genome `[VERIFY]` (Ji et al., CVPR 2020), STTran
`[VERIFY]` (Cong et al., ICCV 2021), DSG-DETR `[VERIFY]`, TEMPURA `[VERIFY]` and OED `[VERIFY]`
learn temporal context *inside* the model. VidOR (Shang et al., 2019; *Annotating Objects and
Relations in User-Generated Videos*; verified on the dataset page) provides frame-resolved
relation intervals with object trajectories.

**Image scene graphs.** EGTR (Im et al., *EGTR: Extracting Graph from Transformer for Scene Graph
Generation*, CVPR 2024, pp. 24229–24238; verified from the official repository) is the one-stage
model we integrate. Visual Genome `[VERIFY]` (Krishna et al., IJCV 2017) is its training data.

**Multi-object tracking and re-identification.**
- ByteTrack `[VERIFY]` (Zhang et al., ECCV 2022) and BoT-SORT `[VERIFY]` (Aharon et al., 2022):
  used through `trackers` 2.6.1 and BoxMOT 25.0.0 (verified packages).
- OSNet `[VERIFY]` (Zhou et al., ICCV 2019): weights from the torchreid model zoo (verified).
- MOT17 `[VERIFY]` (Milan et al., 2016; MOTChallenge archives verified).

**Positive-matrix contraction.** Birkhoff's contraction coefficient for the Hilbert projective metric
`[VERIFY]` (Birkhoff, 1957; see also Seneta, *Non-negative Matrices and Markov Chains*) underlies
§5.

**Distinction.** TAORG studies an *external* layer on top of any model's outputs. We do not claim
that temporal relationships, occlusion handling or dynamic scene graphs are new (L12–L14,
INVALIDATED).

## 3. Problem and canonical stream

**The stream.** A tracker assigns track IDs; a relationship predictor emits scored triplets over
track pairs. The layer receives a canonical stream of `EdgeObservation`s keyed by
(subject track, object track, predicate).

**Three kinds of record.** The observation contract (L57, L60) distinguishes:
1. *observed*;
2. *evaluated negative*: both endpoints were considered, with no relation;
3. *not evaluated*: no record, e.g. an endpoint is missing.

The legacy convention collapsed (3) into visibility UNKNOWN (L55, L57).

**The layer's job.** Emit, per frame, the relationships believed alive.

**Evaluation.** Frame and episode F1 and J, false reactivation, stable termination delay, recurrence
delay, ghost time (`research/metric_definitions.md`, episode and downstream pre-registrations).

## 4. Temporal filters

**Baselines.**
- **NoMemory** and **FixedWindow** (hold W frames).
- **GatedEMA:** an EMA whose decay depends on edge visibility.
- **GatedEMA + hysteresis (B2):** two thresholds.

**V-HPED** (`research/vhped_formulation.md`): a four-state continuous-time filter over
(Z = relationship exists, V = observable) with generator Q.
- **Prediction:** x ← x·expm(QΔ).
- **Update:** by record-type likelihoods (detection, visible absence, occluded absence,
  unknown / no record).
- **Optional components:** flag confusion (M3), occlusion-onset coupling (M4), identity reliability
  (M5), cross-predicate exclusivity (M6).
- **Not a Hawkes process** (L27).

## 5. Certified lazy maintenance (ADSA)

`research/derivations/lazy_maintenance.md` gives the full statements and proofs.

**The constant-evidence map.** Under constant no-record evidence (A1), identity weight 1 (A2), no
cross-predicate factor (A3) and a constant frame interval (A4), a key evolves as x ← xM/‖xM‖ with
M = P(Δ) diag(L_unk^a) > 0.

**Proposition 1 (exact materialisation).** x_{t+n} ∝ x_t Mⁿ.

**Proposition 2 (certificate).** Let x\* be M's Perron vector and p\* its existence marginal.
Birkhoff's theorem makes d_n = d_H(x_n, x\*) non-increasing, and odds(p_n) ≤ odds(p\*)·e^{d_n}.
Hence, if p\* < τ and d_t ≤ logit τ − logit p\*, the key is never emitted again while A1–A4 hold.
**No monotonicity of p_n is assumed**, and the posterior is retained, not deleted.

**Proposition 3 (deadline).** Block contraction coefficients κ(M^m) give a step count n₁ after
which the certificate is guaranteed. A versioned min-heap holds max(n₁, 15). Any evidence change
cancels the deadline (generation invalidation), and at pop time the certificate is re-verified from
the actual state.

**Corollary (termination-delay bound).** Under no-record evidence, a key that was ON is OFF for
good within n₁(d_t) frames.

**Fallbacks.**
- M5 identity evidence varies every frame, so M5 runs eagerly.
- M6 materialises the frozen keys of a pair when another predicate is observed on it.
- A frame-interval change materialises all frozen keys.

**Complexity.** Per frame the eager filter costs O(A_t), where A_t counts every key ever seen. The
lazy filter costs O(R_t + H_t + m_t S³ log n + q_t log H), where H_t is the number of active keys.
**Both pay Ω(E_t) to output E_t edges**, so no sublinear total runtime is claimed.

**Evidence.**

*Synthetic (`results/phase5/`; M3; 1,800 frames; 2 seeds × 14 settings):*
- equivalence: 0 decision mismatches; max |Δscore| = 5.6e-16 (L8);
- key updates per frame: 1,330 → 327 at A = 1,600;
- wall-clock 1.6–1.9× typical and 3.1× with high key retirement;
- **slower** with mostly-live keys (0.79×) and frequent interval changes (0.47×) (L9 revised).

*Real (30 VidOR × EGTR streams; `results/final/dev/e1_real_streams.json`):*
- 0 mismatches and 0 score difference (L71);
- small absolute gains: 0.58 → 0.46 s in total.

## 6. Experimental protocol

**Pre-registration.** Every comparative study was pre-registered: criteria, splits and decision
rules were committed before results. Development-only calibration was followed by a committed
freeze, then held-back evaluation, with completeness accounting and preserved failures.

**Synthetic worlds.** Phase 3/4 corruption manifests and episode worlds; fixed seed sets.

**Real data.**
- **MOT17:** public FRCNN/SDP detections; development {02, 09, 11}; held-back {04, 05, 10, 13}.
- **VidOR:** person–person relations; development: 30 training videos; held-back: 60 validation
  videos (downstream study) plus 40 fresh validation videos (final; never processed, because the
  study stopped).
- **Prior exposure:** all VidOR annotations had been inspected in Phase 5H, so the held-back sets
  are "held back for the study", not untouched.

**Uncertainty.** Cluster bootstraps over tracks (MOT17) or videos (VidOR), with Bonferroni-adjusted
97.5% intervals for registered pairs of tests. These are not intervals over new datasets.

## 7. Results

### 7.1 Synthetic: visibility gating helps only conditionally; complex estimators add nothing

| Finding | Evidence |
|---|---|
| Visibility-gated EMA wins only in matched settings | Phase 3: 10/18 wins and 0 losses, but tuning-dependent (L1, L3, L16) |
| Gating has characteristic failures | Termination under occlusion, all-UNKNOWN streams, ID-switch ghosts (L5, L18) |
| V-HPED rungs give no method contribution | Phase 4: H2 6/21, H4 2 wins / 12 losses. The one robust gain was ID-switch ghost suppression (220 vs 902 GatedEMA ghost frames), offset by relapse (L7, L23, L26) |
| Further estimators lose to simple controls | Episode estimator: STOP (L43–L47). Survival-calibrated hold: STOP, a tuned fixed buffer matches it (L52–L54) |

GatedEMA + hysteresis is a sufficient termination control on these benchmarks (L48).

### 7.2 Real tracker outputs: what is missing, and what can tell

- **The conflation is real.** On MOT17 development sequences with ByteTrack, 39,534 missing-record
  frames comprise 41% genuine gaps, 20% identity takeovers, 26% departures and 13% false tracks
  (L58).
- **Motion-only signals fail.** The identity-suspect cue reaches 0.1–1% recall, and only elapsed
  time separates the classes (AUC 0.62–0.69) (L59).
- **Appearance works (held-back, frozen).**

  | Detector | AUC with elapsed time only | AUC with appearance (OSNet) | ΔAUC [97.5% interval] |
  |---|---|---|---|
  | FRCNN | 0.67 | 0.86 | +0.19 [0.15, 0.24] |
  | SDP | 0.65 | 0.77 | +0.11 [0.05, 0.18] |

  The registered decision is **NARROW** (L61). The evidence is available within 0.2 s of a
  disappearance (L62). It is mostly a *diagnostic* gain rather than better tracking (L63).
- **Interval terminology.** These are track-cluster percentile intervals; they do not capture
  between-sequence variation.

### 7.3 Downstream (controlled, oracle relation predictor on real VidOR tracking)

**Result.** Over 60 held-back videos, the frozen takeover flag cuts identity-ghost time:
- fixed buffer: −72% [63, 80];
- GatedEMA + hysteresis: −76% [67, 83].

All registered harm margins hold (registered GO) (L65).

**Read narrowly:**
- the tuned baselines keep only about 5% of genuine-gap frames, so the absolute persistence margin
  was lenient;
- the flag fires on 32% of genuine-gap frames on VidOR (MOT17: 7–12%) (L67);
- most scored terminations and recurrences were identity-matching flicker (L68);
- the gain is shared by both baselines (L66).

### 7.4 A real scene-graph model

- **Integration.** EGTR's outputs through our adapter are bit-identical to the official evaluation
  path on 200 VG test images. Subset SGDet R@20/50/100 is 21.6/28.1/32.7 vs the authors'
  23.5/30.2/34.4 (L69).
- **On VidOR.** Person–person relation scores are tiny and weakly informative (AUC 0.53–0.69 by
  predicate) (L70).
- **The final evaluation stopped.** The pre-registered predictor-level evaluation (flag benefit, and
  V-HPED vs B2) **stopped at its development gate**, because the registered threshold grid did not
  match the score scale (L72). Whether temporal post-processing improves real VidSGG outputs remains
  **untested** (L6).

## 8. Limitations

1. **No supported novel temporal estimator.** Every estimator-level claim is NOT SUPPORTED or
   synthetic-only.
2. **Annotation-level evidence.** The ghost-suppression result relies on an oracle predictor;
   predictor-level effectiveness is untested.
3. **Exposure, domain and size.** VidOR annotations were previously inspected. The studies cover
   one tracker family with appearance, one re-ID model and one SGG model. MOT17 contributes 7
   sequences.
4. **Lazy maintenance scope.** Its benefit depends on stale-key accumulation and a constant frame
   rate. The M5 identity rung runs eagerly.
5. **A pre-registration error.** The final evaluation's threshold grid ignored the predictor's score
   scale.

## 9. Conclusion

**What holds up.**
- An exact, certified lazy maintenance scheme for a visibility-aware relationship filter.
- Evidence that appearance information, not temporal-filter complexity, is what separates identity
  takeovers from observation gaps on real tracker outputs.

**What does not.** Temporal-filter novelty did not survive pre-registered testing.

**What remains open.** The real-predictor question needs a video scene-graph model, or
rank-normalised scores, under a new registration.

## Reproducibility

- **Code and configurations:** in git.
- **Run records:** under `results/`, with content hashes.
- **Heavy state:** on D: (`docs/SETUP.md`).
- **Runnable pipeline:** `python -m taorg.pipeline run --video V --work W` (`docs/PIPELINE.md`).
