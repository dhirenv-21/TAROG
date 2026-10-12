# TAORG appearance-based tracking audit — Exit report

> Date 2026-10-12, branch `codex/taorg-appearance-audit` (stacked on PR #5).
>
> This was a **tracker-information audit**:
> - real MOT17 public detections (FRCNN, SDP) and frames;
> - ByteTrack, and BoT-SORT (BoxMOT 25.0.0) with appearance off and on;
> - one re-ID model (OSNet x1.0, MSMT17).
>
> **No scene-graph predictor was run, and there are no relationship-persistence results.**
> Frozen and unchanged:
> - observation audit **NOT USABLE** (motion-only ByteTrack);
> - next-method K1 STOP; episode estimator STOP; Phase 5H-1 NARROW.

# Decision: **NARROW** (pre-registered rule, §8; frozen models on held-back sequences)

| Criterion (held-back, V3 records) | FRCNN | SDP |
|---|---|---|
| (a) ΔAUC(T+A − T) ≥ 0.05, 97.5% lower bound > 0 | **+0.194** [0.151, 0.240] ✓ | **+0.112** [0.045, 0.183] ✓ |
| (b) at θ: takeover recall ≥ 0.50 **and** real-gap rejection ≤ 0.15 | 0.561 / 0.068 ✓ | **0.468** / 0.124 ✗ |
| (c) incident-level A AUC at τ = 0.5 s ≥ 0.70 | 0.758 ✓ | 0.758 ✓ |
| (d) `a_loc` coverage ≥ 0.80 | 0.948 ✓ | 0.970 ✓ |
| | **GO** | **NARROW** |

The rule requires GO for **both** detectors, so the decision is **NARROW**. The SDP failure is
narrow and real: at the frozen threshold, fewer than half of SDP takeover frames are flagged.

## Summary answers

| Question | Answer |
|---|---|
| Does appearance improve discrimination beyond elapsed time? | **Yes.** On held-back data, ΔAUC is +0.19 (FRCNN) and +0.11 (SDP). The CIs exclude 0 after Bonferroni correction. It is positive in **all 8** held-back sequence × detector units |
| Is it available early enough to help during a gap? | **Yes, causally.** Evidence exists for about 90% of missing frames already in the first 0.25 s. Incident-level AUC of the appearance-only predictor is 0.77/0.76 at τ = 0.2 s, close to the **retrospective** whole-incident reference (0.81) |
| Does it survive detectors and sequences? | **Partly.** The direction is consistent everywhere, but the size varies: the smallest unit is 13-FRCNN (0.64 → 0.71), and SDP is weaker. The operating point does not transfer stably. On development folds, θ rejected 18% / 27% of real gaps; on held-back data, 7% / 12% |
| What happens without appearance evidence? | About 4–5% of frames have no candidate detection near the last position. There the combined model falls back on elapsed time, and its AUC is unstable (0.58 / 0.81 on 752 / 361 frames). Identity-mixed tracks (AMBIGUOUS, 10% of missing records) are **not addressed** |
| Better tracking, better diagnosis, or both? | **Both, but mostly diagnosis.** Tracking: V3 vs V2 IDSW 363 → 357 (FRCNN) and 504 → 428 (SDP); IDF1 +0.008 / +0.015; takeover frames −2% / −16%. Diagnosis: the **same cue on the appearance-off tracker's records** gives ΔAUC +0.22 / +0.15. The information can be used as a diagnostic **beside** any tracker |
| Does it justify a later downstream relationship experiment? | **Yes, one bounded, pre-registered one** (§9). The usable operating point is modest: about half of takeover frames are flagged at 7–12% false rejection of real gaps |
| Algorithm contribution? | **No.** Comparing a lost track's stored appearance with current detections is the standard re-ID association used by BoT-SORT and DeepSORT. The finding is that this standard information **separates missing-record types** in a way motion-only inputs could not. It is not a new method |

## 1. What actually ran

**Commits:**

| # | Content | Commit |
|---|---|---|
| 1 | Pre-registration and source/version audit (before any comparison) | `ac34d2f` |
| 2 | Tested implementation (+ amendments A1, A2) | `e7ae883` |
| 3 | Parameter freeze after the development gate PASS (before held-back access) | `85a839c` |
| A3 | Post-freeze memory fix (values unchanged; verified) | `1ff4387` |
| 4 | Results, exit report and records | the commit containing this file |

**Data.** `MOT17Det.zip` frames were fetched by HTTP range requests:
- development: 357 MB;
- held-back (after the freeze): 568 MB;
- of a 1.95 GB archive. Per-file SHA-256 values are in `results/appearance/manifests/`.

The labels and public detections are the observation audit's `MOT17Labels.zip`. Licence
CC BY-NC-SA 3.0. No personal video was used.

**Models.**
- BoxMOT 25.0.0 `BotSort` (AGPL-3.0, isolated venv on D:);
- OSNet x1.0 MSMT17 from the torchreid model zoo (SHA-256 `b7d73dc6…`, not trained on MOT17);
- ByteTrack: `trackers` 2.6.1.

**Correction (recorded in the pre-registration §2.1).** `trackers` 2.6.1 `BoTSORTTracker` has
**no** re-ID path, contrary to the observation exit report §9.

**Completeness.**
- Development: 6 units × 3 trackers = 18 runs.
- Held-back: 8 units × 3 trackers = **24/24** runs; missing 0, duplicate 0, failed 0
  (`results/appearance/eval/analysis.json` → `completeness`).

**Failed or aborted runs** (all preserved and none used):
- 2 development out-of-memory aborts. C: is full, so the commit limit binds;
- 1 JSON bug;
- 1 development run that failed implementation check 4 (`run_records_check4_failed.json`;
  amendment A2);
- 1 held-back out-of-memory abort (amendment A3).

**Resources.**
- ReID: 17–53 ms per frame (RTX 4060 Laptop; the 53 ms belongs to the dense 04-SDP);
- tracker: 7–34 ms per frame, with V2 ≈ V3;
- GPU peak ≤ 555 MB; peak working set 6.0 GB (development).

## 2. Implementation verification (pre-registration §6; every unit)

1. **Frame alignment:** image count and size equal `seqinfo` in all 14 units.
2. **Embeddings:** one per detection, unit norm, finite.
3. **Appearance changes the association:**
   - the V3 first-association cost differs from geometry for 1–10% of track–detection pairs;
   - V2 and V3 outputs differ on 0–100% of frames. In **09-FRCNN and 04-FRCNN they are
     identical**: the costs changed, but no assignment did.
4. **Mirror checks:** raw cosine cost = 1 − `s_X`·e, and `smooth_feat` = mirror, both **100%**
   (maximum error 2e-7). This holds after amendment A2: BoT-SORT also seeds `smooth_feat` with the
   pre-confirmation creation detection. The registered cue omits that detection and stays
   tracker-independent; this is disclosed.

## 3. Development (LOSO out-of-fold; `results/appearance/dev/`)

**Gate PASS:**

| Rule | FRCNN | SDP |
|---|---|---|
| G1: ΔAUC(T+A − T) on V3 records | +0.177 [97.5%: 0.102, 0.251] | +0.127 [0.065, 0.187] |
| G2: coverage (pooled) | 0.967 | |

**Signs of fragility already visible before the freeze:**
- At θ the real-gap rejection was 0.176 / 0.274, against a 0.10 target.
- Only 8 FRCNN takeover incidents remained at τ = 0.5 s.

## 4. Held-back results (frozen; `results/appearance/eval/`)

**Frame level, V3 records** (14,467 FRCNN frames, 291 tracks; 11,885 SDP frames, 280 tracks):

| | T | A | T+A | AP (T → T+A) | Brier (T → T+A) |
|---|---|---|---|---|---|
| FRCNN | 0.670 | 0.844 | 0.864 | 0.37 → 0.72 | 0.173 → 0.120 |
| SDP | 0.654 | 0.738 | 0.766 | 0.40 → 0.59 | 0.197 → 0.163 |

**Operating point (θ from development):**
- T+A flags 56% / 47% of takeover frames, wrongly rejecting 6.8% / 12.4% of real-gap frames.
- T flags 21% / 17% while rejecting 9.7% / 7.7%.
- So **takeovers incorrectly retained** are 44% / 53% (T+A) vs 79% / 83% (T).

**Per sequence** (V3, T → T+A):

| Sequence | FRCNN | SDP |
|---|---|---|
| 04 | 0.83 → 0.98 | 0.64 → 0.78 |
| 05 | 0.77 → 0.97 | 0.71 → 0.87 |
| 10 | 0.66 → 0.80 | 0.74 → 0.78 |
| 13 | 0.64 → 0.71 | 0.54 → 0.72 |

On 10-SDP, appearance alone (0.73) is slightly *below* elapsed time (0.74).

**Incident level.** AUC by elapsed time τ, with incident counts (takeovers):

| | τ = 0.2 s | τ = 0.5 s | τ = 1.0 s | Retrospective |
|---|---|---|---|---|
| FRCNN, A | 0.77 (n 265, 34) | 0.76 (179, 22) | 0.81 (114, 17) | 0.81 |
| SDP, A | 0.76 (262, 45) | 0.76 (144, 30) | 0.73 (87, 24) | 0.81 |

The **delay** to A-AUC ≥ 0.70 is 0.2 s, the first grid point. At a fixed τ, elapsed time carries
almost no information (incident AUC 0.44–0.57), which is expected.

**Secondary: the same procedure on other trackers' records** (ΔAUC T+A − T):

| Records | FRCNN | SDP |
|---|---|---|
| V1 ByteTrack | +0.18 [0.13, 0.23] | +0.06 [−0.005, 0.13] |
| V2 appearance off | +0.22 [0.17, 0.27] | +0.15 [0.09, 0.21] |

**Tracking** (`trackers.eval`, TrackEval parity; held-back totals or means; V2 → V3):

| | FRCNN | SDP |
|---|---|---|
| IDSW | 363 → 357 | 504 → 428 |
| IDF1 | 0.566 → 0.574 | 0.658 → 0.673 |
| HOTA | 0.478 → 0.484 | 0.533 → 0.543 |
| MOTA | unchanged | unchanged |
| Common unit SWITCHED | 375 → 369 | 604 → 546 |
| Ground-truth gaps FRAGMENTED | 306 → 301 | 332 → 341 |

V1 differs from V2 in tracker family (no CMC, different thresholds), so V1 → V2 changes are
**not** attributed to appearance.

## 5. Exploratory (labelled; cannot change the decision; `results/appearance/exploratory_proximity.json`)

**Question.** Is the gain merely "some detection is near the last position"?

**Test.** A geometry-only analogue of `a_loc` (the nearest candidate's normalised distance, G),
fitted on development and applied to held-back:

| Predictor | FRCNN | SDP |
|---|---|---|
| T+G | 0.70 | 0.70 |
| T+A | 0.86 | 0.77 |
| T+G+A | 0.86 | 0.76 |

**Reading.** Most of the gain needs **appearance similarity**, not proximity.

## 6. Figures and examples (`results/appearance/plots_dev/`, `plots_eval/`)

1. `fig1_coverage`: availability of appearance evidence vs elapsed time (~0.9 at < 0.25 s,
   → 1.0 by 1.5 s), and the takeover share vs time.
2. `fig2_auc_comparison`: T / A / T+A AUCs, and ΔAUC with 97.5% intervals for V1–V3.
3. `fig3_incident_delay`: incident AUC vs τ, with the RETROSPECTIVE reference.
4. `fig4_examples` (development, predefined rule): annotated, cropped frames, attributed under
   CC BY-NC-SA 3.0. Four cells:
   - **TP**: 11-FRCNN track 56; only 1 incident in the cell;
   - **TN**: 02-FRCNN track 0;
   - **FN**: 02-FRCNN track 22, a takeover missed at P = 0.32 vs θ = 0.33;
   - **FP**: 02-FRCNN track 88, a gap rejected because a similar-looking person stood within the
     gate (`a_loc` 0.18).
   The failures are kept.

## 7. Limitations

1. **Few sequences.** There are 7 MOT17 sequences: 3 development and 4 held-back, all pedestrian
   street scenes from one benchmark.
   - The bootstrap resamples tracks within units. It does not capture between-sequence variance,
     which is visibly large (per-sequence ΔAUC 0.04–0.20).
   - Fitting variance is not included, because out-of-fold scores are held fixed.
2. **The labels make the cue partly structural.** TAKEOVER means another output currently covers
   the same ground-truth person, so a similar-looking detection usually exists nearby. The cue
   measures exactly the inference-time trace of that event. This is legitimate, but the AUC
   should not be read as "appearance predicts the future".
3. **Unstable operating point.** The 0.90-quantile threshold transfers poorly across sequences
   (development held-out folds 18–27% gap rejection; held-back 7–12%). The SDP recall miss (0.47)
   sits close to the 0.50 boundary.
4. **Tracker defaults.** BoT-SORT and ByteTrack defaults were chosen by their authors on MOT17
   data, which includes our held-back sequences. This is the same for V2 and V3; nothing was
   tuned here.
5. **The re-ID model** was trained on MSMT17 (no known MOT17 overlap; licence: non-commercial
   research). Only one model was tested.
6. **Coverage.** AMBIGUOUS (identity-mixed) tracks, GONE and FALSE_TRACK records are outside the
   primary task. Those decisions are still needed downstream.
7. **Reproducibility defect.** The recorded SHA-256 values of `*.csv.gz` files are not content
   hashes, because gzip embeds a timestamp. A3 verification therefore compared **decompressed
   contents**: 54/54 files are identical, and 12/12 association logs are bit-identical.
8. **No relationship result.** The relevance to TAORG is that relationship records inherit track
   missing reasons. Nothing here shows a relationship-persistence gain.

## 8. Three critical reviewer assessments

**Reviewer A (tracking).**
- *Strongest:* a matched appearance on/off ablation with recorded costs and verified mirrors;
  public detections; held-back sequences opened only after a committed freeze.
- *Weakest:* only one tracker family with appearance and one re-ID model. The cue is the standard
  BoT-SORT/DeepSORT distance.
- *Verdict:* credible evidence that this information exists. It is not new tracking technology.

**Reviewer B (scene graphs / TAORG).**
- *Strongest:* it reverses the observation audit's blocker. Identity takeovers *are* observable
  at inference time, early in the gap.
- *Weakest:* no relationship model was run, and half of the takeovers are still missed at a
  usable false-rejection rate.
- *Verdict:* this justifies exactly one downstream test, not a method claim.

**Reviewer C (methodology).**
- *Strongest:* pre-registration before any label; amendments A1–A3 documented with their failure
  records; frozen models; completeness checked; the exploratory proximity control is kept
  separate.
- *Weakest:* there are 3 development sequences. The operating point did not transfer as
  calibrated, and the bootstrap omits between-sequence variance.
- *Verdict:* NARROW is the right label.

## 9. The single justified next step

**One bounded, pre-registered downstream experiment.**

**Treatment.** Feed the **frozen** missing-record classifier (T+A, θ from development) into the
existing termination-aware baseline (GatedEMA + hysteresis / fixed buffer) through the
observation contract (`MissingReason.IDENTITY_SUSPECT`, `EvidenceSource.HEURISTIC`):
- **drop** the relation hold when a takeover is flagged;
- **hold** otherwise.

**Comparison.** Against the unchanged baseline and the fixed buffer, on relationship ghost time
and false termination.

**Requirements:**
- relationship labels on tracked MOT-style video. That means a VidSGG model, or a pre-registered
  relationship proxy, and needs approval;
- the MOT17 sequences used here must not be reused for tuning.

**Stop rule.** If that downstream test shows no gain over the fixed buffer, this line stops.

## 10. Stop

Stopped after the audit, this report and the PR handoff. Not started:
- a scene-graph integration;
- a new temporal estimator;
- any further dataset or cue redesign.

## 11. Terminology clarification (added 2026-10-12, during the downstream study; no number changed)

The intervals in this report are **percentile cluster-bootstrap intervals**.
- **Resampling unit.** Tracks are resampled with replacement *within* each (sequence, detector)
  unit, with 2,000 resamples.
- **Scores held fixed.** The predictor scores are fixed: out-of-fold scores on development, and
  frozen-model scores on held-back data. Model-fitting variance is therefore excluded.
- **Primary ΔAUC.** Reported at the **two-sided 97.5%** level per detector. This is the
  Bonferroni adjustment for two detectors, so the pair has familywise coverage of at least 95%.
  The handoff message's phrase "95% CIs, Bonferroni-corrected" should be read this way.
- **Secondary intervals.** Unadjusted 95%.
- **What the intervals do not capture:**
  - between-sequence variability, which is large here: per-sequence ΔAUC 0.04–0.20;
  - variability over detectors beyond the two tested;
  - variability over tracker families.

They are best described as **within-sample uncertainty over tracks**, not as confidence intervals
for performance on new sequences.
