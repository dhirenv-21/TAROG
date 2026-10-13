# TAORG final evaluation — Pre-registration

> Written 2026-10-13 on branch `codex/taorg-completion`, **before any EGTR prediction was scored and
> before any comparison**. Before this document:
> - the Phase 5 lazy maintenance and benchmark are complete;
> - EGTR is integrated (adapter, smoke test on one VG image and two VidOR frames);
> - the VG subset for the reproduction check is being downloaded.
>
> Unchanged:
> - downstream study: registered GO, narrow, annotation-level;
> - appearance audit: NARROW;
> - motion-only audit: NOT USABLE;
> - episode and K1 studies: STOP;
> - Phase 5H-1: NARROW;
> - no supported novel TAORG temporal algorithm.

## 1. Questions

**Q1 (primary).** Does the frozen appearance takeover flag reduce identity-related relationship
ghosts when relationships come from a **real scene-graph model** (EGTR) rather than an oracle,
without unacceptable harm? This is the single justified next step of the downstream exit report.

**Q2 (primary, method).** Does V-HPED (M3, with certified lazy maintenance) improve relationship
maintenance over GatedEMA + hysteresis (B2) on real predictions? Phase 4 found no method
contribution on synthetic data. This question is asked once, on real data, and a negative answer
is final for this project.

**E1 (engineering).** On the real evaluation streams, lazy V-HPED emits exactly the eager
reference's edges.

## 2. Data and prior exposure

**Videos and annotations.** VidOR v1.0 person–person relations (annotations in `data/vidor/`;
videos from the HF mirror, range-fetched).

**Development.** The downstream study's 30 training-split videos (`sample("training")`). Their
perception outputs and oracle labels were analysed in the downstream study.

**Final evaluation.** **40 fresh validation videos**: `sample("final")`, `random.Random(20261020)`,
drawn from the 203 eligible validation videos **not** in the downstream evaluation sample (overlap
verified 0).
- Their annotations were seen in Phase 5H (aggregate event statistics), so they are **held back
  for this study, not untouched**.
- No video, detection, tracker output or prediction of these 40 videos has been processed.
- Their videos (≈ 105 MB, same source and licence as approved for the downstream study) are
  fetched **after the freeze**.

**Alignment rule.** As in the downstream pre-registration: the decoded frame count and size must
equal the annotation's, otherwise the video is excluded and reported.

## 3. Pipeline (identical inputs for every method)

**Perception.** The downstream pipeline, unchanged:
- torchvision Faster R-CNN v2 COCO persons;
- OSNet x1.0 MSMT17;
- V3 BoT-SORT;
- missing records, cues, and the frozen flag (V3|T+A, θ 0.3757; flag iff `a_loc` is defined and
  P ≥ θ).

**Relationship predictor.** EGTR, official VG checkpoint:
- source pinned at commit `7f87450f`; checkpoint SHA-256 `8926640f…`;
- adapter `taorg.adapters.egtr`;
- run on frames f = 1, 4, 7, … (**stride 3**, ≈ 10 fps), because of EGTR's throughput without the
  compiled kernel (0.28 s/frame).
- The relationship layer and scoring use these sampled frames only.

**Person relations.** For each sampled frame:
- person-class queries (13 VG person classes: boy, child, girl, guy, kid, lady, man, men, people,
  person, player, skier, woman), with object score ≥ 0.1, top 30;
- the full ordered-pair score matrix, using the official ranking quantity: connectivity-weighted
  relation score × both object scores.

**Predicate mapping** (fixed now). VidOR `behind` ← VG `behind`; `in_front_of` ← `in front of`;
`next_to` ← `near`; `watch` ← max(`watching`, `looking at`). These four cover ≈ 80% of VidOR
person–person relation instances.

**Identity mapping.** EGTR person queries are assigned one-to-one (Hungarian, IoU ≥ 0.5) to the
tracker's output detection boxes at that frame. Keys use **tracker IDs**. No annotation is used.

**Stream** (`taorg.experiments.final.stream.build_predicted_stream`):
- **observed** when the triplet score ≥ τ_obs;
- **VISIBLE absence (evaluated negative)** for a known key whose two endpoints are both matched to
  EGTR person queries but which is not predicted;
- **no record (not evaluated)** otherwise.

**Flags.** The relationship step at sampled frame f receives the union of the flags on frames
(previous sampled frame, f].

## 4. Methods

| Id | Method |
|---|---|
| NM | `NoMemory` (raw stream) |
| FB | `FixedWindow` (W in sampled steps) |
| B2 | `GatedEMAHysteresis` |
| VH | `LazyVHPED`, rung M3 (r_ov = r_vo = 0.05, dt_ref = 0.1 s) |
| FB+F, B2+F | `EndpointInvalidation` policy (downstream pre-registration §5), unchanged |

## 5. Calibration (development only; the flag is never used in selection)

1. **τ_obs** ∈ {0.005, 0.01, 0.02, 0.05, 0.1}: maximise the mean NM frame F1 over the 30
   development videos.
2. With τ_obs fixed, maximise **mean J** (unflagged):
   - FB: W ∈ {1, 2, 3, 5, 10, 15, 20, 30, 40} (9 configurations);
   - B2: the 240-configuration grid of `taorg.experiments.final.runner.B2_GRID`;
   - VH: the 288-configuration grid `VH_GRID`.
   - Ties go to the first configuration in grid order. The budgets differ (disclosed).
3. The +F variants reuse the FB and B2 selections.

## 6. Outcomes (scored on sampled frames; ORACLE labels `scoring_labels`)

**Scored keys.** Predicted keys ∪ tracker keys whose annotated relation, among the mapped
predicates, is active at some sampled frame.

**Labels.**
- **alive**: the relation between the endpoints' current identities is annotated active. The
  current identity is the GT match if the track is output, its majority identity if the track is
  in a genuine GAP, and none otherwise.
- **takeover**: an endpoint is in TAKEOVER.
- **gap**: alive, and an endpoint is in GAP.
- **ended_after / started_at**: the relation of the current identities is inactive at the next /
  previous sampled frame. These mark **genuine** changes, not identity-matching flicker.

**Outcomes per video, pooled over videos:**

| Outcome | Definition |
|---|---|
| **IGT** | Identity-ghost seconds: output while an endpoint is in TAKEOVER |
| **PGG** | Output share of genuine-gap frames |
| **STD-gen / FRR-gen** | `genuine_outcomes`: stable termination delay and false reactivation at genuine, visible terminations (min dead span 1/3 s, sustained OFF 0.5 s) |
| **RDD-gen** | Delay to detect genuine recurrences (non-bridged) |
| **Frame P/R/F1** and **J** | `episode_metrics` on alive runs |

## 7. Development gate (decided on development data, before the freeze)

| Rule | Criterion |
|---|---|
| **G1** | Perception and EGTR complete for all 30 development videos; all tests pass |
| **G2 (predictor)** | At the selected τ_obs: NM frame F1 ≥ 0.05, and ≥ 1 predicted key in ≥ 20 videos. Below this, the predictor is too weak for a meaningful temporal evaluation |
| **G3 (exposure)** | IGT(B2, unflagged) ≥ 5 s pooled over development |

**FAIL → STOP:** no final videos are fetched, and a report is written.

## 8. Statistics and decisions (final 40 videos)

**Resampling.** Paired video-cluster bootstrap, 2,000 resamples, seed 20261021. The intervals
capture between-video variation only. They exclude EGTR, detector, tracker and flag
fitting variance (all frozen), and they do not cover domains beyond VidOR.

**Q1** (contrasts C1 = FB+F vs FB and C2 = B2+F vs B2; Bonferroni, **97.5%** intervals). A
contrast passes iff all of the following hold:

| Check | Criterion |
|---|---|
| benefit | Ghost reduction 1 − ΣIGT(+F)/ΣIGT(base) ≥ **0.25**, lower bound > 0 |
| PGG (**relative** margin) | PGG(+F)/PGG(base) lower bound ≥ **0.90** |
| STD-gen | ΔSTD-gen upper bound ≤ **+0.20 s** |
| RDD-gen | ΔRDD-gen upper bound ≤ **+0.20 s** |
| accuracy | Δ frame F1 lower bound ≥ **−0.01** |

Non-estimable quantities fail.

| Decision | Rule |
|---|---|
| **GO** | Both contrasts pass |
| **NARROW** | One passes |
| **STOP** | Neither passes, or the development gate failed |

**Q2.** ΔJ(VH − B2). **SUPPORTED** iff ΔJ ≥ **0.01** with a **97.5%** lower bound > 0, *and* VH is
not worse than B2 on IGT or STD-gen (95% upper bound of the difference ≤ 0). Otherwise **NOT
SUPPORTED**. Also reported (95%, descriptive): VH, B2 and FB vs NM, and B2 vs FB.

**E1.** On every final video, run eager `VHPED` with the selected VH parameters and compare
emitted edges and scores with `LazyVHPED`. Pass: 0 decision mismatches (outside a 1e-9 band
around the threshold) and |Δscore| < 1e-9. Real-stream lazy counters are reported.

## 9. Run matrix and completeness

- **Final:** 40 perception runs; 40 EGTR runs; 40 × 6 = **240** method runs; 40 eager-VH checks.
- **Reported:** expected / actual / missing / duplicate / failed.
- **Resumable**, one video at a time; heavy state on D:; content hashes (gzip mtime 0; array
  byte hashes for `.npz`).

## 10. Examples (predefined; timelines only, no frames redistributed)

From the final split, comparing B2 with B2+F:
- (a) the takeover incident with the median ghost reduction;
- (b) the median-length genuine gap that the flag shortens.

And, comparing VH with B2:
- (c) the genuine termination with the median |ΔSTD|.

Ties go to the first in (video, key, start) order.

## 11. Amendments

(none yet)
