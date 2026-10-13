# TAORG downstream study — Exit report

> Dates 2026-10-12/13; branch `codex/taorg-downstream`, stacked on PR #6.
>
> **Evidence class: a controlled downstream test on real annotations.**
> - Real VidOR videos, a real person detector and the appearance audit's real V3 tracker.
> - Relationship observations come from an **oracle relation predictor** applied to tracker
>   outputs.
> - This is **not** end-to-end scene-graph validation.
> - The annotations were used earlier (Phase 5H), so the evaluation split is **held back for this
>   study, not untouched**.
>
> Unchanged:
> - appearance audit NARROW;
> - motion-only observation audit NOT USABLE;
> - episode study and K1 STOP;
> - Phase 5H-1 NARROW.

# Registered decision: **GO** (both flag contrasts pass; frozen everything; 240/240 runs)

| Contrast (60 evaluation videos) | Ghost reduction [97.5%] | ΔPGG [97.5%] (margin ≥ −0.05) | ΔSTD-g s [97.5%] (≤ +0.20) | ΔRDD s [97.5%] (≤ +0.20) | Pass |
|---|---|---|---|---|---|
| **C1** FB+F vs FB | **0.72** [0.63, 0.80] | −0.014 [−0.021, −0.009] | −0.007 [−0.010, −0.005] | −0.007 [−0.011, −0.003] | ✓ |
| **C2** B2+F vs B2 | **0.76** [0.67, 0.83] | −0.013 [−0.019, −0.008] | −0.000 [−0.001, −0.000] | −0.003 [−0.005, −0.002] | ✓ |

**What this GO does and does not mean.** Read it narrowly. It supports **annotation-level utility
of the appearance information source** for one specific error: holding a relationship on a track
whose person the tracker has re-identified under a new ID. It does **not** support a TAORG
algorithm contribution (§4). The registered harm margins turned out to be lenient in this setting
(§3).

## Summary answers

| Question | Answer |
|---|---|
| Does the flag improve either baseline over its unflagged version? | **Yes, both, on the registered ghost outcome:**<br>• identity-ghost time falls 145 → 40 s (FB) and 110 → 27 s (B2), pooled over 60 videos (≈ 21 min of video);<br>• J rises +0.029 / +0.023 (secondary) |
| Does flagged GatedEMA + hysteresis improve on the equally informed fixed buffer? | B2+F beats FB+F (J +0.16 [0.14, 0.18], STD −0.11 s, IGT −14 s). That is **B2's own advantage over FB**, which is present without the flag (J 0.746 vs 0.580). The **interaction** goes the other way: the flag helps FB *more* (IGT +22 s [10, 38]; J −0.006 [−0.010, −0.002]). The gain is shared, which is **evidence for the information source, not for temporal inference** |
| How much ghost reduction is bought with false terminations? | Per added second of false termination in genuine gaps, the flag removes 2.7 (FB) / 2.4 (B2) ghost-seconds (post-hoc D1). It also loses **25–27% of the baselines' remaining gap persistence**, which was already tiny (PGG 0.053 → 0.039; 0.051 → 0.038) |
| Does the flag remain useful when appearance is unavailable or misleading? | **Unavailable** (17% of missing records): by policy no flag fires, so the behaviour equals the baseline. **Misleading:** on VidOR the frozen flag fires on **32% of genuine-gap frames** (MOT17 held-back: 7–12%) and on 58% of takeover frames; precision for TAKEOVER is 0.45. The threshold-transfer weakness is worse in this domain. The cost stays bounded only because the tuned baselines barely hold through gaps anyway |
| Is recurrence delayed? | **No measurable delay.** ΔRDD is −0.003 to −0.007 s on registered events, and +0.0003 s [0, 0.001] on *genuine* recurrences only (post-hoc D2) |
| Annotation-level utility or predictor-level effectiveness? | **Annotation-level only.** The relation predictor is an oracle whenever both people are tracked. Predictor-level effectiveness is untested |
| Algorithm contribution? | **Not supported** |

## 1. What actually ran

**Commits:**

| # | Content | Commit |
|---|---|---|
| 1 | Data-suitability report and pre-registration | `a326622` |
| 2 | Tested implementation | `bf7749d` |
| 3 | Freeze after the development gate PASS, pushed before any evaluation video was fetched | `0942146` |
| 4 | Results and this report | the commit containing this file |

**Data:**
- VidOR v1.0 annotations (local), persons and person–person relations only;
- videos range-fetched from the HF mirror: development, 30 training-split videos (134 MB
  transferred, 22,017 frames); evaluation, 60 validation-split videos (233 MB, 38,411 frames);
- the sample was seeded and annotation-only (≥ 3 persons, ≥ 1 pp relation, ≤ 60 s);
- all 90 videos decode to exactly the annotated frame count and size.

**Perception** (inference only):
- torchvision Faster R-CNN v2 COCO, persons only (SHA-256 `dd69338a…`);
- OSNet x1.0 MSMT17;
- the appearance audit's V3 BoT-SORT (unchanged);
- the frozen `V3|T+A` flag (SHA-256 `98d7e46a…`, θ = 0.3757).

**Methods:**
- FB (`FixedWindow`) and B2 (`GatedEMAHysteresis`), both unchanged;
- +F = the single pre-registered endpoint-invalidation policy.

**Calibration** (development, unflagged only, max mean J):
- FB W = 5 frames (8 configurations);
- B2 α 0.5, decay_visible 0.3, decay_unknown 0.8, θ_on 0.5 (240 configurations).
- **Both are at the low-hold edge of their grids** (disclosed; not extended).

**Gate PASS:**
- F2: 20,517 takeover-incident runs on scored keys, in 28/30 videos; IGT(B2) 56.8 s;
- F3: coverage 0.775.

**Completeness.** Evaluation perception 60/60; method runs **240/240**; missing 0, duplicate 0,
failed 0.

**Interruptions:**
- The session ended during evaluation perception after 32/60 videos.
- The one partially processed video directory was deleted, and the run resumed from the
  checkpoint at the same frozen commit (clean). Both logs are kept.
- FFmpeg `swscaler` slice warnings appeared while decoding at least one evaluation video. The
  frame counts and sizes still matched.

**Resources:**
- detection 115 ms/frame, ReID 31 ms/frame, tracker 9 ms/frame (RTX 4060 Laptop);
- about 3.5 h of perception in total;
- RAM about 1–4 GB free throughout, so videos were processed one at a time.

## 2. Registered outcomes (evaluation; pooled over 60 videos)

| Method | IGT (s) | PGG | FTT (s) | STD-g (s) | RDD (s) | J |
|---|---|---|---|---|---|---|
| FB | 145.5 | 0.053 | 2550 | 0.164 | 0.137 | 0.580 |
| FB+F | 40.2 | 0.039 | 2589 | 0.157 | 0.131 | 0.609 |
| B2 | 109.6 | 0.051 | 2556 | 0.052 | 0.110 | 0.746 |
| B2+F | 26.6 | 0.038 | 2590 | 0.051 | 0.107 | 0.769 |

**Development** (descriptive, after the freeze): ghost reduction 0.53 (FB) / 0.59 (B2); PGG
0.025 → 0.022 and 0.024 → 0.021. Same direction.

## 3. Why the GO must be read narrowly (post-hoc, labelled; `eval/posthoc.json`)

**D1. The harm margins were lenient here.**
- Calibration chose baselines that drop relations almost immediately. They keep only about 5% of
  genuine-gap frames, so the −0.05 *absolute* PGG margin could hardly bind.
- In relative terms the flag removes about a quarter of what persistence remains.
- The ghost-time removed (83–105 s) is small next to the false-termination time the baselines
  already incur (about 2,550 s).
- A setting with longer holds would expose more of the flag's 32% false-positive rate on genuine
  gaps.

**D2. Most scored terminations and recurrences are matching flicker.**
- 69% of the scored "visible terminations" (2,217 / 3,219) and **97%** of scored "recurrences"
  (21,003 / 21,607) occur while the annotated relation itself continues. The tracker-key truth
  breaks only because an endpoint's IoU match to a VidOR box fails momentarily; 96% of VidOR
  person boxes are tool-generated.
- On **genuine** events only, the flag changes STD-g by −0.001 s and RDD by +0.0003 s, which is
  practically zero.
- So O3/O4 confirm "no harm" but were weak tests of termination and recurrence.

**The incident scale is small.** The median suppressed ghost per takeover incident is 2 frames
(example rule (a), 999 candidates). Because the baselines hold so briefly, most of the flag's
effect is many short trims.

## 4. Information vs temporal method

| Comparison | Result |
|---|---|
| C3 (B2+F − FB+F) | J +0.160 [0.135, 0.184]; IGT −13.6 s [−21.2, −7.8]; PGG −0.001 [−0.004, 0.003] |
| Interaction (flag effect on B2 − flag effect on FB) | IGT +22.3 s [9.9, 37.6]; J −0.006 [−0.010, −0.002] |

**Reading.**
- The flag benefits **both** families, and slightly more for the simpler one.
- The B2-over-FB margin exists independently of the flag.
- Nothing here is TAORG-specific inference: the benefit belongs to the **appearance information
  source**, used through a one-line invalidation rule.

## 5. Flag behaviour on the new domain (evaluation missing records of V3; 113,111)

**Coverage:** `a_loc` is defined for 0.83 of missing records.

| Oracle label | Share of records | Flag rate |
|---|---|---|
| TAKEOVER | 19% | **0.58** |
| GAP | 8% | **0.32** |
| FALSE_TRACK | 57% | 0.14 |
| GONE | 14% | 0.11 |
| AMBIGUOUS | 2% | 0.54 |

**Precision** for TAKEOVER is 0.45.

**Compared with MOT17** (appearance audit held-back, frozen θ): recall 0.47–0.56 at 7–12% gap
rejection. On VidOR, recall is similar, but the false-positive rate on genuine gaps is about 3×
higher. **The threshold does not transfer reliably.** This was not retuned.

## 6. Limitations

1. **Oracle relation predictor.** This is annotation-level utility, not predictor-level
   effectiveness.
2. **Previously inspected annotations.** The evaluation split is held back for this study, not
   untouched.
3. **Low-hold baselines** selected at the grid edge make the PGG harm margin nearly non-binding
   (D1).
4. **Tracker-key truth is flicker-dominated** (D2). Termination and recurrence metrics test
   mostly matching artefacts.
5. **Flag false-positive rate** on genuine gaps is 32% in this domain.
6. **Persons only;** 60 short user videos; one detector; one tracker; one re-ID model.
7. **Intervals.** They are video-level percentile bootstrap intervals. Detector, tracker and flag
   fitting variance are excluded (frozen artifacts), and they do not measure generalisation beyond
   VidOR.
8. **Frames were re-encoded to JPEG** (quality 95) for the reused tracker interface. This is
   identical for all methods.

## 7. Three critical reviewer assessments

**Reviewer A (tracking / perception).**
- *Strongest:* a real detector and tracker on real video; a frozen flag transferred unchanged
  across domains; complete run accounting.
- *Weakest:* the flag's 32% false-positive rate on genuine gaps shows that its calibration does
  not transfer. The benefit relies on baselines that already drop gaps.

**Reviewer B (scene graphs).**
- *Strongest:* the identity-ghost error is real and measurable at the relationship level, and a
  simple invalidation removes about three quarters of it.
- *Weakest:* an oracle predictor; and the termination/recurrence metrics are flicker-dominated.
  Nothing yet shows the effect with real predicted relations.

**Reviewer C (methodology).**
- *Strongest:* pre-registration; the freeze pushed before evaluation videos existed locally; the
  gate decided on unflagged exposure only; post-hoc analyses separated.
- *Weakest:* harm margins were set in absolute terms before the baselines' near-zero persistence
  was known. The registered GO is technically correct but substantively modest.

## 8. Single justified next step

Replace the oracle with **one existing VidSGG predictor's cached predictions** on the same frozen
pipeline and evaluation videos, keeping the flag, policy and baselines frozen. This would test
whether the information-source gain survives realistic relation noise.

That requires integrating a scene-graph model, which is a major scope item needing separate
approval. It should also use **identity-stable truth matching** (or a pre-registered flicker
filter), so that termination and recurrence are tested on genuine events.

If this test shows no ghost reduction within harm margins set *relative* to baseline persistence,
the line stops.

## 9. Stop

Stopped after this report and the PR handoff. No new estimator, Hawkes component, expiry heap or
further dataset was started.
