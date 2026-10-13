# TAORG downstream study — Data-suitability report

> Written 2026-10-12 on branch `codex/taorg-downstream` (from `codex/taorg-appearance-audit`
> @ `b85ffba`). No comparative analysis and no video download had happened when this was written.

## 1. The question this data must support

> Does the **frozen** appearance-based takeover flag (`results/appearance/freeze/frozen_models.pkl`,
> SHA-256 `98d7e46a…5f5a`, model `V3|T+A` with its θ) reduce relationship ghosts without
> unacceptable false termination, beyond strong baselines that receive identical information?

## 2. Requirements and what was verified

| Requirement | MOT17 (local) | VidOR annotations (local, `data/vidor/`) |
|---|---|---|
| Video frames | yes (local) | **no.** Videos are hosted in zips on the HuggingFace mirror `shangxd/vidor`; byte ranges are supported (verified), so selective fetch is possible |
| Persistent object identities | yes (persons) | **yes.** Per-frame boxes with a persistent `tid` (verified in the JSON) |
| Temporally resolved relationship labels | **no.** Person boxes only. Relationships must **not** be inferred from proximity | **yes.** `relation_instances` with `begin_fid` (inclusive) / `end_fid` (exclusive), 50 predicates |
| Genuine termination and recurrence | — | **yes.** Person–person relation keys recur: 33% of keys in validation (2,442 / 7,505), 33% in training (22,163 / 67,306) |
| Identity ambiguity and observation gaps | — | **Plausible, but not established until tracking runs.** 33% of persons in videos with person–person relations have an annotated disappear/reappear gap under the same `tid` (validation 656 / 2,009). Median 2 persons per video; 342 validation / 2,922 training videos have ≥ 3 persons |
| Development/evaluation separation | — | **Video-level split possible** (official training vs validation). **But see §3** |

**Verified annotation facts** (read directly from the JSONs, all videos):
- validation: 835 videos; 556 have person–person (pp) relations (12,333 instances);
- training: 7,000 videos; 4,835 have pp relations (109,322 instances);
- pp predicates are dominated by spatial and attention predicates: `in_front_of`, `next_to`,
  `watch`, `behind`, `away`, `towards`. `hold_hand_of`, `hug`, `push` etc. are rare;
- **96% of person boxes are `generated: 1`**, i.e. produced by the annotation tool's tracker
  (linear / KCF / MOSSE) and not hand-drawn per frame (official page; verified flag counts);
- resolution is mostly 640×480 or 640×360 (some 320×240); median duration about 28 s;
  fps about 30.

## 3. Prior exposure (must be disclosed)

**VidOR annotations were already used in Phase 5H Stage 0 and Phase 5H-1:**
- training videos were used for fitting and selection;
- validation (835) was used as Stage 0 TEST and 5H-1 EVAL-B, at aggregate log-likelihood level.

**Every annotated VidOR video has therefore been inspected by an earlier TAORG study.**
- No VidOR video, detection or tracker output has ever been processed.
- The relationship **annotations** are not untouched.
- A validation split held back from *this* study is **held-back for this study, not untouched
  confirmation**.

The VidOR test split (2,165 videos) has no public annotations.

## 4. Other candidates considered (not inspected; not chosen)

| Candidate | Why not chosen for the smallest viable study |
|---|---|
| **Action Genome** (Charades) | Sparse sampled frames; mostly one actor; the PRD/Memory record it as too sparse for termination dynamics. No multi-person identity ambiguity |
| **VidVRD** | Relations are given per 30-frame segment (coarse timing); ImageNet-VID videos |
| **PVSG** | Pixel-level panoptic labels and a much larger download; licence would need checking. More scope than needed |
| **MOT17 / MOT20 alone** | No relationship labels |

## 5. Smallest viable study

**Scope: person–person relationships only.** The frozen flag uses a *person* re-ID model, so it
is meaningful only for person endpoints. Restricting both endpoints to persons keeps the flag's
domain coherent and avoids a multi-class detection problem.

**Detections.** VidOR has no public detections. One COCO-pretrained detector would run on the
frames, keeping only `person` detections:
- torchvision `fasterrcnn_resnet50_fpn_v2`, COCO_V1 weights;
- 175 MB; COCO val box mAP 46.7;
- BSD-3 code; trained on COCO 2017.

COCO and YFCC100M are both Flickr-sourced. Accidental image overlap with VidOR videos cannot be
excluded, but the detector is not trained on VidOR.

**Tracker and appearance.** The appearance audit's V3 pipeline, unchanged:
- BoxMOT `BotSort` with appearance, the identical settings and the OSNet x1.0 MSMT17 cache;
- the frozen T+A model and θ.

**Relationship observations.** This is a *controlled downstream test on real annotations*, with
real-tracker identity corruption:
- an **oracle relation predictor** applied to tracker outputs;
- a relation (X, Y, p) between tracker tracks is "observed" at frame t iff X and Y are both
  output at t, each is matched (IoU ≥ 0.5) to an annotated person at t, and the annotated relation
  between those persons is active at t;
- there is no synthetic corruption: gaps and identity takeovers come from the real detector and
  tracker;
- **this is not end-to-end scene-graph validation.** The predictor is perfect whenever both
  endpoints are tracked.

**Sample** (annotation-only rule; seeds fixed now):
- pool: videos with ≥ 3 annotated persons, ≥ 1 pp relation and duration ≤ 60 s;
- **development:** 30 of 2,368 training-pool videos (`random.Random(20261013)`): 22,017 frames,
  796 pp relations, **78.5 MB** of video across all 8 training zips;
- **evaluation (held back for this study):** 60 of 263 validation-pool videos
  (`random.Random(20261014)`): 38,411 frames, 1,807 pp relations, **137.7 MB**. Fetched only after
  the freeze.

**Downloads (need approval):**

| Item | Size | Licence / terms |
|---|---|---|
| 90 VidOR videos (selective range fetch) | ≈ 216 MB | The official page gives no licence; it asks for citation (Shang et al. 2019; Thomée et al. 2016). Videos are YFCC100M Flickr uploads under their uploaders' Creative Commons licences. The mirror card says CC-BY-SA-4.0. Research use, no redistribution of videos |
| torchvision Faster R-CNN v2 COCO weights | 175 MB | BSD-3 (torchvision); COCO annotations CC BY 4.0, images under Flickr terms |

**Resources** (estimates, to be measured):
- detection about 15–25 frames/s on the RTX 4060 at 640×480, so about 1 h for 60k frames;
- ReID and tracking add about 40 ms/frame;
- disk under 2 GB on D:;
- free RAM is only about 1 GB, so processing is one video at a time with no parallel jobs.

## 6. Limitations known before starting

1. **Annotations previously inspected** (§3). Held-back ≠ untouched.
2. **Oracle relation predictor.** Results show *annotation-level utility* of the flag, not
   predictor-level effectiveness.
3. **VidOR boxes are mostly tool-generated.** Matching tracker outputs to annotated persons
   inherits their imprecision.
4. **Domain shift for the frozen flag** (MOT17 street crowds → handheld user video, a different
   detector, few persons). This is the point of the test, but the earlier **threshold-transfer
   weakness** (θ rejected 7–27% of real gaps across splits) is expected to recur. Coverage and
   false-positive behaviour will be reported.
5. **Spatial predicates dominate.** Many pp relations (`next_to`, `in_front_of`) persist while both
   people are visible. Takeover ghosts arise only when a person's track identity changes.
6. **Identity ambiguity may be scarce.** With few persons per video, takeovers may be rare. A
   development feasibility gate on the minimum number of affected relation incidents will decide
   whether the evaluation runs at all.

## 7. Recommendation

VidOR is the only locally present relationship dataset that meets every requirement. It needs
about 0.4 GB of downloads. Proceed **only with approval** of both downloads, under the labels:
- **controlled downstream test on real annotations**;
- **held-back for this study, previously inspected annotations**.

If identity takeovers prove too rare on development (the gate), stop.
