# TAORG observation-interface audit — Exit report

> Date 2026-10-12, branch `codex/taorg-observation-audit`.
>
> **The real outputs processed:**
> - the public detections of real detectors (Faster R-CNN, SDP) distributed with MOT17;
> - tracked by an established tracker (ByteTrack, `trackers` 2.6.1);
> - on 3 development sequences.
>
> **No scene-graph predictor was run, and no relationship-performance result exists.**
> Frozen and unchanged: Phase 5H-1 NARROW; episode estimator STOP; K1 next-method STOP.

## Summary

| Question | Answer |
|---|---|
| Was the UNKNOWN conflation a real implementation limitation? | **Yes.** Code trace (audit §2): "no record", "visibility UNKNOWN" and "not evaluated / evaluated negative" collapse into one token in every temporal method. The track-state fields in the schema are ignored. On real tracker outputs, 20% of absence frames are identity takeovers and 41% are real gaps, all presented identically |
| Can inference-time information resolve it? | **No, not with these inputs.** Pre-stated verdict: **NOT USABLE**. Identity-suspect recall is 0.1–1%, and 26–29% of real-gap frames receive drop-type labels |
| Does any benefit survive imperfect association signals? | Not tested: no comparison was justified. ByteTrack reports no association confidence, and none was invented |
| Is any gain due to the contract, a standard rule, or an algorithm? | There is no measured gain. The contract is an **interface fix** that preserves information when an upstream component has it. It is **not an algorithm contribution** |
| Does the fixed buffer still match the more elaborate approach? | Exploratory evidence: elapsed time is the only signal better than chance (AUC 0.62–0.69). That is exactly the information a fixed buffer uses, so nothing in these inputs suggests a richer rule could beat it |
| What remains impossible without better perception? | Telling an identity change from an occlusion, and an exit from a border occlusion. Doing so needs appearance / re-ID evidence or a tracker that reports association confidence |
| **Algorithm contribution justified?** | **No.** |

## 1. Workspace and provenance

- **Workspace.** `D:\ADSA project`; `taorg` imports from D: (verified). The old C: copy was not
  touched.
- **Branch.** Created from `codex/taorg-next-method` @ `ee65631` (PR #4); the PR is stacked on #4.
- **Commits:**

  | # | Content | Commit |
  |---|---|---|
  | 1 | Code-trace audit, sampling rule, pre-stated criteria and contract, before any tracker output | `bbf545e1e59b240d5809e0a8252ab84103f29452` |
  | 2 | Tested adapter/oracle and development audit results | `550f9f3b2584c3db2f8955ca9f64db0f7d6058d4` |
  | 3 | This report, audit part 2, records | the commit containing this file |

  **There is no pre-registration commit and no parameter freeze.** The pre-stated rule (audit §5)
  forbids a downstream comparison when the distinction is not usable. The user's instructions
  agree: "If real inputs do not distinguish the cases, report that limitation and stop."

## 2. Data, models and versions (`results/observation/manifests/sources.json`)

**Data.** MOT17Labels.zip:
- motchallenge.net, 10,107,022 bytes;
- SHA-256 `0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9`;
- CC BY-NC-SA 3.0;
- downloaded with the user's approval in chat.
- Contents: ground truth plus public detections. **No images were downloaded or redistributed.**

**Detectors.** The public detection sets FRCNN and SDP. DPM was excluded by rule: its scores are
not probabilities.

**Tracker.** `trackers` 2.6.1 (Apache-2.0), ByteTrack with library defaults; `supervision`
0.30.9. CPU only, about 2 s per sequence.

**Split.** Seeded by sequence, before any output was produced:
- development {02, 09, 11};
- untouched evaluation {04, 05, 10, 13}, never processed.

**Personal local videos.** Present on the machine but deliberately not used (privacy; unsuitable).

## 3. Information classes (kept separate)

1. **Automatic outputs:**
   - detections;
   - tracker IDs;
   - the tracker's internal lost-buffer state and Kalman predictions (internal; declared).
2. **Derived heuristics, with thresholds fixed a priori:**
   - EXIT;
   - IDENTITY_SUSPECT;
   - OCCLUSION_EVIDENCE.
3. **Oracle (evaluation only):** categories derived from the MOT17 benchmark's manual annotations
   (identities, visibility). No new manual annotation was made; the categorisation rules are the
   "annotation instructions" (audit §5).

## 4. Results (development; details in audit §6)

- **Conflation frequency** (39,534 absence frames): real gaps 41%, identity takeovers 20%, object
  gone 26%, false tracks 13%.
- **Pre-stated criteria:**
  - U1 fails for both detectors: identity-suspect precision 0.04 / 0.56, recall 0.001 / 0.010.
  - U2 fails for both detectors: 29% / 26% of real-gap frames are labelled drop-type.
  - **NOT USABLE.**
- **Exploratory separability:**
  - time since detection is the best single signal (AUC 0.69 / 0.62);
  - geometric identity cues are at chance (IoU with a newer track: 0.53);
  - the tracker's buffer state is inversely informative (0.35 / 0.41).
- **Coverage.** No heuristic fired on 19–21% of absence frames.
- **Absent phenomenon.** Objects leaving and re-entering under the same ground-truth identity do
  not occur in the development sequences.
- **Examples.** A successful occlusion example exists. **No successful identity example exists.**
  Failures are dominated by the drifting Kalman prediction being labelled "out of view".

Figures (`results/observation/plots/`):
1. reason vs oracle confusion;
2. single-signal AUCs;
3. composition vs elapsed time;
4. example timelines.

## 5. The contract (`research/taorg_observation_contract.md`, `src/taorg/schemas/observation.py`)

**What it adds.** Additional types:
- `TrackRecord` and `RelationRecord`;
- `MissingReason`, which has an explicit `UNSPECIFIED` value;
- `EvidenceSource` (NONE / TRACKER / HEURISTIC / ORACLE).

**What it keeps separate:** detector confidence, association reliability (`None` when not
provided), visibility, and relationship existence.

**Legacy path unchanged.** `EdgeObservation` fields, every method, and the meaning of UNKNOWN are
untouched; this is tested.

**Contribution.** An interface fix. It stops the representation from destroying information,
but it **cannot create information the tracker does not provide**.

## 6. Tests

**326 passed** (`python -m pytest -q`): 316 existing plus 10 new.
- 4 contract tests: legacy unchanged, explicit defaults, invalid states rejected.
- 6 adapter/oracle tests:
  - heuristic precedence;
  - record states and times;
  - no association confidence invented;
  - the inference module contains no ground-truth path (AST check);
  - oracle categories;
  - deterministic real-sequence run.

## 7. Limitations

1. Three development sequences, one tracker family, pedestrians only, and box-only tracking (no
   appearance cues).
2. Oracle categories rely on IoU ≥ 0.5 matching and the benchmark's visibility estimates.
3. The heuristic thresholds were fixed a priori and were not tuned. Tuning could move individual
   numbers, but the exploratory AUCs bound what any threshold on these signals can do.
4. ByteTrack's internal state was read through undocumented attributes.
5. This is relevance-by-analogy to TAORG: the relationship layer inherits track-level missing
   reasons, but no relationship predictor was run.

## 8. Three critical reviewer assessments

**Reviewer A (tracking).**
- *Strongest:* real benchmark detections and an established tracker; categories from manual
  annotations; a pre-stated decision rule; a negative result accepted.
- *Weakest:* ByteTrack is motion-only. Trackers with appearance embeddings (BoT-SORT with re-ID,
  DeepSORT) or with association scores might make identity takeovers observable. The conclusion
  is specific to motion-only association.
- *Missing:* a re-ID-capable tracker.

**Reviewer B (scene graphs).**
- *Strongest:* it shows concretely that TAORG's temporal layer discards track state, and that the
  "UNKNOWN" token conflates distinct situations.
- *Weakest:* no relationship predictor was run, so the relationship-level "not evaluated" vs
  "evaluated negative" distinction, arguably the most important one, is still untested.
- *Missing:* outputs of one VidSGG model with per-triple scores.

**Reviewer C (methodology).**
- *Strongest:*
  - the sampling rule and the usable-distinction criteria were committed before any output
    existed;
  - the evaluation sequences were never touched;
  - the exploratory analysis is labelled and not used to rescue the result.
- *Weakest:*
  - with only 3 development sequences, U1/U2 rest on pooled frames, which are correlated within
    tracks;
  - however, the effect sizes (recall ≤ 1%) are far from the thresholds.
- *Convinced?* Yes, that the distinction is not observable from these inputs.

## 9. The single next experiment needed

Repeat this pre-stated audit with an **appearance-based tracker that exposes association
evidence**: BoT-SORT with re-ID embeddings (available in `trackers` as `BoTSORTTracker`, but it
needs frames and a re-ID model), or OC-SORT plus an explicit appearance distance.
- **Cost:** MOT17 frames (the MOT16.zip image archive, 1.95 GB) and a re-ID model. This needs
  approval.
- **Questions it answers:**
  - Does re-ID distance separate TAKEOVER from real gaps (U1/U2)?
  - Only if it does: is a downstream comparison of B2 vs B2 + reason rule vs fixed buffer worth
    pre-registering?

**Alternative.** Integrate one VidSGG model to audit "not evaluated" vs "evaluated negative" at
the relationship level.

## 10. Stop

Stopped after the observation audit and this report, as instructed. Not started:
- a downstream comparison;
- a new estimator;
- a Hawkes mechanism, an online learner, or heap optimisation;
- further downloads or model integration.
