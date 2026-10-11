# TAORG observation-interface audit

> **Part 1** (§1–§5) was written 2026-10-12 **before** any tracker output was produced or
> inspected. **Part 2** (§6 onward) holds the development-sample results.
>
> Frozen and unchanged: Phase 5H-1 NARROW; episode estimator STOP; next-method K1 STOP.

## 1. Hypothesis under audit (not an established explanation)

The next-method study (ledger L55) suggested that the repository's handling of a missing edge
record ("no record ⇒ UNKNOWN") conflates several different situations:
- (a) uncertain visibility while the track identities continue;
- (b) a lost track;
- (c) an unreliable identity association;
- (d) a relationship the predictor or candidate generator simply did not output.

This audit checks:
- **in the code**, where that information is lost or defaulted;
- **on real detector/tracker outputs**, whether inference-time signals can separate a real
  observation gap from an identity-related missing record.

## 2. Current behaviour: where information is lost or defaulted (code references)

| # | Location | Behaviour | Information lost or defaulted |
|---|---|---|---|
| 1 | `src/taorg/schemas/edges.py:57` (`EdgeObservation.observed`) | `observed=False` means "an absence record constructed for temporal processing". `predicate_score` is 0.0 for absences by convention | Nothing distinguishes "the predictor **evaluated** this triple and scored it low" (negative evidence) from "the triple was **never evaluated**" (not a candidate). A missing score becomes 0.0 |
| 2 | `src/taorg/schemas/edges.py:86–106` (`edge_visibility`) | Edge visibility = conservative combination of the subject and object `VisibilityState` | Only three states. No source (tracker, heuristic, oracle), no uncertainty, no reason. `UNKNOWN` covers both "no estimate" and "estimate unreliable" |
| 3 | `src/taorg/temporal/gated_ema.py:162–168` | A key with no record in the frame is treated as `VisibilityState.UNKNOWN` (comment: "Edge not in current frame at all — use UNKNOWN") | **"No record" is collapsed into UNKNOWN visibility.** The same default appears in `va_hmm.py:90`, `hsmm.py:100`, `episode.py:113`, `gated_variants.py:116`, `cure.py:40–41` (`flag_of`) and `vhped.py` (docstring l.84) |
| 4 | `src/taorg/schemas/tracks.py:59–67` (`TrackObservation`) | The schema already carries `detector_confidence`, `tracker_state` (TRACKED / LOST / TENTATIVE / REMOVED), `visibility_state` and `occlusion_ratio` | **No temporal method reads any of these** (grep of `src/taorg/temporal`: only the identity modules `identity.py` / `vhped.py:201` read boxes). Track-level evidence is available in the schema and discarded by every method |
| 5 | `src/taorg/noise/id_switch.py:131–190` | Swaps track IDs in both track and edge records | In the synthetic worlds, an identity swap makes the true key's records **disappear** (renamed to the swapped key). Methods then see "no record ⇒ UNKNOWN" for the true key, and "detections" for a ghost key |
| 6 | `src/taorg/data/synthetic_world.py:23`; `episode_world.py` `render` | Every key gets a record in every frame (absence = VISIBLE, `observed=False`) | Synthetic data never produces "not evaluated". Real VidSGG outputs are top-K or thresholded, so most absent triples are not evaluated at all. The synthetic convention hides that case |
| 7 | `src/taorg/noise/visibility_noise.py:14,101` | The UNKNOWN corruption is applied to the flags of *absence records* | UNKNOWN in the S-conditions means "visibility estimate missing". UNKNOWN from (3) means "no record". **The same token carries two meanings** (the K1 failure, ledger L55) |
| 8 | `src/taorg/adapters/__init__.py` (empty) | No real detector, tracker or VidSGG adapter exists | Nothing in the repository converts real outputs into the contract, so the conflation has never been confronted with real data |

**Conclusion of the code trace** (an observation, not yet an empirical claim): the conflation is
real at the implementation level.
- Three distinct conditions map to one token: "no record", "visibility UNKNOWN" and "evaluated
  negative / not evaluated".
- The available track-state fields are ignored.

Whether *real inputs* contain enough information to separate them is the empirical question
below.

## 3. Real-output sample (sampling rule fixed before any output was produced)

**Data**

| Item | Detail |
|---|---|
| Archive | MOTChallenge **MOT17Labels.zip**: https://motchallenge.net/data/MOT17Labels.zip |
| Size and checksum | 10,107,022 bytes; SHA-256 `0aa79322e91583369f42f17c4d79a0b145380d8732487bba59272048dc82b2b9` |
| Licence | CC BY-NC-SA 3.0 (non-commercial research use) |
| Contents | Benchmark ground truth (identities, per-box visibility fraction, class / ignore flags) and **public detections** for every sequence. No images |

**Detectors.** The public detection sets are real outputs of established detectors, distributed
with the benchmark:
- **FRCNN** (Faster R-CNN): primary;
- **SDP**: secondary, used as a detector-shift check;
- **DPM** is **excluded** by rule: its scores (−0.5 to 3.1) are not probabilities in [0, 1], which
  ByteTrack's thresholds assume.

**Tracker.** Roboflow `trackers` **2.6.1** (Apache-2.0) **ByteTrackTracker** with library
defaults: lost buffer 30 frames at 30 fps (scaled by frame rate), activation 0.7, 2 consecutive
frames to confirm, IoU 0.1, high-confidence 0.6. Dependency: `supervision` 0.30.9. It runs on
CPU, on the boxes only (no images).

**Sequences.**
- All 7 MOT17 **train** sequences (the only ones with public ground truth): 02, 04, 05, 09, 10,
  11, 13. **No clip selection.**
- Seeded split by sequence (`random.Random(20261012).sample`, k = 3):
  - **development {02, 09, 11}**;
  - **untouched evaluation {04, 05, 10, 13}**.
- The split is by video, so adjacent frames never cross splits.

**Required phenomena** (to be counted on development data, and reported if absent):
- temporary detection loss;
- objects leaving and re-entering the frame;
- annotated occlusion (ground-truth visibility);
- ordinary motion;
- identity switches and ambiguous associations (tracker vs ground truth).

**Scope.** No scene-graph predictor is integrated. This study is **detector/tracker-level only**,
and **no relationship-performance results are produced**.

## 4. Information classes (kept separate throughout)

1. **Directly available at inference** (automatic outputs):
   - detections (box, confidence);
   - tracker outputs (`tracker_id` for matched detections; −1 for unconfirmed);
   - the tracker's internal tracklet list after each update: tracks kept in the lost buffer,
     their `time_since_update`, and their **Kalman-predicted box**. This internal state is
     accessible in code but is not part of the documented API; that is declared;
   - image size and frame rate (`seqinfo.ini`).
2. **Derived heuristics** (inference-time, from class 1 only; thresholds fixed here, a priori):
   - **EXIT**: ≥ 50% of the lost track's predicted box lies outside the image, or its centre does.
   - **IDENTITY_SUSPECT**: a confirmed track Y ≠ X, **first output after X's last detection**,
     is detected with IoU ≥ 0.5 against X's predicted box. The reading is that X's object probably
     continues under a new identity.
   - **OCCLUSION_EVIDENCE**: a confirmed track Y ≠ X that **already existed** when X was last
     detected is detected and covers ≥ 30% of X's predicted box (intersection over X's area). The
     reading is that X is probably hidden behind Y.
   - Precedence: EXIT > IDENTITY_SUSPECT > OCCLUSION_EVIDENCE > **UNSPECIFIED**.
   - **TRACK_TERMINATED**: the tracker has deleted X (lost buffer exceeded). This is directly
     available, not a heuristic.
   - **No association-confidence signal is available from ByteTrack.** None is invented.
3. **Ground truth / oracle** (evaluation only; never an input except in the labelled oracle
   diagnostic): the MOT17 benchmark's manual annotations. These are identities, boxes, the
   per-box visibility fraction, and consider/ignore flags, restricted to class 1 pedestrians.

## 5. Oracle categories and the pre-stated observability criteria

**Matching.** Per frame, confirmed tracker outputs are matched to considered ground-truth
pedestrians by Hungarian assignment on IoU ≥ 0.5.
- g(X) is the most frequent ground-truth identity matched to track X.
- X has no g(X) if it is never matched; it is then a **FALSE_TRACK**.

**Absence frames of X.** Frames after X's first detection where X is not output, up to 3 s
after X's last detection.

Each absence frame is labelled (oracle):

| Category | Definition | Correct action for X's record |
|---|---|---|
| FALSE_TRACK | X has no ground-truth identity | drop |
| GONE | g(X) has no considered ground-truth box at t (exited or ended) | drop |
| TAKEOVER | g(X) is matched at t to a different output Y ≠ X (identity fragmentation or switch) | drop: holding X duplicates the object |
| OCCLUDED_GAP | g(X) is present, unmatched, with visibility < 0.5 | hold: real observation gap |
| VISIBLE_MISS | g(X) is present, unmatched, with visibility ≥ 0.5 | hold: real observation gap (detector or association miss) |

**Pre-stated criteria (fixed before the run)** for a **usable inference-time distinction** between
real gaps (OCCLUDED_GAP + VISIBLE_MISS) and identity-related missing records (TAKEOVER), on the
development sample:
- **U1:** IDENTITY_SUSPECT precision for TAKEOVER ≥ 0.70, *and* recall ≥ 0.50.
- **U2:** fewer than 10% of real-gap frames are labelled IDENTITY_SUSPECT or EXIT (a drop-type
  label), so holding is not harmed.
- **U3:** the result holds for both detector sets (FRCNN and SDP), each meeting U1 and U2 on the
  pooled development frames.

| Outcome | Rule |
|---|---|
| **Usable** | U1–U3 all met |
| **Partial** | U2 met, and U1 met for one detector or precision ≥ 0.70 with recall ≥ 0.30 |
| **Not usable** | Otherwise. The task then stops after reporting, with no downstream comparison |

Coverage (the share of absence frames labelled UNSPECIFIED) and the GONE-vs-EXIT agreement are
reported descriptively.

---

*(Part 2, development results: §6 onward.)*
