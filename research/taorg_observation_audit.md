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

## 6. Development results (dev sequences 02, 09, 11 × FRCNN / SDP; `results/observation/dev_audit/`)

### 6.1 Required phenomena (`summary.json` → `phenomena`)

| Phenomenon | Present? | Evidence |
|---|---|---|
| Temporary detection loss, resumed with the same identity | yes | 22–781 resumed absence segments per unit |
| Annotated occlusion | yes | 2,171–11,293 ground-truth boxes with visibility < 0.5 per sequence |
| Ordinary motion without identity change | yes | 10–36 ground-truth persons tracked by exactly one track ID |
| Identity switches / fragmentation | yes | 14–45 ground-truth persons matched to ≥ 2 track IDs; 7,778 TAKEOVER absence frames |
| **Leaving and re-entering the frame** | **absent** | No considered ground-truth identity has two separate presence segments in the dev sequences (MOT convention: a returning person is usually annotated with a new ID). It is reported as absent, not engineered |

### 6.2 How common the conflation is (observation)

| Category | Absence frames | Share |
|---|---|---|
| OCCLUDED_GAP | 14,622 | |
| VISIBLE_MISS | 1,505 | |
| **Real gaps (subtotal)** | **16,127** | **41%** |
| **TAKEOVER** | **7,778** | **20%** |
| GONE | 10,466 | 26% |
| FALSE_TRACK | 5,163 | 13% |
| **Total** | **39,534** | |

Under the legacy contract, all of these reach a temporal method as the same "no record ⇒ UNKNOWN".
**The conflation is real and frequent on real tracker outputs.**

### 6.3 Pre-stated criteria (`summary.json` → `criteria`)

| Detector | IDENTITY_SUSPECT precision / recall (TAKEOVER) | Real-gap frames given a drop-type label | U1 | U2 |
|---|---|---|---|---|
| FRCNN | 0.039 / 0.001 | 29.2% | fail | fail |
| SDP | 0.559 / 0.010 | 25.6% | fail | fail |

# **Verdict: NOT USABLE.**

The available inference-time information does not reliably distinguish real gaps from
identity-related missing records. By the pre-stated rule, **no downstream comparison is run**,
and the untouched evaluation sequences (04, 05, 10, 13) were **never processed**.

### 6.4 What the other signals do (observations)

- **OUT_OF_VIEW_EVIDENCE** is recall-oriented for GONE (0.65 / 0.61), but its precision is low
  (0.46 / 0.37). It is also assigned to 30% / 25% of OCCLUDED_GAP frames. *Explanation
  (hypothesis):* the Kalman prediction of a lost track keeps extrapolating its last velocity and
  drifts off-image, whether or not the person actually left.
- **OCCLUSION_EVIDENCE** is the majority label for real gaps (precision for hold 0.66 / 0.55),
  but it also covers 21–38% of TAKEOVER frames. Another track covering the predicted box looks the
  same whether it is an occluder or the same person under a new ID.
- **The tracker's own state** (in buffer vs deleted) is *inversely* related to correctness: the
  AUC is 0.35–0.41 (`separability.json`). Many real gaps outlast the 30-frame buffer.
- **Unspecified coverage** (no heuristic fired): 21% / 19% of absence frames.

### 6.5 Exploratory separability (defined after the verdict; not a decision input)

Single-signal AUC for TAKEOVER vs real gap (`separability.json`, Fig. 2):

| Signal | FRCNN | SDP |
|---|---|---|
| Time since detection | 0.69 | 0.62 |
| Predicted box outside image | 0.66 | 0.56 |
| IoU with a newer track | 0.53 | 0.53 |
| IoU with an older track | 0.38 | 0.43 |
| Still in buffer | 0.35 | 0.41 |
| Covered by another track | 0.34 | 0.36 |

The composition of missing records vs elapsed time (`hold_vs_time.csv`, Fig. 3): the real-gap
share falls from 0.75 at ≤ 0.1 s to 0.32 at 2–3 s (FRCNN), and from 0.63 to 0.27 (SDP), while the
takeover share rises.

**Interpretation (hypothesis):**
- Elapsed time is the only usable cue, and it is exactly what a fixed buffer already uses.
- Box geometry from a motion-only tracker carries almost no identity information.
- Separating identity changes from gaps needs appearance / re-identification evidence or a
  tracker that reports association confidence. Neither is available here.

### 6.6 Representative examples (predefined rule; `examples.json`, Fig. 4)

| Example | Track | Outcome |
|---|---|---|
| success_occlusion | MOT17-02-FRCNN track 4, frames 228–243 | All 16 frames oracle OCCLUDED_GAP and labelled OCCLUSION_EVIDENCE |
| **success_identity** | **none found** | No absence segment of ≥ 10 frames was majority TAKEOVER *and* majority IDENTITY_SUSPECT |
| fail_takeover | MOT17-02-FRCNN track 20, frames 159–248 | 86/90 frames TAKEOVER, but 77 labelled OUT_OF_VIEW (drifting prediction). 88 such segments exist |
| fail_realgap | MOT17-02-FRCNN track 0, frames 207–296 | All 90 frames OCCLUDED_GAP, but 66 labelled OUT_OF_VIEW. 60 such segments exist |

## 7. Which distinctions can and cannot be made reliably (this sample, these inputs)

**Can be represented** (contract):
- detected vs not detected;
- time since detection;
- tracker lost-buffer vs deleted;
- predicted box;
- "reason unspecified".

**Can be inferred only weakly** (≈ 0.6–0.7 AUC):
- that an absence is more likely an identity change or an exit the longer it lasts.

**Cannot be inferred reliably:**
- identity takeover vs occlusion;
- leaving the image vs occlusion at the image border (the drifting prediction);
- association reliability (not reported by ByteTrack).

**Untested here:**
- relationship-level "not evaluated" vs "evaluated negative" (no scene-graph predictor);
- appearance-based re-identification;
- trackers that report association scores.
