# TAORG observation contract (minimal, backward-compatible)

> **Status:** proposed and implemented in `src/taorg/schemas/observation.py` (2026-10-12).
> The legacy path is untouched.
> - `EdgeObservation`, `TrackObservation` and every temporal method are unchanged.
> - Historical experiments reproduce bit-for-bit.
> - **The meaning of `VisibilityState.UNKNOWN` is not changed.**
>
> The new records are *additional* types. Adapters may emit them; a converter produces legacy
> records exactly as today.

## 1. Principles

1. **Four concepts stay separate:**
   - **detector confidence**: the per-detection score;
   - **track-association reliability**: whether the identity link is trustworthy;
   - **visibility**: whether the object can be seen;
   - **relationship existence**: the scene-graph claim.

   No field stands in for another.
2. **Missing is not negative.** A relationship score that was not evaluated is `None`, not 0.0.
3. **Every inferred field records its source:** `NONE`, `TRACKER` (direct), `HEURISTIC`
   (derived) or `ORACLE` (ground truth; evaluation and diagnostics only).
4. **Unspecified is a value.** `MissingReason.UNSPECIFIED` is the default, and it is what legacy
   data maps to.
5. **No invented reliability.** `association_confidence` stays `None` when the tracker does not
   provide it (ByteTrack does not). Any proxy must be a separately named, separately evaluated
   field.

## 2. Track-level record (`TrackRecord`)

| Field | Type | Meaning | Source in the pilot |
|---|---|---|---|
| `track_id`, `frame_idx`, `timestamp` | int, int, float | identity and time | tracker; sequence |
| `detected` | bool | a detection was associated with this track this frame | tracker output |
| `detector_confidence` | float or None | the associated detection's score; None if not detected | detector |
| `bbox` | 4-tuple or None | the associated detection's box | detector |
| `predicted_bbox` | 4-tuple or None | the tracker's motion prediction while undetected | tracker internal state (declared) |
| `tracker_state` | `TrackerState` | TRACKED / LOST (in buffer) / TENTATIVE / REMOVED (deleted) | tracker |
| `time_since_detection_s` | float or None | seconds since the last associated detection | derived from tracker output |
| `association_confidence` | float or None | identity-link reliability, **if the tracker provides it** | **None** (ByteTrack provides none) |
| `visibility` | `VisibilityState` | visibility evidence | `UNKNOWN` unless a source exists |
| `visibility_source` | `EvidenceSource` | NONE / TRACKER / HEURISTIC / ORACLE | NONE |
| `missing_reason` | `MissingReason` | why the track is undetected (only when `detected=False`) | HEURISTIC or TRACKER, else UNSPECIFIED |
| `missing_reason_source` | `EvidenceSource` | provenance of `missing_reason` | — |

## 3. Relationship-level record (`RelationRecord`, for future VidSGG adapters)

| Field | Meaning |
|---|---|
| `edge_key`, `frame_idx`, `timestamp` | as in `EdgeObservation` |
| `evaluated` | `True` = the predictor scored this triple; `False` = not a candidate; `None` = unknown (legacy) |
| `score` | the predicate score **if evaluated**; `None` otherwise (never a fabricated 0.0) |
| `positive_evidence` | `True` / `False` / `None`: the predictor's own decision, if it has a threshold |
| `missing_reason` | `MissingReason` for the triple (e.g. `NOT_EVALUATED`, `EVALUATED_NEGATIVE`, or a reason inherited from an endpoint track) |
| `subject_track`, `object_track` | references to the endpoints' `TrackRecord`s for that frame |

## 4. `MissingReason` values

| Value | Meaning | Who may set it |
|---|---|---|
| `UNSPECIFIED` | no reason known (legacy default) | — |
| `NOT_EVALUATED` | the triple was not scored (relationship level) | predictor adapter |
| `EVALUATED_NEGATIVE` | the triple was scored below the predictor's decision threshold | predictor adapter |
| `TRACK_TENTATIVE` | the endpoint track is not yet confirmed | tracker |
| `TRACK_LOST` | the endpoint track is in the tracker's lost buffer (no further reason) | tracker |
| `TRACK_TERMINATED` | the tracker deleted the endpoint track | tracker |
| `OCCLUSION_EVIDENCE` | evidence that the object is hidden (e.g. another track covers its prediction) | heuristic / visibility estimator |
| `OUT_OF_VIEW_EVIDENCE` | evidence that the object left the image | heuristic |
| `IDENTITY_SUSPECT` | evidence that the object continues under another identity | heuristic / re-ID |

## 5. Legacy conversion (exact)

`to_legacy_edge` (and the track-presence stream used in the pilot):
- `detected=True` → `observed=True`, `predicate_score=detector_confidence`.
- `detected=False` → **no record**, which is today's behaviour, so legacy methods see `UNKNOWN`.

Legacy experiments never construct the new types. `tests/unit/test_observation_contract.py`
checks:
- the legacy conversion reproduces the existing stream;
- default values are UNSPECIFIED / None;
- UNKNOWN is unchanged.

## 6. What the contract does *not* do

- It does not make any distinction observable. It only stops the interface from destroying the
  information when an upstream component has it.
- Whether real trackers provide such information is the subject of
  `research/taorg_observation_audit.md`.
- Representing the information is an interface fix, **not an algorithm contribution**.
