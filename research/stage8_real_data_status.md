# Stage 8 — Real-Data Experiments: Status

> Date: 2026-10-28. **No real-data experiment has been run.** Nothing in this
> repository reports a real-data number.

## Availability check (web, 2026-10-28)

| Requirement | Status | Source |
|---|---|---|
| Action Genome annotations on this machine | **Absent** (searched Desktop/Downloads/Documents) | local search |
| Charades videos (needed only for base-model runs) | Hosted by AI2; **Non-Commercial Use License** | [allenai.org/data/charades](https://allenai.org/data/charades) |
| Action Genome annotation license | **Not verified** — mirrors of the AG README describe placing annotations under `dataset/ag/annotations`; license text not found in search results | AG GitHub README (mirror) |
| ActionGenome4D (WSGG, arXiv 2603.13185) | Announced in a March 2026 preprint; availability/licence not checked | [arXiv](https://arxiv.org/abs/2603.13185) |
| Base-model predictions (STTran / DSG-DETR / TEMPURA) | Not available; would need checkpoints + Charades frames + GPU inference | — |
| Real tracker outputs with association scores (for IA-VHPED) | Not available. ByteTrack provides detection scores and association by IoU; an identity-continuity score would have to be derived (e.g. matched-box IoU or similarity) — **not yet defined or validated** | — |

## Compatibility with the current adapters

* `src/taorg/data/action_genome.py`: format **assumed**; run `scripts/inspect_action_genome.py`
  first. It emits **no TrackObservations**, so IA-VHPED reduces exactly to V-HPED on AG
  ground-truth streams (identity is perfect there by construction). IA-VHPED's claims can only
  be tested on real data with a real tracker.
* `src/taorg/adapters/precomputed.py` (JSONL) carries visibility per track but **not**
  `identity_confidence` yet; adding an optional `"identity_confidence"` field to `frame` lines is
  a small, backward-compatible extension to do when tracker outputs exist.

## What can be claimed now
Only synthetic evidence (clearly labelled) and algorithmic/theoretical results. The AG pilot
(`scripts/run_pilot_ag.py`) is ready; it does not test IA-VHPED.

## Next actions requiring the user
1. Obtain Action Genome annotations (accept their terms) → inspector → pilot.
2. For identity claims: run a tracker (e.g. ByteTrack) on Charades/AG or VidOR videos, export
   tracks with an association score, define κ from it on *training* videos, then run the IA matrix
   on real streams.
