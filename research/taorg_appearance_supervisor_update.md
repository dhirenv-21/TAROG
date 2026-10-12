# Supervisor update — appearance-based tracking audit (2026-10-12)

**Decision: NARROW** (by the rule fixed before any result). It is GO for the Faster R-CNN
detections and falls just short for SDP.

**What ran.** Real MOT17 video frames and the benchmark's public detections, from two detectors:
- ByteTrack (as before);
- BoT-SORT with its appearance (re-identification) model switched **off** and **on**, otherwise
  identical;
- the re-ID model is a public person re-ID network trained on a different dataset (MSMT17), not
  on MOT17.

The benchmark's ground truth was used **only to score**. We tuned and checked everything on 3
development videos. We froze it (committed and pushed), and only then ran 4 held-back videos.

**Question.** When a tracked person disappears, can the system tell whether they are temporarily
hidden (keep the record) or the tracker has already given them a new ID (drop the record)? Can it
do this better than "how long have they been gone"? Our previous audit found that with box
motion alone, it could not.

**Findings (held-back videos):**
1. **Appearance adds real information.** Comparing the missing person's stored appearance with
   people detected near their last position raised discrimination over elapsed time alone:
   - for Faster R-CNN, from AUC 0.67 to 0.86;
   - for SDP, from 0.65 to 0.77.
   This held in all 8 video × detector combinations.
2. **It is available during the gap, not only afterwards.** It is usable within 0.2 s of the
   person disappearing, for about 95% of missing frames.
3. **It is useful but not decisive.** At a setting that wrongly drops 7–12% of genuine gaps, it
   catches about half of identity takeovers (56% / 47%). Elapsed time catches about 20%.
4. **It works as a diagnostic beside the tracker.** Switching appearance on inside the tracker
   reduces identity switches modestly (−2% / −15%). Most of the value comes from using the
   appearance evidence to **diagnose** missing records, and that works even for a tracker that
   does not use appearance.

**What this is not.**
- It is not a new algorithm: the cue is standard re-ID.
- It is not a relationship result: no scene-graph model was run.

**Proposed next step (needs approval).** One pre-registered downstream test: feed this frozen
"identity-takeover" signal into our existing termination-aware baseline, and check whether it
reduces ghost relationships compared with the simple fixed buffer. If it does not, this line
stops.
