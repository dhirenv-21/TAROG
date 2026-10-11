# Supervisor update — observation-interface audit (2026-10-12)

**What ran on real data:**
- **Automatic outputs.** The public detections of two real detectors (Faster R-CNN and SDP), as
  distributed with the MOT17 benchmark, tracked with ByteTrack (`trackers` 2.6.1). Three
  development videos (MOT17-02, 09, 11) were processed on CPU. Four further videos were reserved
  as untouched evaluation data and never processed.
- **Manual annotations.** Only the benchmark's own ground truth: identities and per-person
  visibility. It was used **only to score** the automatic outputs.
- **Oracle diagnostics.** None was fed to a model.

**Question.** Our last study suggested that TAORG treats every missing observation as "UNKNOWN",
mixing real gaps (occlusion, a missed detection) with identity problems (the tracker gives the
person a new ID). Is that true, and can the tracker's outputs tell the two apart?

**Findings:**
1. **The conflation is real.** In code, every temporal method maps "no record" to UNKNOWN and
   ignores the tracker state stored in the schema. On real tracker output, 41% of missing records
   are real gaps and 20% are identity takeovers, and the methods currently cannot tell them apart.
2. **The available signals cannot separate them.** Our rule was fixed before looking. An
   "identity suspect" cue caught 0.1–1% of the takeovers. About a quarter of real gaps were
   wrongly labelled "left the frame", because the tracker's motion prediction drifts off-image
   during long gaps.
3. **The only weakly useful signal is elapsed time.** It is exactly what a fixed "hold for N
   frames" buffer already uses.

**Contribution.** A small, backward-compatible observation contract that records *why* something
is missing and *where that reason came from*. It prevents information loss when a better tracker
provides the information. It is an interface fix, **not an algorithm contribution**. No
downstream comparison was run, because the pre-stated criteria were not met.

**Next step (needs approval).** Repeat the same audit with an appearance-based tracker that
reports association evidence. This needs MOT17 frames (1.95 GB) and a re-ID model. Alternatively,
integrate one scene-graph model to audit missing relationships directly.
