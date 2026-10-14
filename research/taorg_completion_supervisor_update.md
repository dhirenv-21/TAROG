# Supervisor update — project completion pass (2026-10-14)

**Done and verified.**
1. **Efficient state maintenance (the ADSA core, Phase 5).** The relationship filter now skips
   relationships that are provably dead, without changing any output.
   - It is exactly equal to the full update: 0 differences in 28 synthetic benchmarks and 30 real
     video streams.
   - It does about 4× less per-frame work and runs 1.6–3× faster in typical settings.
   - It is slower when the frame rate varies or when almost every relationship stays active.
   - The proof uses a standard positive-matrix contraction argument: no assumption that scores
     decay smoothly, and no information is deleted.
2. **A real scene-graph model is integrated.** EGTR (CVPR 2024, Apache-2.0) runs through its
   official code. Our wrapper gives bit-identical outputs to the official evaluation, and a small
   benchmark check matches the authors' reported accuracy.
3. **The project is runnable.** One command takes a video through person tracking, EGTR
   relationships and temporal maintenance to saved graphs and an offline viewer. A 241-frame clip
   takes about 4 minutes on the laptop GPU.
4. **Paper draft.** A full working draft states only what the evidence supports.

**What did not work.** The final pre-registered test of the methods on *real predicted*
relationships stopped at its development checkpoint.
- EGTR's scores for person–person relations on these videos are tiny, and the threshold grid I
  registered did not account for that. That was my error, and I did not fix it after the fact.
- Even with a sensible threshold, EGTR's evidence for these relations is weak.
- So the project still has **no evidence on whether temporal post-processing improves real
  predicted relationships**, and **no supported new temporal algorithm**.

**Suggested next step (needs approval).** Re-register the real-predictor test with normalised
scores, ideally with a video scene-graph model. Alternatively, submit the current evidence as a
methods/negative-results paper centred on the certified maintenance scheme and the
missing-record audit.
