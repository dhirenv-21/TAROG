# Supervisor update — next-method study (2026-10-11)

**Synthetic evidence only. This is a research-evidence summary, separate from the demo.**

**What we tried.** We started from the strongest practical baseline, GatedEMA with hysteresis
(B2), and asked whether one small, principled addition could fix its main remaining failure
without new regressions.

**Failure audit** (development seeds only):
- B2 drops relations during long occlusions and during UNKNOWN-visibility gaps.
- Tuning B2 per condition would recover 0.13–0.15 J there, but no single setting works
  everywhere.

**Candidate.** We hold non-visible absences for as long as an online, label-free estimate of
"will it be re-detected?" stays above a threshold. The estimate is a mixture-cure model over
Kaplan–Meier re-detection times. It was pre-registered against:
- B2;
- a tuned fixed per-flag coasting buffer (the established tracker mechanism);
- a geometric adaptive learner (established);
- its own static prior.

**Result: STOP, algorithm contribution UNSUPPORTED** (`research/taorg_next_method_exit_report.md`).
- It fixes the occlusion failure: gap condition J 0.844 → 0.974. But a tuned fixed buffer gets
  most of this (0.944), and its own static prior gets all of it. Online learning added nothing at
  the selected settings.
- It fails badly when "UNKNOWN" means "visibility unknown" (S2: 0.724 → 0.568), because it
  learned from calibration data that UNKNOWN usually means "the track was lost after an identity
  swap".
- It costs extra false persistence after hidden terminations, and 3.7× the latency.

**Most useful lesson.** The input convention "no record = UNKNOWN" mixes two situations whose best
treatment is opposite: identity-swap ghosts should be dropped, and real gaps should be held.
Separating them, ideally with real tracker states, is the most promising next research question.
It needs a new pre-registration.

**Status for the paper.** After three pre-registered method attempts, there is still no supported
algorithm contribution:
- V-HPED: NO;
- episode estimator: STOP;
- this study: STOP.

The evidence supports an evaluation/methodology paper:
- relapse-aware episode metrics;
- the information limits;
- B2 (+ buffer) as the practical method;
- the documented negative results.

Real-data validation with at least one VidSGG model is the gate for any effectiveness claim. **It
is not started and needs approval.**
