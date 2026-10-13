# Supervisor update — downstream relationship study (2026-10-13)

**Registered decision: GO, read narrowly.** A frozen appearance-based "identity takeover" flag
removes about three quarters of identity-related relationship ghosts for **both** baseline
methods. Under the pre-registered harm margins, it does so without measurable harm to gap
persistence, termination delay or recurrence delay.

**What ran.**
- 90 real VidOR videos with person–person relationship annotations: 30 for development and 60
  for evaluation, the evaluation videos downloaded only after everything was frozen.
- A real person detector and the appearance-enabled tracker from the previous audit.
- **Relationships came from an oracle predictor** (the annotations, applied to whatever the
  tracker tracks). This tests the *information*, not a real scene-graph model.
- Two standard baselines (a fixed buffer, and gated EMA with hysteresis), each with and without
  the flag, received identical inputs.

**Findings.**
1. **Ghosts fall sharply for both baselines.** Ghost time drops from 145 to 40 s (fixed buffer)
   and from 110 to 27 s (gated EMA), over about 21 minutes of video.
2. **The gain belongs to the information source, not to a TAORG method.** The flag helps the
   simpler baseline slightly *more*.
3. **The safety margins were easy to meet in this setting.** The tuned baselines already drop
   relationships almost immediately when a person is lost. On this new video domain the flag
   wrongly fires during 32% of genuine gaps (about 3× its rate on MOT17), so with longer-holding
   baselines the cost would be larger.
4. **Termination and recurrence tests were weak.** Most of those events turned out to be
   box-matching flicker. On genuine events, the flag made no difference.

**What this is not.**
- It is not an algorithm contribution.
- It is not evidence with a real relationship predictor.

**Next step (needs approval).** Repeat the frozen test with one existing scene-graph model's
predictions in place of the oracle, with flicker-robust scoring. If the gain disappears, the line
stops.
