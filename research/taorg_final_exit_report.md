# TAORG final evaluation — Exit report

> Date 2026-10-14; branch `codex/taorg-completion`.
> Pre-registration: `research/taorg_final_preregistration.md` (committed in `ed31905` before any
> scoring).

# Registered decision: **STOP** (development gate failed). Q1 and Q2 were **not evaluated**.

| Gate (development, 30 VidOR training videos) | Result | Required |
|---|---|---|
| G1: perception and EGTR complete; tests pass | ✓ (30/30; 7,348 EGTR frames) | — |
| G2: NM frame F1 at the selected τ_obs | **0.0001** (τ_obs = 0.005, the lowest registered) | ≥ 0.05 |
| G2: videos with ≥ 1 predicted key | **1/30** | ≥ 20 |
| G3: identity-ghost exposure, IGT(B2, unflagged) | **0 s** | ≥ 5 s |

**Consequences, per the pre-registration:**
- the 40 fresh final videos were **not downloaded or processed**;
- the flag at predictor level (Q1) and V-HPED vs B2 (Q2) are **not answered**;
- nothing was retuned or redesigned.

## What actually ran

1. **Integration:** EGTR (Apache-2.0, official VG checkpoint). The adapter is bit-identical to the
   official evaluation path on 200 VG test images, and subset recall is consistent with the authors'
   (`research/taorg_egtr_integration.md`).
2. **Development streams:** EGTR on 30 VidOR development videos (every 3rd frame), mapped onto the
   real V3 tracker identities.
3. **Calibration** (registered, unflagged):
   - τ_obs: every registered value gave ≈ 0 predictions;
   - FB, B2 and VH grids at τ_obs = 0.005: J ≤ 0.0006 (`results/final/dev/gate.json`,
     `calibration.csv.gz`).
4. **Exploratory diagnosis** (development only; `diagnosis.json`):
   - identity mapping works (76% of tracks matched to an EGTR person);
   - EGTR's person–person triplet scores are tiny (99.9th percentile 3e-5 to 9e-4), **so the
     registered τ grid was misspecified**. I did not check the score scale before registering;
   - the ranking signal is weak (AUC 0.53–0.69 by predicate);
   - at a 90th-percentile threshold the raw stream's frame F1 is 0.15;
   - raw scores of this size never reach the [0, 1] thresholds of GatedEMA + hysteresis and V-HPED.
5. **E1** (engineering, not a comparison; development streams at the diagnostic threshold;
   `e1_real_streams.json`):
   - lazy V-HPED equals eager **exactly** on 30 real streams: 7,348 steps, 76,570 records,
     0 mismatches, max |Δscore| 0;
   - 1,299 freezes, 1,951 stale heap entries;
   - total time 0.58 s → 0.46 s.
6. **Operational failure:** one calibration attempt aborted out of memory (`load_sgg`
   per-frame decompression; fixed; no result produced; log kept).

## Answers

| Question | Answer |
|---|---|
| Q1: does the flag help with real predicted relations? | **Not tested** (gate failed) |
| Q2: does V-HPED beat GatedEMA + hysteresis on real predictions? | **Not tested.** No V-HPED claim follows. The project's standing evidence is Phase 4's (synthetic; no method contribution) |
| E1: is lazy = eager on real streams? | **Yes** (30 development streams; exact) |
| Is the integrated predictor useful for this question? | **Weakly at best.** EGTR's frame-wise VG predictions carry weak person–person signal on VidOR, on a score scale that the temporal methods' thresholds do not fit |

## The single justified next step (requires a new registration and approval)

**Re-register with predictor-appropriate scoring:**
- τ_obs chosen as a development **quantile**;
- scores rank-normalised to [0, 1] before temporal maintenance;
- ideally a **video** scene-graph model, or a predicate set with stronger image-level signal.

This cannot reuse this registration's decision, and it should not be presented as a continuation of
this test.

## Limitations

- **The pre-registration error is the main cause of the STOP.** The τ grid was not checked against
  the predictor's score distribution. It is reported as such, not repaired after the fact.
- **The diagnosis is development-only and exploratory.**
- **Evidence status:** no claim about temporal methods on real predicted relationships is
  supported or refuted by this study.
