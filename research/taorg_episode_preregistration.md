# TAORG relationship-episode study — Pre-registration

> **Status:** REGISTERED before any episode-study code, calibration or comparison exists.
> **Registered:** 2026-10-11 on branch `codex/taorg-episodes`. The commit hash is recorded in
> `research/taorg_episode_research_log.md`.
> **Config:** `configs/experiments/episodes.yaml` (its SHA-256 is recorded in every stage's metadata).
> **Formulation:** `research/taorg_episode_formulation.md`. **Related work:** `research/taorg_episode_related_work.md`.
> **Decision code:** `src/taorg/experiments/episodes/decision.py`, committed with this file and
> unit-tested.
>
> **Frozen and not reinterpreted:**
> - Phase 4 (METHOD CONTRIBUTION = NO);
> - Phase 5H Stage 0 (WEAK-MIXED);
> - Phase 5H-1 (HAWKES TRACK = NARROW, exploratory, annotation-only);
> - the historical metric definitions (`research/metric_definitions.md`).
>
> This study uses new seeds, a new synthetic world module and new metrics. None of the earlier
> artifacts are modified.

## 1. Question

> Can explicit relationship episodes (an episode terminates once and for all; recurrence is a new
> episode) prevent **false reactivation** and **identity ghosts** without harming **persistence**
> or the detection of **genuine recurrence**?

Sub-questions, each answered by a registered contrast in §8:

| Id | Question | Primary contrast |
|---|---|---|
| Q1 | Does episode handling reduce relapse? | P1 vs B1–B4 on FRR / FPT in C3, C5 (and C2) |
| Q2 | Does it delay genuine recurrence? | P1 vs B1–B4 on recurrence-detection delay in C4, C5 (non-inferiority 0.1 s) |
| Q3 | Does identity handling help beyond the identical addition to GatedEMA? | P2 vs B5 on C6, and the interaction (P2 − P1) − (B5 − B1) |
| Q4 | Does event history outperform simple context? | P4 vs P3 (history retention rule) |
| Q5 | Does the result survive parameter transfer? | T vs baselines on S0–S3 (never tuned) |
| Q6 | Does the full estimator justify its complexity? | the ladder retention rules + latency |

## 2. Hypotheses and a-priori expectations (declared so they cannot be retrofitted)

- **H-E (primary).** The selected episode estimator T (§8.3) beats every simple baseline on the
  relapse, termination and ghost conditions, with no loss on the persistence, recurrence and motion
  controls.
- **Expectations:**
  - GatedEMA + hysteresis (B2) is the strongest competitor on boundary flicker (C5), because its
    on-threshold already demands repeated detections.
  - The HSMM (B4) may match P1 wherever duration statistics matter.
  - Every method is information-limited on hidden termination (C2); only P3/P4 add evidence there,
    and only because visibility is per edge (formulation §11.1).
  - P2 and B5 share the identity module, so P2 ≈ B5 on ghosts is plausible.
  - P4 (event history) is expected **not** to beat P3: the world has no history dependence beyond
    succession timing.
  - **STOP or NARROW are realistic outcomes and will be reported as such.**

## 3. Synthetic episode world (new module `taorg.data.episode_world`)

All values below are copied into the config. Frame rate 30 fps; N = 360 frames (12 s).

**Tracks**
- 2 subjects (class 0) and 4 objects (class 1). All are present and VISIBLE in every frame.
- Boxes: width U[50, 110] px, height = width × U[1.0, 1.8].
- Centres start uniformly in the arena [60, 580] × [60, 420].
- Motion: velocity starts N(0, v0) per axis. Each frame v += N(0, a_std·√Δ) per axis, speed is
  clipped to v_max, and the box reflects at the arena borders.
- Defaults: v0 = 60 px/s, a_std = 80 px/s/√s, v_max = 200 px/s.
- Observed boxes get centre jitter N(0, 0.05 × diagonal) per axis per frame. Ground-truth boxes
  are clean.

**Keys and episodes**
- 8 primary keys (s, o, r) are sampled without replacement, with 6 predicates.
- For each key:
  - The first start is U[0, 45].
  - Each ON duration is U[40, 120] frames.
  - After an episode, with probability p_recur = 0.5 another episode follows after an OFF gap of
    U[10, 60] frames, if it fits (start + 40 ≤ N).
  - Otherwise the key terminates.
  - With probability p_open = 0.2 the last episode is *open*, i.e. it runs to the last frame.
  - Terminating last episodes are truncated to end ≤ N − 1 − 20.

**Succession**
- With probability p_succ = 0.4, each terminated episode of a primary key (s, o, r) spawns an
  episode of the partner key (s, o, partner(r)), with partners {0↔1, 2↔3, 4↔5}.
- It starts at end + δ, δ ∈ U[−8, +4] (overlap or short gap), and lasts U[40, 120] frames,
  truncated at the last frame.
- It is skipped (and recorded) if it would come within 10 frames of an existing episode of that
  key.

**Scores and records**
- Per-key mean score U[0.6, 0.95]; per-frame N(mean, 0.05), clipped to [0.05, 1].
- Every key has a record in every frame: alive frames are observed, dead frames are absence
  records (VISIBLE).
- **False detections draw from the same score distribution as true detections of that key**, so
  scores carry no existence information.

Randomness uses local `random.Random` streams per component, derived with SHA-256
(`derive_seed`), so the same seed gives the same underlying world in every condition (paired
across conditions as well as methods).

## 4. Corruptions (deterministic injectors with manifests)

Order: render → box jitter → temporary gaps → hidden termination → misleading post flags → boundary
flicker → background misses → background false detections → ID switch → i.i.d. flag noise →
bursty flag noise.

**Base, in every condition:**
- **Temporary gaps.** Each episode of length ≥ 20 receives, with probability p_gap = 0.6, one
  OCC-flagged gap of U[3, 30] frames, at least 3 frames from both of its ends.
  - These are the frozen-metric "occlusion" entries, so the frozen Persistence-F1 measures exactly
    persistence through temporary gaps.
- **Background misses.** Visible misses with probability 0.02 per observed record (Phase 3 flicker
  injector).
- **Background false detections.** With probability 0.004 per GT-dead record of a maintained key,
  the record becomes `observed=True` with a score from that key's distribution.

**Conditions** (C = calibrated and pooled for tuning; S = transfer, never tuned):

| Id | Name | What it adds | Role |
|---|---|---|---|
| C1 | gaps | p_gap = 1.0, up to 2 gaps per episode, lengths U[3, 45] | NI: persistence |
| C2 | hidden_termination | Every termination: detections suppressed and flagged OCC from end − U[3, 15] to end + U[15, 45] (clipped 3 frames before the next episode) | benefit: termination |
| C3 | misleading_post | Every termination: dead records flagged OCC for U[15, 60] frames after the end | benefit: false persistence |
| C4 | recurrence | p_recur = 0.9, OFF gaps U[6, 30] | NI: recurrence |
| C5 | boundary_flicker | Within 12 frames of every episode boundary: alive records missed with p = 0.3, dead records falsely detected with p = 0.2 (VISIBLE) | benefit: false reactivation |
| C6 | id_switch | Two same-class identity swaps of 20 frames (Phase 2 injector) | benefit: ghosts |
| C7 | motion | Fast, erratic motion (v0 250, a_std 400, v_max 600), no swaps | NI: identity false alarms |
| S0 | mixed | hidden termination p = 0.5, misleading post p = 0.3, boundary flicker (12, 0.15, 0.10), p_recur 0.6 | transfer reference |
| S1 | mixed + flag noise | S0 + i.i.d. OCC↔VIS flips 0.3 / 0.3 | visibility shift |
| S2 | mixed + UNKNOWN | S0 + 50% of absence flags UNKNOWN | visibility shift |
| S3 | mixed + bursty flags | S0 + bursty flag errors (stationary 0.2, mean burst 10) | visibility shift |

**Manifest types** (none is ever passed to a model):
- `occlusion` (temporary gaps; the frozen Persistence-F1 intervals);
- `termination_occlusion` (C2);
- `post_termination_flags` (C3);
- `boundary_flicker`;
- `false_detection`;
- the existing `flicker`, `id_switch`, `visibility_noise` and `bursty_visibility` types.

## 5. Methods and tuning

| Id | Method | Search |
|---|---|---|
| B1 | GatedEMA | 200 samples (Phase 4 space) |
| B2 | GatedEMA + hysteresis | 200 (B1 draws + θ_on, ρ) |
| B3 | VA-HMM | 200 (Phase 4 space) |
| B4 | Duration-aware HSMM | 200 |
| B5 | GatedEMA + identity | 200 (B1 draws + σ_id, r_id) |
| P1–P4 | Episode ladder | 200 each, nested common random numbers (formulation §7) |
| — | No Memory | 1 (descriptive only) |

The parameter distributions are in the config.
- **Equal budget:** 200 samples per tunable method, so higher-dimensional methods are
  disadvantaged.
- **Seeds:**
  - calibration 5000–5019, evaluation 6000–6049 (50 per condition), smoke 9300–9301 (pipeline
    checks only; never analysed);
  - sampler 2026, bootstrap 4343, half-sample control 2027.
  - These are disjoint from every earlier phase.
- **Fitting data:**
  - The transition statistic E (formulation §6) is fitted on the calibration ground truth of
    C1–C7 (20 seeds × 7 conditions) and saved.
  - Evaluation ground truth is used only by metrics.
- **Selection (one operating point per method, pooled tuning):**
  - For each method, choose the configuration with the highest mean J (§6) over all 140
    calibration instances (C1–C7 × 20 seeds).
  - Tie-breaks: lower mean churn, then lower configuration index.
  - The same parameters are used in every condition, C1–C7 and S0–S3. There is no
    condition-matched retuning.
- **Tuning-capacity control (secondary):** every method is re-selected from the same seeded random
  half (100 indices) of its samples. The decision is recomputed (§8.5).
- Nothing is tuned on evaluation seeds. Selection reads only calibration artifacts.

## 6. Metrics (new; the historical definitions are unchanged and also reported)

Per video. Frames 0 … N−1; GT episodes G(k) are inclusive frame intervals from the lifetimes; the
output O_n is the set of keys emitted at frame n; output **runs** O(k) are maximal runs of
consecutive frames with k ∈ O_n. K_GT = keys with ≥ 1 GT episode. A **ghost key** is one that is
output but not in K_GT.

1. **Frame P/R/F1.** Over all (k, n) with k ∈ K_GT ∪ {output keys}:
   - TP = out ∧ alive, FP = out ∧ ¬alive, FN = ¬out ∧ alive.
   - Undefined ratios are 0. F1 is always defined because K_GT ≠ ∅.
2. **Episode P/R/F1.**
   - IoU(g, o) = |g ∩ o| / |g ∪ o| in frames, within a key.
   - Pairs with IoU ≥ 0.5 are matched greedily by descending IoU (ties: earlier g, then earlier o),
     one-to-one.
   - Precision = matches / (all output runs, ghosts included); recall = matches / (all GT episodes).
   - Zero output runs ⇒ precision 0, F1 0.
3. **J = ½·frame-F1 + ½·episode-F1** (primary quality; higher is better).
   - Always-off gives J = 0.
   - Always-on gives one run per key spanning the video. Its IoU with each episode is below 0.5
     whenever episodes are shorter than half the video, so episode-F1 = 0 and J ≈ ½·frame-F1.
   - Tested in `tests/unit/test_episode_metrics.py`.
4. **Missed-episode rate.** The share of GT episodes with coverage < 0.5, where coverage is the
   fraction of the episode's frames that are output.
5. **Fragmentation.** The mean over GT episodes with coverage ≥ 0.5 of (number of output runs
   intersecting the episode − 1). NaN if there are none.
6. **False-reactivation rate (FRR).**
   - An **evaluable termination** is a GT episode [a, b] with b < N−1 whose following dead interval
     D = [b+1, d] has |D| ≥ 10. Here d = (next GT episode start of k) − 1, or N−1.
   - t_d is the first frame in D with k ∉ O. If there is none, it is a "no drop" (not a
     reactivation; it counts in FPT).
   - Reactivated ⇔ ∃ n ∈ (t_d, d] with k ∈ O_n.
   - FRR = reactivated / evaluable; NaN if there are none. Reactivation frames are also reported.
7. **False-persistence time (FPT).** Output frames of keys in K_GT at GT-dead frames (before the
   first birth, between episodes, after the end), divided by fps, in seconds per video.
8. **Ghost time.** Output frames of ghost keys / fps (seconds per video). Ghost edge-frames are
   also reported (the existing diagnostic) together with swap recall.
9. **Sustained termination delay (STD).**
   - For each evaluable termination: t* = the first n ∈ D such that k ∉ O for all frames in
     [n, min(n + 14, d)] (≥ 0.5 s off, truncated at d).
   - STD = t* − b frames, so 1 is perfect.
   - Censored if there is no such n: STD = |D| + 1, flagged.
   - Reported as the mean in seconds, with the censored count.
   - It is relapse-aware: a dip-and-return does not end the delay.
10. **Recurrence-detection delay (RDD).**
    - A recurrence is a GT episode [a, b] with a preceding GT episode of the same key, separated by
      the dead gap [c, a−1].
    - It is **bridged** if k ∈ O on every frame of the gap: a merge, excluded from the RDD mean and
      reported as a bridge rate.
    - Otherwise RDD = (min{n ∈ [a, b] : k ∈ O_n}) − a frames. It is censored (= b − a + 1) if there
      is no such n.
    - Reported as the mean in seconds, plus the bridge rate.
11. **Onset delay.** Defined as RDD for the first episode of each key (no bridging).
12. **Gap persistence.** The frozen Persistence-F1 over `occlusion` entries, plus gap recall
    TP/(TP + FN) from the same counts.
13. **Churn.** Frozen definition (`taorg.metrics.churn`).
14. **Frozen termination delay and false-persistence rate.** Reported unchanged.
    - These refer to each key's *last* GT alive frame, so with recurrence they describe only the
      final episode. This is a stated limitation of the historical metrics, which is why STD,
      FRR and FPT are added.
15. **Latency.** Frozen per-frame `step()` latency on 10 evaluation instances of C1. Reported:
    mean ms/frame and the ratio to GatedEMA.

## 7. Statistics

- **Unit:** an evaluation video (n = 50 per condition), paired across methods on identical inputs.
- **Paired differences** (treatment − comparator):
  - mean;
  - 95% percentile bootstrap CI (5000 resamples, seed 4343);
  - Cohen's d_z ("undefined" if the SD < 1e-9);
  - two-sided Wilcoxon signed-rank test.
  - Seeds where a metric is NaN in either arm are dropped pairwise, and the count is reported.
- **Multiplicity:**
  - Holm over the family of T-vs-comparator Wilcoxon tests on J (11 conditions × 4 comparators,
    plus B5 in C6/C7 = 46 tests).
  - Holm separately over the ladder ΔJ family (3 contrasts × 7 calibrated conditions = 21).
  - The decisions below use CIs and practical margins, not p-values; Holm-adjusted p-values are
    reported alongside.
- **Pooled ladder contrast:** the per-seed mean of ΔJ over C1–C7 (the same 50 seeds in every
  condition), with a bootstrap over seeds.

## 8. Decision rules (exact; `decision.py`)

### 8.1 Pairwise outcome of treatment X vs comparator Y in condition c

Margins:
- g_J = 0.02 (J gain);
- ni_J = −0.01 (J non-inferiority);
- r_h = 0.25 (relative harm change);
- rdd = 0.1 s (recurrence-delay margin);
- gap = −0.03 (gap-recall non-inferiority);
- churn margin max(0.005, 0.10 × Y's churn).

Target harm endpoint by condition (lower is better):

| C2 | C3 | C5 | C6 | others |
|---|---|---|---|---|
| STD | FPT | FRR | ghost time | none |

**Win** if (A or B) and every applicable guard G holds:
- A: ΔJ ≥ g_J and CI_lo(ΔJ) > 0.
- B: a target exists, mean Δharm ≤ −r_h × mean_Y(harm), CI_hi(Δharm) < 0, and CI_lo(ΔJ) ≥ ni_J.
- Guards G:
  - CI_lo(ΔJ) ≥ ni_J;
  - CI_hi(ΔRDD) ≤ rdd, where RDD is defined (≥ 10 paired seeds);
  - CI_lo(Δgap recall) ≥ gap, where gaps exist;
  - CI_hi(Δchurn) ≤ the churn margin.

**Loss** if any of the following hold:
- ΔJ ≤ −g_J and CI_hi(ΔJ) < 0;
- a target exists and mean Δharm ≥ +r_h × mean_Y(harm) with CI_lo(Δharm) > 0;
- mean ΔRDD ≥ rdd with CI_lo(ΔRDD) > 0.

Loss takes precedence over win.

**Cell outcome** vs a comparator set: win if a win against every member; loss if a loss against
any; otherwise neutral.

### 8.2 Comparator sets

- {B1, B2, B3, B4} in every condition.
- B5 is added in C6 and C7.

### 8.3 Ladder retention and the selected estimator T

Each rule compares a rung with the rung below it, on evaluation data.

- **Identity (P2 vs P1)** is retained iff:
  - on C6, ghost time falls by ≥ 25% with CI_hi < 0 and CI_lo(ΔJ) ≥ −0.01;
  - and there is no ΔJ loss (≤ −0.02, CI_hi < 0) in any of C1–C7.
- **Transition context (P3 vs P2)** is retained iff:
  - in at least one of {C2, C3}, either (ΔJ ≥ 0.02 with CI_lo > 0) or (mean ΔSTD ≤ −0.1 s with
    CI_hi < 0 and CI_lo(ΔJ) ≥ −0.01);
  - and there is no ΔJ loss in C1–C7.
- **Event history (P4 vs P3)** is retained iff:
  - the pooled per-seed ΔJ over C1–C7 has mean ≥ 0.01 and CI_lo > 0;
  - and there is a ΔJ win (≥ 0.02, CI_lo > 0) in at least one of {C2, C3};
  - and there is no ΔJ loss in C1–C7.
  - **History is optional: it must beat simple context on downstream quality.**
- **T** is the highest rung reachable through consecutive retentions starting at P1:
  - P1 if identity is not retained;
  - P2 if identity is retained but context is not;
  - and so on.
  - Rungs above T are reported descriptively.

### 8.4 Study decision (T vs the comparator sets)

BEN = {C2, C3, C5, C6}; NI = {C1, C4, C7}; TRANSFER = {S0, S1, S2, S3}.

| Decision | Rule |
|---|---|
| **GO** | (i) wins in ≥ 3 of the 4 BEN conditions, including ≥ 2 of {C2, C3, C5}; (ii) 0 losses in C1–C7; (iii) ≤ 1 loss in TRANSFER; (iv) T latency ≤ 1 ms/frame and ≤ 20× GatedEMA; (v) the half-sample control does not give STOP |
| **STOP** | (a) wins in ≤ 1 BEN condition; **or** (b) total losses ≥ total wins over all 11 conditions; **or** (c) T fails to beat **B2 (GatedEMA + hysteresis) alone** (pairwise rule 8.1) in every one of C2, C3, C5. In other words, the simple baseline matches the episode estimator on the relapse/termination conditions, and the recommendation is simplification |
| **NARROW** | anything else |

### 8.5 Registered answers

- **Q1:** YES iff P1 has a pairwise win (8.1) against each of B1–B4 in C3 or C5, i.e. in at least
  one of them.
- **Q2:** "delays recurrence" iff P1 has a ΔRDD loss (mean ≥ 0.1 s with CI_lo > 0) against any
  of B1–B4 in C4 or C5.
- **Q3:** YES iff P2 wins against B5 (8.1) in C6 **and** the interaction (P2 − P1) − (B5 − B1) on
  ghost time has CI_hi < 0 (bootstrap over seeds).
- **Q4:** the history retention rule (8.3).
- **Q5:** YES iff T has ≤ 1 loss in TRANSFER **and** wins in ≥ 2 of the 4 TRANSFER conditions.
  If T has no BEN win at all, Q5 is reported as "no in-distribution advantage to transfer".
- **Q6:** "full estimator justified" iff T = P4. Otherwise the report states which mechanisms
  earn their place.
- **Tuning-capacity control:** report the decision and the T selection under the half-sample
  selection.

## 9. Completeness, failures and figures

- **Completeness, checked before any analysis:**
  - expected = 11 conditions × 50 seeds × (10 methods + half-sample variants);
  - actual, missing, duplicate and failed runs are listed;
  - calibration = 140 instances × all configurations;
  - instance manifests are identical across method groups (SHA-256).
- **Representative failures, predefined rule:**
  - For T in each BEN condition and in C4, take the evaluation seed with the largest per-seed J
    regret against the best comparator in that condition (the comparator with the highest mean J).
    Also take the lower-median positive-regret seed.
  - Each is saved as a timeline figure of the three keys with the most T output errors.
  - The result is "No failure observed" if no seed has positive regret.
- **Figures:**
  1. gap persistence by method and condition;
  2. FRR and FPT;
  3. recurrence and onset delay;
  4. ghost time and swap recall;
  5. ladder ΔJ with CIs plus latency (ablation cost);
  6. decision matrix.

## 10. Compute and artifacts

- Local CPU only, ≤ 16 worker processes. Calibration checkpoints per instance (resume skips
  completed instances; a checksum is verified on load).
- Heavy per-instance calibration matrices go to `D:/TAORG_runs/episodes/` (outside git, because the
  system drive is nearly full). A SHA-256 manifest of those files is committed.
- Compact artifacts go to `results/episodes/`: calibration summary, frozen parameters, evaluation
  rows, aggregated statistics, tables and figures.
- Reproduction commands are given in `results/episodes/README.md`.

## 11. Declared limitations

1. Synthetic world only; per-edge visibility; synthetic identity cue.
2. One pooled operating point per method. Condition-matched tuning (Phases 3–4) is not repeated.
3. J is a registered composite. Other weightings could rank methods differently, so its components
   are reported.
4. 200 random samples do not equalise search difficulty, which works against the higher-dimensional
   methods (P2–P4, B4).
5. The world's recurrence, succession and flicker statistics are design choices. Conclusions are
   conditional on them.

## 12. Amendments

None at registration. Any change must be dated, justified, and must state whether it was made
before or after viewing any calibration or evaluation result of the compared methods. Smoke runs
(seeds 9300–9301) are pipeline checks and are never analysed.
