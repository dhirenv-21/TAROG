# Phase 5H-1 exit report — heterogeneity-controlled temporal dependence in VidOR

> **Decision (registered rule, applied unchanged): HAWKES TRACK = NARROW (MIXED).**
> K1, K2, K3 pass; **K4 fails** (0 reproducible individual interactions at FDR ≤ 0.10);
> the family test passes for SPATIAL and MOTION only.
>
> **Evidence class: EXPLORATORY.** Every annotated VidOR video had already been used in Stage 0
> (prereg §2). These are annotation-only results (human-segmented VidOR relation intervals, no
> detector, no tracker). They establish *predictive association in annotation timing*, not
> causation and not real-video effectiveness.
>
> Every number below is copied from `results/phase5h1/full/tables.md` (generated from
> `results/phase5h1/full/analysis.json` by `scripts/phase5h1/result_tables.py`). Table ids
> (T1–T12) refer to that file. Post-hoc diagnostics are labelled **POST-HOC** and never enter
> the decision.

## 1. Pre-registration, freeze and provenance

| Step | Commit | Time (IST) | Content |
|---|---|---|---|
| Pre-registration | `07bf369` | 2026-10-07 09:45 | questions, splits, model ladder, nulls, metrics, decision rule |
| GLM engine | `4d810ce` | 2026-10-07 09:54 | exact segment likelihood, eager reference test, leakage guard |
| Execution infrastructure + A1–A3 | `c546591` | 2026-10-08 09:45 | before any registered fit |
| SELECT entry point clarification | `a88fd91` | 09:47 | `prepare` is the registered SELECT entry; EVAL stays locked |
| Amendment A4 | `8721fcb` | 10:58 | L-BFGS iteration limit 500 → 5000 (diagnostics only, before selection) |
| **Parameter freeze** | **`289319e`** | **11:55** | `results/phase5h1/full/frozen/{frozen.json, params.npz}` + analysis/figure code |
| EVAL unlocked | — | 11:58 | `EVAL_UNLOCK.json` bound to the freeze commit and the freeze-file hash |

- Code hash of all computational modules: `65317bd4489fdae2` (identical in every invocation from
  the first `prepare` to `check`; `provenance/invocations.jsonl`, copied from
  `D:/TAORG_runs/phase5h1/main/invocations.jsonl`). The `git_dirty` flag is true for the A4
  `tune`/`freeze` invocations only because the then-untracked `phase5h1_analysis.py` lay under
  `src/`; it is not part of any fit.
- Config hash after A4: `eee08837…`. Layout hash `8bef01baa3911044` (69 targets, 93 sources,
  504 columns, built from TRAIN only).
- Selected ridge (by total SELECT log-likelihood): G1 3, G2 1, G3 1, G4x 3, G4 3, G4+ 3; class/lag
  variants use G4's 3. None lies at a grid boundary.

## 2. Amendments and execution provenance

| Id | Type | What | Effect on registered science |
|---|---|---|---|
| A1 | technical | exact sufficient-statistic aggregation, chunked design, checkpointed bounded-parallel runner, structural EVAL lock | none: objective identical (max \|Δw\| 1e-10, \|ΔLL\| 2e-10 on 1000 TRAIN videos); serial = parallel on synthetic data |
| A2 | unspecified detail | N-jit construction: censored endpoints pinned; per-video seeding; N-circ applied unchanged with EVAL seed + 100000 | defines the registered null; N-circ's disclosed defect is quantified in §6 |
| A3 | unspecified detail | bootstrap index sharing, family normalisation, K4 τ scan / Spearman definition, hypothesis sidedness, next-event competing set | fixes interpretations before results |
| A4 | technical | `max_iter` 500 → 5000 for every fit | 35/157 early fits had stopped at the limit; after A4, 2208/2208 converged. Selected ridges are **identical** to the superseded 500-iteration attempt (`provenance/selection_maxiter500_superseded.json`) |

**Technical reruns and incidents (all preserved):**
- The first tuning attempt (max_iter 500) is superseded by A4. Its 2208 checkpoints remain under
  the run root and are counted as "unexpected" in T12.
- During that attempt the machine paused for ~34 min (10:09–10:43; every running fit stalled at
  once). A keep-awake hold was requested; no later stall occurred.
- `git commit -a` once swept unrelated working-tree files (Memory.md line, demo outputs) into
  a local commit. It was immediately redone with only the intended file, before any push.
  No research artifact was affected.

## 3. Experiment matrix (T12)

| Stage | Expected | Complete | Missing | Duplicates | Non-converged | Failed jobs |
|---|---:|---:|---:|---:|---:|---:|
| tune (69 targets × 32 specs) | 2208 | 2208 | 0 | 0 | 0 | 0 |
| evaluate (69 × 14 models × 2 halves + 2 next-event) | 1934 | 1934 | 0 | 0 | 0 | 0 |
| nulls (38 × 69 × 4 rungs) | 10488 | 10488 | 0 | 0 | 0 | 0 |
| stability (6 fold sets × 69 × 5 + support) | 2071 | 2071 | 0 | 0 | 0 | 0 |
| hypotheses (observed + 199) | 200 | 200 | 0 | 0 | — | 0 |

Across 406 job executions there were 0 failed and 0 blocked jobs; `all_complete = True`
(`results/phase5h1/full/completeness.json`). No seed was rerun or replaced.

## 4. Sample sizes and support (T1)

- TRAIN 5600 videos (fitting; 328,319 modelled events), SELECT 700, EVAL-A 700 (38,443 events),
  EVAL-B 835 (46,272 events); EVAL 84,715 events over 69 target types.
- Strong imbalance: SPATIAL 62.0% of EVAL events (11 types); ATTENTION_COMMUNICATION 16.9%;
  MOTION 12.4%; CONTACT_MANIPULATION 7.4% across 40 types; VEHICLE_SUPPORT 1.3% (1,108 events).
  The least-supported types have 42–60 EVAL events (e.g. ON(throw) 42, ON(feed) 48).
- **Extraction property (disclosed):** 7,058 annotated EVAL onsets of modelled types coincide
  with the pair-window start, i.e. the moment the pair first co-exists. Under the registered
  event rules (§3: PAIR_BIRTH replaced by elapsed-time bins) these are not modelled events
  (91,773 annotated vs 84,715 modelled).

## 5. Held-out predictive performance (T2, T3)

Per-event held-out log-likelihood (EVAL): G0 −4.372, G1 −3.840, G2 −3.581, G3 −3.502,
G4x −3.435, **G4 −3.419**, G4+ −3.450.

| Comparison (nats/event, 95% video bootstrap) | EVAL | EVAL-A | EVAL-B |
|---|---|---|---|
| heterogeneity G1 − G0 | +0.532 [0.504, 0.562] | +0.539 | +0.527 |
| duration G2 − G1 | +0.259 [0.246, 0.273] | +0.254 | +0.264 |
| simple transitions G3 − G2 | +0.079 [0.072, 0.085] | +0.084 | +0.074 |
| **cross-history \| het + dur: G4x − G2 (K1)** | **+0.145 [0.135, 0.155]** | +0.158 [0.144, 0.173] | +0.135 [0.121, 0.149] |
| **history \| transitions: G4 − G3 (K2)** | **+0.083 [0.076, 0.090]** | +0.091 [0.081, 0.101] | +0.076 [0.067, 0.087] |
| state \| history: G4 − G4x | +0.016 [0.014, 0.019] | +0.017 | +0.016 |
| positive-only constraint: G4+ − G4 | **−0.031 [−0.034, −0.027]** | −0.035 | −0.027 |

- **Effect size:** Δ(G4, G3) is 0.320 of the duration gain Δ(G2, G1), and Δ(G4x, G2) is 0.561.
  Per event, +0.083 nats is a likelihood ratio of e^0.083 ≈ 1.09. Δ(G4, G3) is positive in
  74.7% of EVAL videos.
- **Consistency:** both halves give positive CIs for every history comparison. EVAL-B (the
  official VidOR validation split, never used for selection) is slightly lower than EVAL-A.
- **Secondary (T10, descriptive):** next-event-type log-loss G3 1.043 → G4 0.989 (Δ −0.054
  [−0.059, −0.050]); top-1 accuracy 0.643 → 0.676 (Δ +0.033 [+0.029, +0.036]). The top-1
  improvement is carried by a minority of videos: 42.3% of videos have a positive Δ, 47.4% are
  exact ties and 10.3% are negative (POST-HOC count from the next-event checkpoints).

## 6. Null comparisons (T4)

| | observed | N-circ max | N-jit max | combined max | combined median |
|---|---:|---:|---:|---:|---:|
| Δ(G4x, G2) | +0.145 | +0.015 | +0.031 | +0.031 | +0.021 |
| Δ(G4, G3) | +0.083 | +0.014 | +0.017 | +0.017 | +0.013 |

- Both observed gains are far above all 38 null replicates (figure `fig_null_comparison`).
  The null median is 15% of Δ(G4x, G2) (0.021 / 0.145), against 54% in Stage 0. This is
  consistent with the Stage 0 hypothesis that much of the earlier cross-excitation was pair
  heterogeneity, which the HET block now absorbs.
- **Null quality (negative findings about the controls themselves):**
  - **N-circ inflates events.** Null EVAL has 98,248–98,413 events against 84,715 observed
    (+16%; TRAIN 379,366 vs 328,319 for seed 6200). The registered Stage 0 circular shift turns
    censored interval ends into onset/offset events at random times (disclosed in A2). Its
    per-event Δ is therefore computed on a partly manufactured event set.
  - **N-jit loses events.** Null EVAL has 76,408–76,847 events (−9.3% to −9.8%). Of 214,849 TRAIN
    intervals per replicate, ~8.6% leave the pair window and are dropped, ~25% are clipped,
    ~11.6k offsets become right-censored and ~2.6k onsets become left-censored
    (`attempts.jsonl` fold-null diagnostics). VidOR pair windows are short relative to ±5 s.
  - Neither null is a clean interval-jitter test (A2). They bound how much of the gain
    survives destroying sub-5-s (N-jit) or all (N-circ) relative timing while keeping pair
    composition; they are not exact tests.

## 7. Stability and reproducibility (T6, T7)

**Family test (Δ_F(G4, G3); nats per family event):**

| Family | EVAL-A | EVAL-B | null max | excess vs 10%·Δ_F(G2,G1) | pass |
|---|---|---|---:|---|:-:|
| SPATIAL | +0.115 [0.102, 0.128] | +0.093 [0.081, 0.107] | +0.014 | 0.092 vs 0.023 | ✔ |
| MOTION | +0.105 [0.085, 0.126] | +0.112 [0.074, 0.152] | +0.049 | 0.077 vs 0.021 | ✔ |
| CONTACT_MANIPULATION | +0.058 [0.012, 0.122] | +0.041 [0.014, 0.070] | +0.026 | 0.045 vs 0.057 | ✘ (practical size) |
| ATTENTION_COMMUNICATION | +0.013 [0.006, 0.020] | +0.010 [0.002, 0.017] | +0.01111 | 0.005 vs 0.025 | ✘ (exceeds null max by 0.00001; practical size fails) |
| VEHICLE_SUPPORT | −0.000 [−0.035, 0.035] | +0.024 [−0.006, 0.056] | +0.050 | — | ✘ (CI includes 0) |

**K4 — individual interactions (fails):**
- G4 HIST cells: 18,837. Cells with 5/5 fold sign agreement: real 5,924; N-jit fold nulls
  5,589–5,857.
- The estimated FDR never reaches 0.10 (minimum 0.16, reached at |t| ≥ 53.5 with 5 real cells
  vs 0.8 per null replicate) → **0 discoveries**.
- Inter-fold Spearman of the whole weight vector: real 0.299 > every null (0.255–0.270). This
  half of K4 passes; the discovery half does not.
- **POST-HOC diagnostic** (`posthoc_foldnull_cells.json`): the strongest null cells are
  rare-target cells with near-identical, ridge-shrunk negative weights across folds (tiny SD,
  |t| 35–89), e.g. ON(use) ← OFF(above). The registered t = mean / (SD/√5) is therefore
  inflated by shrinkage in sparse cells whether or not an effect exists. This makes the
  discovery criterion close to unattainable. It is a design limitation of the registered K4,
  recorded here; the decision is not re-scored.
- Strongest real sign-consistent cells (descriptive, **not discoveries**; T7 source):
  OFF(next_to) ← OFF(in_front_of) (≤ 0.5 s, w −2.53), OFF(in_front_of) ← ON(away) (≤ 0.5 s,
  −1.28), OFF(in_front_of) ← ON(next_to) (0.5–2 s, +3.30), ON(next_to) ← ON(behind) (≤ 0.5 s,
  −1.53). All are spatial relabelling patterns.

## 8. Mechanism attribution (T8, ablations)

Δ(G3 + one HIST block, G3), nats/event on EVAL (blocks overlap, so they do not add up to
Δ(G4, G3) = +0.083):

| Block | Δ | Block | Δ |
|---|---|---|---|
| ONSET → OFFSET | **+0.048 [0.044, 0.053]** | lag ≤ 0.5 s | **+0.052 [0.047, 0.057]** |
| ONSET → ONSET | +0.016 | lag 0.5–2 s | **+0.057 [0.051, 0.062]** |
| OFFSET → OFFSET | +0.016 | lag 2–10 s | +0.010 [0.007, 0.012] |
| OFFSET → ON | +0.012 | | |

- The gain is carried by **"another relation starts → this relation ends"** (ONSET → OFFSET)
  at **short lags (≤ 2 s)**. Slow history (2–10 s) adds little.
- **Inhibition matters:** constraining history weights to be non-negative (G4+) loses 0.031
  nats/event (37% of the G4 − G3 gain). Many effective terms are negative, i.e. suppression
  shortly after other events.
- State and history are complementary but mostly overlapping: G4 − G4x = +0.016, while
  G3 − G2 = +0.079.
- **Annotation timing vs scene dynamics (POST-HOC, `posthoc_boundary.json`):**
  - **17.0%** of the 91,773 annotated EVAL events of modelled types (incl. the 7,058
    window-start onsets) tie exactly (same frame) with another predicate's event on the same
    pair, e.g. OFF(away) with OFF(in_front_of), or ON(towards) with
    ON(in_front_of). Ties are excluded from history by the strict-history rule.
  - Only 0.7% fall ≤ 1 frame after a cross-predicate event and 4.3% within (1 frame, 0.5 s];
    50.2% have no cross-predicate event in the previous 10 s.
  - The short-lag gain is therefore best read as the hazard structure *around synchronised
    annotation boundaries*: bursts of co-terminating relabels, then quiet, expressed as strong
    negative short-lag weights. That is a property of how VidOR annotators segment relation
    instances. It may or may not reflect physical scene dynamics. **This data cannot
    distinguish the two.**

## 9. Stage-0-generated hypotheses (T9; model-free; out-of-sample exploratory replication)

| H | Transition (window 0–2 s) | E_in | RR [95% CI] | Holm p | Status |
|---|---|---:|---|---:|---|
| H-a | ON(hold) → OFF(grab) | 78 | 4.93 [2.97, 8.21] | 0.015 | replicates |
| H-b | ON(release) → OFF(hold) | 79 | 44.5 [29.4, 72.7] | 0.015 | replicates |
| H-c | ON(get_off) → OFF(ride) | 1 | — | 0.445 | UNDERPOWERED (predicted by audit) |
| H-d | ON(beneath) → OFF(above) | 0 | — | — | UNDERPOWERED (predicted by audit) |

H-a and H-b are the two predicted contact transitions. A rate ratio of 44.5 for "release starts →
hold ends" is the kind of effect an annotation convention would produce (one label replaces
another). The p-values are at the permutation floor (1/200).

## 10. Runtime and memory (T11)

- Benchmark (1000 TRAIN videos, `provenance/benchmark_train1000.json`): design build 21 s;
  G4 fit ≤ 0.75 s for the largest target; aggregation 64 s. This replaced the earlier
  30–60 CPU-h estimate.
- Registered run, wall clock (`invocations.jsonl`): tune 31 min (8 workers), evaluate 1 min,
  nulls **3.7 h** (7 workers; 38 jobs × ~45–58 min each), stability 8 min, hypotheses 12 s.
  Summed in-worker job time 34.1 h, including the superseded attempt.
- Peak memory per worker ≤ 1.49 GB (null jobs); 7 concurrent workers kept ≥ 9 GB free. The run
  root on D: holds 1.4 GB per TRAIN design (null caches are deleted after each job).
- Per-target G4 fit on full TRAIN (A4 run): median 3.8 s, 90th percentile 18.9 s, max 63 s.

## 11. Failures and negative findings (reported with the same weight as the positives)

1. **K4 fails:** no individual interaction is reproducible at FDR ≤ 0.10. The model-level gain
   is robust; the specific interaction structure is not established. The registered statistic
   is shrinkage-inflated (§7).
2. Only 2 of 5 families pass. CONTACT_MANIPULATION, the family Stage 0 pointed to, fails the
   practical-size criterion (0.045 vs 0.057). VEHICLE_SUPPORT shows no reliable gain.
   ATTENTION_COMMUNICATION is at the null maximum.
3. Positive-only (classical Hawkes) history is clearly worse than signed history (−0.031).
   Classical excitation-only Hawkes is not the right model class here.
4. Both nulls are imperfect: N-circ manufactures +16% extra events, N-jit loses −9.3% to −9.8%.
5. A large share of the structure is synchronised annotation boundaries (17% exact ties).
6. H-c and H-d are untestable on EVAL (1 and 0 events in window).

## 12. Criterion-by-criterion decision (`decide_phase5h1`, unchanged)

| Criterion | Registered definition | Value | Result |
|---|---|---|:-:|
| K1 | CI_lo Δ(G4x, G2) > 0 and Δ > null max | 0.135 > 0; 0.145 > 0.031 | ✔ |
| K2 | CI_lo Δ(G4, G3) > 0 and Δ > null max | 0.076 > 0; 0.083 > 0.017 | ✔ |
| K3 | Δ(G4, G3) − null med ≥ 0.10 · Δ(G2, G1) | 0.070 ≥ 0.026 | ✔ |
| K4 | ≥ 1 discovery at FDR ≤ 0.10 and ρ_real > max ρ_null | 0 discoveries; 0.299 > 0.270 | ✘ |
| Family test | both halves CI_lo > 0, > null max, ≥ 10% of Δ_F(G2, G1) | SPATIAL, MOTION | 2 families |

**GO requires K1–K4 → not met (K4). NARROW requires K1 and ≥ 1 family → met. Label: NARROW.**

## 13. Three critical reviewer assessments

**Reviewer 1 (point processes / statistics):** "The model-level result is convincing *as
prediction*: large, consistent across both halves, far above two nulls, with inhibition properly
represented by an exp-link GLM. But the paper conflates it with 'Hawkes dynamics'. The
history block is a set of lag-window indicators in a multiplicative-intensity model; with signed
weights, short lags and 17% exact ties, this is a relational event model of annotation
boundaries, not self-exciting dynamics. The nulls are flawed in opposite directions (+16% /
−9.3% to −9.8% events), and K4's t-statistic is mis-specified under ridge shrinkage. I would accept a
narrowly worded predictive claim, not a mechanism claim."
*Response:* agreed. Our wording is "cross-predicate event history adds predictive information
about annotated boundaries". No Hawkes-mechanism claim is made, and both null defects and the K4
design flaw are disclosed. A confirmatory protocol should use a support-aware stability
statistic (§15).

**Reviewer 2 (video scene graphs):** "Everything is on human annotations of one dataset whose
relation instances were segmented by annotators who re-label spatial relations jointly. The
SPATIAL/MOTION gains are exactly where annotators change several labels at once. Nothing here
shows that a scene-graph model on real video would benefit. Contact transitions, the
semantically interesting part, fail the practical criterion."
*Response:* agreed on scope. The claim is restricted to annotation timing. Real-video
effectiveness is untested (no detector or tracker was used), and the contact-family result is
reported as a failure.

**Reviewer 3 (research integrity / evaluation):** "EVAL was already seen in aggregate in
Stage 0, the ridge grid was coarse, and there were four amendments. Is this a forking-paths
result?"
*Response:*
- All amendments were committed before the freeze (`289319e`). A4 was decided on optimiser
  diagnostics only and left the selection unchanged.
- The decision code and the analysis/figure scripts were committed before EVAL was unlocked.
- The decision rule was applied unchanged.
- The evidence is still EXPLORATORY by construction (prereg §2). That is why NARROW licenses
  only an independent confirmation, not a paper claim.

## 14. Is a method contribution justified?

**No, not at this stage.**
- NARROW licenses a *focused confirmatory replication*, not a method claim.
- What is supported (exploratory) is an **empirical finding**: on VidOR annotations,
  signed cross-predicate event history at ≤ 2 s improves held-out prediction of relation
  boundaries beyond heterogeneity, duration and current-state controls, mainly for SPATIAL and
  MOTION relations, and mainly via "other relation starts → this one ends".
- The model is a standard multiplicative-intensity point-process GLM (Truccolo et al. 2005;
  Perry & Wolfe 2013); it is not new.
- Classical positive-only Hawkes is worse than signed history. Individual interactions are not
  reproducible at FDR 0.10. The effect may be annotation convention.
- No novelty claim is made.

## 15. Next experiment needed for independent confirmation (proposed; NOT started)

**Confirmatory protocol C1 (to be pre-registered before any data access):**
1. **Data:** an independent dataset with relation start/end annotations from a *different
   annotation process*: ImageNet-VidVRD (segment-level relation annotations) or PVSG (panoptic
   video scene graphs). Annotation density and boundary resolution must be audited first.
   If boundaries are segment-quantised, ties dominate and the test is moot.
2. **Hypotheses (directional, fixed from this report):**
   - C1-a: Δ(G4, G3) > 0 overall.
   - C1-b: the ONSET → OFFSET block at ≤ 2 s carries the largest share.
   - C1-c: signed > positive-only.
   - C1-d: the SPATIAL and MOTION families show Δ_F > 0. Directional, one-sided tests.
3. **Models frozen from Phase 5H-1:** the same ladder, bins and ridge-selection procedure,
   refitted on the new training split. No new model classes.
4. **Nulls repaired** (decided now, before seeing new data):
   - interval jitter of observed endpoints only, within ±J, with window-reflection instead of
     clipping, so event counts are preserved;
   - a "boundary-tie" control that removes events tied with another predicate's event, to
     separate annotation synchronisation from lagged dependence.
5. **Stability:** replace the fold t-statistic with a held-out per-cell likelihood contribution
   on fold-out data, plus a minimum-support rule.
6. **Decision:** confirm if C1-a holds with CI_lo > 0 above both repaired nulls, and C1-b and
   C1-c hold.
7. Only after confirmation: a real-video test with detector/tracker outputs, to ask whether the
   annotation-timing structure helps scene-graph *inference*.

Until then, per the stop condition, the main paper track remains the robustness/evaluation paper
(Phases 3–4), and this result can appear only as an exploratory analysis section.

## 16. Artifacts

- Frozen parameters: `results/phase5h1/full/frozen/` (commit `289319e`).
- Results: `analysis.json`, `tables.md`, `completeness.json`, `posthoc_boundary.json`,
  `posthoc_foldnull_cells.json`, figures `fig_main_comparison`, `fig_null_comparison`,
  `fig_ablation_stability`, `fig_k4_stability` (PNG + PDF) in `results/phase5h1/full/`.
- Provenance: `results/phase5h1/full/provenance/`, plus the run root
  `D:/TAORG_runs/phase5h1/main/` (all checkpoints, `attempts.jsonl`, logs; not in git).
- Code: `src/taorg/hawkes/{glm,nulls1,hyp1,runner,phase5h1_run,phase5h1_analysis}.py`,
  `scripts/phase5h1/{run,benchmark,result_tables,result_figures,posthoc_boundary}.py`.
