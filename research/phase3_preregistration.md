# Phase 3 GO/NO-GO Pilot — Pre-registration

> **Status:** REGISTERED before any calibration or evaluation run.
> **Registered:** 2026-09-28 (commit hash recorded in §12 after the registration commit).
> **Config:** `configs/experiments/phase3_pilot.yaml` (SHA-256 recorded in every stage's metadata).
> **Decision code:** `src/taorg/experiments/phase3/h1.py` (committed together with this file).
>
> No calibration, evaluation, or smoke run had been executed when this document,
> the config and the decision code were committed. The only executions before
> registration were unit tests and a miniature integration test on a different
> tiny world/grid/seed set whose outputs were asserted for completeness only.

---

## 1. Question

**H1.** Does visibility-aware temporal gating provide a statistically and practically
meaningful improvement in the *persistence-versus-termination trade-off* compared with
ordinary (visibility-unaware) temporal smoothing under controlled observation corruption?

- Treatment: **Vis-Gated EMA** (`gated_ema`, Phase 1).
- Comparators for the decision: **EMA**, **Fixed Window**, **HMM** (visibility-unaware smoothers).
- **No Memory** is reported as the zero-persistence / immediate-termination reference
  but is not part of the decision rule (it is not a smoothing method).

H1 is *not* "GatedEMA obtains a higher F1". A method that persists everything trivially
maximises Persistence F1 while failing termination. The decision therefore compares methods
**at matched termination cost** (budget-constrained operating points) and **over whole
trade-off frontiers** (hypervolume).

## 2. Threat declared in advance: the oracle tautology

The Phase 2 occlusion injector flags every suppressed frame `OCCLUDED_OR_UNRELIABLE`, while
genuine terminations remain `VISIBLE`. Under that oracle the visibility flag *is* the answer
to "occluded or ended?", so a visibility-gated method is advantaged **by construction**.
A win in the oracle condition alone is therefore weak evidence. H1 PASS additionally
requires the advantage to survive (i) an imperfect visibility signal and (ii) base-model
flicker (visible false negatives), in which visibility gating is expected to be most fragile.

## 3. Synthetic world (known ground truth)

`taorg.data.synthetic_world` — all values in the config:

- 300 frames at 30 fps (10 s); 2 subject tracks, 4 object tracks, 6 predicates, 8 edges/video.
- Each edge is alive on one contiguous interval. With probability 0.6 it terminates after
  60–180 alive frames and is followed by ≥ 40 frames of *visible* absence; otherwise it stays
  alive to the end.
- Observed predicate scores: per-edge mean ~ U(0.6, 0.95), per-frame noise N(0, 0.05²), clipped.
- The seed determines the video; the same seed gives the same video in every condition and gap.

This world is a controlled abstraction, not a model of any real dataset. Nothing measured on it
may be stated as a property of Action Genome or any other benchmark.

## 4. Conditions

Corruption order: occlusion (Phase 2 injector, 1 gap per edge, ≥2 observed frames before and
after) → ID switch → flicker → visibility noise. Only the occlusion manifest defines
Persistence-F1 intervals (frozen definition §2).

| Name | Role | Gaps (frames) | Corruption beyond occlusion | Tuning |
|---|---|---|---|---|
| C0_oracle | **H1 primary** | 1,2,3,5,10,20 | none (oracle visibility) | own calibration |
| R1_visnoise_0.2 | **H1 robustness** | 1,2,3,5,10,20 | 20% of occluded absences reported VISIBLE and 20% of visible absences reported OCCLUDED | own calibration |
| R2_flicker_0.1 | **H1 robustness** | 1,2,3,5,10,20 | 10% of visible alive frames missed by the base model (still VISIBLE) | own calibration |
| SWV_{0.1,0.3,0.5} | sweep | 1,2,3,5,10,20 | visibility-flag error ρ ∈ {0.1,0.3,0.5} | own calibration |
| SWF_{0.05,0.2,0.3} | sweep | 1,2,3,5,10,20 | flicker q ∈ {0.05,0.2,0.3} | own calibration |
| S1_termination_under_occlusion | stress | 1,2,3,5,10,20 | object OCCLUDED for 15–40 frames right after every termination (scene property) | C0 parameters |
| S2_id_switch_d5 / d20 | stress | none | two same-class ID swaps of 5 / 20 frames | C0 parameters |
| S3_repeated_occlusion | stress | 3, 10 | three gaps per edge (may overlap) | C0 parameters |
| S4_unknown_all / half | stress | 1,2,3,5,10,20 | 100% / 50% of absence records report UNKNOWN | C0 parameters |
| S5_long_absence | stress | 30, 60 | none (long gaps, visibility returns) | C0 parameters |

Stress conditions reuse C0-selected parameters deliberately: they model rare situations a
deployed configuration meets without being re-tuned for them.

## 5. Parameter-selection protocol (fairness)

1. **Disjoint seeds.** Calibration seeds 1000–1019 (20 videos); evaluation seeds 2000–2029
   (30 videos); smoke seeds 9000–9001 (pipeline checks only, never analysed).
2. **Same procedure for every method.** Each method is searched exhaustively over the
   config-declared grid (No Memory 1, Fixed Window 32, EMA 420, Vis-Gated EMA 5040, HMM 720
   configurations). Shared parameters use identical value sets; the Vis-Gated EMA grid contains
   the EMA grid as its `decay_visible == decay_occluded` slice. `decay_unknown` stays at its
   Phase 1 default (= `decay_occluded`).
3. **Calibration objective.** For each calibrated condition, each method, and each termination
   budget B ∈ {2, 5, 10} frames (67 / 167 / 333 ms): among configurations whose mean
   termination delay over all calibration instances (20 seeds × 6 gaps, i.e. a *mixture* of gap
   lengths — gap length is unknown at deployment) is ≤ B, choose maximal mean Persistence F1;
   ties (|ΔF1| ≤ 1e-9) broken by lower churn, lower delay, lower false persistence, then grid
   order. If nothing is feasible, the minimum-delay configuration is used and flagged.
4. **Condition-matched tuning** for all calibrated conditions (each method tuned on data from
   the regime it is evaluated in), removing "the baseline was mis-tuned for this regime" as an
   explanation. Secondary (reported, not decisive): C0-tuned parameters evaluated on R1/R2.
5. **Frontiers.** Each method's calibration Pareto set (max F1, min delay) is evaluated on the
   evaluation seeds; frontier configurations are never re-selected on evaluation data.
6. Nothing is tuned on evaluation instances.

## 6. Metrics

Frozen (research/metric_definitions.md, unchanged): Edge Persistence P/R/F1 (occlusion
intervals only), Relationship Churn Rate, Termination Delay (with censoring), False Persistence
Rate, per-frame latency (mean/median/p95/p99). Per-instance value = frozen per-sequence value.
Reporting convention (not a definition change): Persistence F1 is recorded as NaN (not 0) when an
instance has no occlusion interval; delay/false persistence are NaN when no edge terminates.

Diagnostics (new, exploratory, **never used in the H1 rule**): clean-period recall, spurious
edge-frame rate, ghost edge-frames, post-gap recovery lag, ID-swap recall
(`taorg.experiments.phase3.evaluation`). They exist because the frozen metrics cannot see
drops outside occlusion gaps or hallucinated edges created by ID switches.

## 7. Statistical protocol

- Unit of analysis: evaluation seed (one video; n = 30). Methods are compared on identical
  corrupted instances (paired).
- Descriptive: mean, SD, n, 95% t-interval per method × condition × gap × budget.
- Paired: difference d = Vis-Gated EMA − comparator per seed; mean, SD, 95% percentile
  bootstrap CI (5000 resamples, seeded), Cohen's d_z, two-sided Wilcoxon signed-rank p.
- Multiplicity: Holm over the registered family — all Persistence-F1 tests of Vis-Gated EMA vs
  {EMA, Fixed Window, HMM} in the three H1 conditions × 6 gaps × 3 budgets (162 tests).
  All other tests are exploratory and labelled as such.
- Frontier: normalized dominated hypervolume in (F1 ∈ [0,1]) × (delay ∈ [1, 40] frames) of
  the evaluation means of each method's calibration Pareto set; paired bootstrap over seeds
  (2000 resamples). Secondary frontier: F1 vs false persistence (reported, not decisive).

## 8. H1 decision criteria (exact; implemented in `h1.py`)

Cell = (condition, gap, budget); 18 cells per H1 condition.

**Win vs comparator m** — all of: mean ΔF1 ≥ 0.05; bootstrap CI(ΔF1) lower bound > 0 and
Holm-adjusted p < 0.05; CI(Δdelay) upper ≤ +1 frame; CI(ΔFPR) upper ≤ +0.02;
CI(Δchurn) upper ≤ max(0.005, 0.10 × comparator mean churn).

**Loss vs m** — either (persistence loss) the mirror image of a win (ΔF1 ≤ −0.05, CI upper < 0,
Holm p < 0.05, with the comparator non-inferior on delay/FPR/churn), or (termination/churn loss)
mean ΔF1 < 0.05 and [CI(Δdelay) lower > 2 frames or CI(ΔFPR) lower > 0.05 or CI(Δchurn) lower >
churn margin].

**Cell win** = win vs all three comparators. **Cell loss** = loss vs any comparator.

**Frontier criterion** for (condition, gap): HV(Vis-Gated EMA) − HV(m) ≥ 0.02 with bootstrap CI
lower bound > 0, for every m.

| Decision | Rule |
|---|---|
| **PASS** | C0: ≥ 9/18 cell wins and 0 losses; R1 and R2 each: ≥ 6/18 wins and ≤ 2 losses; frontier criterion in ≥ 4/6 gaps (C0) and ≥ 3/6 gaps (R1, R2 each). |
| **FAIL** | C0 wins < 3, **or** total losses ≥ total wins over C0+R1+R2, **or** C0 frontier criterion in < 2/6 gaps. |
| **MIXED** | anything else (benefit confined to specific regimes, or bought with termination/churn penalties). |

Rationale: ties at very short gaps are expected for every method (any smoother bridges a
1-frame gap), so PASS does not require winning every cell; it requires a consistent advantage
at matched termination cost across a majority of the oracle grid, survival under both kinds of
realistic signal degradation, and frontier-level (not single-point) superiority.

A priori expectation (stated so it cannot be retrofitted): C0 should favour Vis-Gated EMA at
long gaps and tight budgets by construction (§2). R1 and R2 are the informative tests; we
consider a C0-only advantage (i.e. MIXED) a realistic outcome.

## 9. Failure analysis (categories and selection rule)

Categories (`failures.py`): short occlusion (C0, gap 1), long occlusion (C0, gap 20), true
termination (C0), termination under occlusion (S1), ambiguous/UNKNOWN visibility (S4 all),
visibility returning after long absence (S5, gap 60), repeated occlusion (S3, gap 10), ID
switch (S2, 20 frames), visibility-flag noise (R1, gap 10), flicker (R2, gap 10).
Primary-budget (5 frames) operating points. For each category the **worst** and the
**lower-median** positive-severity case over all evaluation seeds are saved (timeline JSON +
figure); "no failure observed" is recorded when no unit has positive severity. Severities are
defined in `failures.py` and reported for every method on the same unit.

## 10. Declared limitations (before results)

1. Synthetic world; no real VidSGG predictions; no real tracker.
2. All Phase 1 methods are per-frame (discrete-time); results are at 30 fps only.
3. The frozen metrics do not penalise drops outside occlusion intervals or hallucinated edges
   (diagnostics are provided but are not decisive).
4. The Vis-Gated EMA grid is larger than the EMA grid (it has one more parameter); selection is
   on calibration data only, so this cannot leak into evaluation, but it gives the treatment
   more tuning freedom.
5. The HMM is the Phase 1 hand-parameterised scaffold (no Baum–Welch); its grid search is the
   only fitting it receives.
6. Visibility noise is i.i.d. per frame; real visibility errors are temporally correlated.

## 11. Amendments

None at registration. Any later change must be dated, justified, and must state whether it was
made before or after viewing evaluation results.

- **A1 — 2026-09-28, before any calibration/evaluation run (engineering only, no protocol change).**
  The first smoke run hung: Windows "spawn" workers re-imported the CLI script, which imported
  SciPy/pandas/matplotlib at module level; with 24 workers the paging file was exhausted and
  workers died mid-import, leaving `multiprocessing.Pool` blocked. Fix: deferred imports in
  `scripts/phase3/run_phase3.py`; default worker count capped at 16. The smoke run (seeds
  9000–9001, grid stride 25) was then repeated and checked only for stage completion and
  artifact presence; its numbers were not inspected.

- **A2 — 2026-09-29, after calibration, before selection/evaluation (engineering only, no protocol
  change).** The first full calibration run (16 workers, started 05:28) stopped progressing after
  250/1080 instances: the log shows 200 instances at 05:48 and 250 only at 09:01, and the run was
  then terminated. The cause was not diagnosed (hypothesis, not verified: host sleep or memory
  pressure — free commit memory was ≈ 1.2 GB, the A1 failure mode). Calibration writes nothing until
  the stage ends, so the aborted run left no data. Calibration was restarted unchanged from clean
  commit `929d082` with 8 workers (09:14–11:19, run `phase3_calibrate_20260929_034436_e4842e`).
  Worker count cannot affect results (`pool.imap` is ordered; each instance is a deterministic
  function of its seed). Check: the C0 selected configurations (11) were recomputed serially on all
  120 C0 calibration instances; max |difference| to the saved means = 1.8e-15.

## 12. Registration record

- Registration commit: recorded in `research/paper_evidence_ledger.md` and `Memory.md` after commit.
