# Phase 3 Exit Report — GO/NO-GO pilot for visibility-aware temporal gating (H1)

> **Evidence class: controlled synthetic evidence.** No real VidSGG model, tracker or dataset was
> used. V-HPED is not implemented; nothing here is evidence about V-HPED, model-agnosticism or
> real-world effectiveness. All numbers are taken from generated artifacts under `results/phase3/`
> (abbreviated `R/`). Date: 2026-09-29.

## 1. Objective

Test the pre-registered hypothesis **H1**: does visibility-aware temporal gating (Vis-Gated EMA,
"GatedEMA") improve the *persistence-versus-termination trade-off* over visibility-unaware
smoothing (EMA, Fixed Window, HMM) under controlled observation corruption, at matched
termination cost and over whole trade-off frontiers? The outcome decides whether building
V-HPED is justified (GO/NO-GO).

## 2. Pre-registration commit

`d8728b3` (2026-09-28): `research/phase3_preregistration.md`, `configs/experiments/phase3_pilot.yaml`
(SHA-256 `0dea058edfc7a3b14246acde0dd5ab915325562c3a507bf46ce1132304a9ad52`, verified identical in
every stage's metadata) and the decision code `src/taorg/experiments/phase3/h1.py`. The protocol,
thresholds, seeds, grids, metrics and tests were not changed after registration.

## 3. Amendments

| # | Date | Timing | Content | Can it influence scientific results? |
|---|---|---|---|---|
| A1 | 2026-09-28 | before any calibration/evaluation | Windows spawn workers exhausted the paging file; heavy imports deferred, ≤ 16 workers (`6b6bced`) | No. Engineering only; smoke numbers were never inspected |
| A2 | 2026-09-29 | after calibration, before selection/evaluation | First calibration run stalled at 250/1080 instances (05:48 → 09:01) and was terminated with no output; re-run unchanged from clean `929d082` with 8 workers | No. Instances are deterministic functions of their seeds and `imap` is ordered. Serial recomputation of the 11 C0 selected configs on all 120 C0 calibration instances matched to 1.8e-15. Cause of stall not diagnosed (hypothesis: host sleep or memory pressure) |

Additions that are **not amendments** because they do not touch the registered rule:
(i) an exploratory formal Pareto-dominance definition (`929d082`, committed before any evaluation
data existed); (ii) `scripts/phase3/freeze_params.py` and `scripts/phase3/check_completeness.py`
(read-only verification/reporting, committed with the freeze in `2ce7d1f`).

## 4. Calibration protocol

Prereg §5. Calibration seeds 1000–1019 (20 videos). The 9 calibrated conditions are C0, R1, R2,
SWV_{0.1,0.3,0.5} and SWF_{0.05,0.2,0.3}, each with gaps {1, 2, 3, 5, 10, 20}: 1080 instances in
total. Every instance was evaluated exhaustively over 6213 configurations (No Memory 1,
Fixed Window 32, EMA 420, Vis-Gated EMA 5040, HMM 720). Run
`phase3_calibrate_20260929_034436_e4842e`, commit `929d082`, clean tree, 7468 s.

**Selection rule:** for each condition × method × budget B ∈ {2, 5, 10} frames, choose the maximal
mean calibration Persistence F1 among configurations with mean delay ≤ B. Ties (|ΔF1| ≤ 1e-9)
are broken by churn, then delay, then false persistence, then grid order. Stress conditions reuse
the C0 selections.

**Calibration verification** (`R/configs/frozen_parameters.json` → `calibration_verification`): all
12 checks pass. Not smoke; full grid; config hash matches; 1080/1080 expected instances with none
missing or duplicated; only calibration seeds; no evaluation seed; every config saw every
instance; no NaN F1/delay. No selection was infeasible.

## 5. Frozen parameter values

Full table with per-selection rationale: `R/tables/frozen_parameters.md`; machine-readable
artifact: `R/configs/frozen_parameters.json`. H1 conditions, budget 5 (primary):

| Condition | Method | Frozen params | Calib F1 | Calib delay | Why (registered rule) |
|---|---|---|---|---|---|
| C0 | Fixed Window | window 5, thr 0.0 | 0.799 | 5.00 | unique max F1 of 5 feasible |
| C0 | EMA | α 1.0, decay 0.8, thr 0.3 | 0.780 | 4.64 | unique max F1 of 180 feasible |
| C0 | Vis-Gated EMA | α 0.2, decay_occ 0.93, **decay_vis 0.0**, thr 0.05 | 1.000 | 1.00 | 282 tied at F1 = 1.0, tie broken by churn → delay → FPR → grid order |
| C0 | HMM | stay 0.99, emit_alive 0.5, emit_dead 0.05, thr 0.7 | 0.799 | 4.96 | 13 tied, tie-break |
| R1 | Vis-Gated EMA | α 0.5, decay_occ 0.99, decay_vis 0.5, thr 0.1 | 0.959 | 4.31 | unique max of 2377 feasible |
| R2 | Fixed Window | window 5 | 0.792 | 4.89 | unique max |
| R2 | EMA | α 1.0, decay 0.8, thr 0.3 | 0.770 | 4.52 | unique max |
| R2 | Vis-Gated EMA | α 1.0, decay_occ 0.95, decay_vis 0.5, thr 0.05 | 1.000 | 4.29 | 33 tied, tie-break |
| R2 | HMM | stay 0.95, emit_dead 0.01, thr 0.3 | 0.792 | 4.88 | 2 tied, tie-break |

(R1 baselines are identical to C0 because visibility-flag noise does not change their input.)

## 6. Parameter-freeze commit

**`2ce7d1f`** — "Phase 3: freeze calibrated parameters before evaluation". It contains the
selection outputs, frozen parameters with rationale and artifact SHA-256s, calibration artifacts,
Amendment A2 and the research-log entry. No evaluation artifact existed at that commit, and
`freeze_params.py` refuses to run if one does.

## 7. Evaluation protocol

Run `phase3_evaluate_20260929_055335_6ac64a` from commit `2ce7d1f` (git_dirty = false), 8 workers,
177 s. Evaluated: all 16 conditions × their gaps × 30 evaluation seeds = 2340 instances. For
each condition, the plan covers the frozen operating points of its tuning source for every
method and budget. For H1 conditions it also covers each method's calibration Pareto set; for
R1/R2 it also covers the C0 operating points (secondary mismatched-tuning analysis). Latency is
measured serially on C0, primary budget (`phase3_latency…`, 900 rows). Analysis, plots, failures
and dominance were computed from saved artifacts only. Nothing was re-tuned, removed or re-seeded.

## 8. Seeds

Calibration 1000–1019 (n = 20); evaluation 2000–2029 (n = 30); smoke 9000–9001 (pipeline checks
only, never analysed). Evaluation rows contain only seeds 2000–2029 and no calibration seed
(completeness check).

## 9. Experiment completeness

`R/tables/completeness.md`, `R/aggregated/completeness.json`: **COMPLETE**.

| | Expected | Actual | Missing | Duplicate | Failed |
|---|---|---|---|---|---|
| (configuration, instance) runs | 156,240 | 156,240 | 0 | 0 | 0 |
| corruption manifests | 2,340 | 2,340 (unique) | 0 | 0 | — |
| latency rows | 900 | 900 | 0 | — | — |

Undefined-by-design values are counted, not hidden: Persistence F1 is undefined for the 660
S2 runs (no occlusion gap). No run had an undefined delay.

## 10. Tests

- Test suite: **182 / 182 pass** (175 at Phase 3 start + 7 dominance tests), run before
  calibration, after the freeze and at exit.
- Reproducibility: the full evaluation stage was re-run into a scratch directory. **0 of
  156,240 rows differ** in any metric or identifier (excluding run ID and wall-clock
  runtime).

## 11. Raw results

`R/raw/evaluation_runs.csv.gz` (156,240 rows: condition, gap, seed, instance, manifest SHA-256,
config, every metric and diagnostic), `R/raw/manifests_evaluation.jsonl.gz` (every corruption
manifest + GT lifetimes), `R/raw/latency.csv.gz`, `R/raw/calibration_summary.csv.gz`, stage
metadata `R/raw/stage_*.json`, logs `R/logs/`.

## 12. Aggregated results

Full per method × condition × gap × budget tables (N, mean, SD, 95% t-CI of every metric):
`R/aggregated/descriptive_selected.csv`, rendered in `R/tables/results_<condition>.md` (16 files).
Primary budget, H1 conditions, matched tuning (No Memory omitted: F1 = 0, delay 1, churn ≈ 0.0145
everywhere):

| Cond | Gap | Method | N | F1 (SD) | F1 CI | Delay (frames) | False persistence | Churn |
|---|---|---|---|---|---|---|---|---|
| C0 | 1 | Fixed Window | 30 | 1.000 (0.000) | [1.000, 1.000] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0000 [0.0000, 0.0000] |
| C0 | 5 | Fixed Window | 30 | 0.889 (0.000) | [0.889, 0.889] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| C0 | 10 | Fixed Window | 30 | 0.571 (0.000) | [0.571, 0.571] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| C0 | 20 | Fixed Window | 30 | 0.333 (0.000) | [0.333, 0.333] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| C0 | 1 | EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0000 [0.0000, 0.0000] |
| C0 | 5 | EMA | 30 | 0.855 (0.031) | [0.844, 0.867] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0129 [0.0118, 0.0140] |
| C0 | 10 | EMA | 30 | 0.550 (0.024) | [0.541, 0.559] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0145 [0.0136, 0.0154] |
| C0 | 20 | EMA | 30 | 0.315 (0.020) | [0.307, 0.323] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0145 [0.0136, 0.0154] |
| C0 | 1 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 1.00 [1.00, 1.00] | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] |
| C0 | 5 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 1.00 [1.00, 1.00] | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] |
| C0 | 10 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 1.00 [1.00, 1.00] | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] |
| C0 | 20 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 1.00 [1.00, 1.00] | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] |
| C0 | 1 | HMM | 30 | 1.000 (0.000) | [1.000, 1.000] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0000 [0.0000, 0.0000] |
| C0 | 5 | HMM | 30 | 0.889 (0.000) | [0.889, 0.889] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| C0 | 10 | HMM | 30 | 0.571 (0.000) | [0.571, 0.571] | 4.95 [4.88, 5.02] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| C0 | 20 | HMM | 30 | 0.333 (0.000) | [0.333, 0.333] | 4.94 [4.88, 5.01] | 0.027 [0.026, 0.029] | 0.0145 [0.0136, 0.0153] |
| R1 | 1 | Fixed Window | 30 | 1.000 (0.000) | [1.000, 1.000] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0000 [0.0000, 0.0000] |
| R1 | 5 | Fixed Window | 30 | 0.889 (0.000) | [0.889, 0.889] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| R1 | 10 | Fixed Window | 30 | 0.571 (0.000) | [0.571, 0.571] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| R1 | 20 | Fixed Window | 30 | 0.333 (0.000) | [0.333, 0.333] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| R1 | 1 | EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0000 [0.0000, 0.0000] |
| R1 | 5 | EMA | 30 | 0.855 (0.031) | [0.844, 0.867] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0129 [0.0118, 0.0140] |
| R1 | 10 | EMA | 30 | 0.550 (0.024) | [0.541, 0.559] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0145 [0.0136, 0.0154] |
| R1 | 20 | EMA | 30 | 0.315 (0.020) | [0.307, 0.323] | 4.82 [4.68, 4.96] | 0.026 [0.025, 0.028] | 0.0145 [0.0136, 0.0154] |
| R1 | 1 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.29 [4.04, 4.53] | 0.023 [0.021, 0.025] | 0.0000 [0.0000, 0.0000] |
| R1 | 5 | Vis-Gated EMA | 30 | 0.992 (0.014) | [0.987, 0.998] | 4.18 [3.99, 4.38] | 0.022 [0.020, 0.024] | 0.0010 [0.0002, 0.0017] |
| R1 | 10 | Vis-Gated EMA | 30 | 0.953 (0.034) | [0.940, 0.966] | 4.36 [4.18, 4.54] | 0.024 [0.022, 0.025] | 0.0043 [0.0034, 0.0052] |
| R1 | 20 | Vis-Gated EMA | 30 | 0.808 (0.083) | [0.777, 0.839] | 4.44 [4.20, 4.67] | 0.024 [0.022, 0.026] | 0.0110 [0.0098, 0.0122] |
| R1 | 1 | HMM | 30 | 1.000 (0.000) | [1.000, 1.000] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0000 [0.0000, 0.0000] |
| R1 | 5 | HMM | 30 | 0.889 (0.000) | [0.889, 0.889] | 5.00 [5.00, 5.00] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| R1 | 10 | HMM | 30 | 0.571 (0.000) | [0.571, 0.571] | 4.95 [4.88, 5.02] | 0.028 [0.026, 0.029] | 0.0145 [0.0136, 0.0154] |
| R1 | 20 | HMM | 30 | 0.333 (0.000) | [0.333, 0.333] | 4.94 [4.88, 5.01] | 0.027 [0.026, 0.029] | 0.0145 [0.0136, 0.0153] |
| R2 | 1 | Fixed Window | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.85 [4.78, 4.92] | 0.027 [0.025, 0.028] | 0.0006 [0.0003, 0.0008] |
| R2 | 5 | Fixed Window | 30 | 0.876 (0.015) | [0.870, 0.881] | 4.85 [4.76, 4.93] | 0.027 [0.025, 0.029] | 0.0149 [0.0140, 0.0158] |
| R2 | 10 | Fixed Window | 30 | 0.556 (0.013) | [0.551, 0.561] | 4.90 [4.85, 4.96] | 0.027 [0.026, 0.029] | 0.0151 [0.0140, 0.0162] |
| R2 | 20 | Fixed Window | 30 | 0.326 (0.007) | [0.323, 0.328] | 4.90 [4.85, 4.96] | 0.027 [0.026, 0.029] | 0.0149 [0.0139, 0.0159] |
| R2 | 1 | EMA | 30 | 0.998 (0.012) | [0.993, 1.002] | 4.67 [4.52, 4.81] | 0.025 [0.024, 0.027] | 0.0008 [0.0004, 0.0011] |
| R2 | 5 | EMA | 30 | 0.844 (0.033) | [0.832, 0.856] | 4.68 [4.50, 4.86] | 0.026 [0.024, 0.027] | 0.0138 [0.0127, 0.0150] |
| R2 | 10 | EMA | 30 | 0.535 (0.029) | [0.524, 0.545] | 4.73 [4.58, 4.88] | 0.026 [0.024, 0.027] | 0.0153 [0.0142, 0.0165] |
| R2 | 20 | EMA | 30 | 0.307 (0.023) | [0.298, 0.316] | 4.71 [4.58, 4.85] | 0.026 [0.024, 0.027] | 0.0152 [0.0141, 0.0162] |
| R2 | 1 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.33 [4.22, 4.43] | 0.023 [0.022, 0.024] | 0.0006 [0.0004, 0.0009] |
| R2 | 5 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.33 [4.19, 4.47] | 0.023 [0.022, 0.025] | 0.0004 [0.0002, 0.0007] |
| R2 | 10 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.38 [4.27, 4.49] | 0.024 [0.022, 0.025] | 0.0007 [0.0003, 0.0011] |
| R2 | 20 | Vis-Gated EMA | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.38 [4.29, 4.48] | 0.024 [0.022, 0.025] | 0.0005 [0.0002, 0.0008] |
| R2 | 1 | HMM | 30 | 1.000 (0.000) | [1.000, 1.000] | 4.85 [4.78, 4.92] | 0.027 [0.025, 0.028] | 0.0006 [0.0003, 0.0008] |
| R2 | 5 | HMM | 30 | 0.876 (0.015) | [0.870, 0.881] | 4.85 [4.76, 4.93] | 0.027 [0.025, 0.029] | 0.0149 [0.0140, 0.0158] |
| R2 | 10 | HMM | 30 | 0.556 (0.013) | [0.551, 0.561] | 4.89 [4.82, 4.96] | 0.027 [0.025, 0.029] | 0.0152 [0.0141, 0.0163] |
| R2 | 20 | HMM | 30 | 0.326 (0.007) | [0.323, 0.328] | 4.90 [4.85, 4.96] | 0.027 [0.026, 0.029] | 0.0151 [0.0141, 0.0161] |

## 13. Confidence intervals

Descriptive: 95% t-intervals (above and in `R/tables/results_*.md`). Paired differences: 95%
percentile bootstrap, 5000 resamples, seeded. Hypervolume: paired bootstrap, 2000 resamples.
Many C0 intervals have zero width because both methods are deterministic functions of gap length
in this world. They express the absence of seed variability, not precision about a population.

## 14. Effect sizes

Budget 5, gaps 5/10/20 (all other cells: `R/tables/effect_sizes.md`, `R/aggregated/paired_tests.csv`):

| Cond | Gap | Comparator | ΔF1 [CI] | d_z | Holm p | ΔDelay [CI] | ΔFPR [CI] | ΔChurn [CI] |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | Fixed Window | +0.111 [+0.111, +0.111] | undef. (SD≈0) | <0.001 | -4.00 [-4.00, -4.00] | -0.028 [-0.029, -0.027] | -0.0145 [-0.0153, -0.0137] |
| C0 | 10 | Fixed Window | +0.429 [+0.429, +0.429] | undef. (SD≈0) | <0.001 | -4.00 [-4.00, -4.00] | -0.028 [-0.029, -0.026] | -0.0145 [-0.0153, -0.0137] |
| C0 | 20 | Fixed Window | +0.667 [+0.667, +0.667] | undef. (SD≈0) | <0.001 | -4.00 [-4.00, -4.00] | -0.028 [-0.029, -0.027] | -0.0145 [-0.0153, -0.0137] |
| C0 | 5 | EMA | +0.145 [+0.134, +0.156] | 4.73 | <0.001 | -3.82 [-3.95, -3.69] | -0.026 [-0.028, -0.025] | -0.0129 [-0.0139, -0.0118] |
| C0 | 10 | EMA | +0.450 [+0.442, +0.458] | 19.06 | <0.001 | -3.82 [-3.95, -3.69] | -0.026 [-0.028, -0.025] | -0.0145 [-0.0153, -0.0137] |
| C0 | 20 | EMA | +0.685 [+0.678, +0.692] | 33.88 | <0.001 | -3.82 [-3.95, -3.68] | -0.026 [-0.028, -0.025] | -0.0145 [-0.0153, -0.0137] |
| C0 | 5 | HMM | +0.111 [+0.111, +0.111] | undef. (SD≈0) | <0.001 | -4.00 [-4.00, -4.00] | -0.028 [-0.029, -0.026] | -0.0145 [-0.0153, -0.0137] |
| C0 | 10 | HMM | +0.429 [+0.429, +0.429] | undef. (SD≈0) | <0.001 | -3.95 [-4.00, -3.87] | -0.028 [-0.029, -0.026] | -0.0145 [-0.0153, -0.0137] |
| C0 | 20 | HMM | +0.667 [+0.667, +0.667] | undef. (SD≈0) | <0.001 | -3.94 [-4.00, -3.88] | -0.027 [-0.029, -0.026] | -0.0145 [-0.0153, -0.0137] |
| R1 | 5 | Fixed Window | +0.103 [+0.098, +0.108] | 7.21 | <0.001 | -0.82 [-1.00, -0.64] | -0.006 [-0.007, -0.004] | -0.0135 [-0.0144, -0.0127] |
| R1 | 10 | Fixed Window | +0.382 [+0.369, +0.393] | 11.32 | <0.001 | -0.64 [-0.81, -0.48] | -0.004 [-0.005, -0.003] | -0.0102 [-0.0111, -0.0092] |
| R1 | 20 | Fixed Window | +0.474 [+0.444, +0.502] | 5.71 | <0.001 | -0.56 [-0.78, -0.34] | -0.004 [-0.005, -0.002] | -0.0035 [-0.0044, -0.0026] |
| R1 | 5 | EMA | +0.137 [+0.126, +0.148] | 4.30 | <0.001 | -0.64 [-0.81, -0.46] | -0.004 [-0.005, -0.003] | -0.0119 [-0.0130, -0.0108] |
| R1 | 10 | EMA | +0.403 [+0.391, +0.414] | 11.92 | <0.001 | -0.46 [-0.61, -0.33] | -0.003 [-0.004, -0.002] | -0.0102 [-0.0111, -0.0092] |
| R1 | 20 | EMA | +0.493 [+0.463, +0.519] | 6.16 | <0.001 | -0.39 [-0.59, -0.17] | -0.002 [-0.004, -0.001] | -0.0035 [-0.0044, -0.0026] |
| R1 | 5 | HMM | +0.103 [+0.098, +0.108] | 7.21 | <0.001 | -0.82 [-1.00, -0.64] | -0.006 [-0.007, -0.004] | -0.0135 [-0.0144, -0.0127] |
| R1 | 10 | HMM | +0.382 [+0.369, +0.393] | 11.32 | <0.001 | -0.59 [-0.78, -0.42] | -0.004 [-0.005, -0.003] | -0.0102 [-0.0111, -0.0092] |
| R1 | 20 | HMM | +0.474 [+0.444, +0.503] | 5.71 | <0.001 | -0.51 [-0.71, -0.29] | -0.003 [-0.005, -0.002] | -0.0035 [-0.0044, -0.0026] |
| R2 | 5 | Fixed Window | +0.124 [+0.119, +0.130] | 8.11 | <0.001 | -0.52 [-0.61, -0.43] | -0.004 [-0.004, -0.003] | -0.0145 [-0.0153, -0.0136] |
| R2 | 10 | Fixed Window | +0.444 [+0.439, +0.449] | 34.33 | <0.001 | -0.52 [-0.62, -0.43] | -0.004 [-0.004, -0.003] | -0.0144 [-0.0152, -0.0136] |
| R2 | 20 | Fixed Window | +0.674 [+0.672, +0.677] | 99.09 | <0.001 | -0.52 [-0.61, -0.43] | -0.004 [-0.004, -0.003] | -0.0145 [-0.0153, -0.0136] |
| R2 | 5 | EMA | +0.156 [+0.144, +0.167] | 4.77 | <0.001 | -0.35 [-0.44, -0.27] | -0.002 [-0.003, -0.002] | -0.0134 [-0.0144, -0.0124] |
| R2 | 10 | EMA | +0.465 [+0.455, +0.475] | 16.28 | <0.001 | -0.35 [-0.43, -0.27] | -0.002 [-0.003, -0.002] | -0.0146 [-0.0155, -0.0137] |
| R2 | 20 | EMA | +0.693 [+0.685, +0.701] | 29.83 | <0.001 | -0.33 [-0.42, -0.24] | -0.002 [-0.003, -0.002] | -0.0147 [-0.0156, -0.0138] |
| R2 | 5 | HMM | +0.124 [+0.119, +0.130] | 8.11 | <0.001 | -0.52 [-0.61, -0.43] | -0.004 [-0.004, -0.003] | -0.0145 [-0.0153, -0.0137] |
| R2 | 10 | HMM | +0.444 [+0.439, +0.449] | 34.33 | <0.001 | -0.51 [-0.60, -0.42] | -0.003 [-0.004, -0.003] | -0.0145 [-0.0153, -0.0137] |
| R2 | 20 | HMM | +0.674 [+0.672, +0.677] | 99.09 | <0.001 | -0.52 [-0.61, -0.43] | -0.004 [-0.004, -0.003] | -0.0146 [-0.0154, -0.0138] |

**Caveat:** where the paired SD is ≈ 0 (floating-point noise), d_z is undefined. The generated
table prints values of ~10^15 or ±inf for these cells; they must not be quoted.

## 15. Holm-corrected tests

Registered family: Persistence F1 of Vis-Gated EMA vs {EMA, Fixed Window, HMM} × 3 H1 conditions
× 6 gaps × 3 budgets = **162 tests** (two-sided Wilcoxon, Holm). 97 significant at 0.05: 94
favour Vis-Gated EMA, **3 favour the comparators** (R1, budget 2, gap 1: ΔF1 = −0.032
[−0.048, −0.018], Holm p = 0.033, against each comparator). 6 significant tests have
|ΔF1| < 0.05, i.e. statistically but not practically significant (R2 budget 5 gap 3: +0.016
vs EMA). All other tests (other metrics, No Memory, sweeps, stress) are exploratory.

## 16. Persistence results

Fig. A (`R/plots/figA_persistence_f1_vs_gap.png`; per-budget versions in `supplementary/`).
Visibility-unaware methods lose persistence roughly as the fraction of the gap beyond their delay
budget: at budget 5, F1 ≈ 0.89 / 0.57 / 0.33 at gaps 5 / 10 / 20. Vis-Gated EMA: C0 1.000
everywhere (tautological); R1 0.992 / 0.953 / 0.808; R2 1.000 at every gap. At gaps < budget, all
smoothers bridge and are tied.

## 17. Termination results

Fig. B (false persistence) and Fig. C (delay). At the primary budget (5 frames), Vis-Gated EMA's
delay and false persistence are **lower** than every comparator's in all 54 H1 comparisons
(Δdelay −0.33 to −4.0 frames; every CI upper bound < 0). At budgets 2 and 10 this is **not
uniform**: in R1, Vis-Gated EMA's mean delay is up to +0.26 frames (budget 2) and +0.69 frames
(budget 10) *higher*. The largest Δdelay CI upper bound is **+0.96 frames, just inside the +1-frame
non-inferiority margin**. ΔFPR is at most +0.005 (CI upper ≤ 0.007, margin 0.02). No cell violated
a margin, but budget 10 in R1 is close to one.
**Termination fails badly in two stress conditions** (§24): termination under occlusion and
UNKNOWN visibility.

## 18. Churn results

Fig. D. Budget 5: Vis-Gated EMA churn 0.000–0.011 vs 0.013–0.015 for the smoothers in the H1
conditions. **Exception (secondary analysis):** C0-tuned Vis-Gated EMA under flicker has churn
0.154–0.178, about 10–12× the baselines (`R/tables/tuning_mismatch.md`). Known metric property:
churn averages never-GT edges in with churn 0, so ID-switch ghosts dilute it (prereg §6).

## 19. ID-switch results

S2, C0-tuned, budget 5 (`R/tables/stress.md`; diagnostics are non-decisive):

| Swap length | Method | True-key swap recall [CI] | Ghost edge-frames [CI] | Spurious rate |
|---|---|---|---|---|
| 5 | Fixed Window / EMA / HMM | 0.79 / 0.73 / 0.79 | 39.9 / 38.5 / 39.5 | 0.041–0.043 |
| 5 | Vis-Gated EMA | 0.888 [0.834, 0.943] | **152.4 [118.6, 186.2]** | 0.096 |
| 20 | Fixed Window / EMA / HMM | 0.26 / 0.25 / 0.26 | 112.0 / 110.8 / 111.8 | 0.092–0.093 |
| 20 | Vis-Gated EMA | **0.864 [0.795, 0.933]** | **251.1 [209.1, 293.1]** | 0.153 |

Both effects come from one code path, verified by reading `gated_ema.py`: a key with no record
in a frame is treated as UNKNOWN, and `decay_unknown` defaults to `decay_occluded`. So both the
true key and the ghost key are held.

## 20. Flicker results

R2 (q = 0.1) passes the robustness criterion (10 wins, 0 losses, frontier 5/6). SWF sweep
(matched tuning, budget 5): Vis-Gated EMA F1 ≥ 0.993 at every gap for q ∈ {0.05, 0.2, 0.3}, with
delay 3.98–4.42 vs 4.51–5.27 for the smoothers. The baselines stay ≤ 0.883 at gap ≥ 5
(`R/tables/sweeps.md`, Fig. F). The result depends on `decay_visible` > 0 being selected; the
oracle-tuned `decay_visible = 0` produces the churn failure in §18.

## 21. Visibility-noise results

R1 (ρ = 0.2) passes the robustness criterion with the thinnest margin in the rule (frontier
4/6 vs ≥ 3 required; gaps 1 and 2 fail with ΔHV 0.007 and 0.015 < 0.02). SWV sweep at gap 10,
budget 5: F1 0.998 (ρ 0.1), 0.953 (ρ 0.2), 0.888 (ρ 0.3), **0.582 (ρ 0.5)** vs 0.571 for Fixed
Window. No advantage remains when the flag is uninformative. Mismatched tuning (C0 parameters on
R1): F1 0.864 / 0.703 / 0.511 / 0.346 at gaps 1 / 5 / 10 / 20 vs Fixed Window 1.000 / 0.889 /
0.571 / 0.333, i.e. **worse than the visibility-unaware baselines** at gaps ≤ 10.

## 22. Pareto analysis

Fig. E (`R/plots/figE_pareto_gap10.png`); dominance: `R/tables/dominance.md`. Formal definition
(`src/taorg/experiments/phase3/dominance.py`, exploratory): set dominance of calibration-Pareto
frontiers evaluated on held-out seeds, inside the hypervolume box, with bootstrap p_dom.
Classes: A = dominates every simpler method at every (condition, gap) with p_dom ≥ 0.95;
B = dominates all of them in at least one (condition, gap); C = otherwise.

| Condition | Class | Where dominance fails |
|---|---|---|
| C0 | A | — (tautological) |
| R1 | B | gaps 1, 2, 3 (vs EMA, Fixed Window, HMM; p_dom ≤ 0.34) |
| R2 | B | gap 1 (all), gap 20 (EMA, Fixed Window; p_dom 0.37) |
| **Overall** | **B** | |

**Answer to Step 6: B — condition-specific improvement.** GatedEMA does *not* Pareto-dominate the
simpler methods under the formal definition, except in the oracle condition. It is never
dominated either (reverse p_dom = 0 everywhere): the frontiers cross at short gaps.

## 23. Hypervolume analysis

`R/tables/hypervolume.md`. Normalised dominated HV (F1 × delay ∈ [1, 40] frames). Minimum HV gain
of Vis-Gated EMA over the strongest comparator [95% bootstrap CI]:

| Gap | C0 | R1 | R2 |
|---|---|---|---|
| 1 | 0.013 [0.011, 0.014] ✗ | 0.007 [0.005, 0.008] ✗ | 0.008 [0.003, 0.011] ✗ |
| 2 | 0.023 [0.022, 0.024] ✓ | 0.015 [0.014, 0.017] ✗ | 0.021 [0.019, 0.022] ✓ |
| 3 | 0.034 [0.032, 0.035] ✓ | 0.025 [0.023, 0.027] ✓ | 0.031 [0.029, 0.033] ✓ |
| 5 | 0.057 [0.055, 0.058] ✓ | 0.043 [0.040, 0.045] ✓ | 0.052 [0.049, 0.054] ✓ |
| 10 | 0.110 [0.107, 0.112] ✓ | 0.083 [0.080, 0.086] ✓ | 0.106 [0.102, 0.109] ✓ |
| 20 | 0.211 [0.211, 0.211] ✓ | 0.157 [0.152, 0.162] ✓ | 0.210 [0.209, 0.212] ✓ |

(✓ = registered frontier criterion ΔHV ≥ 0.02 with CI lower bound > 0 against every comparator.)
Several differences at gaps 2–3 are just above the 0.02 margin. The frontier count is sensitive
to that margin.

## 24. Failure cases

Categories and selection rule: prereg §9; artifacts `R/failures/` (timeline JSON + PNG/PDF,
worst and lower-median positive case per category), index `R/failures/failure_index.json`.
Budget 5, C0-tuned parameters unless the category is R1/R2.

| Category | Units with GatedEMA severity > 0 | GatedEMA mean severity | Baselines mean | Saved cases |
|---|---|---|---|---|
| Short occlusion (C0, g1) | 0 / 240 | 0 | 0 (No Memory 1) | none — no failure observed |
| Long occlusion (C0, g20) | 0 / 240 | 0 missed frames | ≈ 16 | none — no failure observed |
| Repeated occlusion (S3, g10) | 0 / 240 | 0 | 17.4–18.1 | none — no failure observed |
| True termination (C0) | 144 / 144 | delay 1.0 | 4.8–5.0 | worst s2000, typical s2014 (not a failure: minimum possible delay) |
| **Termination under occlusion (S1)** | 144 / 144 | **delay 27.5** | 4.8–5.0 | worst s2015 (41 frames), typical s2014 (27) |
| **UNKNOWN visibility (S4-all)** | 144 / 144 | **delay 38.3** | 4.8–5.0 | worst s2003 (41), typical s2029 (39) |
| **Long absence (S5, g60)** | 237 / 237 | 24.0 missed + recovery frames | 56–58 | worst s2006 (40), typical s2008 (23) |
| **ID switch (S2, 20 frames)** | 58 / 60 | 170.5 ghost edge-frames | 117.8–118.3 | worst s2024 (498), typical s2020 (153) |
| Visibility-flag noise (R1, g10) | 65 / 240 | 0.88 missed frames | ≈ 6.0–6.2 | worst s2003 (7), typical s2005 (3) |
| Flicker (R2, g10) | 22 / 240 | 0.10 flips | ≈ 2.1 | worst s2001 (2), typical s2010 (1) |

**Observation vs hypothesised explanation:**

1. *Termination under occlusion.* OBS: every terminating edge whose object becomes occluded is
   held about 27 frames (maximum 41), about 5.5× the smoothers, with false persistence 0.18 vs
   0.03. HYP: occluded absence is, by design, not evidence of termination, so any visibility-gated
   rule is blind to endings that coincide with occlusion. The duration is set by
   `decay_occluded`. This is a structural limitation, not a tuning artifact (not separately
   tested).
2. *UNKNOWN visibility.* OBS: delay 38.3, false persistence 0.26 when every absence is UNKNOWN;
   harmless when only half are (delay about 2.1). HYP: `decay_unknown` defaults to
   `decay_occluded`, so "no information" is treated as "occluded". A separate UNKNOWN policy is
   an untested design choice.
3. *ID switches.* OBS: better true-key recall, but 2.2–3.8× ghost edge-frames. HYP: the
   no-record → UNKNOWN code path (verified in code) holds ghost keys; persistence is applied to
   identity errors as well as to occlusions.
4. *Long absence.* OBS: F1 0.998 at 30 frames but 0.750 at 60 frames. HYP: a finite hold horizon
   of about ln(0.05/score)/ln(0.93) ≈ 38 frames for the frozen parameters.
5. *Visibility-flag noise and flicker.* OBS: residual misses of 0.88 frames per edge (R1) and
   0.10 flips (R2), far below the baselines. The noise-driven failures appear under **mismatched
   tuning** (§18, §21), not under matched tuning.
6. *Short occlusion.* No failure, and no advantage either: every smoother bridges a 1-frame gap.
   In R1 at budget 2, gap 1, the noisy flag costs 0.032 F1.

## 25. H1 criterion-by-criterion decision

Rule: prereg §8, implemented in `src/taorg/experiments/phase3/h1.py` (registered in `d8728b3`);
output `R/aggregated/h1_decision.json`, `R/aggregated/h1_cells.csv`, `R/tables/h1_decision.md`.
Cell wins are exactly the cells with gap ≥ budget in all three conditions.

| # | Criterion | Required threshold | Observed result | Satisfied? | Evidence artifact |
|---|---|---|---|---|---|
| P1 | C0 cell wins | ≥ 9 / 18 | 10 / 18 | Yes | `h1_decision.json` → wins.C0_oracle |
| P2 | C0 cell losses | 0 | 0 | Yes | → losses.C0_oracle |
| P3 | C0 frontier criterion | ≥ 4 / 6 gaps | 5 / 6 (fails gap 1: ΔHV 0.013 vs EMA) | Yes | → hv_gaps_won; `hypervolume.csv` |
| P4 | R1 cell wins | ≥ 6 / 18 | 10 / 18 | Yes | → wins.R1_visnoise_0.2 |
| P5 | R1 cell losses | ≤ 2 | 0 | Yes | → losses |
| P6 | R1 frontier criterion | ≥ 3 / 6 | 4 / 6 (fails gaps 1, 2) | Yes (margin 1 gap) | → hv_gaps_won |
| P7 | R2 cell wins | ≥ 6 / 18 | 10 / 18 | Yes | → wins.R2_flicker_0.1 |
| P8 | R2 cell losses | ≤ 2 | 0 | Yes | → losses |
| P9 | R2 frontier criterion | ≥ 3 / 6 | 5 / 6 (fails gap 1) | Yes | → hv_gaps_won |
| F1 | FAIL if C0 wins < 3 | — | 10 | Not triggered | → fail_checks |
| F2 | FAIL if total losses ≥ total wins (C0+R1+R2) | — | 0 losses vs 30 wins | Not triggered | → fail_checks |
| F3 | FAIL if C0 frontier < 2 / 6 | — | 5 / 6 | Not triggered | → fail_checks |

Per-cell win conditions (ΔF1 ≥ 0.05; CI lower > 0; Holm p < 0.05; CI upper Δdelay ≤ +1;
CI upper ΔFPR ≤ +0.02; CI upper Δchurn ≤ max(0.005, 10%)) are evaluated per comparator in
`h1_cells.csv`. The thresholds in `h1_decision.json → criteria` match prereg §8 exactly.

## 26. Final decision

# **H1 = PASS**

The PASS is mechanically correct under the registered rule and is not reinterpreted. Its
*informativeness* is narrower than the word suggests (C0 circular; win count fixed by the
gap ≥ budget structure; dependence on condition-matched tuning), as recorded in §27–§30.

## 27. Evidence-ledger changes

`research/paper_evidence_ledger.md` (change log at the bottom):

| Claim | Before | After |
|---|---|---|
| L1 gating improves trade-off (controlled corruption) | UNTESTED | **SUPPORTED** (synthetic, matched tuning only) |
| L2 holds with imperfect visibility | UNTESTED | **PARTIAL** |
| L3 holds under flicker | UNTESTED | **SUPPORTED** (synthetic, matched tuning only) |
| L4 bridging ⇒ delay coupling | UNTESTED | **PARTIAL** (empirical; not exact on the mean for HMM; no proof) |
| L5 failure modes characterised | UNTESTED | **SUPPORTED** (as characterisation) |
| L10 protocol reproducible | PARTIAL | **SUPPORTED** (synthetic pipeline) |
| L16 robust to tuning mismatch | new | **NOT SUPPORTED** |
| L17 formal Pareto dominance | new | **NOT SUPPORTED** (class B only) |
| L18 safe under termination-in-occlusion / UNKNOWN / ID switch | new | **NOT SUPPORTED** |
| L19 advantage with uninformative flag | new | **NOT SUPPORTED** (expected negative control) |
| L20 negligible latency overhead | new | **PARTIAL** (synthetic scale) |
| L21 needed beyond a visibility-aware probabilistic baseline | new | **UNTESTED** |

L1 was not upgraded on p-values alone. The registered practical-effect, CI, non-inferiority and
frontier criteria were all met in all three conditions with large effects. Its scope is
explicitly limited.

## 28. Claims now supported (and only in this scope)

On the controlled synthetic world, at 30 fps, with i.i.d. corruption and condition-matched
tuning:
- Gating on an informative visibility signal improves persistence through occlusion without
  violating the registered delay/false-persistence/churn non-inferiority margins against EMA,
  Fixed Window and HMM (lower delay at the primary budget; up to +0.69 frames mean at budget 10
  in R1) (L1), including under 10–30% base-model flicker (L3).
- The gain tracks the informativeness of the visibility signal and disappears when the flag is
  uninformative (dose–response, L19 as negative control).
- The failure modes of visibility gating are characterised (L5).
- The Phase 3 pipeline is reproducible bit-for-bit (L10).

## 29. Claims still unsupported

Model-agnostic effectiveness (L6), any real-data effect, V-HPED vs GatedEMA (L7), lazy/eager
equivalence and complexity (L8, L9), protocol novelty (L11), frame-rate invariance (L15),
robustness to tuning mismatch (L16, NOT SUPPORTED), formal Pareto dominance (L17, NOT SUPPORTED),
safety under termination-in-occlusion / UNKNOWN / ID switches (L18, NOT SUPPORTED), the need for
this gating form over a visibility-aware HMM (L21), and a proof of the bridging–delay coupling
(L4).

## 30. Reviewer attack

### Reviewer A — Dynamic / video scene graph expert

- **Strongest aspect:** separating "gap bridging" from "termination" and reporting both at matched
  cost is more honest than the usual recall-only temporal-consistency evaluation. The failure
  catalogue (ghosts after ID switches, termination during occlusion) is useful to practitioners.
- **Weakest aspect:** there is no scene graph model. Eight independent edges with i.i.d. scores
  are not a VidSGG output. Real predictions have correlated predicate confusions, many
  simultaneous candidate predicates per pair, and missing detections that remove *all* edges of
  an object at once.
- **Confounds:** visibility in the synthetic world is generated by the same process that removes
  observations. In real systems visibility comes from a tracker whose failures correlate with the
  very events of interest (occluders, fast motion, termination by walking away).
- **Unsupported claims to watch for:** any wording suggesting occlusion handling is new (ledger
  L12 is INVALIDATED) or that the method improves VidSGG.
- **Missing experiments:** Action Genome with ≥ 2 base models (e.g. a STTran-style and a
  DSG-DETR/OED-style model); tracker-derived visibility; comparison with object-permanence /
  4D scene-graph approaches on the same outputs; per-predicate analysis (contact vs attention
  predicates behave differently).
- **Is H1 convincing?** As a synthetic sanity check, yes. As evidence for the paper's real-world
  motivation, no.
- **Response / plan:** Phase 6–7 real-data evaluation with base-model-agnostic adapters;
  object-level occlusion (all edges of an occluded object suppressed together) added to the
  synthetic world before Phase 4 conclusions are drawn.

### Reviewer B — Temporal modelling / algorithms expert

- **Strongest aspect:** the matched-budget design exposes a real structural fact. Visibility-unaware
  memory couples gap length to termination delay, and an extra information channel is required to
  break that coupling.
- **Weakest aspect:** that fact is nearly a theorem, and the experiment mostly confirms it. The
  win count equals the number of (gap ≥ budget) cells, so H1 PASS was close to guaranteed given
  an informative flag. The bound itself (L4) is unproven and, as stated, not exact on the mean for
  the HMM.
- **Confounds:** the comparison gives the treatment information the comparators do not receive.
  The fair baseline is a visibility-aware HMM or Bayesian filter, i.e. the same information in a
  principled model. Without it, "gating" and "using visibility at all" cannot be separated. The
  treatment grid is also 12× larger than EMA's.
- **Unsupported claims:** that continuous-time or Hawkes dynamics are needed. Phase 3 shows that
  a one-parameter-per-state per-frame rule already reaches F1 ≥ 0.95 under the tested
  degradations, which leaves little headroom.
- **Missing experiments:** a visibility-aware HMM; frame-rate variation; a proof of the coupling
  bound with precise conditions; an analysis of the hold horizon (closed-form expiry as a function
  of the decay rates and threshold).
- **Is H1 convincing?** Convincing that visibility information helps. Not convincing that anything
  beyond a trivial gate is needed.
- **Response / plan:** Phase 4 must include a visibility-aware HMM baseline and must be
  pre-registered against GatedEMA on conditions where GatedEMA demonstrably fails (§31). Write
  the L4 proof in `research/derivations/`.

### Reviewer C — Statistics / experimental methodology

- **Strongest aspect:** pre-registration before calibration, disjoint seeds, a frozen-parameter
  commit before evaluation, complete-matrix verification, Holm over a declared family,
  non-inferiority margins, a bit-identical rerun, and negative results retained. This is
  unusually careful.
- **Weakest aspect:** the unit of analysis is a synthetic video. In C0 there is **no
  between-video variance** (zero-width CIs, undefined d_z), so the p-values are driven by 30
  identical differences. They quantify seed-level Monte-Carlo noise of the simulator, not
  generalisation. Two of three H1 conditions are near-deterministic for the baselines.
- **Confounds:** the definition of "win" interacts with the grid (gap ≥ budget). The frontier
  criterion has differences within 0.001–0.005 of the 0.02 margin at gaps 2–3, so the frontier
  counts are margin-sensitive. Condition-matched tuning presumes knowledge of the regime; the
  secondary mismatched analysis reverses the R1 result.
- **Unsupported claims:** "statistically significant" should not be used as a synonym for
  important. 6 of 97 significant tests are below the practical margin, and 3 significant results
  go against the treatment.
- **Missing experiments:** a regime-agnostic tuning protocol (calibrate one configuration on a
  mixture of regimes), correlated noise models, sensitivity of the PASS to the ΔHV margin and to
  the budget set, and more heterogeneous worlds, so that seed variance means something.
- **Is H1 convincing?** The procedure is sound and the decision is traceable. The evidence is
  strong about the simulator and weak about the world.
- **Response / plan:** in Phase 4, report the mixed-regime tuning result as a *primary* outcome,
  and pre-register margin-sensitivity analyses. Treat C0 as a unit test in the paper, not as a
  result.

## 31. Recommended Phase 4 (design only — NOT started, NOT implemented)

H1 = PASS, so Phase 4 (V-HPED) is justified **as a question, not as an expectation**. The
central question must be:

> **Does V-HPED provide a meaningful improvement beyond the much simpler GatedEMA baseline?**

Phase 3 shows little headroom where GatedEMA already works: F1 0.95–1.00 at lower delay in
R1/R2 with matched tuning. Improvements must therefore be sought, and pre-registered, where
GatedEMA **fails**. If V-HPED cannot beat GatedEMA there, the Hawkes complexity is not justified,
and the paper should become a protocol + analysis paper with GatedEMA as the method.

Proposed (to be pre-registered before any V-HPED tuning):
1. **Comparators:** GatedEMA (primary), a *visibility-aware HMM* (new, needed to separate
   "probabilistic model" from "Hawkes dynamics"), EMA, Fixed Window, HMM.
2. **Conditions, derived from Phase 3 failures and fixed now, before V-HPED exists:**
   (a) one configuration tuned on a mixture of regimes and evaluated on all of them (addresses
   L16); (b) temporally correlated (Markov/bursty) visibility errors; (c) termination under
   occlusion (S1); (d) UNKNOWN/no-record handling and ID-switch ghosts (S2, S4); (e) long absences
   beyond the GatedEMA hold horizon (S5); (f) frame rates 15/30/60 fps; (g) object-level
   occlusion (all edges of an object together). The Phase 3 H1 conditions are kept as
   **non-inferiority** checks: V-HPED must not lose there.
3. **Candidate V-HPED components** (each justified only by an ablation against GatedEMA):
   continuous-time edge dynamics with visibility-conditioned decay (frame-rate invariance);
   cross-predicate excitation/inhibition (for example, a new predicate on the same pair as
   termination evidence during occlusion, targeting S1); hysteresis (churn under flicker without
   regime-specific tuning); closed-form expiry (a hold horizon derived rather than fixed);
   lazy graph maintenance with an eager reference (ADSA component, correctness-tested
   separately).
4. **Decision rule (sketch):** GO for V-HPED only if it beats GatedEMA by a pre-set practical
   margin on frontier HV in a majority of failure-derived conditions, with no loss in the
   Phase 3 conditions and no loss to the visibility-aware HMM. Otherwise record V-HPED as NOT
   JUSTIFIED.
5. **Risk to guard against:** choosing conditions after seeing V-HPED results. Mitigation:
   register the Phase 4 conditions from this report before implementing V-HPED.

## 32. Journal-paper roadmap

**ALREADY SUPPORTED (synthetic scope only)**
- Visibility gating improves the persistence/termination trade-off at matched cost (L1, L3).
- The gain is dose-dependent on flag informativeness (L19 negative control).
- Failure-mode characterisation of visibility gating (L5).
- Reproducible corruption/metric/statistics pipeline (L10).

**NEEDS PHASE 4**
- V-HPED vs GatedEMA on failure-derived conditions (L7); visibility-aware HMM baseline (L21).
- Robustness to tuning mismatch / regime-agnostic tuning (L16).
- Frame-rate invariance of continuous-time dynamics (L15).
- Correlated visibility errors and object-level occlusion.

**NEEDS REAL-DATA VALIDATION**
- Action Genome (annotation availability and licence to verify), with ≥ 2 sufficiently
  different base VidSGG models (pipeline: base model → TAORG / V-HPED → temporal relationship
  outputs).
- Tracker-derived visibility and its real error rates.
- Model-agnostic claim (L6): cannot be SUPPORTED until multiple base models are evaluated.

**NEEDS ABLATION**
- Each V-HPED component vs GatedEMA; `decay_unknown` policy; no-record handling; hysteresis;
  cross-predicate terms; closed-form expiry vs fixed threshold.
- Sensitivity of conclusions to the ΔHV margin, the budget set and the grid size (tuning
  freedom).

**NEEDS MATHEMATICAL PROOF**
- Bridging–delay coupling for visibility-unaware memory, with correct conditions (L4).
- V-HPED closed-form decay/expiry, softplus-likelihood integral, threshold domain (Memory open
  decisions 3–4).
- Lazy/eager equivalence and O((k_obs+k_vis+k_exp) log A) per-frame cost (L8, L9).

**NEEDS LITERATURE VERIFICATION**
- Every `[VERIFY]` citation in `paper/02_introduction.md` and `03_related_work.md`.
- Novelty of the matched-termination-cost robustness protocol for VidSGG (L11).
- Prior visibility-/occlusion-aware tracking-by-detection persistence (e.g. track "lost" states)
  and object-permanence / 4D scene-graph work, to position the contribution precisely.

**NEEDS WRITING**
- Abstract (only after real-data results), final contribution list, related work, method
  (after Phase 4), theory section (after proofs), real-data experiments, discussion. The
  synthetic section (`paper/07_synthetic_results.md`) exists as a draft; figures need a cosmetic
  fix (Fig. E panel titles overlap).

## 33. All commits created in this session

| Commit | Message |
|---|---|
| `2ce7d1f` | Phase 3: freeze calibrated parameters before evaluation |
| `867b7be` | Phase 3: evaluation run and pre-registered analysis outputs |
| (this report) | Phase 3: exit report, evidence ledger, results notes, paper sections (hash in `Memory.md` / `git log`) |

Earlier Phase 3 commits: `d8728b3` (pre-registration), `6b6bced` (A1), `39bb238`
(reporting/paper skeleton/ledger), `929d082` (dominance definition).

## 34. Final git status

After the exit-report commit, the working tree is clean except for two untracked files that
predate the pre-registration: `scripts/run_pilot.py` and `tests/fixtures/pilot_fixtures.py`
(2026-09-28 01:36–01:38, an earlier ad-hoc pilot runner superseded by `scripts/phase3/`). They
were neither used, deleted nor committed; the user should decide their fate.

## 35. Final test count

**182 passed, 0 failed** (`python -m pytest -q`).

---

**STOP.** Phase 3 is complete. V-HPED has not been implemented, Phase 4 has not been started,
Action Genome has not been integrated, and the research hypothesis has not been modified. Awaiting
explicit approval.
