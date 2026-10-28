# TAORG Research — Claims Evidence Ledger

> **Purpose:** Track the evidential status of every paper claim.
> Each claim must eventually map to specific experiments, tables, or
> figures with traceable run IDs.
>
> Do NOT mark a claim as "supported" without linking to actual experiment artifacts.
> Negative or neutral results must also be recorded here.

## Claims Registry

| ID | Claim | Status | Evidence | Run IDs | Notes |
|----|-------|--------|----------|---------|-------|
| H1 | Visibility gating improves persistence/termination trade-off vs visibility-unaware methods | UNTESTED | — | — | Central hypothesis. Tested in Phase 3 pilot. |
| H2 | Cross-predicate Hawkes interactions provide measurable benefit beyond gated EMA | UNTESTED | — | — | May fail; if so, scope claims accordingly. |
| H3 | Learned temporal parameters outperform hand-set parameters | UNTESTED | — | — | Requires Phase 4+ V-HPED implementation. |
| H4 | Lazy decay + expiry scheduling matches eager reference within tolerance | SUPPORTED (synthetic traces; proof under conditions L1 rho_unk=0, L2 gamma=0) | research/derivations/lazy_expiry.md; tests/property/test_lazy_equivalence.py (24 randomized streams, 0 decision mismatches); benchmark 0 mismatches over 16 grid points | `outputs/runs/complexity/20261027T202322Z/results.json` | Exact only under L1-L2; otherwise falls back to eager. |
| C1 | Lazy/heap achieves O((k_obs+k_vis+k_exp) log A) vs O(A) eager | SUPPORTED with refinement | Derived as O(k_touch P + (k_touch + k_wake)(P + log H)) vs Theta(A P); counters match; latency faster for touch fraction <= 0.5 (about 1.3x-5.5x), SLOWER (about 0.77x) at touch fraction 1 | `outputs/runs/complexity/20261027T202322Z/results.json` | Machine-dependent latency; counters machine-independent. |

## Synthetic sanity signals (NOT paper evidence — claims above stay UNTESTED)
Source: `outputs/runs/vhped_synthetic/run_20261027T194003Z.json` (3 seeds, 20/10/20 synthetic videos, threshold chosen on val by frame F1; git repo not present in this checkout so commit = unknown).

| Signal | Observation (test, mean ± std) | Bears on |
|---|---|---|
| Removing visibility collapses occlusion persistence | V-HPED-noVis persistence F1 0.497±0.122 vs full 0.942±0.011 | H1 |
| Fitted V-HPED drops terminated edges faster than Gated EMA | termination delay 3.23±0.23 vs 5.58±0.28 frames; false persistence 0.072 vs 0.119 | H1/H2 |
| Hawkes terms vs fitted VT-HMM (no excitation) is a TRADE-OFF, not a win | VT-HMM: frame F1 0.967 vs 0.957, persistence F1 0.944 vs 0.942; V-HPED: delay 3.23 vs 5.29 | H2 (mixed) |
| Cross-predicate terms add little over self-only | delay 3.23±0.23 vs 3.62±0.14; frame F1 equal within std | H2 (weak) |
| Learned gamma[putting_down→holding] recovers the planted cue | 0.85 / 0.86 / 0.69 across seeds (other off-diagonals ~0) | H2 mechanism |
| Fitted > hand-set | persistence F1 0.942 vs 0.919; delay 3.23 vs 4.41 | H3 |
| V-HPED churn higher than Gated EMA | 0.048±0.013 vs 0.035±0.012 | negative |

## Synthetic Phase 8 ablation matrix (NOT paper evidence on real data)
Source: `outputs/runs/20261027_202753_a7314fe5` (5 predeclared seeds x 2 conditions, 20/10/20 videos, val-selected thresholds, paired bootstrap over test videos). Generated tables: `paper/generated/`.

| Signal | Occlusion + flicker | + ID switches | Bears on |
|---|---|---|---|
| Full V-HPED vs Gated EMA, frame F1 (paired diff, 95% CI) | +0.008 [+0.006, +0.010], 5/5 seeds | **-0.015 [-0.018, -0.012], 0/5 seeds** | H1/H2 |
| Full V-HPED vs Gated EMA, termination delay | -1.63 [-1.87, -1.38] frames | -1.57 [-1.78, -1.36] frames | H1/H2 |
| Full vs VT-HMM (no Hawkes), frame F1 | **-0.011 [-0.013, -0.009]** | -0.031 [-0.035, -0.028] | H2 (negative on accuracy) |
| Full vs VT-HMM, termination delay | -1.90 [-2.14, -1.67] | -0.83 [-0.99, -0.65] | H2 (positive on termination) |
| Full vs self-only (cross terms), frame F1 | +0.003 [+0.002, +0.004] | +0.003 [+0.002, +0.003] | H2 (small) |
| Fitted vs hand-set, frame F1 | +0.008 [+0.006, +0.010] | **-0.033 [-0.036, -0.029]** | H3 (condition-dependent) |
| ECE of alive probability (full fit) | 0.030 (Gated EMA score 0.286) | **0.211** (hand-set 0.028) | calibration |
| Visibility removed, persistence F1 | 0.571 vs 0.951 | 0.432 vs 0.895 | H1 |


| Date | Result | Experiment | Impact |
|------|--------|------------|--------|
| 2026-10-28 | Synthetic ID switches: fitted V-HPED loses frame F1 to Gated EMA, EMA and hand-set V-HPED; ECE collapses to 0.21 | `outputs/runs/20261027_202753_a7314fe5` | H3 does not hold under identity noise; robust fitting needed |
| 2026-10-28 | VT-HMM (visibility, no Hawkes) has the best frame F1 under occlusion + flicker | `outputs/runs/20261027_202753_a7314fe5` | Hawkes terms trade accuracy for faster termination |
| 2026-10-28 | Lazy V-HPED is slower than eager (about 0.77x) when every pair has a record every frame | `outputs/runs/complexity/20261027T202322Z/results.json` | Efficiency claim must be stated for sparse touch patterns only |
| 2026-10-28 | Synthetic: fitted VT-HMM (no Hawkes) beats full V-HPED on frame F1 (0.967 vs 0.957); V-HPED churn higher than Gated EMA | `outputs/runs/vhped_synthetic/run_20261027T194003Z.json` | H2 must be argued as a persistence/termination trade-off, not a uniform win |

---

## Update Protocol
1. Add new claims when they emerge from experiments.
2. Update status: UNTESTED → TESTING → SUPPORTED / REFUTED / INCONCLUSIVE.
3. Link to run IDs and artifact paths.
4. Never delete refuted claims — mark them and record the evidence.
