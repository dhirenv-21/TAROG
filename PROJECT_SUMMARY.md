# TAORG / V-HPED — Project Summary (for external novelty review)

**Project:** TAORG — Temporal Adaptive Object Relationship Graph
**Method:** V-HPED — Visibility-aware Hawkes-Process Edge Dynamics
**Context:** Semester-3 Advanced Data Structures & Algorithms (ADSA) research project, written as a conference-paper-style research codebase (Python, NumPy, PyTorch).
**Status (2026-10-28):** method, proofs, efficient data structure, evaluation pipeline and paper draft are done; **all accuracy results so far are on synthetic data**. Real-data experiments (Action Genome, base models such as STTran/TEMPURA, real tracker noise) are implemented but **not yet run** (no data on the machine).

---

## 1. Problem

Video scene graph generation (VidSGG) models (e.g. STTran, DSG-DETR, TEMPURA) predict subject–predicate–object relationships *per frame*. Occlusion, missed detections and tracker ID switches make continuing relationships flicker:

`person-holding-cup → missing → person-holding-cup → missing`

Temporal smoothing (fixed window, EMA) hides flicker but keeps relationships alive after they truly end, because it cannot tell **why** an edge stopped being observed.

**Research question:** when a relationship edge stops being observed, can a visibility-aware temporal point-process model tell *observation failure* (occlusion, missed detection, tracker loss) apart from *genuine termination* (both objects clearly visible, relationship gone)?

V-HPED is a **model-agnostic post-processing layer** placed after any VidSGG model.

## 2. Method (V-HPED)

Per subject–object pair and predicate `p`, frame interval `Δ`, visibility regime `v ∈ {VISIBLE, OCCLUDED, UNKNOWN}`:

1. **Hawkes excitation.** Traces of past detections decay with a visibility-dependent rate (`β_occ < β_vis`, so memory survives occlusion):
   `e_q ← e_q·exp(−β_v Δ)`, then `e_q ← e_q + mark` when predicate `q` is detected.
   Detection intensity: `λ_p = softplus(μ_p + Σ_q α[q,p] e_q)` (cross-predicate excitation/inhibition; softplus keeps it valid for any sign of α).
2. **Cross-predicate termination hazard (new).** `h_p = softplus(η_p + Σ_q γ[q,p] e_q)` — e.g. observing "putting down" can raise the hazard that "holding" ends.
3. **Visibility-thinned observation model (core idea).** Absence is *not* modelled as a Hawkes event. Instead, if the relationship is alive, the base model's detections are the Hawkes process thinned by visibility `ρ_v` (`ρ_vis = 1`, `ρ_occ, ρ_unk` learned); if dead, only spurious detections at rate `ε`:
   `P(detect | alive) = 1 − exp(−ρ_v λ_p Δ)`, `P(detect | dead) = 1 − exp(−ρ_v ε Δ)`.
4. **Alive/dead filter.** A two-state continuous-time Markov chain (rates `h_p` alive→dead, `r` dead→alive) filtered with the exact forward algorithm. Output = posterior `P(alive)`, thresholded to keep/drop the edge.
5. **Learning.** Exact, differentiable discrete-time log-likelihood (PyTorch) → unsupervised maximum likelihood on training videos only; threshold chosen on validation. (Frame-constant intensity makes the compensator exact, avoiding the softplus-integral problem.)

**Proved properties** (documented derivations, each with a unit test):
- **P1 Exact censoring:** if `ρ_v = 0`, an absence carries zero evidence.
- **P2 Self-calibrating negative evidence:** a *visible* miss has log Bayes factor `−(λ_p − ε)Δ` — evidence that the relationship ended is proportional to the current Hawkes intensity. Densely/recently detected relationships are dropped after few visible misses; sparsely detected ones are not over-penalised. (Gated EMA applies the same decay regardless of history.)
- **P3 Nesting:** with `α = γ = 0`, `ρ ≡ 1` the model is exactly a 2-state HMM (checked numerically to 1e-10) → clean ablation ladder: HMM ⊂ VT-HMM (+visibility) ⊂ V-HPED-self (+self-excitation) ⊂ V-HPED (+cross terms).
- **P4** Cross-predicate independence when cross terms are off.
- A termination-delay bound under stated assumptions.

## 3. ADSA contribution: exact lazy maintenance + wake-up heap

The eager reference updates every active pair every frame: `Θ(A·P)` (A active pairs, P predicates).

- **Lemma (untouched semigroup):** if (L1) `ρ_unk = 0` and (L2) `γ = 0`, a pair with no record in a frame evolves in closed form, depending only on elapsed time:
  `a(t) = π + (a(t₀) − π)·exp(−(h + r)(t − t₀))`, `π = r/(h + r)`, with zero likelihood contribution.
- The trajectory is monotone ⇒ each predicate crosses the output threshold / prune level at most once, at a closed-form time.
- **Per-pair wake-up min-heap:** one authoritative wake-up per pair (earliest crossing). Early wake-ups are safe (pop only re-evaluates the closed form); superseded entries are skipped via a wake id (version); at most one push per touch or wake-up; amortized rebuild keeps heap size `O(A)`.
- **Theorem:** under L1–L2, lazy and eager emit identical edges, scores and log-likelihood. If L1/L2 fail, automatic exact fallback to eager.
- **Complexity:** `O(k_touch·P + (k_touch + k_wake)(P + log H))` per frame vs `Θ(A·P)` eager; output enumeration `O(|S|)` reported separately.
- **Verification:** randomized eager-vs-lazy harness (24 streams, irregular timestamps, pairs dropping in/out, mixed visibility) — 0 decision mismatches.

**Measured (one Windows machine, P = 26, up to A = 3000 pairs):**

| fraction of pairs touched per frame | speed-up of lazy over eager |
|---|---|
| 0.05 | ≈ 4.5–5.5× |
| 0.20 | ≈ 2.3–2.7× |
| 0.50 | ≈ 1.3× |
| 1.00 | **≈ 0.77× (slower)** |

0 decision mismatches across all 16 benchmark points.

## 4. Evaluation infrastructure

- Canonical edge/frame schema; frozen **adapter contract** + JSONL adapter so any base model's predictions plug in.
- Deterministic occlusion / ID-switch injection with saved corruption manifests (all methods see identical corrupted input).
- Frozen metrics: edge persistence F1 inside occlusion gaps, termination delay, false-persistence rate, churn, ECE/Brier of the alive score, latency. Vectorised threshold sweep proven equal to the frozen metric code by tests.
- Baselines: no memory, fixed window, EMA, visibility-gated EMA (critical baseline), 2-state HMM.
- Action Genome loader + format inspector + full Phase-3 go/no-go pilot (5 predeclared seeds × 6 gap lengths × 9 methods, validation-only threshold selection, hypervolume trade-off comparisons) — **implemented, tested on a fake dataset, not yet run on real data.**
- 240 automated tests; every paper number generated from run artifacts (no hand-typed results).

## 5. Results so far (SYNTHETIC ONLY)

Synthetic scenario: pairs with 3 predicates ("holding", a short "putting down" cue before holding ends, independent "looking at"), heterogeneous detector reliability (0.95/0.75/0.5), spurious detections, pair-level occlusion gaps of 3–20 frames; second condition adds tracker ID switches. 5 seeds, 20 train / 10 val / 20 test videos per seed, paired bootstrap 95% CIs over test videos.

**Condition A: occlusion + detector flicker**

| Method | Frame F1 ↑ | Persistence F1 ↑ | Termination delay (frames) ↓ | False persistence ↓ | Churn ↓ | ECE ↓ |
|---|---|---|---|---|---|---|
| No memory | 0.774 | 0.000 | 1.02 | 0.018 | 0.257 | 0.227 |
| Fixed window | 0.903 | 0.505 | 4.31 | 0.131 | 0.032 | 0.170 |
| EMA | 0.931 | 0.732 | 7.83 | 0.135 | 0.045 | 0.204 |
| Vis-Gated EMA | 0.950 | 0.922 | 5.05 | 0.105 | 0.040 | 0.286 |
| HMM | 0.890 | 0.269 | 2.37 | 0.041 | 0.088 | 0.206 |
| V-HPED hand-set | 0.950 | 0.921 | 4.17 | 0.072 | 0.036 | 0.013 |
| VT-HMM (fit, no Hawkes) | **0.969** | 0.951 | 5.26 | 0.089 | 0.050 | 0.054 |
| V-HPED w/o visibility | 0.922 | 0.571 | 3.92 | 0.080 | 0.036 | 0.060 |
| V-HPED self-only | 0.955 | 0.946 | 3.53 | 0.074 | 0.041 | 0.033 |
| **V-HPED full (fit)** | 0.958 | **0.951** | **3.37** | 0.077 | 0.038 | 0.030 |
| V-HPED lazy-exact (fit) | 0.956 | 0.948 | 3.63 | 0.075 | 0.039 | 0.030 |

Paired differences (full V-HPED minus other):
- vs Gated EMA: frame F1 **+0.008** [+0.006, +0.010] (5/5 seeds); termination delay **−1.63** [−1.87, −1.38] frames.
- vs VT-HMM (no Hawkes): frame F1 **−0.011** [−0.013, −0.009] (V-HPED worse); delay **−1.90** frames (V-HPED faster).
- vs self-only (cross-predicate terms): frame F1 +0.003 [+0.002, +0.004]; delay −0.15 frames (small).
- Fitted vs hand-set: frame F1 +0.008, delay −0.73 frames.
- In an earlier 3-seed synthetic run, the planted cue "putting down → holding ends" was recovered as a positive γ (0.69–0.86 across seeds, other cross terms ≈ 0).

**Condition B: + tracker ID switches (negative result)**

| Method | Frame F1 | Termination delay | ECE |
|---|---|---|---|
| Vis-Gated EMA | 0.891 | 3.74 | 0.192 |
| V-HPED hand-set | **0.909** | 3.74 | 0.028 |
| VT-HMM (fit) | 0.907 | 2.98 | 0.237 |
| V-HPED full (fit) | 0.875 | **2.12** | 0.211 |

Fitted V-HPED loses frame F1 to Gated EMA (−0.015 [−0.018, −0.012], 0/5 seeds) and to its own hand-set version, and its calibration collapses. Unsupervised fitting absorbs spurious ID-switch edges.

## 6. Honest limitations

1. No real-data results yet; all accuracy claims are synthetic.
2. Hawkes terms buy faster termination and better calibration, **not** uniformly higher accuracy (VT-HMM wins frame F1 in condition A).
3. Fitted parameters are not robust to ID switches.
4. Self-calibrating evidence can cause churn on dense relationships after a few misses.
5. Lazy evaluation is exact only under L1–L2 and is slower when every pair is touched every frame; occluded-but-present pairs are still updated eagerly.
6. Building blocks (Hawkes processes, HMMs, thinning, heaps) are standard; the claimed novelty is their combination as an observation model for relationship persistence, the self-calibrating visible-absence evidence, the cross-predicate hazard, and the exactness conditions + proof for lazy maintenance.

## 7. Closest prior work (for the reviewer to compare against)

- VidSGG: Action Genome (Ji et al., CVPR 2020), STTran (Cong et al., ICCV 2021), DSG-DETR (Feng et al., WACV 2023), TEMPURA (Nag et al., CVPR 2023).
- Point processes on graphs: Hawkes (1971), Neural Hawkes (Mei & Eisner, NeurIPS 2017), Know-Evolve (Trivedi et al., ICML 2017), DyRep (Trivedi et al., ICLR 2019).
- Thinning (Lewis & Shedler 1979); HMM forward algorithm (Rabiner 1989); calibration/ECE (Guo et al., ICML 2017); tracking noise: ByteTrack (Zhang et al., ECCV 2022).
- Not yet done: a systematic search for occlusion-aware / termination-aware temporal VidSGG methods published 2023–2026.

## 8. Question for the reviewer

Please rate the novelty (research contribution, not engineering effort) of:
(a) the visibility-thinned Hawkes observation model with self-calibrating visible-absence evidence,
(b) the cross-predicate termination hazard,
(c) the exact lazy-maintenance conditions + wake-up heap,
and the project overall, and say what prior work it most overlaps with.
