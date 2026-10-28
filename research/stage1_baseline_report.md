# Stage 1 — Repository Audit and Baseline Verification

> Date: 2026-10-28. No code was modified in this stage. This report freezes the
> baseline that every later extension must be compared against.

## 0. Verification performed in this stage

| Check | Result |
|---|---|
| Full test suite (`PYTHONPATH=src python -m pytest`) | **240 passed, 0 failed, 0 skipped** (20.9 s). No pre-existing failures. |
| Reproduction of a saved result | Re-ran ablation-matrix cell (ID-switch condition, seed 42) from `outputs/runs/20261027_202753_a7314fe5/config.resolved.yaml`: **max absolute difference 0** over all 11 methods × 7 metrics. |
| Git provenance | **Not a git repository** → every run records `git_commit = unknown`. Baseline is identified by run directories below, not a commit. |
| Package install | `taorg` is not pip-installed; tests and scripts need `PYTHONPATH=src`. |

Baseline artifacts (frozen — do not overwrite):

| Artifact | Path |
|---|---|
| Phase 8 synthetic ablation matrix (5 seeds × 2 conditions × 11 methods) | `outputs/runs/20261027_202753_a7314fe5/` |
| Phase 5 eager-vs-lazy scaling benchmark | `outputs/runs/complexity/20261027T202322Z/` |
| Early 3-seed synthetic V-HPED benchmark | `outputs/runs/vhped_synthetic/run_20261027T194003Z.json` |

## 1. Architecture and module dependencies

```
data/action_genome.py ─┐            adapters/base.py (contract) ── adapters/precomputed.py (JSONL)
                       ▼                         ▼
              schemas/{tracks,edges,runs}  ◄── canonical FrameGraph stream
                       ▼
noise/{occlusion,id_switch,manifest}  (deterministic corruption + manifests)
                       ▼
temporal/  no_memory · fixed_window · ema · gated_ema · hmm      (baselines, untouched)
temporal/vhped/
   params ─► dynamics ─► alive ─► model (eager oracle, VHPED)
   visibility ─────────────────►   │
   expiry_heap ─► lazy (LazyVHPED, subclass of VHPED)
   likelihood (PyTorch mirror of model) ─► trainer (MLE)
                       ▼
metrics/{persistence,churn,termination,latency}  (frozen)  + calibration (frozen) + threshold_sweep (vectorised, tested equal)
                       ▼
experiments/{pilot, ablation_matrix, vhped_benchmark, complexity_benchmark, synthetic_eval, artifacts, pilot_figures}
scripts/{run_pilot_ag, inspect_action_genome, run_ablation_matrix, run_complexity_benchmark, run_vhped_synthetic, make_paper_artifacts}
paper/{main.tex, references.bib, generated/}
```

Size: ~10.2k lines of Python (src + scripts + tests). Registry `METHODS`: `no_memory, fixed_window, ema, gated_ema, hmm, vhped, vhped_lazy`.

## 2. Mathematical definition of V-HPED (as implemented)

Per pair `(i,j)`, predicate `p ∈ {0..P−1}`, frame interval `Δ_n = t_n − t_{n−1}` (first frame `1/fps`), pair regime `v_n ∈ {VIS, OCC, UNK}` (most conservative of the pair's records; no record ⇒ UNK). Sources: `dynamics.py`, `alive.py`, `model.py`, derivation `research/derivations/vhped_model.md`.

**Hawkes excitation** (left-limit, then add events after the update):
```
e_q ← e_q · exp(−β_{v_n} Δ_n)                       β = [β_vis, β_occ, β_unk] > 0
λ_p = softplus(μ_p + Σ_q α[q,p] e_q)                 detection intensity (1/s), > 0 for any α
h_p = softplus(η_p + Σ_q γ[q,p] e_q)                 termination hazard (1/s), > 0
e_q ← e_q + o_n^q · w_n^q                            w = predicate score (marks) or 1
```

**Visibility-thinned observation probabilities** (`ρ_vis ≡ 1`; `ρ_occ, ρ_unk ∈ [0,1]`):
```
P(o=1 | alive, v) = 1 − exp(−ρ_v λ_p Δ)      P(o=1 | dead, v) = 1 − exp(−ρ_v ε Δ)
```

**Alive/dead transition** (exact 2-state CTMC, rates `h_p` alive→dead, `r` dead→alive):
```
a⁻ = π + (a − π) exp(−(h_p + r) Δ),   π = r / (h_p + r)
```
Birth: a predicate is tracked from its first observation with `a = π₀`, then gets the same predict+update (matches HMM baseline).

**Update / likelihood:** `c = a⁻ L1 + (1 − a⁻) L0`, `a = a⁻ L1 / c`, `log p(o) = Σ_n Σ_tracked log c`.
Output score = `a`; edge emitted iff `a ≥ τ`; predicates with `a < prune_below` are untracked.

**Parameter estimation** (`trainer.py`): unsupervised maximum likelihood of the observation stream on training videos (Adam, 200 iterations, lr 0.05, L2 1e‑3 on off-diagonal α and all γ). Constraints by reparameterisation (softplus for β, ε, r; sigmoid for ρ, π₀). Ablation flags: `use_visibility`, `use_excitation`, `use_cross_predicate`, `fit_gamma`, `rho_unk_fixed`. Videos batched by padding with `dt = 0` steps (proved no-op, tested). Threshold `τ` selected on validation by frame F1.

## 3. Lazy maintenance (`lazy.py`, `expiry_heap.py`, derivation `lazy_expiry.md`)

* Conditions: **(L1)** `ρ_unk = 0`, **(L2)** `γ = 0` ⇒ an untouched pair evolves in closed form `a(t) = π + (a(t₀) − π) e^{−(h+r)(t−t₀)}` (semigroup), contributes 0 log-likelihood.
* Monotone ⇒ unique crossing of `τ` / `prune_below`, closed-form delay.
* **Per-pair wake-up heap:** one authoritative `(time, wake_id)` per pair; push only if earlier than pending; early wake-ups re-validate from the closed form; superseded entries skipped by id; rebuild scan only after the heap doubles (drop stale if > 3/4 stale).
* Fallback to eager updates when L1/L2 fail (`lazy_exact = False`).
* Claimed complexity: `O(k_touch·P + (k_touch + k_wake)(P + log H))` per frame vs `Θ(A·P)` eager; output enumeration `O(|S|)` separate.

## 4. Data, corruption, adapters, baselines, metrics

* **Synthetic generator** (`vhped_benchmark.generate_video`): pairs × 3 predicates (holding with probability 0.6 of termination in frames 40–120; 6-frame "putting down" precursor; independent "looking at"); per-edge detector reliability ∈ {0.95, 0.75, 0.5}; spurious detections 0.02/frame; pair-level occlusion gaps {3,5,10,20}, 2 per pair; track observations for ID-switch injection. ID-switch condition: `inject_id_switch`, 2 swaps × 5 frames, same class.
* **Real-data adapters:** Action Genome loader (format **assumed**, inspector script provided); frozen adapter contract + JSONL precomputed-prediction adapter. **No real data or base-model predictions exist on this machine.**
* **Baselines:** no memory, fixed window (w=5), EMA (α 0.3, decay 0.9), visibility-gated EMA (0.7 / 0.95), 2-state HMM (hand-set). V-HPED variants: hand-set, VT-HMM (fit, α=γ=0), no-visibility (fit), self-only (fit), full (fit), lazy-exact (fit, γ=0, ρ_unk=0).
* **Metrics (frozen):** persistence P/R/F1 inside occlusion intervals; churn; termination delay (+censoring); false-persistence rate; latency; ECE/Brier (frozen 2026‑10‑28 before first use). Frame-level F1 is used for threshold selection only (auxiliary, not frozen).
* **Protocol:** fit on train, threshold on val, report test; paired design (identical corrupted streams); 5 predeclared seeds `[42,123,456,789,2024]`; paired bootstrap (10 000) over test videos.

## 5. Test coverage (240 tests)

| Area | File | Tests |
|---|---|---|
| Baselines | `unit/test_baselines.py` | 42 |
| V-HPED math (propositions, nesting = HMM, NumPy = PyTorch likelihood, finite-difference gradient, extreme inhibition, ablation masks, padding) | `unit/test_vhped.py` | 37 |
| Eager vs lazy equivalence (24 randomized streams), closed form, crossing times, heap | `property/test_lazy_equivalence.py` | 35 |
| Threshold sweep = frozen metrics; post-hoc thresholding valid per method | `unit/test_threshold_sweep.py` | 23 |
| Frozen metrics edge cases | `unit/test_metrics.py`, `unit/test_calibration.py` | 25 |
| Corruption determinism | `unit/test_corruption.py` | 18 |
| Schemas, config, utils | 3 files | 34 |
| AG loader (fake tree), adapters | 2 files | 12 |
| Integration (pilot, matrix, benchmark, artifacts, Phase 2 eval) | 5 files | 14 |

**Not covered:** visibility *misclassification*, memory usage, real data, heap-size bound `H = O(A)` (no test), irregular timestamps beyond the lazy harness and padding test, ID-switch-specific metrics.

## 6. Reproducible baseline metrics (synthetic; test split; mean over 5 seeds)

### Condition A — occlusion + detector flicker

| Method | Frame F1 | Pers. F1 | Term. delay | False pers. | Churn | ECE | ms/frame |
|---|---|---|---|---|---|---|---|
| No memory | 0.774 | 0.000 | 1.02 | 0.018 | 0.257 | 0.227 | 0.00 |
| Fixed window | 0.903 | 0.505 | 4.31 | 0.131 | 0.032 | 0.170 | 0.01 |
| EMA | 0.931 | 0.732 | 7.83 | 0.135 | 0.045 | 0.204 | 0.03 |
| Vis-Gated EMA | 0.950 | 0.922 | 5.05 | 0.105 | 0.040 | 0.286 | 0.04 |
| HMM | 0.890 | 0.269 | 2.37 | 0.041 | 0.088 | 0.206 | 0.03 |
| V-HPED hand-set | 0.950 | 0.921 | 4.17 | 0.072 | 0.036 | 0.013 | 0.36 |
| VT-HMM (fit) | **0.969** | 0.951 | 5.26 | 0.089 | 0.050 | 0.054 | 0.40 |
| V-HPED no-vis (fit) | 0.922 | 0.571 | 3.92 | 0.080 | 0.036 | 0.060 | 0.36 |
| V-HPED self-only (fit) | 0.955 | 0.946 | 3.53 | 0.074 | 0.041 | 0.033 | 0.35 |
| **V-HPED full (fit)** | 0.958 | 0.951 | 3.37 | 0.077 | 0.038 | 0.030 | 0.38 |
| V-HPED lazy-exact (fit) | 0.956 | 0.948 | 3.63 | 0.075 | 0.039 | 0.030 | 0.43 |

### Condition B — + tracker ID switches

| Method | Frame F1 | Pers. F1 | Term. delay | False pers. | Churn | ECE |
|---|---|---|---|---|---|---|
| Vis-Gated EMA | 0.891 | 0.878 | 3.74 | 0.061 | 0.050 | 0.192 |
| HMM | 0.861 | 0.258 | 2.29 | 0.039 | 0.063 | 0.172 |
| V-HPED hand-set | **0.909** | 0.876 | 3.74 | 0.060 | 0.029 | 0.028 |
| VT-HMM (fit) | 0.907 | 0.763 | 2.98 | 0.044 | 0.043 | 0.237 |
| V-HPED self-only (fit) | 0.873 | 0.877 | 2.29 | 0.039 | 0.060 | 0.210 |
| **V-HPED full (fit)** | 0.875 | 0.895 | **2.12** | 0.038 | 0.059 | 0.211 |

Paired (full V-HPED − Gated EMA), 95% bootstrap CI over test videos: A: frame F1 **+0.008 [+0.006, +0.010]**, delay **−1.63 [−1.87, −1.38]**; B: frame F1 **−0.015 [−0.018, −0.012]**, delay −1.57 [−1.78, −1.36].

### Efficiency (one Windows machine, P = 26)
Lazy/eager speed-up at A = 3000: f = 0.05 → 4.46×, 0.2 → 2.27×, 0.5 → 1.26×, 1.0 → **0.77× (slower)**. Decision mismatches: 0 / 16 grid points.

## 7. Diagnostic of the main known failure (ID switches) — measured, not yet tested causally

Mean fitted parameters (full V-HPED, 5 seeds):

| Parameter | Condition A | Condition B (ID switches) | Hand-set |
|---|---|---|---|
| `prior_alive π₀` | 0.983 | 0.983 | 0.5 |
| `ρ_unk` | 0.500 (= init; **no UNK frames in A ⇒ no gradient**) | **0.006** | 0.5 |
| `ρ_occ` | 0.001 | 0.001 | 0.05 |
| selected `τ` | 0.5–0.8 | **0.9 in all 5 seeds (upper end of grid)** | — |

Observations: (i) a learned birth prior of ≈0.98 means an edge appearing on a *new* pair — exactly what an ID switch produces — is believed almost immediately; (ii) the pair the identity was taken from has no records ⇒ UNK, and with `ρ_unk ≈ 0` it is fully censored (kept alive); (iii) validation pushes `τ` to the grid boundary, consistent with miscalibrated new-edge beliefs (ECE 0.211 vs 0.028 hand-set).
**Hypothesis H-ID (to test in Stage 3):** the ID-switch failure is driven mainly by over-confident births (`π₀`) plus censoring of the abandoned pair, not by the Hawkes terms. Falsifier: pinning `π₀` to 0.5 (or an identity-aware birth) does not recover frame F1 / ECE in condition B.

## 8. Required summary

### 8.1 Existing capabilities
1. Exact, tested V-HPED filter + differentiable likelihood + MLE trainer with ablation switches.
2. Exact lazy maintenance with proof (under L1–L2), wake-up heap, and a 0-mismatch equivalence harness.
3. Full evaluation stack: deterministic corruption with manifests, frozen metrics, vectorised threshold sweep proven equal to the frozen code, paired bootstrap, validation-only threshold selection.
4. Real-data readiness: AG loader + inspector, frozen adapter contract, JSONL adapter, AG pilot runner (tested on a fake tree).
5. Paper pipeline generating every number from artifacts; results reproduce bit-exactly.

### 8.2 Existing limitations
1. **No real-data result of any kind.** All accuracy evidence is synthetic.
2. Fitted V-HPED fails under ID switches (frame F1 and calibration).
3. VT-HMM (no Hawkes) has the best frame F1 in condition A; Hawkes terms buy faster termination and calibration, not accuracy.
4. Cross-predicate terms give a small effect (+0.003 frame F1).
5. Lazy is slower than eager when every pair is touched every frame; OCC-touched pairs are updated eagerly.
6. V-HPED is ~10× slower per frame than EMA-family baselines in Python (0.38 vs 0.04 ms/frame on synthetic videos).
7. Memory usage is not measured anywhere.
8. Visibility misclassification and irregular frame rates are not part of the synthetic experiments.

### 8.3 Mathematical assumptions currently in force
1. Discrete time: intensity and hazard constant within a frame (left-limit); exact for the discrete model, `O(βΔ)` approximation of continuous time.
2. The regime of an interval is the regime at its **right** endpoint.
3. Detections given alive/dead and regime are conditionally independent across predicates and frames given the traces (Hawkes dependence enters only through `e`).
4. Pairs are conditionally independent of each other (no coupling across pairs — relevant to ID switches, which move evidence *between* pairs).
5. Visibility labels are treated as correct (no visibility-error model).
6. Births condition on the first observation (likelihood is conditional on birth times).
7. Lazy exactness: L1 `ρ_unk = 0`, L2 `γ = 0`; decisions compared with tolerance 1e‑9 near the threshold.
8. Marks (detector scores) affect excitation only, not the emission likelihood.

### 8.4 Potentially redundant components
| Component | Status | Recommendation |
|---|---|---|
| `experiments/synthetic_eval.py` + `metrics/latency.measure_latency` | Phase 2 pipeline; superseded by `ablation_matrix.py` for experiments; still used by its tests | Keep (baseline record); mark as legacy |
| `scripts/run_vhped_synthetic.py` + `vhped_benchmark.run_benchmark` | Superseded by the ablation matrix (which reuses `generate_video`) | Keep for provenance of the early run; do not extend |
| `model.py` vs `likelihood.py` | Intentional duplication (NumPy oracle vs PyTorch), tested equal | Keep |
| `VersionedHeap` uses `EdgeKey(s, o, −1)` as a pair key | Works, but a sentinel predicate id is a schema smell | Minor; leave unless the heap is extended |
| `tracking/` package | Empty | Leave (Phase 7 placeholder) |

### 8.5 Unverified or partially verified claims
| Claim | Status |
|---|---|
| H1–H3 on real data | **UNTESTED** |
| Novelty of the visibility-thinned Hawkes observation model, cross-predicate hazard, lazy exactness | **Not verified against literature** — only a recall-based related-work list exists; no systematic 2023–2026 search has been done (Stage 2) |
| Heap size `H = O(A)` amortized | Argued in `lazy_expiry.md`, **not tested** |
| Complexity bound | Supported by counters on synthetic traces; no adversarial workload tested |
| "Self-calibrating evidence" causes churn on dense relationships | Illustrated by one qualitative example; not quantified |
| AG annotation format | **Assumed**, not inspected |
| Bibliography details (esp. DSG-DETR authors) | Not verified against publishers |
| Termination-delay bound (Cor. 2a) | Proof only; no numerical test |

### 8.6 Reproducible baseline metrics
Section 6 tables, regenerated from the frozen run directories above; reproduction verified bit-exactly for one cell in this stage.

## 9. Next recommended action (Stage 2)
Literature review and novelty-gap analysis. This needs web access to primary sources
(CVF Open Access, NeurIPS/ICML/ICLR proceedings, arXiv). Without it, the novelty
audit must be labelled incomplete. Priority questions: (a) has any VidSGG / dynamic
scene-graph work modelled relationship termination with visibility-dependent
observation likelihoods; (b) prior use of thinned point processes as emission
models for latent on/off states; (c) identity-uncertainty-aware relationship
filtering (relevant to the measured ID-switch failure).
