# V-HPED — Model Derivation (Phase 4 reference model)

> **Status:** Specification + derivation for the eager reference implementation in
> `src/taorg/temporal/vhped/`. Propositions below are proved for the
> *discrete-time* model that the code implements. Empirical claims (H1–H3) remain
> UNTESTED until real-data runs exist — see `research/claims.md`.

This document resolves the Phase 4 "before coding" decisions in `Phases.md`:
visible-negative-evidence model, state/intensity/alive relationship, softplus
likelihood treatment, parameter constraints, visibility-dependent beta, and
threshold domain.

---

## 1. What is standard vs. what is ours

| Component | Status |
|---|---|
| Multivariate exponential-kernel Hawkes intensity | Standard (Hawkes 1971; nonlinear/softplus link as in e.g. neural Hawkes) |
| Thinning of a point process by a detection probability | Standard point-process operation |
| Two-state alive/dead hidden Markov filter | Standard (forward algorithm) |
| **Visibility-thinned Hawkes emission for scene-graph edges** — the Hawkes intensity defines the *expected observation rate* of a live relationship, and visibility thins it | **Proposed contribution** |
| **Self-calibrating negative evidence** — a visible absence carries log-evidence proportional to the current Hawkes intensity, so the strength of "it ended" depends on the edge's own observation history | **Proposed contribution** (follows from the above) |
| **Cross-predicate termination hazard** — observing predicate `q` on a pair can raise/lower the hazard that predicate `p` ends (e.g. *putting down* → *holding* ends) | **Proposed contribution** |
| **Visibility-dependent excitation memory** (`beta_vis`, `beta_occ`) — memory of past evidence is preserved through occlusion | **Proposed contribution** (PRD FR-5) |

The paper must not claim invention of Hawkes processes, HMMs, or thinning.
The claim is the *combination as an observation model for VidSGG edge
persistence*, its exact likelihood, and the nesting/censoring properties below.

---

## 2. Notation

- Pair `(i, j)`, predicate set `{0..P-1}`. Frames at timestamps `t_0 < t_1 < ...`,
  `Δ_n = t_n - t_{n-1}` (seconds; first frame uses `1/fps`).
- Pair visibility regime `v_n ∈ {VIS, OCC, UNK}` (from `EdgeObservation.edge_visibility`;
  a tracked pair absent from frame `n` is `UNK` — never silently `VIS`, PRD FR-2).
- `o_n^p ∈ {0,1}`: predicate `p` observed on the pair at frame `n`; mark `w_n^p ∈ [0,1]`
  (predicate score, or 1 when marks are disabled).
- Hidden state `A_n^p ∈ {alive, dead}`; filtered belief `a_n^p = P(A_n^p = alive | o_{≤n})`.

Kept separate in code and docs (Rules §2.1):

| Quantity | Symbol | Domain | Code |
|---|---|---|---|
| excitation trace | `e_q` | `≥ 0` | `dynamics.PairTraces` |
| pre-activation | `z_p` | `ℝ` | `dynamics.preactivation` |
| event intensity | `λ_p = softplus(z_p)` | `> 0` (1/s) | `dynamics.intensity` |
| termination hazard | `h_p = softplus(η_p + Σ_q γ[q,p] e_q)` | `> 0` (1/s) | `dynamics.hazard` |
| alive probability | `a_p` | `[0,1]` | `alive.filter_step` |
| keep/drop threshold | `τ` | applied to `a_p` | `VHPED.score_threshold` |

---

## 3. Dynamics

**Excitation traces (per pair, per source predicate `q`).** Between frames:

`e_q ← e_q · exp(-β_{v_n} Δ_n)`, then after frame `n` is processed: `e_q ← e_q + o_n^q w_n^q`.

`β_v > 0` depends on the regime: `β_OCC < β_VIS` lets memory of past evidence
survive occlusion. The regime used for the interval `(t_{n-1}, t_n]` is `v_n`
(the regime observed at the end of the interval) — a documented modelling choice.

**Intensity.** `z_p = μ_p + Σ_q α[q,p] e_q`, `λ_p = softplus(z_p)`.
Because `softplus > 0` for every real input, **negative α (inhibition) can never
produce an invalid intensity** (Rules §2.3). Tested in `test_vhped.py`.

**Termination hazard.** `h_p = softplus(η_p + Σ_q γ[q,p] e_q)`.

Both `λ` and `h` are evaluated from the **left-limit** traces (after decay, before
adding frame-`n` events) and held constant over the frame interval.

---

## 4. Observation model — resolving "visible absence is negative evidence"

A standard Hawkes likelihood only models positive event arrivals. We do **not**
treat absence as a Hawkes event. Instead:

> If the relationship is **alive**, the base model's detections of it are a
> point process with intensity `λ_p`, **thinned** by the visibility-dependent
> detection probability `ρ_v` (`ρ_VIS = 1`, `ρ_OCC, ρ_UNK ∈ [0,1]` learned).
> If the relationship is **dead**, detections are spurious with rate `ε`, thinned the same way.

With the intensity piecewise-constant over a frame, the probability of at least
one detection in the frame is exact:

```
p1 = P(o=1 | alive, v) = 1 - exp(-ρ_v λ_p Δ)
p0 = P(o=1 | dead,  v) = 1 - exp(-ρ_v ε  Δ)
```

**Softplus integral (Rules §2.2).** Because the code is a *discrete-time*
model with frame-constant intensity, the compensator over a frame is exactly
`ρ_v λ_p Δ`; no closed-form integral of `softplus(μ + Σ α e^{-βs})` is needed or
assumed. Relative to a continuous-time process this is a left-Riemann
approximation with per-frame error `O(β Δ)` in the exponent; at 30 fps and
`β ≤ 5/s` this is `≤ 0.17` relative in the trace over one frame. The fitted
model *is* the discrete model, so its likelihood is exact.

**Transition (predict).**

```
a⁻ = π + (a - π) · exp(-(h_p + r) Δ),     π = r / (h_p + r)
```

This is the **exact** transition of the two-state continuous-time Markov chain
with rates `h_p` (alive→dead) and `r` (dead→alive) held constant over the frame;
for `r = 0` it is `a · exp(-h_p Δ)`. (An earlier draft used the competing-risk
approximation `a e^{-hΔ} + (1-a)(1-e^{-rΔ})`, which is not a semigroup in Δ and
broke exact lazy evaluation for `r > 0`; replaced 2026-10-28. Results with
`r = 0`, the default, are unchanged.) `r` is a small re-birth rate. A predicate is first tracked when first observed
with `a = π_0` and then receives the same predict/update step (this matches the
existing `HMM` baseline's birth semantics, which enables Proposition 3).

**Update.** `L1 = p1^o (1-p1)^{1-o}`, `L0 = p0^o (1-p0)^{1-o}`,

```
c_n = a⁻ L1 + (1 - a⁻) L0           (one-step predictive evidence)
a   = a⁻ L1 / c_n
```

**Log-likelihood** of a video = `Σ_n Σ_{tracked p} log c_n^p` — the exact forward
algorithm likelihood of the discrete model (conditional on birth times; the
birth event's own evidence is included). Implemented twice:
`alive.filter_step` (NumPy, inference) and `likelihood.sequence_log_likelihood`
(PyTorch, differentiable); a test asserts they agree to 1e-9.

---

## 5. Propositions (discrete model)

**Proposition 1 (exact censoring under occlusion).** If `ρ_v = 0` then for any
absence `o = 0`: `L1 = L0 = 1`, hence `a = a⁻`. Under full occlusion the belief
evolves *only* through the prior transition (hazard and re-birth); the absence
carries zero evidence.
*Proof.* `p1 = p0 = 1 - e^0 = 0`. ∎ (Tested.)

**Proposition 2 (self-calibrating negative evidence).** For a visible absence
(`ρ = 1`, `o = 0`) the log Bayes factor against "alive" is

```
log L1/L0 = -λ_p Δ - log(1 - p0) = -(λ_p - ε) Δ.
```

So the evidence that a relationship ended is **linear in the current Hawkes
intensity**: an edge that was observed densely and recently (large `λ_p`)
is dropped after few visible misses, while an edge whose detections have always
been sparse (small `λ_p`) is not over-penalised by one miss. Gated EMA, by
contrast, applies the same multiplicative decay regardless of history.
*Proof.* Direct substitution, `1 - p1 = e^{-λΔ}`, `1 - p0 = e^{-εΔ}`. ∎

**Corollary 2a (visible-termination delay bound).** Assume no new events,
regime VIS throughout, cross terms zero, `r = 0`, and `λ_p(t) ≥ λ_min > ε` over
the window. After `k` visible misses the posterior odds satisfy

```
odds_k ≤ odds_0 · exp(-k (λ_min - ε) Δ)
```

so the edge is dropped (`a < τ`) after at most
`k* = ⌈ (log odds_0 - log(τ/(1-τ))) / ((λ_min - ε) Δ) ⌉` frames.
Hazard only strengthens this (it moves mass alive→dead). Note `λ_p` itself
decays during the misses, so `λ_min` must be taken over the window
(e.g. `softplus(μ_p)` if `α[·,p] ≥ 0`). A delay bound **does not exist** when
`λ_min ≤ ε` — the model then genuinely cannot tell the edge ended.

**Proposition 3 (nesting).** With `α = 0, γ = 0` (no Hawkes), `ρ ≡ 1`
(no visibility), constant frame period, the model is *exactly* the existing
2-state `HMM` baseline with
`stay = π + (1-π)e^{-kΔ}`, `resurrect = π(1 - e^{-kΔ})` with `k = softplus(η) + r`, `π = r/k`,
`emit_obs_alive = 1 - e^{-softplus(μ)Δ}`, `emit_obs_dead = 1 - e^{-εΔ}`,
`prior = π_0`. (Tested numerically against `taorg.temporal.hmm.HMM`.)

This gives the ablation ladder used for H1–H3:

```
HMM  ⊂  VT-HMM (+visibility, α=γ=0)  ⊂  V-HPED-self (+self-excitation)  ⊂  V-HPED (+cross α, γ)
```

**Proposition 4 (cross-predicate independence when disabled).** If the
off-diagonal of `α` and `γ` is zero then the belief trajectory of predicate `p`
is a function only of `(o^p, w^p, v)`. (Tested: adding/removing observations of
another predicate on the same pair leaves `a^p` unchanged.)

---

## 6. Parameter constraints (trainer)

| Parameter | Constraint | Transform |
|---|---|---|
| `μ_p, η_p, α, γ` | real | identity |
| `β_VIS, β_OCC, β_UNK` | `> 0` | softplus + 1e-6 |
| `ρ_OCC, ρ_UNK` | `[0,1]` | sigmoid |
| `ε, r` | `> 0` | softplus |
| `π_0` | `(0,1)` | sigmoid |

Ablations fix parameters by masking gradients (`use_visibility=False` pins
`ρ ≡ 1`, ties all β to `β_VIS`; `use_cross_predicate=False` zeros the
off-diagonals of α, γ; `use_excitation=False` zeros α and γ).

Fitting is unsupervised maximum likelihood over the *observation stream* of
training videos only (FR-6). It does not need ground-truth alive labels.
Threshold `τ` must be selected on validation data, never test.

---

## 7. Known limitations / open items (record in Memory.md)

1. The regime for an interval is taken from its right endpoint. Alternative
   (left endpoint) changes results at regime boundaries; untested which is better.
2. Unsupervised MLE may trade off `ε` against `λ` (identifiability). Synthetic
   recovery tests only check *behaviour*, not exact parameter recovery.
3. Pruning beliefs below `prune_below` breaks exact equivalence with the
   likelihood (a pruned edge re-born later restarts at `π_0`). Equivalence
   tests run with `prune_below = 0`.
4. The eager reference updates every active edge each frame (`O(A·P)` per
   frame). Lazy materialisation + expiry heap is Phase 5: belief no longer
   decays as a single exponential under visible absence, so the expiry time
   requires a root-finder or a bound (Corollary 2a), not a closed form.
5. Detector score marks enter excitation only, not the emission likelihood.
