# IA-VHPED — Identity-Aware V-HPED (Stage 4 design and derivation)

Implementation: `src/taorg/temporal/vhped_ia/`. Original V-HPED (`temporal/vhped/`)
is unchanged and remains the comparison baseline.

## 1. Problem and hypothesis

Tracker identity switches move relationship evidence between pairs: the real pair goes
missing (or visibly absent) while still holding, and a phantom pair appears and later
vanishes. V-HPED treats every record as evidence about the pair named by its track labels,
so (i) the real pair gets spurious absence evidence, (ii) the phantom pair gets spurious
births, and (iii) maximum likelihood absorbs both into global parameters (Stage 3: `ρ_unk → 0`).

**Hypothesis.** If an inference-time *identity-continuity score* `κ ∈ [0,1]` per track and
frame is available (e.g. tracker association confidence), weighting relationship evidence by
the posterior probability that the pair's identities are the ones its history refers to
prevents (i)–(iii) — see HA1–HA4 in `research/stage3_direction_selection.md`.

## 2. Variables

| Symbol | Meaning | Domain |
|---|---|---|
| `κ_{i,n}` | identity-continuity score of track label `i` at frame `n` (from the tracker; 1 if not supplied) | `[0,1]` |
| `S_{i,n}` | latent: label `i` refers to its *home* object (the object its history was built on) | `{0,1}` |
| `ζ_{i,n}` | `P(S_{i,n}=1 | κ_{i,≤n})` — track identity belief | `[0,1]` |
| `ζ_{ij,n}` | pair identity belief `= ζ_{i,n} ζ_{j,n}` | `[0,1]` |
| `S_{ij,n}` | latent pair consistency `= S_i S_j` | `{0,1}` |
| `A_{p,n}` | alive state of predicate `p` on the pair (as in V-HPED) | `{0,1}` |
| `φ` | foreign-detection rate: detections on an inconsistent pair that carry no information about its own relation | `≥ 0` (1/s) |
| `b` | belief that a predicate first seen on an inconsistent pair is alive | `(0,1)` |
| `ω` | spontaneous identity-recovery rate (config, default 0) | `≥ 0` |

All V-HPED symbols (`λ, h, ρ, ε, r, π₀, e, β, α, γ, μ, η`) keep their meaning.

## 3. Identity dynamics (per track, independent across tracks)

A *jump* (reassignment of the label to a different object) occurs at frame `n` with
probability `1 − κ_{i,n}`. We use the two-object swap model: a jump toggles `S_i`
(a label taken by another object returns home at the next jump). Between frames the label
recovers its home object with rate `ω`.

```
ζ_{i,n}  = κ_{i,n} ζ_{i,n−1} + (1 − κ_{i,n})(1 − ζ_{i,n−1})          (jump toggles)
ζ_{i,n} ← 1 − (1 − ζ_{i,n}) exp(−ω Δ_n)                              (recovery)
ζ_{i,first} = 1;  a track absent from a frame gets no κ update (only recovery)
```
Validity: convex combinations of probabilities ⇒ `ζ ∈ [0,1]`.
**Assumption I1:** `κ` is a calibrated non-jump probability. **I2:** jumps are independent
across tracks and of relationship states. **I3:** two-object swaps (toggle). All three are
modelling choices; I1 is tested by an informativeness sweep incl. an uninformative `κ`.

## 4. Observation model given identity

For a pair at frame `n` with records `o_p` (observed or absent) for tracked/born predicates
`p ∈ T_n` and regime `v`:

* if `S_{ij}=1` (consistent): `o_p` follows V-HPED's thinned-Hawkes emission
  `L_p(o_p | A_p)` with `P(o=1|alive) = 1−e^{−ρ_v λ_p Δ}`, `P(o=1|dead) = 1−e^{−ρ_v ε Δ}`;
* if `S_{ij}=0` (inconsistent): records are produced by a foreign pair and are
  independent of this pair's `A`: `Q(o_p=1) = 1 − e^{−ρ_v φ Δ}` (same thinning).

**Assumption I4:** predicates are conditionally independent given `(S_{ij}, A)` and the traces
(as in V-HPED). **I5:** `S_{ij,n}` is independent of `A` and of past relationship observations
given `κ` (identity is the tracker's business).

## 5. Exact one-step joint posterior (within a frame)

Let `a⁻_p` be the V-HPED predicted belief (CTMC transition), `c_p = a⁻_p L_p(o_p|1) + (1−a⁻_p) L_p(o_p|0)`
and `post_p = a⁻_p L_p(o_p|1)/c_p` the V-HPED update. For born predicates, `a⁻_p` is the
V-HPED birth prediction from `π₀` (consistent branch) and `b` (inconsistent branch).

Joint predictive probability of the frame's records for the pair:
```
P(o | past) = ζ_{ij} Π_{p∈T} c_p  +  (1 − ζ_{ij}) Π_{p∈T} Q(o_p)
```
Posterior that the pair is consistent:
```
w = ζ_{ij} Π c_p / P(o | past)
```
Posterior alive belief (exact marginal of the joint posterior over (S_ij, A_p)):
```
a_p = w · post_p + (1 − w) · ā_p,      ā_p = a⁻_p   (tracked)   or   b   (born)
```
Trace update uses the expected genuine increment: `e_q ← e_q + w · o_q · mark_q`.

Validity: `P(o|past)` is a mixture of two valid joint distributions over records ⇒ a valid
probability; `w, a_p ∈ [0,1]`; `λ, h, Q ≥ 0`. No future frame is used: `ζ_{ij,n}` uses
`κ_{≤n}`, all other terms are V-HPED's causal quantities.

**Approximation (documented).** The posterior over `S_{ij}` (`w`) is used within the frame
but *not* carried to the next frame: the next frame's prior is the κ-only `ζ`. The full joint
filter over `(S_i, S_j, A_{1..P})` would carry it; we drop it for (a) per-track rather than
per-pair identity state and (b) cost. The alive marginal `a_p` carried forward is exact for
this frame's posterior; the dropped quantity is the correlation between `S` and `A`.

**Reduction.** `κ ≡ 1 ⇒ ζ ≡ 1 ⇒ w ≡ 1 ⇒` IA-VHPED ≡ V-HPED (same beliefs, traces,
log-likelihood). Tested exactly.

## 6. Likelihood and estimation

`log p(o_{1:T}) = Σ_n Σ_pairs log P(o_n | past)` with the mixture above; differentiable in
all V-HPED parameters plus `φ` (softplus) and `b` (sigmoid). `ζ` depends only on `κ` and `ω`
(fixed), so it is precomputed. Training on train videos only; thresholds on validation.

**Contamination argument (why HA4 is expected).** A record on a pair with `ζ_{ij} ≈ 0`
contributes `log Q(o)` — independent of V-HPED's parameters — so phantom births and
identity-induced absences no longer pull `ρ_unk, π₀, μ, α` towards explaining them.
With `ζ ≈ 1` the gradient is V-HPED's. This is a property of the likelihood, not a proof
of better accuracy.

## 7. Inference algorithm and complexity

Per frame: (1) update `ζ_i` for tracks present (`O(#tracks)`); (2) per touched pair, compute
`a⁻, c_p, post_p` (V-HPED, `O(P)`), `Q`, `w`, and the mixture (`O(P)`); (3) emit.
Total per frame: V-HPED cost + `O(#tracks)`.

**Proposition (lazy exactness preserved).** Under L1 (`ρ_unk = 0`) and L2 (`γ = 0`), an
untouched pair has no records, so `T`-products are empty, `P(o|past) = 1`, `w` is irrelevant,
and its evolution is V-HPED's closed form (`lazy_expiry.md` Lemma), which does not involve
`ζ`. Hence the lazy wake-up heap remains exact for IA-VHPED; identity updates cost
`O(#tracks present)` per frame. (Tested by an eager-vs-lazy harness on streams with κ < 1.)

## 8. Identifiability notes

* `φ` vs `ε`: both are "detections not caused by this pair's live relation"; they are separated
  only through `ζ` (φ acts when `ζ < 1`). With uninformative `κ` (constant), `ζ` is constant and
  `φ, b` are weakly identified — HA3 predicts no gain then.
* `b` is identified only by births on low-`ζ` pairs.
* `κ` calibration is assumed (I1); a miscalibrated `κ` (e.g. always 0.5) degrades to diluting
  all evidence by a constant factor — tested via the uninformative setting.

## 9. Expected failure modes

1. Tracker gives no or uninformative confidence ⇒ no gain (HA3).
2. Multi-object swaps / identity merges violate the toggle model (I3).
3. Very frequent jumps ⇒ evidence diluted everywhere ⇒ slower termination.
4. Real-object disappearance coinciding with low κ is attributed to identity, delaying
   termination.

## 10. Ablations

| Ablation | Question |
|---|---|
| IA-VHPED with `κ ≡ 1` | Equals V-HPED? (regression) |
| IA-VHPED with uninformative κ | Is any gain due to identity information? (HA3) |
| IA without identity-aware births (`b` tied to `π₀`) | Do births matter? |
| IA with κ-only weight (`w := ζ`, no within-frame posterior) | Does the exact one-step posterior matter? |
| IA lazy-exact variant (`γ=0`, `ρ_unk=0`) + lazy implementation | Accuracy cost of the efficient variant; equivalence |
| Swap frequency 0 / 2 / 4 per video; κ informativeness 0 / 0.5 / 1 | Dose-response |

## 11. IA-v1 failure and the calibrated jump model (IA-v2)

**IA-v1 result (negative, run `outputs/runs/20261028_050001_0c54b136`).** With the raw mapping
`P(jump) = 1 − κ` (assumption I1), fitted IA-VHPED was *worse* than fitted V-HPED under ID
switches (frame F1 0.829 vs 0.875 at 2 swaps, informative κ) and its results were almost
identical for informative and uninformative κ.

**Diagnosis.** The simulated tracker score is a *confidence*, ≈ U(0.9, 1.0) on normal frames.
Read as a calibrated non-jump probability it implies a ≈5% jump chance per track per frame,
whereas true jumps occur at ≈0.3% of track-frames. Under the toggle model the fixed point of
`ζ ← κζ + (1−κ)(1−ζ)` with `κ < 1` every frame is `ζ = 1/2`, so every pair's identity belief
decays towards 1/4 within tens of frames and *all* evidence is diluted. I1 was violated by
construction — exactly failure mode §9.1.

**IA-v2: learned calibration.** Replace I1 by a two-parameter calibration learned with the
other parameters by maximum likelihood on training videos:
```
J(κ) = σ(a + b · logit(1 − κ)),   b ≥ 0 (softplus),   J(1) = J(None) = 0
ζ ← (1 − J) ζ + J (1 − ζ)
```
`J` is monotone non-increasing in `κ` (b ≥ 0), so a lower tracker score never means a lower
jump probability. `ζ` is computed inside the PyTorch likelihood, so `(a, b)` receive gradients
through every pair's mixture weight; NumPy and PyTorch agree to 1e‑9 (test). The reduction to
V-HPED still holds when no score is supplied or `κ ≡ 1`. Identifiability of `(a, b)` comes
only from how well identity-diluted evidence explains the observation stream; with
uninformative κ the calibration should learn `b ≈ 0` and a small `a` (HA3).
