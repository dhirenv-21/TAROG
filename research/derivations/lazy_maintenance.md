# Lazy state maintenance for V-HPED: assumptions, proofs, complexity (Phase 5)

Implementation: `src/taorg/temporal/lazy_vhped.py` (`LazyVHPED`) and
`src/taorg/temporal/expiry_queue.py` (`ExpiryQueue`). The eager reference is
`src/taorg/temporal/vhped.py`, which is unchanged. The heap is an **implementation mechanism, not a
novelty claim**. The filter is the four-state continuous-time filter of
`research/vhped_formulation.md`, **not** an exponential Hawkes process. No exponential expiry formula
is used.

## 1. Setting

**Eager step.** For each maintained key e at frame t (formulation §2–§8):
- predict: x′ = x P(Δ);
- update: x ∝ c · L ⊙ x′ (the factor c applies to the Z = 1 states only);
- emit e iff p = x[Z=1] > τ.

**What drives L.**
- L comes from the record type: detection, VISIBLE absence, OCCLUDED absence, UNKNOWN absence or
  no record. Under M5 it is mixed with the identity weight w; with Δ ≠ Δ_ref it is tempered by the
  exponent a = Δ/Δ_ref.
- c is the M6 cross-predicate factor.

**The eager reference never prunes.** Every key ever observed is updated in every frame.

## 2. Assumptions under which a key may be left untouched

For a key e and frames t + 1, …, t + n, the lazy filter may skip e only if all of the following
hold:

| | Assumption |
|---|---|
| (A1) | e receives the **no-record** likelihood L_unk, or an explicit UNKNOWN absence record, which has the same likelihood. No detection, and no VISIBLE or OCCLUDED absence |
| (A2) | The identity weight is w = 1. In M5 it is not, so M5 is an **eager fallback**: no key is ever frozen |
| (A3) | c = 1: no other predicate is observed on e's pair while cross-predicate coupling is on (M6) |
| (A4) | The frame interval Δ is constant (one "regime"), so P(Δ), the tempering exponent a and L_unk^a are constant |

Under (A1)–(A4), x_{k+1} ∝ x_k M with **M = P(Δ) diag(L_unk^a)**, a fixed 4×4 matrix.

**M is strictly positive.**
- Q is an irreducible CTMC generator, so P(Δ) = expm(QΔ) has all entries > 0.
- L_unk = (1−f, 1−f, 1−ε, m) has all entries > 0 (formulation §4).
- Therefore M > 0 entrywise.

## 3. Exact materialisation (unchanged from `research/derivations/vhped_complexity.md` §4)

**Proposition 1.** Under (A1)–(A4), x_{t+n} = x_t Mⁿ / ‖x_t Mⁿ‖₁.

*Proof.* Normalisation multiplies by positive scalars, which commute with the linear map. ∎

**Implementation.**
- Spans of ≤ 16 frames are replayed with the eager arithmetic.
- Longer spans use `propagate_constant` (repeated squaring with rescaling, O(S³ log n)), which is
  tested against stepwise propagation to rtol 1e-9.

## 4. A certificate that a key can never be emitted again (no monotonicity assumed)

Let x\* be the Perron left eigenvector of M (normalised to sum 1), and p\* = x\*[Z=1] its
fixed-point marginal. Let d_H(x, y) = max_i log(x_i/y_i) − min_i log(x_i/y_i) be Hilbert's
projective metric on the positive orthant.

**Lemma 1 (Birkhoff).** For positive M, d_H(xM, yM) ≤ κ(M) d_H(x, y), where
κ(M) = tanh(Δ(M)/4) < 1 and Δ(M) = max_{i,j,k,l} log(M_ik M_jl / (M_il M_jk)) is the projective
diameter. Since x\* M ∝ x\*, the sequence d_n = d_H(x_n, x\*) is non-increasing, and
d_{n+m} ≤ κ(M^m) d_n for every block length m.

**Lemma 2 (odds bound).** If d_H(x, y) ≤ D, then odds(p(x)) ≤ e^D · odds(p(y)).

*Proof.* Let r_i = x_i/y_i ∈ [a, b] with log(b/a) ≤ D. Then

  (x₂+x₃)/(x₀+x₁) ≤ b(y₂+y₃) / (a(y₀+y₁)). ∎

**Proposition 2 (certificate).** Suppose (A1)–(A4) hold from frame t on, p\* < τ, and
d_t ≤ T := logit(τ) − logit(p\*). Then p_{t+k} ≤ τ for every k ≥ 0 while the assumptions hold, so
e is never emitted.

*Proof.*

  logit(p_{t+k}) ≤ logit(p\*) + d_{t+k}   (Lemma 2)
               ≤ logit(p\*) + d_t      (Lemma 1)
               ≤ logit(τ). ∎

**Notes.**
- The marginal trajectory p_k may be non-monotone. The certificate bounds the whole future, not
  the next step.
- If p\* ≥ τ, no certificate exists: the regime keeps the key above threshold forever, and the key
  stays eager.
- Implementation: T includes a safety margin of 1e-9 in log-odds; at pop time, d is recomputed
  from the actual state.

**Proposition 3 (certification deadline).** If d_t > T, then d_{t+n} ≤ T is guaranteed for

  n₁ = min over m of m · ⌈log(T/d_t) / log κ(M^m)⌉,

where the minimum is taken over the computed block lengths m ∈ {1, 2, 4, …, 2¹⁴} with
κ(M^m) < 1. This follows from Lemma 1 applied block-wise.

**Scheduling.**
- `ExpiryQueue` holds the deadline max(n₁, 15). The 15-frame minimum quiet period is a cost policy
  that suppresses freeze/thaw churn; it does not affect correctness.
- Any evidence change cancels the deadline (generation invalidation).
- At pop time, the key is frozen only if the recomputed d ≤ T; otherwise it is rescheduled.

**Termination-delay corollary (explicit assumptions).**
- Suppose a key is ON (p > τ) when its relation stops being observed. Suppose further that from
  then on it receives only no-record evidence under (A1)–(A4), with p\* < τ.
- Then it is OFF for good by frame t + n₁(d_t). This is an **upper bound** on the stable
  termination delay of the no-record regime.
- It does not cover VISIBLE-absence evidence, which terminates faster, or M5/M6 effects.

## 5. Freezing never deletes the posterior

A frozen key keeps (x_t, t, regime). It is **materialised** exactly (Proposition 1) and returned
to the active set when any of the following happens:
- it receives any record other than UNKNOWN. Recurrence and reactivation are therefore exact;
- another predicate is observed on its pair under M6 (A3 breaks);
- the frame interval changes. A4 breaks, so all frozen keys are materialised under the old regime;
- its belief is read. This is a non-mutating evaluation.

## 6. Equivalence statement (what is guaranteed)

**What is guaranteed.** For M2–M6, every emitted edge and its score equal the eager reference's
up to floating-point error:
- active keys use the same arithmetic in the same order;
- frozen keys provably emit nothing (Proposition 2);
- materialisation is exact in real arithmetic (Proposition 1). Its floating-point error is
  O(log n · ε) for the closed form and O(n · ε) for stepwise propagation.

**Where decisions can differ.** Only if an eager score lies within that error of τ.

**Verified** (`tests/unit/test_lazy_vhped.py`):
- randomized traces for M2–M5, M6 with random exclusivity, interval jitter and visibility records;
- recurrence after freezing;
- deadline cancellation and stale entries;
- a 20,000-frame trace;
- certificate soundness on 50 random maps;
- the benchmark (28 runs): **0 decision mismatches**, maximum |Δscore| = 5.6e-16.

## 7. Complexity per frame

Notation: R_t = records, H_t = active keys (recently changed, emitted, or awaiting certification),
F_t = frozen keys, E_t = emitted edges (E_t ≤ H_t), m_t = materialisations, q_t = heap operations.

| | Eager | Lazy |
|---|---|---|
| Key updates | O(A_t) = all keys ever seen | O(H_t) |
| Indexing records | O(R_t) | O(R_t) |
| Materialisation | — | O(m_t · S³ log n) |
| Heap | — | O(q_t log Hp), heap size ≤ 3·scheduled + 64 (compaction) |
| Regime change | — | O(F_t) materialisations + O(S³ · 15) for a new certificate (eigendecomposition + 15 squarings), cached per interval |
| **Output** | Ω(E_t) | Ω(E_t) |

- **Space:** both are O(A · S). Lazy adds O(heap) and the frozen index.
- **Output enumeration.** Emitting every active edge in every frame costs Ω(E_t) for both filters.
  **No sublinear total runtime is claimed**: the saving is in updates to keys that are neither
  emitted nor changing.

## 8. Benchmark (synthetic; `scripts/phase5/benchmark.py`; `results/phase5/`)

**Configuration.**
- Rung M3 parameters; 1,800 frames at 30 fps; 60 tracks; seeds 53001 and 53002.
- Settings vary: candidate keys A ∈ {100, 400, 1,600, 6,400}, ON rate, occlusion rate, retirement
  rate, predicate count, and interval jitter.
- Timings are wall-clock on an AMD 7000-series laptop CPU under Windows. They are **not
  portable**.

**Results** (benchmark v2 at commit `e9a08a9`; 28 runs):

| Setting | Key updates / frame (eager → lazy) | Emitted / frame | Speed-up |
|---|---|---|---|
| A = 100 / 400 / 1,600 / 6,400 | 84→19 / 344→82 / 1,330→327 / 4,743→1,164 | 7 / 32 / 129 / 462 | 1.65 / 1.72 / 1.75 / 1.62 |
| sparse observations (p_on 0.01) | 1,324 → 297 | 63 | 1.89 |
| many retirements (p_retire 0.02) | 1,322 → 172 | 29 | **3.10** |
| few retirements (p_retire 0.0005) | 1,322 → 866 | 493 | **0.79 (slower)** |
| interval jitter 1% / 10% of frames | 1,325 → 471 / 1,327 → 1,163 | 129 | 1.06 / **0.47 (slower)** |

**First run.** `benchmark_v1_unoptimised.jsonl` is kept: same equivalence, 0.3–3.0×. It came
before the cost-only optimisations (pure-Python 4-state distance, conditional cancel, minimum quiet
period).

**Reading.** Laziness pays when stale keys accumulate. It costs extra when most keys are alive and
emitted (output-bound), and under variable frame rate, where every interval change thaws all frozen
keys.

## 9. Limitations

1. **M5 is fully eager.** The identity weight relaxes every frame, so no constant-evidence span
   exists.
2. **Variable frame rate defeats the regime.** A per-interval certificate cache with exact
   piecewise materialisation would need storing the interval sequence; it is not implemented.
3. **Certificate conservativeness.** Δ(M^m) is computed in floating point. The 1e-9 log-odds margin
   and the re-verification at pop time guard against error, but this is not interval arithmetic.
4. **Python constant factors dominate** at these sizes. The operation-count reduction (≈ 4×) is
   larger than the wall-clock gain (≈ 1.7×).
5. **Benchmarks are synthetic.** Real streams are measured in the final evaluation.
