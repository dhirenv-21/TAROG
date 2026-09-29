# V-HPED complexity and eager/lazy equivalence (Phase 4)

Refers to `research/vhped_formulation.md` §10 and `src/taorg/temporal/vhped.py` (eager
reference). The lazy/heap mechanism is **not** implemented in Phase 4 (it is Phase 5). This note
fixes what Phase 5 must preserve.

## 1. Notation

At frame t: R_t records (detections plus absence records), A_t maintained edges, K_t tracks,
P = number of predicates, S = 4 joint states (Z, V).

## 2. Time per frame (eager reference, as implemented)

| Operation | Cost |
|---|---|
| Transition P(Δ) = expm(QΔ) | O(S³), once per distinct Δ (cached; one value at constant fps) |
| Index records by key | O(R_t) |
| Identity update (M5) | O(K_t) |
| Per maintained edge: predict (S² = 16 mult-adds), likelihood lookup / mixture (S), tempering (S pow, only if Δ ≠ Δ_ref), normalise (S) | O(A_t · S²) |
| Cross-predicate factor (M6) | O(Σ_e \|obs on e's pair\|) ≤ O(A_t · P) |
| **Total** | **O(S³·[new Δ] + R_t + K_t + A_t·(S² + P))** = O(R_t + K_t + A_t·P) for fixed S |

## 3. Space

O(A·S) edge states + O(K) identity beliefs and last boxes + O(P²) exclusivity + O(#distinct Δ ·
S²) cached transitions. The eager reference never prunes, so A counts every key ever observed in
the video.

Pruning is **not exact**. The belief of an edge with no evidence converges geometrically to the
stationary distribution π at rate ≥ min(μ+ν, γ_o+γ_v) (the spectral gap of Q). It reaches π only
in the limit. A pruning rule "drop e once ‖x_e − π‖₁ < δ and p_e ≤ τ; re-initialise at π on
re-detection" changes any later belief by at most δ in total variation (the filter update is a
contraction in the Hilbert projective metric for positive likelihoods). It changes no output
decision unless the belief lies within δ of τ. This bound is stated, not yet tested.

## 4. Lazy materialisation: exact equivalence

**Claim.** For an edge that receives the *same* likelihood vector L (after identity tempering
and evidence-rate tempering) for n consecutive frames with the same Δ, the eager recursion
x_{k+1} = (x_k P) ⊙ L / ‖(x_k P) ⊙ L‖₁ gives

  x_{k+n} = x_k Mⁿ / ‖x_k Mⁿ‖₁,  M = P · diag(L).

**Proof.** Write the unnormalised step as y ↦ y M. Normalisation is a positive scalar c_k; since
(c y) M = c (y M), the product of the n scalars factors out of the linear map. The normalised
iterate after n steps is therefore x_k Mⁿ divided by its ℓ₁ norm. ∎

**Numerics.** `propagate_constant` computes Mⁿ by repeated squaring with max-rescaling after
every product, O(S³ log n). `tests/unit/test_vhped.py::test_lazy_equals_eager_within_tolerance`
checks 20 random generators, likelihoods and n ≤ 400: agreement to rtol 1e-9, atol 1e-12. The
eager path accumulates O(n·ε_mach) relative rounding error, the lazy path O(log n·ε_mach).

**When constancy holds:**
- M2–M4: during any run of identical records (no record, or the same flag), at constant fps.
- M5: only while u_s = u_o = 1. Otherwise u relaxes every frame and L changes; the steady state
  u ≈ 0.9994 of the jitter-free case is not exactly 1, so exact lazy spans require either
  freezing u below a tolerance or piecewise evaluation.
- M6: only while no other predicate on the pair is detected.

## 5. Expiry scheduling (what a heap would need)

For M2 with pure prior propagation (OCC flag with f = ε, or L ∝ 1), the marginal is
p(t) = π + (p₀ − π) e^{−(μ+ν)t}, which is monotone. The drop time
T = ln((p₀ − π)/(τ − π)) / (μ+ν) is exact and can be keyed in a min-heap. For a general constant
L, p along x Mⁿ is a ratio of sums of exponentials in n, which is not necessarily monotone. A heap
key would then need either (i) a monotonicity certificate (e.g. L_{Z=1} ≤ L_{Z=0} componentwise,
which makes the Z-marginal non-increasing), or (ii) a conservative lower bound with lazy
re-validation on pop. Both are Phase 5 work. **The heap is an implementation mechanism, not a
novelty claim.**

## 6. Measured

Latency of the eager reference is reported in `research/phase4_exit_report.md` §19
(`results/phase4/tables/latency.md`).
