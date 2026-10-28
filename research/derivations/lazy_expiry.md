# Lazy V-HPED — Exactness, Expiry Scheduling, and Complexity (Phase 5)

> Implementation: `src/taorg/temporal/vhped/lazy.py`, `expiry_heap.py`.
> Oracle: eager `VHPED` (`model.py`). Equivalence harness:
> `tests/property/test_lazy_equivalence.py`. Benchmark:
> `scripts/run_complexity_benchmark.py`.
> Status: proofs below are for the discrete-time model as implemented; the
> empirical evidence is one machine's benchmark (see §7) — latency is
> machine-dependent, operation counts are not.

## 1. Setting

The eager reference processes, at every frame `n`, **every active pair**
(pairs created by an observation and not yet pruned). A pair is *touched* at
frame `n` if the frame contains at least one record (observed or absent) for
it; otherwise it is *untouched* and the eager model treats it with regime
`UNK` and no observations.

Let `A` = active pairs, `P` = predicates, `k_touch(n)` = touched pairs in
frame `n`, `k_wake(n)` = valid wake-ups popped at frame `n`.

## 2. Lazy-exactness conditions

* **(L1)** `ρ_UNK = 0` — a pair absent from the frame carries no evidence
  (Proposition 1 of `vhped_model.md`: exact censoring).
* **(L2)** `γ = 0` — the termination hazard does not depend on the traces,
  so `h_p = softplus(η_p)` is constant.

No condition on `r` is needed because the predict step is the exact
two-state CTMC transition (`vhped_model.md` §4).

If (L1) or (L2) fails, `LazyVHPED` sets `lazy_exact = False` and updates
every active pair every frame (identical to eager; tested).

## 3. Lemma (untouched evolution is a semigroup)

Under (L1)–(L2), one eager frame step of an untouched pair over interval `Δ`
maps, for every tracked predicate `p`,

```
e    ← e · exp(-β_UNK Δ)
a_p  ← π_p + (a_p − π_p) · exp(−k_p Δ),   k_p = h_p + r,  π_p = r / k_p
```

and adds `log c = 0` to the log-likelihood.

*Proof.* With `ρ = 0` and no observation both emission likelihoods equal
`exp(0) = 1`, so the update leaves the predicted belief unchanged and the
evidence is 1. `h_p` is constant by (L2). The predict step is the CTMC
transition stated. ∎

Both maps satisfy `F(Δ₁) ∘ F(Δ₂) = F(Δ₁ + Δ₂)`. Hence after any sequence of
untouched frames from time `t₀` to `t` the state depends only on `t − t₀`:

```
a_p(t) = π_p + (a_p(t₀) − π_p) · exp(−k_p (t − t₀))          (★)
```

## 4. Expiry / wake-up scheduling

**Monotonicity.** (★) is monotone in `t`, converging to `π_p`. For a level
`L` (output threshold `τ`, or prune level `ε_prune`), the side of `L` changes
at most once, at

```
d*_p(L) = ln( (a_p(t₀) − π_p) / (L − π_p) ) / k_p     if a_p(t₀) and π_p lie on opposite sides of L
d*_p(L) = ∞                                          otherwise
```

(`crossing_delay` in `expiry_heap.py`; brute-force tested).

**Decision timing.** Eager keeps an edge at frame time `t` iff `a_p(t) ≥ τ`
and prunes iff `a_p(t) < ε_prune`. For a downward crossing these change
exactly for `t > t₀ + d*`; so an event is applied at the first frame with
timestamp **strictly greater** than its scheduled time. (For an upward
crossing — possible only when `r > 0` and `π_p > τ` — eager keeps at
`t ≥ t₀ + d*`; the lazy model differs only if a frame lands *exactly* on the
crossing time, a measure-zero tie covered by the declared tolerance.)

**Per-pair wake-up heap.** Each pair holds one authoritative wake-up
`w = t₀ + min_p min(d*_p(τ), d*_p(ε_prune))` with a wake id. Rules:

1. On touch: compute `w_new`. If a pending wake-up `w_old ≤ w_new` exists,
   keep it (it is *early*, which is safe); else push `w_new` with a new id.
2. On pop (time `< t_frame`, id current): re-evaluate every tracked
   predicate with (★) at `t_frame`, apply threshold/prune changes,
   materialize, and push the next wake-up from the new state.
3. Entries whose id is no longer current are skipped (stale).

*Safety of early wake-ups.* A wake-up only re-evaluates (★) exactly, so
waking early changes nothing except cost. *Completeness.* The pending
wake-up is never later than the earliest true crossing: rule 1 keeps
`min(w_old, w_new)`; rule 2 recomputes from the exact state. So no crossing
is applied late. ∎

**Theorem (equivalence).** Under (L1)–(L2) and exact arithmetic, `LazyVHPED`
emits the same edge set with the same alive scores as `VHPED` at every
frame, and accumulates the same log-likelihood.
*Proof sketch.* Touched pairs: materialize with (★) to the previous frame
time (Lemma), then run the identical eager step. Untouched pairs: their
eager state equals (★) (Lemma); their output membership changes only at
crossings, which are applied at the correct frame (§4). Pruning likewise.
Untouched frames add 0 to the log-likelihood. ∎

**Floating point.** Eager composes per-frame exponentials; lazy uses one
exponential. Declared tolerance: scores rel 1e-9 / abs 1e-12; decisions may
differ only where `|a − τ| < 1e-9`. The harness (24 randomized streams with
irregular timestamps, pair drop-outs, mixed visibility, `r ∈ {0, 0.4}`,
three thresholds, three prune levels) observed **no** decision differences.

## 5. Complexity

Per frame, with heap size `H`:

| | Eager | Lazy |
|---|---|---|
| state maintenance | `Θ(A · P)` | `O(k_touch · P + (k_touch + k_wake) · (P + log H))` |
| output enumeration (separate, Rules §9) | `O(|S|)` | `O(|S|)` |

* Each touch costs one eager pair step (`O(P)`), one materialization
  (`O(P)`), one membership refresh (`O(P)`), and **at most one** heap push.
* Each valid wake-up costs `O(P)` re-evaluation + one push. Stale pops are
  bounded by the total number of pushes.
* Total pushes ≤ (touches + wake-ups); hence over a video the heap work is
  `O((K_touch + K_wake) log H)`, where `K` are totals over frames.
* Rebuild: the stale-entry scan runs only when the heap has doubled since the
  last check (amortized `O(1)` per push), and drops stale entries when they
  exceed 3/4 of the heap, so `H = O(live wake-ups) = O(A)` amortized.
* Memory: `O(A · P)` state + `O(H)`.

This refines the PRD's `O((k_obs + k_vis + k_exp) log A)` claim: `k_obs + k_vis`
corresponds to `k_touch` (a pair with any record, observed or visible/occluded
absence), `k_exp` to `k_wake`, and the extra factor `P` is the per-pair
predicate vector. Candidate-pair generation (building the frame's records) is
outside both counts.

**Number of wake-ups.** Each pair has at most `2·|tracked|` true crossings
between touches (monotone (★)), plus at most one early wake-up per touch.
So `K_wake ≤ K_touch + 2·P·(number of untouched stretches)`.

## 6. Termination-delay proposition (explicit assumptions)

From `vhped_model.md` Corollary 2a: assume (i) the edge was last observed at
`t₀` with posterior odds `o₀`; (ii) the pair stays **touched with regime VIS**
and the predicate is absent for `k` frames of length `Δ`; (iii) no new events
on the pair (traces only decay); (iv) `γ = 0` and `r = 0`; (v)
`λ_p(t) ≥ λ_min > ε` on the window; threshold domain = alive probability `τ`.
Then the edge is dropped after at most
`k* = ⌈(ln o₀ − ln(τ/(1−τ))) / ((λ_min − ε) Δ)⌉` frames. Under regime UNK with
(L1) the drop time is instead the closed-form `d*_p(τ)` of §4, depending only
on `h_p` (and `r`). Under OCC (`ρ_OCC ≈ 0`) the delay grows like
`1/(ρ_OCC (λ − ε) + h_p)` — the intended persistence through occlusion.

## 7. Empirical check (machine-dependent)

`scripts/run_complexity_benchmark.py`, P = 26, 2 predicates per pair, 5
warm-up frames (all pairs touched) + 60 frames, touch fraction `f`:

* decisions: 0 mismatches against eager in all 16 grid points;
* lazy/eager pair steps ≈ `(5 + 60 f)/65` (counter, machine-independent);
* latency: lazy faster for `f ≤ 0.5` (≈1.3× at 0.5, ≈2.5× at 0.2, ≈5× at
  0.05); **slower (≈0.78×) at `f = 1`**, where there is nothing to skip and
  scheduling overhead remains. Run artifact path and exact numbers: see
  `Memory.md` / `research/claims.md`.

## 8. Limitations

1. Pairs touched with regime OCC are updated eagerly although `ρ_OCC ≈ 0`;
   extending laziness to OCC requires `ρ_OCC = 0` (exact censoring) and a
   regime-specific closed form — straightforward but not implemented.
2. With `γ ≠ 0` the untouched hazard is `softplus(η + c e^{−βt})`, whose
   integral has no elementary closed form; lazy mode falls back to eager.
3. Output enumeration is `O(|S|)` per frame in both versions because scores
   are emitted every frame; a consumer that only needs set *changes* could
   use the event stream instead.

## 9. Are L1 and L2 necessary? (Stage 4, Direction D)

**Proposition (necessity, generic case).** Let a tracked predicate of an untouched pair
have belief `a ∈ (0,1)`, traces `e ≠ 0`, `λ ≠ ε`, `h > 0`. If `ρ_UNK > 0` (L1 fails) or
`γ[·,p] ≠ 0` (L2 fails), the eager filter's belief after elapsed time `D` depends on how
`D` is split into frames. Hence no function of `(state at t₀, D)` reproduces it, and an
elapsed-time closed form — the basis of exact wake-up scheduling — cannot exist.

*Why.* (L1 fails) each untouched frame multiplies the odds by `exp(−ρ_UNK (λ(t_n) − ε) Δ_n)`
with `λ(t_n)` the left-limit intensity of a decaying trace; the product is a Riemann sum of
`λ` that depends on the partition, and it is interleaved with the hazard step, which does not
commute with odds multiplication. (L2 fails) the predict step multiplies by
`exp(−h(t_n) Δ_n)` with `h(t_n) = softplus(η + γ e(t_n))` varying with the decaying trace —
again a partition-dependent Riemann sum. *Check:* `test_violating_a_condition_makes_belief_partition_dependent`
(partitions `[0.3]`, `[0.1]×3`, `[0.05, 0.25]`, `[0.01]×30` give beliefs differing by
> 1e‑4), while with L1+L2 the four partitions agree to 1e‑12
(`test_conditions_hold_belief_depends_only_on_elapsed_time`).
*Degenerate exceptions:* `e = 0` (no excitation left), `λ ≡ ε`, or `h = r = 0` — then
the partition dependence vanishes. So L1/L2 are necessary in the generic case, not merely
convenient. This is a statement about this discrete-time filter, proved by the argument
above and checked numerically; it is not a general theorem about point-process filters.

**Extension to IA-VHPED.** The identity mixture does not change untouched evolution under
L1 (all likelihood factors are 1, so `a = a⁻` whatever `ζ`); lazy exactness carries over
(`identity_aware.md` §7; tested by `test_lazy_ia_matches_eager_ia`). Identity beliefs are
per-track and cost `O(#tracks present)` per frame.

## 10. When is lazy guaranteed faster? Adversarial workloads

With per-step costs `c_E` (eager pair step), `c_L ≥ c_E` (lazy touched step: eager step +
materialization + refresh + scheduling) and `c_W` (wake-up), lazy is faster on a frame iff
`c_L k_touch + c_W k_wake < c_E A`. Using `k_wake ≤ k_touch + (true crossings)`, a sufficient
condition is `k_touch / A < c_E / (c_L + c_W)`. Measured on this machine `c_L/c_E ≈ 1.3`,
so the break-even touch fraction lies between 0.5 and 1 (§7: 1.26× at 0.5, 0.77× at 1).
**Adversarial workloads:** (i) every pair touched every frame (`f = 1`): no skipping, pure
overhead; (ii) pairs touched every other frame: every touch reschedules and roughly every
wake-up is early — measured in `scripts/run_complexity_benchmark.py --pattern alternating`.
No workload makes lazy asymptotically worse than eager by more than the constant
`(c_L + c_W)/c_E`, because each frame's lazy work is bounded by `O(A)` touches plus at most
one wake-up per touch or true crossing.
