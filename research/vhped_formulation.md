# V-HPED formulation (Phase 4) — written BEFORE implementation

> Status: **frozen with the Phase 4 pre-registration** (`research/phase4_preregistration.md`).
> Any change after the pre-registration commit is an amendment. The name "V-HPED"
> (Visibility-aware Hawkes-Process Edge Dynamics) is **provisional**. §9 argues that the Hawkes
> label is not justified by this formulation and should be dropped unless a later formulation
> earns it.

## 0. Design goal, derived from the Phase 3 failures

Phase 3 (`research/phase3_exit_report.md`) showed that deterministic visibility gating
(GatedEMA) helps when the visibility flag is informative, but that it fails structurally in
five places:

| Phase 3 failure | Root cause in GatedEMA | Mechanism proposed here |
|---|---|---|
| A. Termination while occluded (delay ≈ 27 vs 5 frames) | Occluded absence is never evidence; the hold length is a per-frame decay constant tuned on a regime without such events | Explicit termination hazard during unobserved periods (§3, M2) and an occlusion-onset termination coupling (§6, M4) |
| B. All-UNKNOWN visibility (delay ≈ 38) | UNKNOWN is decayed like OCCLUDED by default | UNKNOWN = *uninformative about visibility*; latent visibility is inferred (§4, M2/M3) |
| C. ID-switch ghosts (2–4×) | A key with no record is held as UNKNOWN; memory transfers blindly across identities | Identity-reliability-tempered likelihoods (§7, M5) |
| D. Flicker / tuning shift (≈ 10× churn) | Visible absence = hard termination evidence when `decay_visible = 0` | Probabilistic miss model: visible absence is *finite* evidence whose weight is a likelihood ratio (§4) |
| E. Parameter transfer | Per-frame constants; flag treated as truth | Parameters in physical time (s⁻¹) and flags as noisy sensors (§4, §5, M3) |

Two distinctions are built into the model:

- **NO OBSERVATION ≠ NEGATIVE RELATIONSHIP EVIDENCE.** A missing detection is negative evidence
  only in proportion to the belief that the relationship *could have been seen*.
- **OCCLUDED ≠ TERMINATED.** Occlusion removes evidence; it does not by itself end a
  relationship. Termination is a latent event with its own hazard.

## 1. Quantities (kept strictly separate)

For an edge e = (u, v, r) (subject track u, object track v, predicate r) and physical time t (s):

| Symbol | Meaning | Type |
|---|---|---|
| Z_e(t) ∈ {0,1} | the relationship **exists** at t | latent |
| V_e(t) ∈ {0,1} | the relationship is **observable** by the base model at t (both endpoints visible) | latent |
| C_k(t) ∈ {0,1} | track k's **identity** is correct at t | latent |
| D_{e,t} ∈ {0,1}, s | **relationship evidence**: the base model detected e (with score s) | observed |
| F_{e,t} ∈ {VIS, OCC, UNK} | **visibility evidence**: the flag on an absence record (UNK also covers "no record") | observed |
| p_e(t) = P(Z_e(t)=1 \| history) | **relationship existence probability** (the output belief) | derived |
| Z: 1 → 0 transition | **relationship termination** | latent event |
| D = 0 | **observation absence** | observed |

Output rule: e is emitted at t iff p_e(t) > τ.

## 2. Model ladder (each rung must earn its place; see `research/vhped_ablation_plan.md`)

| Rung | Name | Adds |
|---|---|---|
| M0 | GatedEMA (Phase 1) | reference: deterministic per-flag decay |
| M1 | Visibility-aware HMM | discrete-time 2-state HMM whose absence likelihood depends on the flag (§8) |
| M2 | Continuous-time probabilistic edge dynamics | latent (Z, V) continuous-time Markov chain, rates in s⁻¹, evidence-rate likelihoods; flags trusted as exact when VIS/OCC, UNK inferred |
| M3 | + uncertain visibility | flags are noisy sensors of V with confusion rates r_ov, r_vo |
| M4 | + termination hazard at occlusion onset | coupled transition (Z,V): (1,1) → (0,0) |
| M5 | + identity reliability | likelihoods tempered by P(identities correct) |
| M6 | + cross-predicate interaction | exclusivity evidence between predicates on the same pair |

## 3. Prior dynamics (M2): continuous-time Markov chain

Z and V follow independent two-state continuous-time Markov chains:

- Z: 1 → 0 at rate μ (termination hazard, s⁻¹); 0 → 1 at rate ν (birth/re-activation).
- V: 1 → 0 at rate γ_o (becomes occluded); 0 → 1 at rate γ_v (becomes visible).

Joint state order (Z, V) ∈ {(0,0), (0,1), (1,0), (1,1)}. Generator Q = Q_Z ⊕ Q_V
(Kronecker sum), transition matrix P(Δ) = exp(QΔ).

**Derivation: absence of evidence relaxes to the prior, not to FALSE.** Marginalising V, and with
no evidence over [t, t+Δ], the Kolmogorov forward equation for p(t) = P(Z=1) is
dp/dt = ν(1 − p) − μp, whose solution is

  p(t+Δ) = π + (p(t) − π) · e^{−(μ+ν)Δ},  with π = ν / (μ + ν).

So in the absence of evidence the belief moves toward the stationary prior π. It moves toward
0 only if π = 0. With π = 0.5 the belief moves toward maximal uncertainty (UNKNOWN), as required.

**Closed-form hold horizon.** If τ > π and p(t₀) = p₀ > τ, the edge is dropped after

  T_hold = ln((p₀ − π)/(τ − π)) / (μ + ν)  seconds,

a quantity in physical time. The Phase 3 hold horizon of GatedEMA was
ln(τ/s₀)/ln(decay_occluded) *frames*. For this reason the parameter search enforces τ > π + 0.01
(otherwise an unobserved edge is never dropped).

**Termination during occlusion.** μ > 0 means termination probability grows with elapsed
unobserved time. The belief does not stay fixed under occlusion; this is the principled
replacement for a magic timeout. Stated before any data: in the Phase 3/4 synthetic world, a
termination followed by occlusion is observationally identical to an occlusion gap until the
durations diverge. Therefore **no estimator can separate them before that point without
additional evidence**, and on condition F1 (termination during occlusion) we expect every
well-tuned hold rule, including M0 and M1, to lie on nearly the same frontier. M4 (§6) and M6
(§10) are the only rungs that add information-bearing structure; M4's structure is a prior
assumption, not evidence.

## 4. Observation model (M2, M3)

Per frame at time t, with Δ = t − t_prev, for each maintained edge:

| Evidence | L(Z=1,V=1) | L(Z=1,V=0) | L(Z=0,V=1) | L(Z=0,V=0) |
|---|---|---|---|---|
| detected (D=1) | 1 − m | ε | f | f |
| absent, flag VIS | m (1 − r_vo) | (1 − ε) r_ov | (1 − f)(1 − r_vo) | (1 − f) r_ov |
| absent, flag OCC | m r_vo | (1 − ε)(1 − r_ov) | (1 − f) r_vo | (1 − f)(1 − r_ov) |
| absent, flag UNK / no record | m | 1 − ε | 1 − f | 1 − f |

- m = P(miss | exists, observable), the base-model miss/flicker rate.
- f = P(detection | not existing), the false-positive rate.
- ε = P(detection | exists, not observable) = 10⁻³ (fixed).
- r_ov = P(flag VIS | V=0), r_vo = P(flag OCC | V=1): flag confusion rates. In **M2 both are
  fixed at 10⁻⁶** (flags trusted as exact); in **M3 they are tuned** (uncertain visibility).
  UNK carries no information about V (its probability is the same under V = 0 and 1, so it
  cancels).

**Likelihood ratio of an absence for existence**, if V were known to be visible with
probability q:

  LR_absent = P(D=0 | Z=1) / P(D=0 | Z=0) = (q·m + (1−q)(1−ε)) / (1 − f).

- q = 1: LR = m/(1−f) ≪ 1. Visible absence is strong but *finite* negative evidence; its
  strength is controlled by m, so flicker (visible misses) no longer implies hard termination.
- q = 0: LR ≈ 1. Occluded absence is **no evidence**, and the belief follows the prior dynamics
  of §3.

This is the formal content of "no observation ≠ negative evidence". The exact filter does not
use a plug-in q: it propagates the joint (Z, V) distribution, so q is the filtered posterior
P(V=1 | history) and a single flag that contradicts a long run is automatically discounted by
the persistence of V (rates γ_o, γ_v). This is the graceful degradation required when the flag
is unreliable, and it replaces the `if visible: … else: …` of GatedEMA.

**Evidence rate (frame-rate invariance).** Consecutive frames are not conditionally independent
views, so each per-frame likelihood is tempered: L ← L^{Δ/Δ_ref}, with Δ_ref = 1/30 s. This is
a power-likelihood modelling assumption, flagged as such. It makes evidence accumulate per
second rather than per frame, and at 30 fps it is exactly the untempered model. Together with
§3, the whole model is parameterised in physical time, which is the only property that
distinguishes M2 from M1 at a fixed frame rate (§8).

**Filter recursion (exact, per edge):**
x_t ∝ L_t^{Δ/Δ_ref} ⊙ (P(Δ)ᵀ x_{t−1}), with x a 4-vector, normalised to sum 1.
p_e(t) = x_t(1,0) + x_t(1,1). A new key is initialised at the stationary distribution of Q when
it is first observed, then updated with that observation.

## 5. What M3 changes

Only r_ov and r_vo become free parameters. With r = 10⁻⁶ (M2), a false VIS flag during an
occlusion is treated as certain visibility, which gives strong negative evidence (the Phase 3
R1 failure of C0-tuned GatedEMA). With calibrated r, the same flag is weighed against the
V-persistence prior.

## 6. M4: termination hazard at occlusion onset

The generator is modified so that from (Z=1, V=1) the chain moves to (1,0) at rate (1−η)γ_o
and to (0,0) at rate ηγ_o (termination coinciding with loss of view, e.g. an object carried out
of view), with η ∈ [0, 0.9]. P(Δ) = expm(QΔ) is computed once per distinct Δ and cached.
η = 0 recovers M3. Prior expectation, stated before data: onset coupling multiplies the belief
by the same factor at *every* occlusion onset, so it shifts the hold horizon for gaps and for
terminations alike. In a world where the two are indistinguishable at onset, we expect **no
frontier gain** from M4 on F1 (§3). The rung is retained in the ladder because Phase 4 is meant
to test it, not assume it.

## 7. M5: identity reliability

Per track k, u_k(t) = P(C_k(t) = 1).

- Recovery (predict): u ← 1 − (1 − u) e^{−r_id Δ}.
- Switch evidence (update) from the normalised bounding-box centre displacement
  d_k = ‖c_k(t) − c_k(t_prev)‖ / diag_k(t). Under "no switch" d ~ half-normal(σ_id); under
  "switch" d ~ Uniform[0, 2]. With a fixed switch hazard h = 0.03 s⁻¹ (per-frame prior
  h_Δ = 1 − e^{−hΔ}), the posterior switch probability is
  s = h_Δ · 0.5 / (h_Δ · 0.5 + (1 − h_Δ) · φ(d; σ_id)), and u ← u (1 − s).
- Edge weight w_e = u_u · u_v. Every per-frame likelihood of e is tempered as a mixture:
  L_eff = w_e · L + (1 − w_e) · 1. With probability 1 − w_e the record belongs to a different
  physical relationship and is uninformative about this one.

Effect: during a suspected switch, an absent true key receives little negative evidence (so it
is held), and a newly appearing ghost key receives little positive evidence (so it is slow to be
born). Outside suspected switches, M5 = M4. The cue is not an oracle: Phase 4 adds
bounding-box jitter (σ = 0.15 box diagonals) to **every** condition, so M5 pays for false
alarms everywhere and gains only where switches occur.

## 8. The visibility-aware HMM baseline (M1) and its relation to M2

M1: Z only, discrete time. p_stay = P(Z_t=1 | Z_{t−1}=1), p_birth = P(Z_t=1 | Z_{t−1}=0).
Emissions: P(D=1 | Z=1) = 1 − θ_F on absence-bearing frames, where θ_F = P(absent | Z=1, flag F)
for F ∈ {VIS, OCC, UNK}, and P(D=1 | Z=0) = f. The flag is used *probabilistically*, through
three tuned likelihoods, not as a switch. Parameters: p_stay, p_birth, prior, θ_VIS, θ_OCC,
θ_UNK, f, τ.

**Proposition (stated a priori).** At a fixed frame rate, M2 restricted to frames with a VIS or
OCC flag is a special case of M1: V is then known, so P(D=0 | Z=1, flag) is m for VIS and
1 − ε for OCC. M2 differs from M1 only (i) during UNK/no-record runs, where M2 carries visibility
belief forward through the V chain while M1 uses a memoryless θ_UNK, and (ii) under a frame-rate
change. *Expectation declared before data: M2 ≈ M1 on all 30 fps conditions except transients
after flag changes; M2 > M1 only on the frame-rate-shift conditions.* If the data show a larger
M2–M1 gap, that is a finding to explain, not a confirmation.

## 9. Cross-predicate interaction (M6) and the "Hawkes" question

**Exclusivity statistic** (learned, not declared). From the *calibration* ground truth of the
condition, for predicates r ≠ r':
E[r, r'] = n_succ(r → r') / (n_prior(r → r') + 1), where n_prior counts births of r' on a pair
that previously carried r, and n_succ counts those births occurring within one frame after r
ended on that pair. In worlds without predicate succession, E ≈ 0 and M6 = M5.

**Evidence.** When (u, v, r') is detected at t, every maintained edge (u, v, r) with r ≠ r'
receives the likelihood factor L(Z_r=1) = (1 − w_x · E[r, r'])^{Δ/Δ_ref}, L(Z_r=0) = 1, with
w_x ∈ [0, 1] tuned. Observing a successor predicate is evidence that r ended, even when r's own
absence is uninformative (UNK or noisy flags).

**Is this a Hawkes process?** A multivariate Hawkes process specifies the conditional
*intensity of events*, λ_r(t) = μ_r + Σ_{r'} Σ_{t_j < t} α_{r r'} κ(t − t_j). Here:

- nothing models the *intensity* of detection events;
- the cross-type influence acts on the *log-odds of a latent state*, as a Bayesian likelihood
  factor;
- its effect decays through the prior relaxation of §3 (exponential kernel e^{−(μ+ν)Δ} on the
  log-odds deviation from the prior), which is only formally analogous to an exponential
  Hawkes kernel.

**Conclusion (a priori): the formulation is a continuous-time factorial HMM filter with
identity-tempered likelihoods and optional cross-predicate evidence coupling. It is not a Hawkes
process.** Only inhibition is modelled, because only inhibitory structure is present in the H3
world; excitation is untested. If M6 fails H3, it is removed. Either way, the method should be
renamed (e.g. "visibility-aware continuous-time edge filter") unless a genuine self-exciting
intensity model is introduced and tested later.

**Novelty (to verify in the literature, not claimed):** factorial HMMs, continuous-time Markov
chains, tempered/power likelihoods and mixture likelihoods for association uncertainty are all
standard. Any contribution would lie in the decomposition for relationship edges and in the
evaluation, not in the inference algorithm.

## 10. Complexity (derived before implementation; checked in the exit report)

Let R_t be the records at frame t, A_t the maintained edges, K_t the tracks, and S = 4 the
joint states.

- **Time per frame (eager reference):** O(A_t · S²) for prediction and update, plus O(K_t) for
  M5 identity updates, plus for M6 O(Σ_pairs |obs on pair| · |maintained edges on pair|), which
  is O(A_t · P) in the worst case (P = predicates). expm(QΔ) is O(S³) once per distinct Δ
  (cached; one value at a constant frame rate).
- **Space:** O(A · S) for edge states, O(K) for identities, O(P²) for E. The reference
  implementation does not prune edges, so A counts keys ever observed in the video. It is bounded
  by the number of distinct (u, v, r) keys the base model outputs. Pruning is exact only for keys
  whose state equals the stationary distribution, which it approaches geometrically but never
  reaches. Any pruning rule is therefore an approximation with an explicit tolerance.
- **Lazy materialisation.** For an edge that receives the same evidence type for n consecutive
  frames (e.g. no record, with w_e constant), x_{t+n} ∝ (diag(L)^{a} P(Δ)ᵀ)ⁿ x_t, which is
  computable by repeated squaring in O(S³ log n) or by eigendecomposition. It is equal to the
  eager recursion in exact arithmetic; in floating point both are within O(n·ε_mach) relative
  error after normalisation. M5 breaks constancy (u_k relaxes each frame), so lazy spans are
  valid only for M2–M4, or for M5 when u = 1 exactly. Expiry: for pure prior propagation in M2
  the marginal p is monotone (closed form of §3), so the drop time is exact. For general constant
  evidence p_n converges geometrically to a fixed point but need not be monotone, so an expiry
  heap would need a monotonicity check or conservative bounds. **Implementing the lazy/heap
  mechanism is Phase 5 and is not done in Phase 4; the heap is an implementation mechanism, not
  a novelty claim.**
