# TAORG relationship-episode estimator: formulation

> **Status:** FROZEN with the episode-study pre-registration (`research/taorg_episode_preregistration.md`).
> Written before any implementation (2026-10-11). Any later deviation is an amendment.
> Related work and novelty check: `research/taorg_episode_related_work.md`. Summary there: every
> mechanism below is standard, and combining them is **not** claimed as a method novelty.

## 0. Scope

This formulation post-processes a per-frame relationship stream (the repository's `FrameGraph`
protocol: `reset_video / step / finalize_video`). It is a small eager reference: every maintained
key is updated on every frame, with no heap or lazy evaluation.

Phase 4 conclusions stay frozen (METHOD CONTRIBUTION = NO; V-HPED relapse, L23). So do Stage 0
(WEAK-MIXED) and Phase 5H-1 (NARROW, exploratory, annotation-only). This study asks a different
question and reinterprets none of them.

## 1. Five quantities that must not be conflated

For a relationship key k = (s, o, r) (subject track, object track, predicate) at frame n with
timestamp t_n (seconds; Δ_n = t_n − t_{n−1}, and Δ_0 = 1/30 s):

| # | Quantity | Symbol | Type | Source at inference |
|---|---|---|---|---|
| 1 | **Relationship existence** | Z_n(k) ∈ {0, 1} | latent | inferred from detections |
| 2 | **Observability** of an absence | F_n(k) ∈ {VIS, OCC, UNK} | observed covariate | the absence record's visibility flag (edge visibility = conservative combination of subject/object flags; no record ⇒ UNK) |
| 3 | **Identity reliability** | w_n(k) = u_n(s)·u_n(o) ∈ [0, 1] | derived | track boxes and timestamps only (§5) |
| 4 | **Termination of the current episode** | event E_end(k, j) at frame n | decision | the within-episode filter falls below θ_end (§3) |
| 5 | **Birth of a subsequent episode** | event E_birth(k, j+1) at frame n | decision | a sequential test on detections after the previous termination (§4) |

- Existence is a **belief** about the world.
- Termination and birth are **decisions** of the estimator. They define the output episodes.
- Observability is treated as an exogenous flag, not as a latent variable. This is deliberate.
  Phase 4 modelled V as latent with confusable flags, and the exact filter then *explained away*
  absences and relapsed (L23).
- Identity reliability only modulates *how much* a record's evidence counts. It never creates or
  deletes evidence.

**Mode.** Each maintained key is either ACTIVE (inside episode j) or IDLE. The output at frame n
contains k if and only if k is ACTIVE after processing frame n.

## 2. Observation model (per frame, per key)

Records: DET (`observed=True`), or an absence with flag F, or no record (treated as UNK).

| Record | P(· ∣ Z = 1) | P(· ∣ Z = 0) | Likelihood ratio Λ |
|---|---|---|---|
| DET | 1 − m | f | (1 − m)/f |
| absence, F = VIS | θ_VIS = m | 1 − f | m/(1 − f) |
| absence, F ∈ {OCC, UNK} | θ_F = m + a_F (1 − f − m) | 1 − f | θ_F/(1 − f) |

- m is the miss probability while visible and f the false-detection probability, both per frame.
- a_F ∈ [0, 1] sets how uninformative an occluded or unknown absence is: a_F = 1 gives Λ = 1 (no
  evidence) and a_F = 0 makes it as informative as a visible absence.
- **Constraint:** θ_F ≤ 1 − f, so every absence has Λ ≤ 1. An absence can never raise the belief
  that the relation exists. (The existing VA-HMM baseline does not impose this: θ_occ ∈ [0.5, 1]
  can exceed 1 − f.) The flag is conditioned on, so Λ is a conditional likelihood given the flag.
- Detection scores are **not** used by the episode estimator or the HMM/HSMM baselines. GatedEMA
  uses them by construction. In the study world, false detections have the same score distribution
  as true ones (pre-registration §3), so scores carry no existence information and this choice is
  not a confound.

**Identity tempering** (rungs P2+, baseline B5 analogue): L̃_z = w·L_z + (1 − w), so
Λ̃ = (w L₁ + 1 − w)/(w L₀ + 1 − w). Λ̃ lies between 1 and Λ: identity-unreliable evidence is
shrunk toward "no evidence" and never flips sign.

## 3. Within-episode existence filter (termination)

Inside episode j of key k, the latent existence follows a **pure death process**: the relationship
ends at hazard μ_n (s⁻¹), and Z = 0 is **absorbing within the episode**. The episode cannot
"come back"; a recurrence is a new episode (§4).

    predict:  p⁻_n = p_{n−1} · exp(−μ_n Δ_n)
    update:   p_n  = p⁻_n Λ̃_n / (p⁻_n Λ̃_n + (1 − p⁻_n))          [equivalently with L̃₁, L̃₀]
    terminate when p_n < θ_end  →  the episode closes at frame n (k leaves the output at n)

Hazard (the ladder rungs differ only here and in §5):

    μ_n = μ · (1 + κ_c · c_n + κ_h · η_n)
    c_n = Σ_{r' ≠ r : (s,o,r') ACTIVE after frame n−1}  E[r][r']                      (P3+, simple transition context)
    η_n = Σ_{births b of (s,o,r'), r' ≠ r, at t_b ≤ t_{n−1}}  E[r][r'] · exp(−(t_n − t_b)/τ_h)   (P4, event history)

- P1 and P2 use κ_c = κ_h = 0.
- c_n uses only the estimator's own decisions on the same pair at the previous frame, so the
  result does not depend on the order in which keys are processed within a frame.
- η_n uses the estimator's own birth times, not ground-truth onsets.
- E (§6) is a fixed parameter fitted on calibration ground truth.

**Lemma 1 (no relapse without detections).** If frames n+1, …, n+L contain no detection of k,
then p is non-increasing over them.
*Proof:* exp(−μΔ) ≤ 1 and Λ̃ ≤ 1 for every absence (constraint, tempering preserves it), and
p ↦ pΛ/(pΛ + 1 − p) is non-decreasing in p and ≤ p when Λ ≤ 1. ∎

Consequences:
- The Phase 4 relapse mechanism (rising belief during flagged absences) cannot occur.
- A terminated episode stays terminated by construction: its state is discarded.
- **The only way a key re-enters the output is a new birth (§4), and births require detections.**
  A false reactivation therefore needs false detections to pass the birth test. The experiment
  measures how often that happens (pre-registration §6).

**Hold horizon.** After a long run of detections the predicted odds are capped at
≈ 1/(e^{μΔ} − 1) ≈ 1/(μΔ). This is because Z = 0 is absorbing and the prior predictive always
leaks mass. So after a detection the odds are about Λ_det/(μΔ). Under absences with ratio Λ the
episode terminates after the smallest L with

    Λ_det/(μΔ) · Λ^L · exp(−μ L Δ) < θ_end/(1 − θ_end).

- Under VIS absences (Λ = m/(1 − f) ≪ 1) this is a few frames.
- Under OCC absences with a_OCC → 1 it is ≈ (1/μ)·log(Λ_det/(μΔ) · (1 − θ_end)/θ_end) seconds.

This is the explicit persistence-versus-termination trade-off that hidden terminations exploit.
No estimator that sees only OCC absences can separate a hidden termination from an occlusion
before their durations diverge (Phase 4 formulation §3). The transition context (P3/P4) is the
only rung that adds evidence there.

**Numerics.** p is clipped to [1e−12, 1 − 1e−12]. The update is computed in probability space
with float64. Λ is never 0 or ∞ (m, f ∈ (0, 1)).

## 4. Birth of a (subsequent) episode

An IDLE key runs a **CUSUM** (Page 1954) on the per-frame log-likelihood ratio of the same
observation model:

    λ_n = log Λ̃_n,      C_n = max(0, C_{n−1} + λ_n)
    birth when C_n ≥ h_eff(t_n),   h_eff(t) = h + h_r · exp(−(t − t_close)/τ_r)

- t_close is the time the key's previous episode terminated.
- For a key that has never been ACTIVE, h_eff = h (no refractory term).
- On birth, episode j+1 starts with a **fresh** state: p⁻ := 0.5 (fixed), updated with frame n's
  record (a detection, since λ_n > 0 is needed to cross), and C is no longer used. Because
  Λ_det > 1, p_n ≥ 0.5, so with θ_end ≤ 0.5 the new episode is ACTIVE at frame n.
- On termination, C := 0.
- A key is first maintained (IDLE, C = 0) at its first detection. Before that it has no state,
  which is equivalent because absences only lower C.

**Genuine recurrence.** It is always possible: ⌈h_eff/λ_det⌉ consecutive detections cause a birth.
The **recurrence-detection delay** this costs is the price of flicker protection, and the
experiment measures it.

**How flicker is prevented from creating repeated false episodes:**
1. An isolated false detection adds λ_det. Each following visible absence subtracts
   |log(m/(1 − f))|, so C returns to 0 within a few frames unless h ≤ λ_det.
2. Near a boundary, where flicker clusters (the relation just ended, the detector chatters), the
   refractory term raises the threshold to h + h_r and lets it decay with time constant τ_r.
3. Within an episode, a short run of visible misses must push p below θ_end against the
   accumulated odds (hold horizon above). One or two misses do not terminate a well-established
   episode unless m is tuned to be very small.

**How incorrect termination is handled.** A false termination (the relation still holds) closes
episode j. When detections continue, the CUSUM births episode j+1 after ⌈h_eff/λ_det⌉ detections.
The output therefore has a gap, which is a fragmentation. Online processing cannot rewrite past
output, so episodes are **not** merged retroactively. The refractory term deliberately makes this
recovery slower right after a termination. This trade-off (fragmentation and recovery versus false
reactivation) is measured by the episode metrics; it is not assumed away.

## 5. Identity reliability (P2+; the identical module is used by baseline B5)

Per track i: centre c, box diagonal D, velocity v (px/s) and reliability u ∈ [0, 1]. New tracks
start with u = 1 and v = 0. At each frame (Δ = Δ_n):

    recovery:   u ← 1 − (1 − u)·exp(−r_id Δ)                         (all known tracks)
    residual:   d = ‖c − (c_prev + v Δ)‖ / D                         (tracks seen before)
    densities:  g₀(d) = exp(−d²/(2σ²)) / (2πσ²),  σ = σ_id·sqrt(max(1, Δ/Δ_ref))   [2-D Gaussian, per unit area]
                g₁(d) = 1/(π d_max²) for d ≤ d_max, 0 otherwise                       [uniform jump in a disc]
    switch:     h = 1 − exp(−H Δ);   s = h g₁ / (h g₁ + (1 − h) g₀)   (s := 1 if both densities vanish)
    reliability u ← u·(1 − s)
    velocity:   if s < 0.5: v ← v + β((c − c_prev)/Δ − v), else v unchanged;   c_prev ← c

- Fixed constants: H = 0.05 s⁻¹, d_max = 3 diagonals, β = 0.5, Δ_ref = 1/30 s.
- Tunable: σ_id and r_id.
- Edge reliability is w(k) = u(s)·u(o). A track that is not observed in a frame keeps its state.
- Ordinary smooth motion is predicted by v, so it produces small residuals. Swapped identities
  jump to another track's position. Fast or erratic motion can produce false alarms (condition
  C7 measures this).
- This is a deliberately simple cue: there is no re-association, appearance model or gating
  between tracks.

## 6. Transition context statistic E (fitted, never inferred from test data)

    E[r][r'] = n_succ(r → r') / (n_term(r) + 1)

- n_term(r) counts **calibration ground-truth** terminations of r (episode end before the last
  frame).
- n_succ(r → r') counts those where an episode of r' ≠ r on the same pair starts within
  [end − 10, end + 15] frames.
- It is computed once from the calibration seeds of the calibrated conditions, saved as an
  artifact, and passed to P3/P4 as a constructor parameter.
- Evaluation ground truth is never read (tested).

## 7. The proposed ladder (cumulative)

| Rung | Name | Adds | Tunable parameters |
|---|---|---|---|
| P1 | Episode estimator | §2–4 | m, f, a_OCC, a_UNK, μ, θ_end, h, h_r, τ_r |
| P2 | + identity reliability | §5 tempering | + σ_id, r_id |
| P3 | + simple transition context | κ_c c_n in μ_n | + κ_c |
| P4 | + event-history context | κ_h η_n in μ_n | + κ_h, τ_h |

**Common random numbers.** Rung P(k+1) configuration i uses the same base draws as rung Pk
configuration i, plus draws for its added parameters from a separate stream. Ladder differences
therefore reflect the mechanism, not the sampling luck.

## 8. Baselines (all through the same interface)

| Id | Method | Notes |
|---|---|---|
| B1 | GatedEMA | Phase 1 implementation, unchanged |
| B2 | GatedEMA + hysteresis | Same score; output turns on at s ≥ θ_on, off at s < θ_off = ρ·θ_on, ρ ∈ (0, 1); otherwise it keeps the previous output. Shares base draws with B1 |
| B3 | VA-HMM | Phase 4 M1, unchanged |
| B4 | Duration-aware HSMM | Explicit-duration two-state (ON/OFF) forward filter. Age bins of 1/30 s up to 6 s (the last bin absorbs older ages); Weibull dwell hazards (shape, scale) for ON and OFF; the same flag-conditioned emissions and parameter ranges as the VA-HMM (θ_vis, θ_occ, θ_unk, f); output iff P(ON) > threshold. The key is initialised at its first detection with P(ON, age 0) = prior and P(OFF, age ≥ 6 s) = 1 − prior. Irregular Δ advances ⌈Δ/Δ_ref⌉ bins, which is an approximation |
| B5 | GatedEMA + identity | The §5 module. On detection s ← s + wα(x − s); on absence s ← s·d^w (w = 1 recovers GatedEMA). Shares base draws with B1 |
| — | No Memory | Descriptive reference only |

The VA-HMM and HSMM can relapse: their p_birth or OFF hazard re-enters ON, and their emission
allows Λ_OCC > 1. B1, B2 and B5 relapse only through detections. That is exactly what the study
is designed to measure.

## 9. What is and is not an observable input

- **Allowed at inference:** the per-frame records (observed flag, flags), track boxes, timestamps,
  and fixed parameters (including E).
- **Never allowed:** simulator truth, lifetimes, corruption manifests or labels, evaluation ground
  truth, future frames.
- **Tests enforce:**
  - causality: altering frames after n does not change output at frames ≤ n;
  - the method constructors receive only parameters;
  - E is computed from calibration seeds only.

## 10. Complexity (eager reference)

- **Per frame:** O(K_n + T_n + K_n·R) for K_n maintained keys, T_n tracks and R predicates per
  pair (context sums), plus O(B_pair) for the event-history kernels (births per pair; small).
- **Memory:** O(K + T + births).
- **HSMM baseline:** O(K_n·A) with A = 180 age bins.
- Measured latency is reported. The heap and lazy versions are out of scope, as the user
  instructed.

## 11. Known limitations, declared before results

1. Visibility is per edge (the inherited Phase 3/4 simplification). This makes context
   evidence (P3/P4) available when the predecessor relation is hidden but its successor is
   visible. Any P3/P4 gain depends on this world assumption.
2. The identity cue is a constant-velocity residual on synthetic boxes, not a real tracker.
3. Lemma 1 rules out flag-driven relapse but not detection-driven false births. Performance
   against confident false detections is purely empirical.
4. Termination under occlusion is information-limited for every method (§3).
5. All evidence is synthetic.
