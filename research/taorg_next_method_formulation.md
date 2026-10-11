# B2 + survival-calibrated non-visible hold (K1): formulation

> **Status:** FROZEN with `research/taorg_next_method_preregistration.md` (2026-10-11), before any
> registered comparison. Eager reference: `src/taorg/temporal/cure.py` (`B2SurvivalHold`).
> Motivation and selection: `research/taorg_next_method_failure_audit.md`.
>
> No novelty is claimed for:
> - Kaplan–Meier estimation;
> - mixture-cure models;
> - Bayes-optimal stopping with a monotone posterior;
> - online learning of detection profiles.
>
> The only candidate contribution is their use as a **label-free, online-recalibrated
> termination rule for relationship persistence**. The pre-registered experiment tests whether
> even that is worth anything beyond a tuned fixed coasting buffer.

## 1. Inputs available at inference time

Per frame n, with timestamp t_n in seconds:
- for each relationship key k, a record that is either a detection (`observed=True`, score x), an
  absence with a visibility flag F ∈ {VIS, OCC, UNK}, or no record (treated as UNK);
- fixed parameters, including a prior fitted on **unlabeled calibration input streams**.

The method does **not** use:
- ground truth, lifetimes, manifests or corruption labels;
- future frames;
- track boxes (there is no identity component).

## 2. State variables

| Variable | Meaning |
|---|---|
| s_k | B2's GatedEMA score (unchanged B2 update) |
| ON_k | The output state (B2's hysteresis state, overridden as in §4) |
| τ_k | Time of k's last detection |
| run_k = (F, t₀, counted, closed) | The current **non-visible absence run** of an ON key: its stratum F (flag of its first non-visible absence frame), start reference t₀ = τ_k, the bins already added to the risk set, and whether it reached the horizon |
| Y_F[l], d_F[l], l = 1..H | Per-video online risk-set counts: runs at risk at bin l, and re-detections at bin l |
| h⁰_F[l] | Prior per-bin re-detection hazard (fitted once on calibration input streams) |

## 3. Sampling model (explicit)

Run length is measured in **sampling bins** of Δ_ref = 1/30 s: l = round((t − τ_k)/Δ_ref). The
horizon is H = round(3 s / Δ_ref) = 90 bins.

The hazards h_F[l] are defined per bin of the *reference sampling grid*. With irregular
timestamps, a frame advances the run clock by the rounded number of elapsed bins, and all skipped
bins enter the risk set without an event.

**This is not frame-rate invariance.** The re-detection hazard per bin depends on the detector's
per-frame miss process, which changes with the sampling rate. At another frame rate, the prior
must be refitted on streams with that rate (or rebinned with an explicit model of per-frame
detection). This study runs at 30 fps only.

## 4. Update equations and decision

**(a) B2 score (unchanged):**
- detection: s ← αx + (1 − α)s;
- absence with flag F: s ← s·d_F (with d_VIS, d_OCC, d_UNK);
- GatedEMA prunes s < 1e−8.

**(b) Run bookkeeping** (independent of the output once a run has started, so that dropping an
edge does not censor its own learning signal):
- A run starts at the first frame on which an ON key has a non-visible absence (F ∈ {OCC, UNK}).
  It enters the risk set for bins 1..l₀.
- On each later frame without a detection, bins counted+1 .. min(l, H) enter the risk set.
- At the next detection, at bin L ≤ H, an event is recorded: d_F[L] += 1, Y_F[L] += 1.
- A run that reaches l ≥ H is closed and treated as "no return within H".

**(c) Hazard and survival, shrunk to the prior** (κ = prior strength in pseudo-runs):

    ĥ_F(l) = (d_F[l] + κ·h⁰_F[l]) / (Y_F[l] + κ),      Ŝ_F(l) = Π_{j ≤ l} (1 − ĥ_F(j)),   Ŝ_F(0) = 1

**(d) Mixture-cure posterior** that the run is a temporary gap (re-detected within H), given no
re-detection in its first l bins:

    q_F(l) = 1 − Ŝ_F(H) / Ŝ_F(l)        (l < H);     q_F(l) = 0  for l ≥ H

This follows from the cure model S(l) = (1 − π) + π·S_G(l) with S_G(H) = 0: then
1 − π = S(H) and P(gap | no return through l) = π·S_G(l)/S(l) = 1 − S(H)/S(l).

**(e) Decision for key k at frame n:**
- If k is ON and the record is a non-visible absence: **ON iff q_F(l) ≥ τ**.
- Otherwise: B2's own rule.
  - Detection or visible absence: OFF if s < θ_off = ρ·θ_on; ON from OFF if s ≥ θ_on.
  - A pruned score means OFF.

## 5. Variants (same interface; used as ablations and comparators)

| Mode | Hold rule on non-visible absences | Role |
|---|---|---|
| `km` (**K1**) | q_F(l) ≥ τ with the online-shrunk KM hazards | proposed |
| `static` | q_F(l) ≥ τ with the prior hazards only | ablation: online learning |
| `pooled` | `km` with OCC and UNK in one stratum | ablation: stratification |
| `geometric` (GEO) | q = πr^l/(πr^l + 1 − π), r = 1 − p_F; p_F = (Σd + κp⁰)/(ΣY + κ), π_F = (Σd + κπ⁰)/(Σd + n_cens + κ) | closest established *adaptive* learner (per-frame detection-profile learning; geometric return times) |
| `buffer` (BUF) | l ≤ N_F, with N_OCC and N_UNK tuned | closest established *rule* (per-flag coasting buffer, as in tracker `max_age` / `track_buffer`) |

## 6. Handling the hard cases

**Missing observations.** No record is treated as UNK, which is the same stratum as an UNKNOWN
flag.

**Uncertain visibility.** OCC and UNK runs are separate strata, each learned online. A visible
absence keeps B2's rule, because a visible absence is direct termination evidence.

**Identity changes.** These are not modelled. After an identity swap, a ghost key has no records,
so its run is UNK. The hold rule then applies to it, and ghost output may **lengthen**. C6 is
therefore a registered harm check (ghost time).

**Termination:**
- after a visible absence, B2's rule applies (fast);
- under OCC/UNK flags, termination happens when the hold expires;
- a hidden termination (C2) is still information-limited.

**Recurrence.** Births go through B2's on-threshold, unchanged. A recurrence whose dead gap is
OCC- or UNK-flagged can be **bridged** (merged) if the hold outlasts the gap.

**Counterexample to the self-supervision (development iteration 1).** Re-detection does not imply
persistence: a visible termination followed by recurrence produces "returns". This is why
visible runs are excluded from learning.

## 7. Parameter estimation and calibration

**Prior h⁰_F, p⁰_F, π⁰_F.**
- Source: the calibration seeds' **input streams** only (episode-study C1–C7 × seeds 5000–5019),
  with no labels.
- Population: every run starting at the first non-visible absence after a detection of any key.
  This population is independent of any method's output.
- Saved as an artifact.

**τ, κ (K1, static, pooled, GEO), N_OCC, N_UNK (BUF), and B2's base parameters:**
- 200 random samples per method, with common random numbers;
- selected by the maximum pooled calibration J over C1–C7;
- ties broken by lower churn, then lower index.

**Horizon.** H = 3 s, fixed (not tuned).

## 8. Complexity

Per frame:
- **Time:** O(K_n) for B2 and bookkeeping, where K_n is the number of maintained keys. Risk-set
  updates are amortised O(1) per bin per run (each bin of each run enters once). Each hold query
  costs O(H) to evaluate q_F(l), cached per (stratum, l) per frame. Total O(K_n + min(K_n, Q_n)·H),
  where Q_n is the number of distinct (stratum, l) queries.
- **Memory:** O(K + |strata|·H).

A cumulative-product cache would make each query O(1); it is not needed at this scale.

## 9. Mathematical properties (what is and is not claimed)

**P1 (structural; trivial).**
- *Claim:* q_F(l) is non-increasing in l within a run, for fixed hazards, because Ŝ is
  non-increasing.
- *Consequence:* within a non-visible run, an ON key is dropped at most once and stays OFF until
  a detection.
- This is a structural fact, not a statistical guarantee.

**P2 (equivalence; important for the falsification design).**
- *Claim:* with fixed hazards (`static`), the rule equals a per-stratum maximum hold length
  L*_F = max{l < H : q_F(l) ≥ τ}, i.e. a per-flag coasting buffer.
- *Consequence:* the static variant has exactly the expressive power of BUF. Its only difference
  is that L*_F is derived from unlabeled calibration streams rather than tuned on labeled
  calibration data.
- *Therefore any advantage of K1 over BUF must come from online adaptation.*

**P3 (the property investigated): Bayes optimality of the posterior-threshold hold under the
mixture-cure model.**

*Setting.* Consider one non-visible run. It is a gap (re-detected at a random bin T ≤ H, with
survival S_G) with probability π, otherwise terminated. Per-bin losses:
- c_fp for each output bin while terminated;
- c_fn for each non-output bin while the gap is still open (bins before re-detection).

Policies observe only "no detection yet", so every non-anticipating policy is a hold length L.

*Expected cost of hold length L:*

    C(L) = c_fp·(1 − π)·L + c_fn·π·Σ_{l=L+1}^{H} S_G(l)

    C(L) − C(L−1) = c_fp·(1 − π) − c_fn·π·S_G(L)

S_G is non-increasing, so the increments are non-decreasing and C is discretely convex. Its
minimiser is

    L* = max{l : c_fn·π·S_G(l) ≥ c_fp·(1 − π)} = max{l : q(l) ≥ τ},   τ = c_fp/(c_fp + c_fn)

because q(l) = π·S_G(l)/(π·S_G(l) + 1 − π).

*Hence:* with known hazards, holding while q ≥ τ minimises the expected per-run frame cost among
all policies that can only stop holding, and dropping irreversibly costs nothing within the run.

*Assumptions:*
- runs are independent and identically distributed within a stratum;
- strata are correct;
- terminated runs are never re-detected within H, and gaps always are (S_G(H) = 0);
- the loss is per bin and additive;
- the cost of re-birth after a premature drop is ignored;
- the episode-level IoU effects inside J are ignored.

*Counterexamples where the property says nothing:*
- a recurrence after a short termination (a "return" that is not persistence);
- gaps longer than H (labelled terminated);
- false detections ending a terminated run;
- flag noise mixing strata (S1);
- regimes that change within a video;
- J's episode term penalising fragmentation non-additively.

*Plug-in.* With the estimated hazards, the rule is the plug-in Bayes rule. Its consistency
follows from the consistency of Kaplan–Meier hazard estimates under independent censoring (a
standard result; not claimed). Per-video sample sizes are small (tens of runs; audit O4), which is
why the hazards are shrunk to the prior.

*Tested numerically:* `tests/unit/test_cure.py` checks
- by brute force, on known distributions, that argmin_L C(L) equals the threshold rule;
- P1;
- P2.

**Not claimed:** frame-rate invariance; robustness to non-stationarity; any guarantee on J.

## 10. Limitations declared before results

1. Synthetic world (the episode-study world); per-edge visibility.
2. The adaptation is per video (statistics reset at each video), so learning is limited to tens of
   runs.
3. The prior is fitted on the calibration streams, which share the world with evaluation
   (different seeds).
4. The development pilot suggests most of the gain is achievable by a static rule, equivalently a
   tuned buffer.
