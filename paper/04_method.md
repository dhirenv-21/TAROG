# 4. Method — WORKING DRAFT (established components only)

> Status: defines notation and the **already implemented** Phase 1 components.
> **V-HPED is not implemented and not presented as a method here** (Phase 4 has not been
> approved). Everything below matches `src/taorg/temporal/` at the Phase 3 commit.

## 4.1 Setting and notation

Time is continuous and measured in **seconds**, $t \in \mathbb{R}_{\ge 0}$. A base model
emits observations at frame times $t_k$ (for a fixed frame rate $f$, $t_k = k\,\Delta$ with
$\Delta = 1/f$; irregular $t_k$ are allowed by the notation).

- **Scene graph.** $G_t = (V_t, E_t)$: $V_t$ tracked entities, $E_t$ output relationship edges.
- **Edge.** $e = (u, v, r)$ — subject track $u$, object track $v$, predicate $r$.
- **Observation evidence.** At $t_k$: indicator $o_e(t_k) \in \{0,1\}$ (the base model
  asserts $e$) and score $s_e(t_k) \in [0,1]$ (defined when $o_e(t_k)=1$).
- **Visibility.** Per track $\nu_u(t) \in \{\textsf{VIS}, \textsf{OCC}, \textsf{UNK}\}$
  (visible; occluded-or-unreliable; unknown), derived from tracker/detector signals by a
  configurable policy. Edge visibility (conservative rule):
  $\nu_e = \textsf{VIS}$ iff $\nu_u = \nu_v = \textsf{VIS}$; $\nu_e = \textsf{OCC}$ if either is
  $\textsf{OCC}$; otherwise $\textsf{UNK}$. `UNK` never silently becomes `VIS` (PRD FR-2).
- **Temporal state.** $x_e(t) \in [0,1]$ — an alive *score*. It is not a calibrated
  probability unless a calibration step is added (calibration/ECE deferred).
- **Last observation time.** $\tau_e(t) = \max\{t_k \le t : o_e(t_k) = 1\}$.
- **Output decision.** $a_e(t) \in \{0,1\}$ with activation threshold $\theta_{\text{on}}$ and
  termination threshold $\theta_{\text{off}} \le \theta_{\text{on}}$ (hysteresis):
  $a_e(t) = 1$ if $x_e(t) \ge \theta_{\text{on}}$; $a_e(t)=0$ if $x_e(t) < \theta_{\text{off}}$;
  otherwise $a_e(t) = a_e(t^-)$. **All Phase 1 methods use a single threshold**
  $\theta_{\text{on}} = \theta_{\text{off}} = \theta$ (no hysteresis).

Kept strictly separate (Rules.md §2.1): event intensity $\lambda$ (Hawkes, Phase 4 only),
pre-activation state $z$, alive score $x$, and the keep/drop threshold $\theta$.

## 4.2 Visibility-gated exponential dynamics (Vis-Gated EMA)

Between observations, with visibility regime $\nu$ constant on an interval, the state decays as

$$ \frac{dx_e}{dt} = -\lambda_{\nu}\, x_e \quad\Rightarrow\quad x_e(t) = x_e(t_0)\, e^{-\lambda_{\nu}(t-t_0)}, $$

with regime-specific rates $\lambda_{\textsf{VIS}}$ (visible absence: evidence of termination,
fast) and $\lambda_{\textsf{OCC}}$ (occluded absence: censored evidence, slow);
$\lambda_{\textsf{UNK}} = \lambda_{\textsf{OCC}}$ by the Phase 1 default. For piecewise-constant
regimes on intervals $I_1,\dots,I_m$ with no observation,
$x_e(t) = x_e(\tau_e)\exp\!\big(-\sum_j \lambda_{\nu_j}|I_j|\big)$.

**Discrete implementation (Phase 1).** At a fixed frame rate the implementation applies a
per-frame factor $d_\nu = e^{-\lambda_\nu \Delta}$ (so $\lambda_\nu = -f \ln d_\nu$; $d_\nu = 0$
means instantaneous reset) using the visibility reported at that frame, and on observation
applies $x \leftarrow \alpha\, s + (1-\alpha)\, x$. Two consequences must be stated in the paper:

1. The decay step equals the continuous model only if the regime is constant over each
   inter-frame interval.
2. The observation update is **frame-rate dependent**: a frame-rate-invariant analogue needs
   $\alpha(\Delta) = 1 - e^{-\kappa\Delta}$. The Phase 1 baselines are therefore not
   frame-rate invariant; Phase 3 results hold at 30 fps only.

## 4.3 Visibility-unaware reference methods

- **No Memory:** $a_e(t_k) = o_e(t_k)$.
- **Fixed Window:** $a_e(t_k) = 1$ iff $o_e(t_j)=1$ for some $t_j \in (t_k - W\Delta, t_k]$.
- **EMA:** as §4.2 with a single decay rate for every absence regardless of visibility.
- **HMM:** 2-state (alive/dead) forward filter over the binary observation sequence with
  hand-set transition/emission parameters; output $P(\text{alive}\mid o_{1:k}) \ge \theta$.
  Visibility is not used.

## 4.4 V-HPED

[TBD — Phase 4, pending approval.] Open decisions that must be resolved before any V-HPED
equation is written here (Memory.md): treatment of visible-absence negative evidence; mapping
from intensity to alive score; softplus-intensity likelihood integral; threshold domain and
expiry derivation; constraints on regime parameters.
