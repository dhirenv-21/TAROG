# 5. Theoretical Analysis — PLACEHOLDERS

> Nothing in this file is a theorem. Each item is a **PROPOSITION CANDIDATE**; items with a
> sketch still need a complete, checked proof with all assumptions stated. The heap / priority
> queue is standard and is not claimed as novel (C2 concerns its correct use for
> visibility-regime-dependent expiry, not the data structure).

Notation from §4. Absence dynamics: $x(t) = x(t_0)e^{-\lambda_\nu (t-t_0)}$ within a regime.

---

### P1 — Closed-form decay under piecewise-constant visibility
**PROPOSITION CANDIDATE — PROOF SKETCH, FULL PROOF REQUIRED.**
If no observation of $e$ occurs on $[t_0, t]$ and the regime is piecewise constant with
finitely many switches, then $x(t) = x(t_0)\exp(-\sum_j \lambda_{\nu_j}|I_j|)$.
*Sketch:* solve the linear ODE on each interval and chain by continuity of $x$.
*Assumptions to state:* regimes are known at switch times; no observation-triggered jump.

### P2 — Expiry-time computation
**PROPOSITION CANDIDATE — PROOF SKETCH, FULL PROOF REQUIRED.**
Continuous time, constant regime with $\lambda_\nu > 0$ and $x(t_0) \ge \theta > 0$: the edge
expires at $t^\star = t_0 + \lambda_\nu^{-1}\ln(x(t_0)/\theta)$; after a regime switch at
$t_s < t^\star$ the expiry is recomputed from $x(t_s)$ (strict monotone decay ⇒ unique
crossing). Discrete per-frame factor $0 < d < 1$, $x_0 \ge \theta$: the first frame with
$x_0 d^k < \theta$ is $k^\star = \lfloor \ln(\theta/x_0)/\ln d \rfloor + 1$ (for $d=0$,
$k^\star = 1$). Under oracle visibility after a genuine termination, $k^\star$ is exactly the
frozen Termination Delay of Vis-Gated EMA (drop at $t_{\text{end}} + k^\star$).
*Edge cases to prove/test:* $x_0 = \theta$ exactly; $\lambda_\nu = 0$ (no expiry);
floating-point near-ties.

### P3 — Lazy/eager decision equivalence
**PROPOSITION CANDIDATE — PROOF REQUIRED.**
Claim to be made precise: materialising $x$ only at events (observation, visibility switch,
scheduled expiry, query) via P1, and scheduling expiries via P2, yields the same keep/drop
decisions as eager per-frame updating, under exact arithmetic and a stated event ordering.
With floating point the claim must carry a tolerance: decisions may differ when
$|x - \theta| < \varepsilon$ (product of per-frame factors vs. a single exponential). The
randomized eager-vs-lazy harness (Phase 5) is the empirical check.

### P4 — Memory bound
**PROPOSITION CANDIDATE — PROOF REQUIRED.**
State is $O(A)$ for $A$ non-expired edges; a lazily invalidated heap can hold stale entries, so
the bound needs either an indexed heap or a compaction policy (e.g. rebuild when stale > live),
giving $O(A)$ amortised heap size. The exact policy is a Phase 5 decision.

### P5 — Per-update computational complexity
**PROPOSITION CANDIDATE — PROOF REQUIRED.**
Target statement: per frame $O((k_{\text{obs}} + k_{\text{vis}} + k_{\text{exp}})\log A)$ for
lazy maintenance vs. $O(A)$ eager, where $k_{\text{obs}}$, $k_{\text{vis}}$, $k_{\text{exp}}$ count
observed edges, visibility-regime changes affecting active edges, and expiries. The count must
include heap invalidations and stale pops, and must exclude candidate-pair generation
(Rules.md §9). Not yet derived.

### P6 — Hysteresis / minimum dwell time
**PROPOSITION CANDIDATE — PROOF SKETCH, FULL PROOF REQUIRED.**
With $\theta_{\text{off}} < \theta_{\text{on}}$ and absence decay rates bounded by
$\lambda_{\max}$, an edge activated at $x \ge \theta_{\text{on}}$ remains active for at least
$\lambda_{\max}^{-1}\ln(\theta_{\text{on}}/\theta_{\text{off}})$ seconds without observations,
which bounds output flips in a window $T$ by $2\lceil T / \text{dwell}\rceil$ + (number of
observation-driven activations). *Note:* no Phase 1 method has hysteresis; this is a
candidate design element for Phase 4, and its churn benefit is UNTESTED.

### P7 — Persistence–termination coupling of visibility-unaware memory
**PROPOSITION CANDIDATE — PROOF SKETCH, FULL PROOF REQUIRED.**
*Motivation:* explains the structure of the Phase 3 oracle-condition results analytically.
Let a method's absence update be identical for every absence frame regardless of visibility
(EMA, Fixed Window, the Phase 1 HMM), with a keep/drop rule monotone in the number of
consecutive absences. If the state at the start of a $g$-frame gap is no larger than the
state at a genuine termination, then *bridging the gap implies a termination delay of at
least $g+1$ frames.*
*Sketch (EMA):* bridging requires $x_g d^g \ge \theta$; with $x_T \ge x_g$,
$x_T d^g \ge \theta$, so the edge is still output $g$ frames after $t_{\text{end}}$; hence
delay $\ge g+1$. Fixed Window: bridging needs $W > g$ and delay $= W$.
*For visibility gating* the gap uses $d_{\textsf{OCC}}$ and termination uses $d_{\textsf{VIS}}$, so
the two are decoupled **exactly when the visibility signal is correct**; with per-frame
visibility errors the coupling re-enters (heuristically, the expected log-decay per gap frame
becomes $(1-\rho)\ln d_{\textsf{OCC}} + \rho\ln d_{\textsf{VIS}}$ — heuristic, not proven).
*Caveat:* the proposition is close to tautological; its value is explanatory (it predicts that
any advantage of gating under oracle visibility is structural), not a contribution in itself.
