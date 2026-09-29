# Paper Evidence Ledger

> **Rule:** the manuscript may state a claim only if its status here is **SUPPORTED**, and only
> within the scope written in "Current Evidence". Allowed statuses: UNTESTED, PARTIAL,
> SUPPORTED, NOT SUPPORTED, INVALIDATED. Evidence must point to generated artifacts, never to
> hand-copied numbers.
>
> Companion files: `research/claims.md` (older Phase 0 ledger, kept for history),
> `research/phase3_preregistration.md`, `research/paper_results_notes.md`.

_Last updated: 2026-09-29 (after Phase 3 evaluation; run `phase3_evaluate_20260929_055335_6ac64a`,
frozen-parameter commit `2ce7d1f`, artifacts commit `867b7be`). All Phase 3 evidence is **controlled
synthetic evidence** (`R/` = `results/phase3/`)._

| # | Claim | Hypothesis | Evidence Required | Experiment | Current Evidence | Status | Paper Section |
|---|---|---|---|---|---|---|---|
| L1 | Visibility-aware gating improves the persistence/termination trade-off over visibility-unaware temporal smoothing under controlled corruption | H1 | Registered PASS on Phase 3 synthetic sweep (oracle + degraded visibility + flicker), matched termination cost, frontier HV | Phase 3 pilot | Registered rule → **PASS** (`R/aggregated/h1_decision.json`): 10/18 wins, 0 losses in C0, R1, R2; frontier 5/6, 4/6, 5/6 gaps. Scope: synthetic world, 30 fps, i.i.d. corruption, **condition-matched tuning**. Caveats: C0 is tautological (prereg §2); wins are exactly the gap ≥ budget cells; advantage vanishes at ρ = 0.5 and reverses under tuning mismatch (L16) | SUPPORTED (synthetic, matched tuning only) | §7 |
| L2 | The gating advantage holds with an imperfect visibility signal | H1 (robustness) | R1 criterion + ρ sweep | Phase 3 R1, SWV | R1 (ρ = 0.2) meets the robustness criterion (10 wins, 0 losses, frontier 4/6: the thinnest margin in the rule). Advantage shrinks monotonically with ρ and is gone at ρ = 0.5 (`R/tables/sweeps.md`). Significant small loss at gap 1, budget 2 (ΔF1 −0.032, Holm p 0.033). i.i.d. noise only; C0-tuned parameters are worse than the baselines on R1 | PARTIAL | §7 |
| L3 | The gating advantage holds under base-model flicker (visible misses) | H1 (robustness) | R2 criterion + q sweep | Phase 3 R2, SWF | R2 meets the criterion (10 wins, 0 losses, frontier 5/6); with matched tuning F1 ≥ 0.993 at all gaps for q ≤ 0.3 at equal or lower delay (`R/tables/sweeps.md`). Requires `decay_visible` > 0: C0-tuned parameters produce ≈ 10–12× churn under flicker (`R/tables/tuning_mismatch.md`) | SUPPORTED (synthetic, matched tuning only) | §7 |
| L4 | Visibility-unaware memory couples gap bridging to termination delay (bridging g frames ⇒ delay ≥ g+1) | P7 | Proof + empirical check in C0 | Phase 3 C0 + derivation | Empirically near-exact on evaluation frontiers (`R/aggregated/frontier_points.csv`): Fixed Window = g+1, EMA ≥ g+1, HMM mean slightly **below** g+1 (1.988 at g = 1, 5.993 at g = 5), so the statement as written is not exact on the mean for score-dependent methods. No proof yet. Explains why the H1 wins are exactly the gap ≥ budget cells | PARTIAL | §5 |
| L5 | Known failure modes of visibility gating (termination under occlusion, UNKNOWN visibility, ID switches) | — | Stress conditions + failure cases | Phase 3 S1–S5 | Characterised (`R/tables/stress.md`, `R/failures/`): S1 delay 27.4 vs ≈ 5 frames, false persistence 0.18 vs 0.03; S4-all delay 38.3, false persistence 0.26; S2 ghost edge-frames 2.2–3.8× the baselines; S5 gap 60 F1 0.75 (finite hold horizon) | SUPPORTED (as a failure characterisation, synthetic) | §7, §9 |
| L6 | Model-agnostic post-processing improves real VidSGG outputs without retraining | C1 | Action Genome + ≥1 base model, same protocol | Phase 6–7 | — | UNTESTED | §7 |
| L7 | V-HPED improves beyond Vis-Gated EMA | H2/H3 | Phase 4 + Phase 8 ablations | Phase 4–8 | V-HPED not implemented | UNTESTED | §7, §8 |
| L8 | Lazy decay + expiry scheduling matches the eager reference | H4 / P3 | Randomized eager-vs-lazy harness | Phase 5 | — | UNTESTED | §5 |
| L9 | Lazy maintenance costs O((k_obs+k_vis+k_exp) log A) per frame | P5 | Derivation + scaling benchmark | Phase 5 | — | UNTESTED | §5 |
| L10 | The corruption protocol (manifests, persistence/termination metrics, matched-cost comparison) is reproducible | C3 | Determinism tests + rerun equality | Phase 2–3 | Phase 2 determinism tests pass; calibration aggregates re-derived serially (max abs diff 1.8e-15); full evaluation re-run bit-identical except wall-clock runtime (see exit report §10); completeness 156240/156240 | SUPPORTED (synthetic pipeline) | §6 |
| L11 | The robustness protocol is novel for VidSGG | C3 | Literature check | Literature review | Not yet checked | UNTESTED | §3 |
| L12 | "First to maintain relationships through occlusion" | — | — | Literature review | 4DST / World Scene Graph Generation addresses persistence through disappearance | INVALIDATED | never |
| L13 | Compiled event-query matching as a research contribution | — | — | Literature review | Covered by SASE, Cayuga, TESLA/T-Rex | INVALIDATED | never (engineering only) |
| L14 | Temporal relationships / dynamic scene graphs are novel | — | — | Literature review | STTran, DSG-DETR, TEMPURA, OED, 4DST, … | INVALIDATED | never |
| L15 | Continuous-time formulation gives frame-rate invariance | Phase 4 | Frame-rate ablation | Phase 4/8 | Phase 1 methods are not frame-rate invariant | UNTESTED | §4, §8 |
| L16 | Visibility gating is robust to tuning mismatch (parameters tuned in one regime used in another) | — | Prereg §5.4 secondary analysis | Phase 3 R1/R2 mismatched | C0-tuned GatedEMA on R1: F1 0.703 / 0.511 at gaps 5 / 10 vs 0.889 / 0.571 (Fixed Window, HMM), i.e. worse; on R2: churn 0.15–0.18 vs 0.015 (`R/tables/tuning_mismatch.md`) | NOT SUPPORTED | §7, §9 |
| L17 | GatedEMA Pareto-dominates the visibility-unaware baselines (formal set dominance, `dominance.py`, exploratory) | — | Class A in the frontier dominance analysis | Phase 3 dominance | Overall class **B** (condition-specific): A in C0 only (tautological), B in R1 (no dominance at gaps 1–3) and R2 (gaps 1, 20); never dominated in reverse (`R/tables/dominance.md`) | NOT SUPPORTED (only "condition-specific") | §7 |
| L18 | Visibility gating is safe when termination coincides with occlusion, when visibility is UNKNOWN, or under ID switches | — | S1, S4, S2 | Phase 3 stress | Fails in all three (see L5) | NOT SUPPORTED | §9 |
| L19 | The advantage survives an uninformative visibility flag | — | ρ = 0.5 sweep point | Phase 3 SWV_0.5 | No advantage (F1 0.582 vs 0.571 at gap 10; 0.886 vs 0.889 at gap 5). Negative control: consistent with the gain coming from the visibility signal | NOT SUPPORTED (expected) | §7 |
| L20 | Visibility gating adds negligible per-frame cost over EMA | — | Latency, primary-budget points | Phase 3 latency | 0.0084 vs 0.0085 ms/frame mean (8 synthetic edges, one machine; `R/tables/latency.md`). Not representative of real graph sizes | PARTIAL | §7 |
| L21 | Visibility-aware memory is needed beyond a visibility-aware *probabilistic* baseline (e.g. visibility-conditioned HMM) | — | Comparison with a visibility-aware HMM | not run | No visibility-aware comparator other than GatedEMA exists yet | UNTESTED | §7 |

## Change log

- 2026-09-29 (Phase 3 results): L1 UNTESTED→SUPPORTED (scoped); L2 UNTESTED→PARTIAL; L3
  UNTESTED→SUPPORTED (scoped); L4 UNTESTED→PARTIAL; L5 UNTESTED→SUPPORTED (failure
  characterisation); L10 PARTIAL→SUPPORTED (synthetic pipeline); added L16–L21. L1 was not
  upgraded on p-values alone: the registered criteria (effect ≥ 0.05 F1, CI, Holm, non-inferiority
  margins on delay/FPR/churn, frontier HV) were met in all three H1 conditions with large effects
  and tight CIs. Its scope is limited because of the C0 tautology, the tuning dependence (L16) and
  the ρ = 0.5 null (L19).
