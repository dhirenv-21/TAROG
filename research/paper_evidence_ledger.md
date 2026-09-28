# Paper Evidence Ledger

> **Rule:** the manuscript may state a claim only if its status here is **SUPPORTED**, and only
> within the scope written in "Current Evidence". Allowed statuses: UNTESTED, PARTIAL,
> SUPPORTED, NOT SUPPORTED, INVALIDATED. Evidence must point to generated artifacts, never to
> hand-copied numbers.
>
> Companion files: `research/claims.md` (older Phase 0 ledger, kept for history),
> `research/phase3_preregistration.md`, `research/paper_results_notes.md`.

_Last updated: 2026-09-28 (pre-Phase-3-results state)._

| # | Claim | Hypothesis | Evidence Required | Experiment | Current Evidence | Status | Paper Section |
|---|---|---|---|---|---|---|---|
| L1 | Visibility-aware gating improves the persistence/termination trade-off over visibility-unaware temporal smoothing under controlled corruption | H1 | Registered PASS on Phase 3 synthetic sweep (oracle + degraded visibility + flicker), matched termination cost, frontier HV | Phase 3 pilot | — | UNTESTED | §7 |
| L2 | The gating advantage holds with an imperfect visibility signal | H1 (robustness) | R1 criterion + ρ sweep | Phase 3 R1, SWV | — | UNTESTED | §7 |
| L3 | The gating advantage holds under base-model flicker (visible misses) | H1 (robustness) | R2 criterion + q sweep | Phase 3 R2, SWF | — | UNTESTED | §7 |
| L4 | Visibility-unaware memory couples gap bridging to termination delay (bridging g frames ⇒ delay ≥ g+1) | P7 | Proof + empirical check in C0 | Phase 3 C0 + derivation | — | UNTESTED | §5 |
| L5 | Known failure modes of visibility gating (termination under occlusion, UNKNOWN visibility, ID switches) | — | Stress conditions + failure cases | Phase 3 S1–S5 | — | UNTESTED | §7, §9 |
| L6 | Model-agnostic post-processing improves real VidSGG outputs without retraining | C1 | Action Genome + ≥1 base model, same protocol | Phase 6–7 | — | UNTESTED | §7 |
| L7 | V-HPED improves beyond Vis-Gated EMA | H2/H3 | Phase 4 + Phase 8 ablations | Phase 4–8 | V-HPED not implemented | UNTESTED | §7, §8 |
| L8 | Lazy decay + expiry scheduling matches the eager reference | H4 / P3 | Randomized eager-vs-lazy harness | Phase 5 | — | UNTESTED | §5 |
| L9 | Lazy maintenance costs O((k_obs+k_vis+k_exp) log A) per frame | P5 | Derivation + scaling benchmark | Phase 5 | — | UNTESTED | §5 |
| L10 | The corruption protocol (manifests, persistence/termination metrics, matched-cost comparison) is reproducible | C3 | Determinism tests + rerun equality | Phase 2–3 | Phase 2 determinism tests (same seed ⇒ same manifest) pass | PARTIAL | §6 |
| L11 | The robustness protocol is novel for VidSGG | C3 | Literature check | Literature review | Not yet checked | UNTESTED | §3 |
| L12 | "First to maintain relationships through occlusion" | — | — | Literature review | 4DST / World Scene Graph Generation addresses persistence through disappearance | INVALIDATED | never |
| L13 | Compiled event-query matching as a research contribution | — | — | Literature review | Covered by SASE, Cayuga, TESLA/T-Rex | INVALIDATED | never (engineering only) |
| L14 | Temporal relationships / dynamic scene graphs are novel | — | — | Literature review | STTran, DSG-DETR, TEMPURA, OED, 4DST, … | INVALIDATED | never |
| L15 | Continuous-time formulation gives frame-rate invariance | Phase 4 | Frame-rate ablation | Phase 4/8 | Phase 1 methods are not frame-rate invariant | UNTESTED | §4, §8 |
