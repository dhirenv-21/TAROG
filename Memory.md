# TAORG / V-HPED — Working Memory for Claude Code

> This file becomes the compact cross-session source of truth **after coding begins**. It is intentionally small and factual. Claude Code should update it after meaningful completed tasks. Do not paste full logs or duplicate the whole PRD here.

## Current status
- Stage: **Phase 1 complete**
- Active phase: **Phase 1 — Canonical edge stream + reference temporal baselines** ✅
- Next phase: **Phase 2 — Noise protocol + custom metrics**
- Last updated: 2026-09-28
- Paper intent: **Yes — conference/workshop-oriented research paper is a required project outcome.**

## Project identity
- Framework: TAORG — Temporal Adaptive Object Relationship Graph
- Proposed algorithm: V-HPED — Visibility-aware Hawkes-Process Edge Dynamics
- Central question: distinguish occlusion/tracker-caused missing edges from genuine relationship termination.
- ADSA focus: lazy temporal-state maintenance + expiry min-heap + formal complexity/termination analysis.

## Non-negotiable research constraints
- Research correctness/reproducibility before demo polish.
- Visibility-gated EMA is a critical baseline.
- Full V-HPED must earn its claim through ablation; do not assume it beats gated EMA.
- Keep eager reference implementation for correctness testing before heap optimization.
- Every paper number must trace to run artifacts.
- Negative results must be retained.

## Current open decisions
1. Exact mathematical treatment of visible-but-absent negative evidence.
2. Exact alive-score/probability mapping.
3. Correct likelihood integral with softplus intensity.
4. Exact threshold domain and expiry derivation.
5. Constraints/learning strategy for beta_vis and beta_occ.
6. Dataset availability/annotation verification for ActionGenome4D.
7. Exact recent WSGG baseline to reproduce.

## Completed work
- [x] Project research scope defined.
- [x] PRD drafted.
- [x] Architecture drafted.
- [x] Coding/research rules drafted.
- [x] Phased plan drafted.
- [x] Visual/paper design guidance drafted.
- [x] Repository initialized (Phase 0).
- [x] pyproject.toml with setuptools, pytest, ruff.
- [x] Canonical schemas: TrackObservation, EdgeKey, PairKey, EdgeObservation, TemporalEdgeState, ExpiryEntry, FrameGraph.
- [x] Config loading (PyYAML) with merge and validation.
- [x] Seed utility (Python random + NumPy + PyTorch).
- [x] Git commit capture utility.
- [x] Structured logging setup.
- [x] Run metadata schema (RunMetadata).
- [x] Run artifact writer (RunArtifactWriter).
- [x] TemporalEdgeModel protocol (base interface).
- [x] Synthetic fixtures (occlusion + termination scenarios).
- [x] Example experiment config.
- [x] .gitignore.
- [x] README.md.
- [x] AGENTS.md.
- [x] research/metric_definitions.md stub.
- [x] research/claims.md evidence ledger.
- [x] research/experiment_registry.csv stub.
- [x] research/derivations/ and paper_notes/ directories.
- [x] Package skeleton: adapters, data, metrics, noise, tracking sub-packages.
- [x] Phase 0 — 36 tests passing (unit + integration).
- [x] NoMemory baseline (src/taorg/temporal/no_memory.py).
- [x] FixedWindow baseline (src/taorg/temporal/fixed_window.py).
- [x] EMA baseline (src/taorg/temporal/ema.py).
- [x] GatedEMA baseline (src/taorg/temporal/gated_ema.py) — CRITICAL research baseline.
- [x] HMM 2-state scaffold (src/taorg/temporal/hmm.py).
- [x] Temporal method METHODS registry (src/taorg/temporal/__init__.py).
- [x] Baseline configs (configs/temporal/baselines.yaml).
- [x] Comprehensive baseline test suite — 42 tests covering protocol compliance, canonical input, decay behavior, visibility gating, determinism, reset, thresholds, occlusion bridging, termination detection.
- [x] Phase 1 — 78 total tests passing (36 Phase 0 + 42 Phase 1).
- [ ] Metrics frozen.
- [ ] Pilot completed.
- [ ] V-HPED reference implemented.
- [ ] Expiry heap implemented/proved.
- [ ] Base models integrated.
- [ ] Final experiments completed.
- [ ] Paper drafted.

## Latest verified experimental results
None yet. Do not invent results.

## Decisions made during implementation
Add short entries only, newest first:

2026-09-28 — Gated EMA: decay_unknown defaults to decay_occluded (conservative). Warns if gating is effectively disabled. — src/taorg/temporal/gated_ema.py
2026-09-28 — GatedEMA uses EdgeObservation.edge_visibility for combined visibility (conservative: VISIBLE only if both subject+object VISIBLE). — src/taorg/temporal/gated_ema.py
2026-09-28 — HMM uses hand-set parameters (scaffold); Baum-Welch deferred to Phase 4+ if needed. — src/taorg/temporal/hmm.py
2026-09-28 — METHODS registry in temporal/__init__.py for config-driven method selection without method-specific branches. — src/taorg/temporal/__init__.py
2026-09-28 — All baselines use the same FrameGraph->FrameGraph interface (TemporalEdgeModel protocol). No method-specific preprocessing. — src/taorg/temporal/
2026-09-27 — Config library: PyYAML — Simpler than OmegaConf, fewer dependencies, sufficient for Phase 0. Revisitable later. — src/taorg/config.py
2026-09-27 — Python 3.10+ — Compatible with dataclass features (slots, frozen), type union syntax. Final pin deferred to base-model compatibility check. — pyproject.toml
2026-09-27 — Build backend: setuptools.build_meta — Standard, minimal. — pyproject.toml
2026-09-27 — EdgeKey uses NamedTuple (not dataclass) — Needed for hashability and zero-overhead dict key usage. — src/taorg/schemas/edges.py
2026-09-27 — Edge visibility derived conservatively — Edge is VISIBLE only if both subject and object are VISIBLE. — src/taorg/schemas/edges.py

## Known issues / blockers
- No git repo initialized yet. Not blocking but should be done before experiments.
- No external datasets. Not needed until Phase 3.

## Next action
Implement **Phase 2** from `Phases.md`: deterministic occlusion injection, deterministic ID-switch injection, saved corruption manifests, Edge Persistence precision/recall/F1, Relationship Churn Rate, Termination Delay, false-persistence measure, calibration metric scaffold, latency instrumentation. Freeze metric definitions in `research/metric_definitions.md` before pilot results.

## Update protocol for Claude Code
After each meaningful task:
1. Update `Current status` / `Active phase` if changed.
2. Move finished items in `Completed work`.
3. Record important design decisions.
4. Record new blockers/open questions.
5. Add only verified experimental results, with run IDs/paths.
6. Set one concrete `Next action`.
7. Keep this file concise; detailed derivations belong under `research/derivations/`, and experiment details belong in run artifacts.
