# TAORG / V-HPED — Working Memory for Claude Code

> This file becomes the compact cross-session source of truth **after coding begins**. It is intentionally small and factual. Claude Code should update it after meaningful completed tasks. Do not paste full logs or duplicate the whole PRD here.

## Current status
- Stage: **Phase 2 complete**
- Active phase: **Phase 2 — Noise protocol + custom metrics** ✅
- Next phase: **Phase 3 — Go/no-go pilot**
- Last updated: 2026-09-28
- Paper intent: **Yes — conference/workshop-oriented research paper is a required project outcome.**

## Git provenance
- Repository: local + remote (https://github.com/dhirenv-21/TAROG.git)
- Branch: `main`
- Latest commit: `60060c5` (Phase 2 tests)
- Baseline commit: `770caaa` (Phase 1 complete)
- Identity: dhiren2125@gmail.com / dhirenv-21 (repo-local)

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
- [x] Run metadata schema (RunMetadata) with git_dirty and git_branch.
- [x] Run artifact writer (RunArtifactWriter).
- [x] TemporalEdgeModel protocol (base interface).
- [x] Synthetic fixtures (occlusion + termination scenarios).
- [x] Example experiment config.
- [x] .gitignore.
- [x] README.md.
- [x] AGENTS.md.
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
- [x] Comprehensive baseline test suite — 42 tests.
- [x] Phase 1 — 78 total tests passing.
- [x] Git repository initialized (local + remote).
- [x] Metric definitions FROZEN in research/metric_definitions.md (Persistence F1, Churn Rate, Termination Delay, False Persistence Rate, Latency).
- [x] Corruption manifest schema (src/taorg/noise/manifest.py) — JSON-serializable, load/save.
- [x] Deterministic occlusion injector (src/taorg/noise/occlusion.py) — configurable gap lengths, edge selection, manifest generation.
- [x] Deterministic ID-switch injector (src/taorg/noise/id_switch.py) — bounded duration, manifest with ID mappings.
- [x] Edge Persistence F1 metric (src/taorg/metrics/persistence.py) — evaluated over corrupted intervals only.
- [x] Relationship Churn Rate metric (src/taorg/metrics/churn.py) — consecutive state flips, normalized.
- [x] Termination Delay metric (src/taorg/metrics/termination.py) — with censoring for never-dropped edges.
- [x] False Persistence Rate metric (src/taorg/metrics/termination.py) — post-termination persistence fraction.
- [x] Latency instrumentation (src/taorg/metrics/latency.py) — mean/median/p95/p99.
- [x] End-to-end synthetic evaluation pipeline (src/taorg/experiments/synthetic_eval.py).
- [x] RunMetadata extended with git_dirty and git_branch fields.
- [x] get_git_branch utility (src/taorg/utils/git.py).
- [x] Corruption configs (configs/corruption/defaults.yaml).
- [x] Corruption tests — 18 tests (manifest, occlusion, ID-switch).
- [x] Metric tests — 20 tests (persistence F1, churn, termination delay, false persistence).
- [x] Integration tests — 5 tests (synthetic eval pipeline).
- [x] Phase 2 — 121 total tests passing (78 Phase 0+1 + 43 Phase 2).
- [ ] Pilot completed.
- [ ] V-HPED reference implemented.
- [ ] Expiry heap implemented/proved.
- [ ] Base models integrated.
- [ ] Final experiments completed.
- [ ] Paper drafted.

## Latest verified experimental results
None yet. Synthetic evaluation pipeline produces test outputs only — not paper evidence. Do not invent results.

## Decisions made during implementation
Add short entries only, newest first:

2026-09-28 — Metric definitions FROZEN: Edge Persistence F1 evaluated over corrupted intervals only (not full sequence). Churn excludes initial appearance/final disappearance. Termination delay uses censoring for never-dropped edges. — research/metric_definitions.md
2026-09-28 — Occlusion injection suppresses observations + sets visibility to OCCLUDED while keeping GT relationship conceptually alive. Deep-copies input. — src/taorg/noise/occlusion.py
2026-09-28 — ID-switch swaps track IDs between co-occurring tracks for bounded durations. Updates both TrackObservations and EdgeObservations. — src/taorg/noise/id_switch.py
2026-09-28 — Corruption manifest is JSON-serializable with version field for forward compatibility. — src/taorg/noise/manifest.py
2026-09-28 — RunMetadata extended with git_dirty (bool) and git_branch (str) for full provenance. — src/taorg/schemas/runs.py
2026-09-28 — Git repo initialized with local identity (dhiren2125@gmail.com). Remote: https://github.com/dhirenv-21/TAROG.git — .git/config
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
- Calibration/ECE metric deferred (requires probabilistic method output).
- No external datasets. Not needed until Phase 3.

## Next action
Implement **Phase 3** from `Phases.md`: go/no-go pilot with synthetic corruption sweep across baselines. Produce preliminary results comparing all 5 baselines across gap lengths [1,2,3,5,10,20] with multiple seeds. These are pilot results for internal assessment, not final paper numbers.

## Update protocol for Claude Code
After each meaningful task:
1. Update `Current status` / `Active phase` if changed.
2. Move finished items in `Completed work`.
3. Record important design decisions.
4. Record new blockers/open questions.
5. Add only verified experimental results, with run IDs/paths.
6. Set one concrete `Next action`.
7. Keep this file concise; detailed derivations belong under `research/derivations/`, and experiment details belong in run artifacts.

