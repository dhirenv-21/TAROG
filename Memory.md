# TAORG / V-HPED — Working Memory for Claude Code

> This file becomes the compact cross-session source of truth **after coding begins**. It is intentionally small and factual. Claude Code should update it after meaningful completed tasks. Do not paste full logs or duplicate the whole PRD here.

## Current status
- Stage: **All offline work for Phases 3-10 done (V-HPED, lazy heap, adapters, ablation matrix, paper draft). Blocked on data: AG pilot, base models, real tracker noise.**
- Active phase: **Phase 3 pilot (needs Action Genome annotations)**; Phases 4, 5 exit gates met on synthetic data
- Next phase: **Phase 6 base-model predictions through the frozen adapter contract**
- Last updated: 2026-10-28
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
Items 1–3 and 5 are RESOLVED for the discrete-time model in `research/derivations/vhped_model.md` (visibility-thinned Hawkes emission + alive/dead filter; alive score = posterior P(alive); frame-constant intensity makes the compensator exact; beta_v learned via softplus). Threshold domain = P(alive). Still open:
4. Expiry derivation for the lazy heap — posterior under visible absence is not a single exponential; needs a root-finder or the Corollary 2a bound.
- Regime of an interval taken from its right endpoint (untested alternative: left endpoint).
- Identifiability of eps vs lambda under unsupervised MLE.
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
- [x] V-HPED eager reference implemented (src/taorg/temporal/vhped/: params, dynamics, visibility, alive, model, likelihood, trainer) and registered as METHODS["vhped"].
- [x] V-HPED derivation + propositions (censoring, self-calibrating evidence, nesting = HMM, cross-predicate independence) — research/derivations/vhped_model.md.
- [x] V-HPED config with hand-set reference + ablation ladder — configs/temporal/vhped.yaml.
- [x] Phase 4 tests — tests/unit/test_vhped.py (35) + tests/integration/test_vhped_benchmark.py (3); NumPy inference == PyTorch likelihood to 1e-9; nesting equals HMM baseline to 1e-10; finite-difference gradient check.
- [x] Synthetic train/val/test benchmark — src/taorg/experiments/vhped_benchmark.py, scripts/run_vhped_synthetic.py.
- [x] Phase 3 pilot pipeline: AG loader (src/taorg/data/action_genome.py), schema inspector (scripts/inspect_action_genome.py), vectorised threshold sweep equal to frozen metrics (src/taorg/metrics/threshold_sweep.py), pilot runner with V-HPED fitting + ablations (src/taorg/experiments/pilot.py, scripts/run_pilot_ag.py), figures (src/taorg/experiments/pilot_figures.py), config (configs/experiments/pilot_action_genome.yaml, configs/datasets/action_genome.yaml).
- [x] Pilot smoke-tested end to end on a FAKE AG tree only (tests/fixtures/fake_action_genome.py); fake numbers are meaningless.
- [x] 192 total tests passing (2026-10-28, run with PYTHONPATH=src).
- [x] Phase 5: LazyVHPED + per-pair wake-up heap (src/taorg/temporal/vhped/lazy.py, expiry_heap.py), proofs + complexity (research/derivations/lazy_expiry.md), randomized eager-vs-lazy harness (tests/property/), scaling benchmark (outputs/runs/complexity/20261027T202322Z).
- [x] Alive filter predict step changed to exact two-state CTMC transition (identical results for rebirth_rate = 0).
- [x] Phase 6 offline: adapter contract frozen (src/taorg/adapters/base.py) + JSONL precomputed adapter (src/taorg/adapters/precomputed.py).
- [x] Calibration metric (ECE, Brier) frozen in metric_definitions.md §6 before first use (src/taorg/metrics/calibration.py).
- [x] Phase 8 synthetic ablation matrix incl. ID switches, 5 seeds, paired bootstrap (outputs/runs/20261027_202753_a7314fe5).
- [x] Phase 9 scripts: scripts/make_paper_artifacts.py -> paper/generated/ (tables, figures, numbers.tex, results_text.tex, provenance.json).
- [x] Phase 10: paper/main.tex + references.bib (not compiled: no LaTeX on this machine), research/reproducibility_audit.md.
- [x] 240 tests passing (2026-10-28).
- [ ] Base models integrated.
- [ ] Final experiments completed.
- [ ] Paper drafted.

## Latest verified experimental results
2026-10-28 SYNTHETIC ablation matrix `outputs/runs/20261027_202753_a7314fe5` and complexity benchmark `outputs/runs/complexity/20261027T202322Z`: see research/claims.md (incl. negative ID-switch result). Not real-data evidence.
2026-10-28 SYNTHETIC sanity benchmark only (`outputs/runs/vhped_synthetic/run_20261027T194003Z.json`, log in same folder): see research/claims.md 'Synthetic sanity signals'. Not paper evidence.
Previously: none. Synthetic evaluation pipeline produces test outputs only — not paper evidence. Do not invent results.

## Decisions made during implementation
Add short entries only, newest first:

2026-10-28 — AG loader format is ASSUMED from the public toolkit layout; must be verified with scripts/inspect_action_genome.py on the real files before interpreting the pilot. Person = track 0, object track = 1 + class index (one instance per class per frame; duplicates counted, first kept). frame_idx = position among annotated keyframes; fps = 30 (unverified). Boxes not loaded. — src/taorg/data/action_genome.py
2026-10-28 — Pilot sweeps thresholds post hoc (one threshold-0 run per method). Verified exact for no_memory, ema, gated_ema, hmm, vhped. FixedWindow is NOT sweepable (thresholds best window score, emits latest score) and is rejected by the pilot. — src/taorg/experiments/pilot.py
2026-10-28 — Pilot threshold selection: per (seed, method) on validation (10% of AG train, split_seed 0), max frame-level F1 averaged over gap lengths; test curves used only for trade-off reporting (hypervolume of false-persistence vs persistence-F1). — src/taorg/experiments/pilot.py
2026-10-28 — Likelihood batching pads videos with dt = 0 steps (exact no-op), so variable-length AG videos fit in one batch. — src/taorg/temporal/vhped/likelihood.py

2026-10-28 — V-HPED negative evidence: absence is NOT a Hawkes event; the Hawkes intensity is the detection rate of a live edge, thinned by visibility rho_v (rho_vis = 1). Visible-absence log Bayes factor = -(lambda - eps) dt. — research/derivations/vhped_model.md §4
2026-10-28 — V-HPED birth semantics match HMM baseline (prior, then predict+update) so alpha=gamma=0, rho=1 nests exactly into HMM. — src/taorg/temporal/vhped/model.py
2026-10-28 — Cross-predicate termination hazard gamma[q,p] added (beyond PRD's alpha). — src/taorg/temporal/vhped/dynamics.py
2026-10-28 — MLE is unsupervised (observation-stream likelihood, train videos only); threshold chosen on val by frame-level F1 (auxiliary, not a frozen metric). — src/taorg/temporal/vhped/trainer.py

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
- No Action Genome annotations on this machine; pilot cannot run until they are placed at data/action_genome/annotations (or --root).
- AG relation 'termination' per frozen metric = last GT observation before the video ends; this includes relations that end because the object becomes invisible. Not separated in the pilot (metric frozen); consider for interpretation.
- Calibration/ECE metric deferred (requires probabilistic method output).
- No external datasets. Not needed until Phase 3.

## Next action
0) Real data is the only blocker. Then:  Place AG annotations, run `python scripts/inspect_action_genome.py --root data/action_genome` and check the report against the loader assumptions. 2) Run `python scripts/run_pilot_ag.py --workers <n>` (full predeclared seeds). 3) Review hypervolume comparisons for the go/no-go gate. Previous note: Implement **Phase 3** from `Phases.md`: go/no-go pilot with synthetic corruption sweep across baselines. Produce preliminary results comparing all 5 baselines across gap lengths [1,2,3,5,10,20] with multiple seeds. These are pilot results for internal assessment, not final paper numbers.

## Update protocol for Claude Code
After each meaningful task:
1. Update `Current status` / `Active phase` if changed.
2. Move finished items in `Completed work`.
3. Record important design decisions.
4. Record new blockers/open questions.
5. Add only verified experimental results, with run IDs/paths.
6. Set one concrete `Next action`.
7. Keep this file concise; detailed derivations belong under `research/derivations/`, and experiment details belong in run artifacts.

