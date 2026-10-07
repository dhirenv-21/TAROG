# TAORG / V-HPED — Working Memory for Claude Code

> This file becomes the compact cross-session source of truth **after coding begins**. It is intentionally small and factual. Claude Code should update it after meaningful completed tasks. Do not paste full logs or duplicate the whole PRD here.

## Current status
- Stage: **Phase 4 complete — H2 = MIXED, H3 = MIXED, H4 = FAIL; METHOD CONTRIBUTION = NO** (synthetic only)
- Active phase: none — **awaiting user approval** (recommended pivot: evaluation/failure-analysis paper)
- Next phase: undecided (Phase 5 lazy/heap = ADSA deliverable; real-data validation of the protocol)
- Last updated: 2026-09-29
- Paper intent: **Yes — conference/workshop-oriented research paper is a required project outcome.**

## Git provenance
- Repository: local + remote (https://github.com/dhirenv-21/TAROG.git)
- Branch: `main`
- Latest commits: `d8728b3` Phase 3 pre-registration; `2ce7d1f` parameter freeze; `867b7be` evaluation artifacts; exit-report commit follows (see `git log`)
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
- [x] Phase 3 synthetic world, flicker / visibility-noise / UNKNOWN injectors, condition specs (src/taorg/experiments/phase3/, src/taorg/noise/).
- [x] Phase 3 pre-registration (`d8728b3`), calibration (1080 instances x 6213 configs), freeze (`2ce7d1f`), evaluation (156,240 runs, complete), analysis, failures, exit report.
- [x] Pilot completed — 182 tests passing.
- [x] Phase 4: pre-registration `bad0ef9`, VA-HMM (M1), V-HPED eager reference M2–M6 (`src/taorg/temporal/vhped.py`), baseline freeze `07b01c2`, V-HPED freeze `490b28c`, evaluation complete (478,620 runs), exit report `research/phase4_exit_report.md`. 215 tests passing.
- [x] V-HPED reference implemented (eager; Phase 4).
- [ ] Expiry heap implemented/proved.
- [ ] Base models integrated.
- [ ] Final experiments completed.
- [ ] Paper drafted.

## Latest verified experimental results
Phase 3 (controlled SYNTHETIC evidence; run `phase3_evaluate_20260929_055335_6ac64a`, commit `2ce7d1f`,
seeds 2000–2029; details `research/phase3_exit_report.md`, artifacts `results/phase3/`):
- H1 = PASS by the registered rule: 10/18 wins, 0 losses in C0, R1 (vis-noise 0.2), R2 (flicker 0.1); frontier 5/6, 4/6, 5/6.
- Caveats: C0 tautological (decay_visible = 0 selected); wins == gap >= budget cells; advantage gone at vis-noise 0.5; oracle-tuned GatedEMA worse than baselines on R1 and ~10x churn on R2 (tuning dependence).
- Failures: termination under occlusion (delay 27.4 vs ~5), all-UNKNOWN visibility (38.3), ID-switch ghosts (2.2-3.8x), hold horizon ~38 frames.
- Formal Pareto dominance: class B (condition-specific), not A.
- Evaluation re-run bit-identical (0/156,240 rows differ).

Phase 4 (SYNTHETIC; `results/phase4/`): H2 MIXED (6/21 wins: F3 g20, F5 ×3, F8 ×2), H3 MIXED (T1 only), H4 FAIL (2 wins / 12 losses).
- Only robust gain: ID-switch ghost suppression (F5 ghosts 220 vs 902 GatedEMA / 575 VA-HMM at equal F1/delay/FPR).
- M3+ filters RELAPSE after the first post-termination drop (F1 98–100%); frozen delay metric is blind to it; FPR 2–5× on F1/F7/F8.
- VA-HMM ≈ GatedEMA; matched tuning removes Phase 3 failures A/B; M4 not retained; continuous time hurts at 15 fps.

Phase 5H (Hawkes feasibility, Track B) — DESIGN ONLY, 2026-09-30: HAWKES TRACK = MODIFY. Events = relation onsets/offsets (not frames); data = VidOR/PVSG (Action Genome too sparse); maths (censored/missing/noisy/inhibitory/marked Hawkes) already exists; identity-weighted history = marked Hawkes. Stage 0 DONE on VidOR ground truth (annotations only): HAWKES SIGNAL = WEAK-MIXED (C1, C2 pass; C3 excess/self = 0.07 < 0.10; C4 fails vs heterogeneity null). ~54% of the multivariate gain reproduced by the circular-shift null. Report research/phase5h_stage0_report.md. Awaiting user decision; no visibility-aware Hawkes / detector work started.
Phase 5H-1 (2026-10-07): pre-registered (`07bf369`, EXPLORATORY: no untouched annotated VidOR data); point-process GLM engine with exact likelihood + tests (`4d810ce`); smoke on 300 TRAIN videos only. Full registered run NOT started (est. 30-60 CPU-h; needs parallel fitting). Supervisor showcase in results/phase5h1/showcase + research/phase5h1_supervisor_update.md. Branch phase5h-hawkes-feasibility is pushed (user authorised GitHub updates). See research/phase5h_hawkes_design.md, research/phase5h_hawkes_literature.md.

## Decisions made during implementation
Add short entries only, newest first:

2026-09-29 — Phase 4 ran baseline group first (calibrate→freeze→evaluate) before any V-HPED run; equal 200-sample random search per method. — research/phase4_preregistration.md
2026-09-29 — Obsolete pre-registration pilot files archived to archive/pre_registration_pilot/ (not deleted). — research/phase4_preregistration.md §12
2026-09-29 — `render_video` extracted from `generate_video` (Phase 3 outputs verified identical). — src/taorg/data/synthetic_world.py

2026-09-29 — Phase 3 amendment A2: stalled calibration run terminated (no output) and re-run with 8 workers; serial recomputation matched to 1.8e-15. — research/phase3_preregistration.md §11
2026-09-29 — Evaluation artifacts committed before interpretation; logs force-added (gitignored *.log) for audit trail without editing .gitignore. — results/phase3/

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
- GatedEMA treats edge keys with no record as UNKNOWN and decay_unknown defaults to decay_occluded → holds ID-switch ghosts and all-UNKNOWN streams (Phase 3 S2/S4). Untested alternative policy; Phase 4 ablation.
- Cohen's d_z is printed as ~1e15/inf when paired SD ≈ 0 (reporting issue; not in H1 rule).
- Fig. E panel titles overlap (cosmetic).
- Frozen termination-delay metric (first drop) is gameable by non-monotone estimators (relapse); any further method comparison needs a relapse-aware metric, pre-registered.
- V-HPED evidence-rate tempering over-weights per-frame misses at low fps; flag reliabilities are tuned, not estimated online.
- Calibration/ECE metric deferred (requires probabilistic method output).
- No external datasets. Not needed until Phase 3.

## Next action
WAIT for explicit user approval. Options in research/phase4_exit_report.md §26–§28 (pivot to evaluation/failure-analysis paper; relapse-aware Phase 4b; real-data protocol validation). Do not start Action Genome or Phase 5 without approval.

## Update protocol for Claude Code
After each meaningful task:
1. Update `Current status` / `Active phase` if changed.
2. Move finished items in `Completed work`.
3. Record important design decisions.
4. Record new blockers/open questions.
5. Add only verified experimental results, with run IDs/paths.
6. Set one concrete `Next action`.
7. Keep this file concise; detailed derivations belong under `research/derivations/`, and experiment details belong in run artifacts.

