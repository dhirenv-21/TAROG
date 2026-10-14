# TAORG / V-HPED — Working Memory for Claude Code

> This file becomes the compact cross-session source of truth **after coding begins**. It is intentionally small and factual. Claude Code should update it after meaningful completed tasks. Do not paste full logs or duplicate the whole PRD here.

## Current status
- Stage: **Project completion pass done** (branch `codex/taorg-completion`): Phase 5 lazy maintenance COMPLETE; EGTR integrated and source-verified; runnable pipeline; final predictor-level evaluation **STOP at its development gate**; paper draft. Earlier: downstream GO (narrow), appearance NARROW, observation NOT USABLE
- Earlier: next-method K1 STOP / UNSUPPORTED; episode study STOP; Phase 5H-1 HAWKES TRACK = NARROW (exploratory, annotation-only); Phase 4 METHOD CONTRIBUTION = NO (synthetic); Stage 0 WEAK-MIXED
- Active phase: none — **awaiting user review** of the completion PR. No supported novel temporal estimator.
- Last updated: 2026-10-14
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
- [x] Relationship-episode study (pre-registered ladder P1–P4 vs 5 baselines; decision STOP).
- [x] Next-method study (failure audit of B2; K1 survival hold; decision STOP, contribution UNSUPPORTED).
- [x] Observation-interface audit on real MOT17 tracker outputs (distinction NOT USABLE; contract added).
- [x] Appearance-based tracking audit (BoT-SORT appearance off/on + OSNet; MOT17 held-back; decision NARROW).
- [x] Downstream relationship study (VidOR, oracle predictor, frozen flag; registered GO; no algorithm contribution).
- [x] Expiry heap implemented/proved (certified lazy maintenance; research/derivations/lazy_maintenance.md).
- [x] Base model integrated: EGTR (VG), adapter bit-identical to the official path (research/taorg_egtr_integration.md).
- [x] Final experiments: registered final evaluation run to its development gate -> STOP (research/taorg_final_exit_report.md).
- [x] Paper drafted (paper/taorg_paper_draft.md; working draft, [VERIFY] references remain).

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
Phase 5H-1 registered run (2026-10-08; freeze `289319e`; `results/phase5h1/full/`; report `research/phase5h1_exit_report.md`): **NARROW**.
- Δ(G4x,G2) +0.145 [0.135, 0.155], Δ(G4,G3) +0.083 [0.076, 0.090] nats/event on 84,715 EVAL events; null max 0.031 / 0.017 (38 nulls); K3 pass.
- K4 FAIL: 0 discoveries (min FDR 0.16); registered fold t-statistic is ridge-shrinkage-inflated in sparse cells (post-hoc).
- Families passing: SPATIAL, MOTION (contact fails practical size). Gain = ONSET→OFFSET at lags ≤ 2 s; signed > positive-only (−0.031).
- 17.0% of events tie exactly with another predicate's boundary (annotation convention risk). N-circ +16% events, N-jit −9.3% to −9.8%.
- Method contribution NOT justified. Heavy run state: D:/TAORG_runs/phase5h1/main (outside git).

Project completion (2026-10-14; branch `codex/taorg-completion`; commits ed105f5..):
- Phase 5 COMPLETE: LazyVHPED (Birkhoff/Hilbert certificate, exact materialisation, versioned deadline heap; M5 eager fallback). 28/28 synthetic runs 0 mismatches (max |dscore| 5.6e-16); key updates ~4x fewer; wall 1.6-3.1x typical, 0.79x mostly-live keys, 0.47x frequent interval jitter. results/phase5/.
- Phase 6: EGTR (Apache-2.0; src 7f87450f; ckpt sha256 8926640f) in D:/TAORG_runs/egtr/venv (transformers 4.18; separate from BoxMOT env). Adapter bit-identical to the official path on 200 VG test images; subset R@20/50/100 21.6/28.1/32.7 vs authors 23.5/30.2/34.4. results/egtr/.
- Final evaluation (prereg ed31905): development gate FAIL (registered tau grid >= 0.005 vs EGTR person-person scores ~1e-6..1e-3; NM F1 0.0001) -> Q1 (flag at predictor level) and Q2 (V-HPED vs B2) STOP, final 40 videos never fetched. Exploratory dev diagnosis: 76% tracks matched to EGTR persons; AUC 0.53-0.69; NM F1 0.15 at 90th-pct tau. E1 lazy=eager exact on 30 real streams.
- Runnable: python -m taorg.pipeline run --video V --work W (two stage venvs); smoke run 241-frame clip 3m45s (results/pipeline_smoke/). B2/VH need [0,1]-scaled scores (raw EGTR scores never reach thresholds).

Downstream study (2026-10-13; prereg `a326622`, impl `bf7749d`, freeze `0942146`; `results/downstream/`; report `research/taorg_downstream_exit_report.md`): **registered GO, narrow**.
- VidOR person–person relations; 30 dev (training) + 60 eval (validation) videos, range-fetched (annotations previously used in Phase 5H: held back, not untouched); torchvision FRCNN v2 COCO persons + V3 BoT-SORT + frozen V3|T+A flag (sha256 98d7e46a…); oracle relation predictor on tracker outputs.
- FB W=5 / B2 (α 0.5, dv 0.3, du 0.8, θ_on 0.5) selected on unflagged dev J — both at low-hold grid edges.
- Eval (240/240 runs): ghost reduction FB 0.72 [97.5% 0.63, 0.80], B2 0.76 [0.67, 0.83]; ΔPGG −0.014/−0.013; ΔSTD-g, ΔRDD ≈ 0 → both contrasts pass.
- Narrow reading: baseline PGG ≈ 0.05 (absolute margin lenient; relative loss 25–27%); flag fires on 32% of genuine-gap frames on VidOR (MOT17 7–12%); 69% of scored terminations / 97% of recurrences are matching flicker (post-hoc); flag helps FB more than B2 (information source, not TAORG inference).

Appearance audit (2026-10-12; prereg `ac34d2f`, impl `e7ae883`, freeze `85a839c`, A3 fix `1ff4387`; `results/appearance/`; report `research/taorg_appearance_audit_exit_report.md`): **NARROW**.
- BoxMOT 25.0.0 BotSort (V2 appearance off / V3 on, identical settings, ECC CMC) + OSNet x1.0 MSMT17 (sha256 b7d73dc6…; not MOT17-trained); ByteTrack V1; MOT17Det frames fetched by HTTP range (dev 357 MB, held-back 568 MB after freeze).
- Held-back V3 TAKEOVER vs GAP: AUC T 0.670 → T+A 0.864 (FRCNN, +0.194 [97.5% 0.151, 0.240]); 0.654 → 0.766 (SDP, +0.112 [0.045, 0.183]); 8/8 units positive. At frozen θ: recall 0.56 / 0.47, gap rejection 0.07 / 0.12 (SDP fails recall ≥ 0.50 → NARROW).
- Available during the gap: coverage 0.95 / 0.97; incident A AUC 0.77 / 0.76 at τ = 0.2 s (retrospective 0.81).
- Mostly diagnosis, not tracking: same cue on V2 records ΔAUC +0.22 / +0.15; V3 vs V2 IDSW 363→357 / 504→428. Exploratory: proximity-only cue explains little (T+G 0.70 vs T+A 0.86/0.77).
- trackers 2.6.1 BoTSORTTracker has NO re-ID (correction to observation report §9). Not an algorithm contribution (standard re-ID distance).

Observation audit (2026-10-12; commits `bbf545e` (criteria before outputs), `550f9f3` (adapter + dev results); `results/observation/`; report `research/taorg_observation_exit_report.md`): **NOT USABLE**.
- Real outputs: MOT17 public FRCNN/SDP detections + ByteTrack (trackers 2.6.1, defaults); dev seqs 02/09/11; eval seqs 04/05/10/13 untouched; data MOT17Labels.zip (10.1 MB, CC BY-NC-SA 3.0) at D:/TAORG_runs/observation.
- Conflation real: absence frames = real gaps 41%, identity takeovers 20%, gone 26%, false tracks 13%.
- Identity-suspect recall 0.1-1%; 26-29% of real gaps labelled drop-type (drifting Kalman predictions); only elapsed time separates (AUC 0.62-0.69) = fixed-buffer information.
- New contract (src/taorg/schemas/observation.py): interface fix, legacy unchanged; not an algorithm contribution.

Next-method study (2026-10-11; prereg `fe084a6`, impl `ed9a021`, freeze `8a7c916`, eval `320bc2e`; `results/next_method/`; report `research/taorg_next_method_exit_report.md`): **STOP / UNSUPPORTED**.
- Audit (dev seeds 7000-7019): B2's main failure = non-visible-absence hold mismatch (matched headroom C1 0.134, S2 0.150 J).
- K1 (B2 + online KM mixture-cure hold): C1 +0.129 J vs B2 but S2 -0.156, C3 -0.021, C2/S0 FPT +3-4 s; equal to its static prior (kappa 838 selected); the fixed buffer BUF matches or beats it (S2 0.864 vs 0.568); latency 3.71x B2.
- The 'no record = UNKNOWN' convention conflates swap ghosts and unknown-visibility gaps (ledger L55).
- Relocation: repository copied to D:\ADSA project; editable install re-pointed; checkpoints verified; about 1.6 GB RAM free, so <= 4 workers.

Relationship-episode study (2026-10-11; prereg `42b61ea`, impl `e7e7804`, freeze `d027634`, eval `c312fb9`; `results/episodes/`; report `research/taorg_episode_exit_report.md`): **STOP**.
- 11,000/11,000 evaluation rows complete (11 conditions × 50 seeds × 20 variants); calibration 140 instances × 1801 configs.
- T = P1 episode estimator (identity rung not retained: C6 ΔJ −0.037); context retained vs P2 only; history not retained (pooled ΔJ +0.00004).
- B2 GatedEMA+hysteresis never beaten in C2/C3/C5; beats P1 in C2 (STD 0.665 vs 0.922 s) and C5 (J 0.886 vs 0.823, FRR 0.142 vs 0.283).
- P1 highest mean J in 6/7 calibrated conditions but fails churn guard vs VA-HMM/HSMM (0.014 vs 0.007); no recurrence delay; worse ghosts (8.23 vs ~2.3 s/video; post-hoc: ghosts linger after swaps).
- Latency: P1 0.016 ms/frame (2.1× GatedEMA), HSMM 0.112 (14.5×).

## Decisions made during implementation
Add short entries only, newest first:

2026-10-14 — Completion: EGTR checkpoint (456 MB), VG annotations (19.6 MB) and 200 VG test images (range fetch) downloaded with user approval; EGTR venv D:/TAORG_runs/egtr/venv; LazyVHPED added (eager VHPED untouched); final evaluation sample 'final' = 40 fresh validation videos (never processed). — research/taorg_project_completion.md
2026-10-13 — Downstream study: VidOR videos (≈367 MB transferred) and torchvision FRCNN v2 COCO weights (175 MB) downloaded with user approval; heavy state D:/TAORG_runs/downstream; deterministic gzip (mtime 0) + content hashes adopted. Appearance-audit intervals are track-cluster percentile bootstrap, 97.5% per detector (Bonferroni), not plain 95% CIs (its exit report §11). — research/taorg_downstream_preregistration.md
2026-10-12 — Appearance audit: BoxMOT installed in isolated venv D:/TAORG_runs/appearance/venv (AGPL-3.0, --no-deps; C: full); OSNet x1.0 MSMT17 weights and MOT17Det frames (range-fetched) downloaded with user authorisation; heavy state on D:. Amendments A1 (strict conf > 0.5), A2 (smooth_feat creation seed; cue unchanged), A3 (post-freeze memory fix, outputs verified identical). — research/taorg_appearance_audit_preregistration.md §13
2026-10-12 — Installed trackers 2.6.1 (+supervision 0.30.9, av, pydeprecate) and downloaded MOT17Labels.zip with user approval; personal local videos deliberately not used. — research/taorg_observation_research_log.md
2026-10-11 — Workspace moved (copied) to D:\ADSA project; `pip install -e . --no-deps` re-pointed the package; the old C: copy still exists (the user will delete it). — research/taorg_next_method_failure_audit.md §0
2026-10-11 — Episode study: one pooled operating point per method (max calibration J over C1–C7); heavy per-instance checkpoints on D:/TAORG_runs/episodes with committed SHA-256 manifests; ≤ 6 workers (C: nearly full, pagefile pressure). — research/taorg_episode_preregistration.md, results/episodes/README.md
2026-10-08 — Phase 5H-1 amendments A1–A4 (aggregation/runner; N-jit pins censored endpoints; analysis details; max_iter 5000), all before freeze `289319e`. — research/phase5h1_preregistration.md §11
2026-10-08 — Heavy run state on D: (C: nearly full); compact artifacts in results/phase5h1/full/. — research/phase5h1_exit_report.md §16
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
- EGTR triplet scores are ~1e-6..1e-3 (ranking scores): threshold-based temporal methods (B2, V-HPED) need rank-normalised [0,1] scores; the registered final tau grid was misspecified (pre-registration error).
- LazyVHPED is slower than eager under frequent frame-interval changes (each change thaws all frozen keys) and with mostly-live keys.
- C: drive is full: the Windows commit limit binds (OOM at ~4 GB free RAM). Keep one tracker run in memory; avoid parallel jobs. Recorded *.csv.gz SHA-256 values include gzip timestamps (not content hashes).
- GatedEMA treats edge keys with no record as UNKNOWN and decay_unknown defaults to decay_occluded → holds ID-switch ghosts and all-UNKNOWN streams (Phase 3 S2/S4). Untested alternative policy; Phase 4 ablation.
- Cohen's d_z is printed as ~1e15/inf when paired SD ≈ 0 (reporting issue; not in H1 rule).
- Fig. E panel titles overlap (cosmetic).
- Frozen termination-delay metric (first drop) is gameable by non-monotone estimators (relapse); any further method comparison needs a relapse-aware metric, pre-registered.
- V-HPED evidence-rate tempering over-weights per-frame misses at low fps; flag reliabilities are tuned, not estimated online.
- Calibration/ECE metric deferred (requires probabilistic method output).
- Phase 5H-1 K4 statistic (fold t = mean/(SD/√5)) is inflated by ridge shrinkage in sparse cells; redesign before any confirmatory study.
- Registered N-circ null manufactures events from censored intervals (+16%); N-jit clipping loses ~9.5% of events.
- No external datasets. Not needed until Phase 3.
- Episode estimator: 'no record' is treated as UNKNOWN (weak evidence), so ghost keys linger after identity swaps; the visible-miss likelihood selected on pooled J fragments episodes under boundary flicker (episode exit report §7–§8).
- Episode ladder contrasts compare selected operating points: mechanism and base-parameter selection are confounded (post-hoc E1).

## Next action
WAIT for user review of the completion PR (Phase 5 complete; EGTR integrated; final evaluation STOP at gate; paper draft). Single justified next step (needs a NEW registration + approval): predictor-level test with rank-normalised scores and a development-quantile tau, ideally with a video scene-graph model; do not reuse the stopped registration. Earlier: WAIT for user review of the downstream PR (registered GO, narrow; no algorithm contribution). Single justified next step (needs approval; major scope): replace the oracle relation predictor with one existing VidSGG model's cached predictions on the same frozen pipeline, with identity-stable/flicker-robust scoring and harm margins relative to baseline persistence; stop the line if no gain. Earlier: WAIT for user review of the appearance-audit PR (decision NARROW). Single justified next step (needs approval): one pre-registered downstream test feeding the FROZEN T+A takeover flag into GatedEMA+hysteresis / fixed buffer via the observation contract, measuring relationship ghost time and false termination; requires relationship labels (VidSGG model or a pre-registered proxy) and must not tune on the MOT17 sequences used here; stop the line if no gain over the fixed buffer. Earlier: Single next experiment (needs approval: MOT17 frames 1.95 GB + re-ID model): repeat the pre-stated audit with an appearance-based tracker exposing association evidence (e.g. BoT-SORT + re-ID); alternatively integrate one VidSGG model to audit 'not evaluated' vs 'evaluated negative'. Earlier: WAIT for user review of the next-method PR (STOP / UNSUPPORTED) and of the episode-study PR. No algorithm contribution is currently supported. The most promising new question (needs a new prereg) is to separate 'no record / track lost' from 'visibility UNKNOWN', ideally using real tracker states. Earlier: WAIT for user review of the episode-study PR (decision STOP). Recommendation: use GatedEMA + hysteresis as the termination-aware baseline and the episode metrics in the robustness-evaluation paper; do not propose the episode estimator as a method. Phase 5H-1 NARROW decision also still pending. Do not start detector integration, another dataset, heap optimisation, or a later phase without approval.

## Update protocol for Claude Code
After each meaningful task:
1. Update `Current status` / `Active phase` if changed.
2. Move finished items in `Completed work`.
3. Record important design decisions.
4. Record new blockers/open questions.
5. Add only verified experimental results, with run IDs/paths.
6. Set one concrete `Next action`.
7. Keep this file concise; detailed derivations belong under `research/derivations/`, and experiment details belong in run artifacts.


2026-10-07 - Added scripts/phase5h1/visual_demo.py: visual wrapper around the existing synthetic Stage 0 demo (not a Phase 5H-1 comparison). Generated PNG/PDF/JSON under outputs/showcase_demo. Verified run: planted alpha 0.6 recovered as 0.583; independent control 0.000. Five Stage 0 tests passed. Frozen research artifacts unchanged.
