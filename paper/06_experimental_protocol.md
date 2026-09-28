# 6. Experimental Protocol — DRAFT

> Written from the frozen Phase 2 protocol (research/metric_definitions.md) and the Phase 3
> pre-registration (research/phase3_preregistration.md). No results in this section.

## 6.1 Evaluation principle

Temporal post-processors are compared on **identical corrupted inputs** (paired design) and at
**matched termination cost**. Persistence alone is not a meaningful target: a method that never
drops an edge maximises persistence while failing termination. Every comparison therefore
reports persistence together with termination delay, false persistence and churn, and the
central comparison is either (i) at a fixed termination-delay budget, or (ii) over the whole
persistence–termination frontier.

## 6.2 Controlled corruption

All corruption is deterministic given a seed and is recorded in a versioned JSON manifest
(`taorg.noise.manifest.CorruptionManifest`) so that every method receives exactly the same
corrupted stream and every corruption can be audited.

| Corruption | Operator | Ground truth |
|---|---|---|
| Occlusion gap of $g$ frames | observation suppressed, object visibility set `OCCLUDED_OR_UNRELIABLE` (Phase 2) | unchanged — relationship still alive |
| ID switch of $d$ frames | two same-class track IDs swapped in tracks and edges (Phase 2) | unchanged |
| Base-model flicker, rate $q$ | observed edge suppressed while both objects stay `VISIBLE` (Phase 3) | unchanged |
| Visibility-flag noise, rate $\rho$ | absence-record visibility flipped OCC↔VIS (Phase 3) | unchanged |
| UNKNOWN visibility, rate $u$ | absence-record visibility set `UNKNOWN` (Phase 3) | unchanged |
| Termination under occlusion | object occluded for 15–40 frames right after a genuine termination (scene property) | relationship ended |

Corruption sub-seeds are derived from the scenario seed with SHA-256, never from Python's
salted `hash()`, so manifests are identical across processes and machines (tested).

## 6.3 Synthetic world (Phase 3)

Ground truth with exactly known relationship lifetimes (`taorg.data.synthetic_world`): 300
frames at 30 fps, 8 edges per video among 2 subject and 4 object tracks, 60% of edges
terminating after 60–180 alive frames followed by ≥ 40 visible-absence frames, noisy observed
scores. This world isolates temporal behaviour from perception errors; it is not a model of
any real dataset.

## 6.4 Metrics (frozen before the pilot)

- **Edge Persistence P/R/F1** — frame-level, micro-averaged over occlusion intervals only.
  Because ground truth is alive inside every injected interval, precision is 1 whenever
  anything is output there, and F1 is a monotone function of persistence recall.
- **Relationship Churn Rate** — output-state flips per consecutive frame pair within each
  edge's GT window (first to last GT-alive frame), averaged over edges; initial appearance and
  final disappearance are not counted.
- **Termination Delay** — $t_{\text{drop}} - t_{\text{end}}$ in frames; 1 = dropped on the
  first post-termination frame; never-dropped edges are censored at the sequence end and
  reported separately.
- **False Persistence Rate** — fraction of post-termination frames on which the edge is still
  output.
- **Latency** — wall-clock time of the temporal module's `step()` per frame (mean, median,
  p95, p99), excluding base-model inference.
- **Diagnostics (not frozen, never decisive):** clean-period recall, spurious edge-frame rate,
  ghost edge-frames, post-gap recovery lag, ID-swap recall. They expose failures the frozen
  metrics cannot see (drops outside occlusion gaps; hallucinated edges after ID switches).

Known metric properties to disclose: (i) Persistence F1 ignores everything outside occlusion
intervals; (ii) the churn implementation includes never-GT predicted edges with churn 0 in its
mean, so ID-switch ghosts *dilute* churn; (iii) termination metrics only consider each edge's
final GT termination.

## 6.5 Parameter selection

Disjoint calibration (20 videos) and evaluation (30 videos) seeds. Each method is searched
exhaustively over a declared grid with identical value sets for shared parameters (the
Vis-Gated EMA grid contains the EMA grid). For each termination budget
$B \in \{2, 5, 10\}$ frames, the configuration with the highest mean calibration Persistence F1
among those with mean calibration delay $\le B$ is selected (ties: churn, delay, false
persistence, grid order). Calibration pools all gap lengths because gap length is unknown at
deployment. Tuning is condition-matched (each method is tuned on calibration data from the
regime it is evaluated in); stress conditions reuse oracle-condition parameters. Frontiers
are each method's calibration Pareto set, evaluated on held-out seeds.

## 6.6 Statistical analysis

Unit: evaluation video (n = 30), paired across methods. Descriptive mean, SD and 95%
t-intervals; paired differences with 95% percentile-bootstrap CIs (5000 resamples), Cohen's
$d_z$, and two-sided Wilcoxon signed-rank tests with Holm correction over the registered
family. Frontiers are compared by normalized dominated hypervolume
(F1 × delay ∈ [1, 40] frames) with a paired bootstrap over videos. Practical-significance and
non-inferiority margins are fixed in advance (ΔF1 ≥ 0.05; delay +1 frame; false persistence
+0.02; churn max(0.005, 10%)).

## 6.7 Reproducibility

Every stage writes a provenance record (run ID, UTC timestamps, git commit/branch/dirty flag,
command line, config SHA-256, Python/NumPy/SciPy/matplotlib versions, platform). Raw
per-(configuration, instance) results, all manifests and GT lifetimes are saved; every table
and figure is regenerated from these files by `scripts/phase3/run_phase3.py`. Seeds are
predeclared in `configs/experiments/phase3_pilot.yaml`; no seed is rerun selectively.

## 6.8 Real-data protocol

[TBD — Phase 6–7: Action Genome with base-model predictions; tracker-derived visibility;
real ByteTrack failure cases.]
