# TAORG / V-HPED — Architecture

## 1. Architectural principle
Build a **research pipeline with replaceable adapters**, not a monolithic app. The temporal module must operate on a canonical intermediate representation so it can be attached to STTran, DSG-DETR, TEMPURA, or precomputed predictions without rewriting the algorithm.

## 2. High-level data flow

```text
Raw video / dataset annotations
        |
        v
Dataset loader -------------------------------+
        |                                      |
        v                                      |
Detector / GT boxes                            |
        |                                      |
        v                                      |
Tracker (e.g. ByteTrack)                       |
        |                                      |
        +--> visibility/track-state signals    |
        |                                      |
        v                                      |
Base VidSGG model / precomputed predictions    |
        |                                      |
        v                                      |
Canonical FrameGraph / EdgeObservation stream <+
        |
        +--> optional deterministic noise injector
        |
        v
Temporal method interface
  | no-memory
  | fixed-window
  | EMA
  | visibility-gated EMA
  | HMM/2-state
  ` V-HPED
        |
        v
Robust temporal scene graph
        |
        +--> benchmark metrics
        +--> robustness metrics
        +--> calibration metrics
        +--> latency/complexity logs
        v
Run artifacts -> aggregation -> paper tables/figures
```

## 3. Technical stack
### Core
- Python 3.10 or 3.11; choose one after checking base-model compatibility and pin it.
- PyTorch for tensors/optimization and integration with existing VidSGG code.
- NumPy for lightweight numerical operations.
- SciPy only where justified for optimization/integration/statistics.
- OpenCV for video/box utilities when needed.
- PyYAML or OmegaConf for configs; choose one and use it consistently.
- pandas for result aggregation, not core per-frame inference.
- matplotlib for research plots.
- pytest for tests.

### Tracking / VidSGG integration
- ByteTrack through a maintained compatible implementation or adapter.
- Existing STTran / DSG-DETR / TEMPURA repositories should be integrated through adapters or isolated vendor/external environments rather than copied and rewritten unnecessarily.

### Experiment/reproducibility utilities
- standard `logging` module or a small structured logging wrapper
- JSON/JSONL + CSV for durable result artifacts
- Git commit capture through subprocess
- optional TensorBoard/W&B only if the team explicitly chooses it; local artifacts remain mandatory even if an online tracker is used

## 4. Canonical domain objects
Use typed dataclasses or equivalent validated structures.

### `TrackObservation`
- video_id
- frame_idx
- timestamp
- track_id
- class_id
- bbox_xyxy
- detector_confidence
- tracker_state
- optional occlusion/visibility features

### `EdgeKey`
Immutable key:
- subject_track_id
- object_track_id
- predicate_id

For cross-predicate state, also expose a pair key `(subject_track_id, object_track_id)`.

### `EdgeObservation`
- `EdgeKey`
- predicate_score
- observed: bool
- visibility_state
- source model metadata

### `TemporalEdgeState`
At minimum:
- pair/edge key
- predicate state vector `z` or sufficient statistics
- last_update_time
- visibility regime
- alive score/probability
- scheduled expiry
- version/generation counter

### `ExpiryEntry`
- expiry_time
- edge key
- version

When popped, ignore entries whose version is no longer current.

## 5. Temporal-method interface
All methods implement a common contract similar to:

```python
class TemporalEdgeModel(Protocol):
    def reset_video(self, video_id: str) -> None: ...
    def step(self, frame: FrameGraph) -> FrameGraph: ...
    def finalize_video(self) -> None: ...
    def state_dict(self) -> dict: ...
```

Trainable methods additionally expose `fit(...)` or a separate trainer. Evaluation code must not contain method-specific branches beyond configuration/registration.

## 6. V-HPED internal design
Split V-HPED into four concerns:
1. **State dynamics:** materialize decay and apply predicate events.
2. **Visibility policy:** select regime/parameters and define treatment of absence.
3. **Alive mapping:** convert state/intensity to keep/drop score.
4. **State scheduler:** lazy materialization + expiry heap.

Do not bury all four in one class. This is essential for ablations and mathematical testing.

### Cross-predicate dynamics
For each object pair maintain a predicate vector. An observed predicate `q` can update target predicates through `alpha[q, p]`. Negative alpha is permitted only if the chosen positive-intensity formulation remains valid.

### Visibility regime transitions
On `VISIBLE <-> OCCLUDED_OR_UNRELIABLE`:
- materialize state to transition time
- change regime/decay behavior
- invalidate old expiry entry by incrementing version
- compute a new expiry if one exists

### Expiry scheduling caveat
The proposed closed-form threshold crossing is straightforward only under specified monotone exponential decay assumptions. Before using it, derive the exact threshold in the state domain being scheduled. If multiple exponentials, softplus mapping, inhibition, or changing regimes break the simple formula, implement a correct solver/bound or narrow the proposition's assumptions. Keep an eager reference implementation as the correctness oracle.

## 7. Proposed repository structure

```text
taorg/
├── README.md
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
├── Memory.md
├── pyproject.toml
├── requirements/                  # only if split envs are necessary
│   ├── core.txt
│   └── baselines.txt
├── configs/
│   ├── datasets/
│   ├── base_models/
│   ├── temporal/
│   ├── noise/
│   └── experiments/
├── src/taorg/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── tracks.py
│   │   ├── edges.py
│   │   └── runs.py
│   ├── data/
│   │   ├── action_genome.py
│   │   ├── actiongenome4d.py
│   │   ├── vidor.py
│   │   └── pvsg.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── sttran.py
│   │   ├── dsg_detr.py
│   │   └── tempura.py
│   ├── tracking/
│   │   ├── base.py
│   │   ├── bytetrack.py
│   │   └── visibility.py
│   ├── noise/
│   │   ├── occlusion.py
│   │   ├── id_switch.py
│   │   └── manifest.py
│   ├── temporal/
│   │   ├── base.py
│   │   ├── no_memory.py
│   │   ├── fixed_window.py
│   │   ├── ema.py
│   │   ├── gated_ema.py
│   │   ├── hmm.py
│   │   └── vhped/
│   │       ├── model.py
│   │       ├── dynamics.py
│   │       ├── visibility.py
│   │       ├── alive.py
│   │       ├── likelihood.py
│   │       ├── trainer.py
│   │       └── expiry_heap.py
│   ├── metrics/
│   │   ├── scene_graph.py
│   │   ├── persistence.py
│   │   ├── churn.py
│   │   ├── termination.py
│   │   ├── calibration.py
│   │   └── latency.py
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── artifacts.py
│   │   └── aggregate.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── git.py
├── scripts/
│   ├── prepare_data.py
│   ├── generate_corruptions.py
│   ├── run_experiment.py
│   ├── run_pilot.py
│   ├── benchmark_heap.py
│   └── aggregate_results.py
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   └── fixtures/
├── research/
│   ├── claims.md
│   ├── metric_definitions.md
│   ├── derivations/
│   ├── experiment_registry.csv
│   └── paper_notes/
├── outputs/                       # gitignored except tiny examples
│   ├── runs/
│   ├── figures/
│   └── tables/
└── external/                      # gitignored/submodules as licensing permits
```

## 8. Configuration hierarchy
A final run config should resolve all of:
- dataset + split
- base model/checkpoint
- tracker/GT tracks
- visibility policy
- temporal method + hyperparameters
- noise protocol + seed
- metric config
- hardware/runtime settings

The resolved config must be copied into the run artifact directory.

## 9. Run artifact structure

```text
outputs/runs/<run_id>/
├── config.resolved.yaml
├── metadata.json
├── metrics.json
├── per_video_metrics.csv
├── timing.json
├── corruption_manifest.json       # if used
├── warnings.log
└── optional_predictions.*
```

## 10. Testing architecture
### Unit
- exponential decay/materialization
- alpha interaction direction/indexing
- visibility transitions
- heap version invalidation
- metric toy cases
- corruption determinism

### Property/equivalence
Generate random event traces and assert optimized lazy implementation matches eager reference within tolerance.

### Integration
Tiny synthetic video/edge fixture passes through canonical schema -> corruption -> temporal method -> metrics.

### Regression
Freeze small expected outputs for critical metrics and pilot aggregation.

## 11. Complexity instrumentation
Do not rely only on wall-clock time. Record:
- active edges `A`
- observed edges `k_obs`
- visibility changes `k_vis`
- valid expirations `k_exp`
- stale heap pops
- heap size
- number of state materializations
- CPU/GPU timing separately where relevant

This allows the ADSA complexity claim to be empirically checked.

## 12. Paper architecture
The codebase must support a clean paper narrative:
1. Problem and failure mode
2. Visibility-aware observation model
3. Multivariate temporal dynamics
4. Efficient lazy state maintenance
5. Complexity/bound under explicit assumptions
6. Robustness protocol and metrics
7. Experiments and ablations
8. Limitations/negative results

Every final table/figure should be generated from stored experiment artifacts by script.
