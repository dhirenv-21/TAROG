# TAORG / V-HPED — Development Phases

## General rule
Do not ask Claude Code to build the entire project in one prompt. Work phase-by-phase. Each phase has an exit gate. The paper is a first-class deliverable, so evidence/provenance is collected from the beginning.

## Phase 0 — Repository + research contract
### Goal
Create a clean, testable research repository before integrating heavy models.

### Build
- repository skeleton from `Architecture.md`
- `pyproject.toml`, formatter/linter only if lightweight, pytest
- canonical schemas
- config loading
- seed utility
- run metadata/artifact writer
- tiny synthetic fixture
- `.gitignore`
- `research/metric_definitions.md` stub
- `research/claims.md` ledger

### Exit gate
- imports work
- tests run locally
- a dummy experiment creates a complete run artifact folder
- no external dataset required

## Phase 1 — Canonical edge stream + reference temporal baselines
### Goal
Establish the common interface and simple baselines before V-HPED.

### Build
- FrameGraph/EdgeObservation representations
- no-memory
- fixed-window
- EMA
- visibility-gated EMA
- simple 2-state/HMM baseline or scaffold if dependency decisions are pending
- eager temporal update path

### Tests
- toy edge appears/disappears
- visible absence decays faster than occluded absence for gated EMA
- methods consume identical canonical input

### Exit gate
Baseline behavior is deterministic and metrics can consume its outputs.

## Phase 2 — Noise protocol + custom metrics
### Goal
Build the experimental apparatus before the proposed model.

### Build
- deterministic occlusion injection
- deterministic ID-switch injection
- saved corruption manifests
- Edge Persistence precision/recall/F1
- Relationship Churn Rate
- Termination Delay
- false-persistence measure
- calibration metric scaffold
- latency instrumentation

### Required research action
Freeze exact metric definitions in `research/metric_definitions.md` before pilot results are interpreted.

### Exit gate
Hand-constructed toy sequences produce manually verifiable metric values, and the same seed creates the same corruption manifest.

## Phase 3 — Go/no-go pilot
### Goal
Test the central visibility hypothesis cheaply.

### Data
Prefer Action Genome with ground-truth boxes/relations or precomputed canonical predictions. Avoid spending the first two weeks reproducing all heavy base models if the pilot can be tested without them.

### Compare
- no memory
- EMA
- visibility-gated EMA

### Corruptions
Occlusion gaps: 1, 2, 3, 5, 10, 20 frames; multiple predetermined seeds.

### Analyze
Plot persistence F1 against termination/false-persistence behavior across thresholds and occlusion lengths.

### Go gate
Visibility gating demonstrates a stable improvement in the persistence/termination trade-off rather than a single cherry-picked operating point.

### Pivot gate
If the signal is weak/inconsistent, stop expansion of V-HPED and prioritize a tracker-noise robustness benchmark paper/project. Record the result honestly in `Memory.md` and `research/claims.md`.

## Phase 4 — Mathematical V-HPED reference model
### Goal
Implement the proposed temporal model correctly in eager form before optimization.

### Before coding
Resolve/document:
- exact visible-negative-evidence model
- state/intensity/alive-probability relationship
- softplus likelihood integral treatment
- parameter constraints
- beta visibility behavior
- threshold domain

### Build
- multivariate predicate state
- excitation/inhibition matrix alpha
- visibility regimes
- alive mapping/calibration hook
- likelihood/training code
- eager inference reference

### Tests
- zero-alpha reduces to independent predicate dynamics
- no cross-predicate effect when disabled
- inhibition cannot create invalid intensity
- gradients finite on toy data
- toy likelihood checked numerically where possible

### Exit gate
Toy synthetic sequences can be fit and recovered behavior is sensible; no unresolved mathematical shortcut is silently embedded.

## Phase 5 — Lazy decay + expiry heap (core ADSA phase)
### Goal
Implement and validate the data-structure contribution.

### Build
- lazy materialization
- expiry min-heap
- generation/version invalidation
- visibility-transition rescheduling
- instrumentation counters
- eager-vs-lazy randomized equivalence harness

### Formal work
Write derivations in `research/derivations/`:
- update complexity
- memory complexity
- conditions for threshold-crossing/expiry calculation
- termination-delay proposition with explicit assumptions

### Benchmarks
Synthetic traces varying:
- active edges A
- observed edges per frame
- visibility transitions
- expiry rates
- predicate count

### Exit gate
Optimized decisions match eager reference within declared tolerance; complexity and latency results are reproducible.

## Phase 6 — Base model integration
### Goal
Attach temporal methods to real VidSGG predictions.

### Priority order
1. Reproduce/integrate one base model completely.
2. Freeze adapter contract.
3. Add STTran / DSG-DETR / TEMPURA as feasible.

Team member 4 should start dependency/reproduction work early in parallel, but core temporal development should not block on all three models.

### Exit gate
For each claimed base model, baseline evaluation matches a credible expected range under the same dataset/protocol, or discrepancies are documented rather than hidden.

## Phase 7 — Dataset expansion + tracker noise
### Goal
Move from synthetic pilot to full robustness evaluation.

### Build/evaluate
- Action Genome standard evaluation
- ActionGenome4D only after annotation/availability verification
- VidOR or PVSG for multi-person ID-switch tests as appropriate
- ByteTrack real failure cases
- controlled ID-switch durations

### Exit gate
Noise experiments have paired corruption manifests and multiple seeds; real-tracker analysis includes categorized failure examples.

## Phase 8 — Full ablation matrix
### Compare
- no memory
- fixed window
- EMA
- visibility-gated EMA
- HMM/2-state
- V-HPED no visibility
- V-HPED no cross-predicate alpha
- V-HPED hand-set/no MLE
- full V-HPED
- verified closest recent method if reproducible

### Report
- standard scene-graph accuracy
- persistence
- churn
- termination
- calibration
- latency
- mean ± std over fixed seed set

### Exit gate
All planned ablations use the same evaluation definitions and final configs are frozen.

## Phase 9 — Paper figures, tables, and statistical checks
### Build
Scripts only; no hand-transcribed core numbers.

### Required figures
- method/pipeline diagram
- key persistence-vs-termination trade-off figure
- performance vs occlusion length
- churn comparison
- termination-delay distribution/summary
- latency/scaling plot
- selected qualitative sequences showing occlusion vs true termination

### Required tables
- standard benchmark accuracy
- robustness results
- ablations
- efficiency

### Statistical practice
Use paired comparisons where the same corrupted clips/seeds are evaluated by each method. Report variability and avoid significance claims without an appropriate test.

## Phase 10 — Paper drafting + reproducibility audit
### Paper sections
- Abstract
- Introduction
- Related Work
- Method
- Algorithm/Data Structure
- Experimental Protocol
- Results
- Ablations
- Limitations
- Conclusion

### Audit checklist
- every quantitative claim maps to run IDs
- equations match implementation
- metric definitions match code
- dataset statements are cited/verified
- no test leakage
- seeds/configs saved
- negative findings represented
- limitations explicit

### Final deliverables
- reproducible code repository
- environment/setup instructions
- configs for headline experiments
- raw/aggregated result provenance
- paper draft
- ADSA complexity/proof material
