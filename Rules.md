# TAORG / V-HPED — Rules for Claude Code and Human Contributors

## 0. Prime directive
This is **conference-paper research code**. Optimize for correctness, reproducibility, falsifiability, and traceability before convenience or visual polish.

Claude Code must read `PRD.md`, `Architecture.md`, `Rules.md`, `Phases.md`, and the current `Memory.md` before making substantial changes.

## 1. Research integrity rules
1. Never fabricate metrics, citations, dataset properties, experiment outputs, benchmark numbers, or completed work.
2. Never write a paper claim as established merely because it appears in the project proposal. Mark unverified claims as hypotheses/TODOs.
3. Never claim SOTA without verified comparable evidence.
4. Do not hide failed runs or negative ablations.
5. Do not tune on the final test split.
6. Every reported number must be traceable to saved run artifacts.
7. If a mathematical derivation is uncertain, stop and flag it rather than encoding it as fact.
8. The paper must distinguish standard Hawkes-process machinery from this project's actual contributions.

## 2. Mathematical correctness rules
1. Keep these concepts separate in code and docs:
   - Hawkes/event intensity `lambda`
   - pre-activation/state `z`
   - edge-alive probability/score
   - keep/drop threshold
2. A softplus intensity is positive, but its likelihood integral is not automatically the same closed form as a linear exponential Hawkes process. Implement a justified integral treatment.
3. Negative `alpha` values require a formulation that preserves valid intensity; test extreme cases.
4. Visible absence is not automatically a standard Hawkes event. Explicitly define how negative evidence enters the model.
5. The expiry formula may only be used under assumptions that make the threshold crossing valid. State and test those assumptions.
6. The termination-delay proposition must specify initial state, visibility regime, no-new-event assumptions, threshold domain, and parameter constraints.
7. Maintain a simple eager implementation as a reference oracle before optimizing.

## 3. Coding rules
- Python type hints on public functions and core data structures.
- Prefer dataclasses for canonical research records.
- Small, testable modules; avoid god classes.
- No hard-coded absolute paths.
- No magic constants in algorithms; put experimental parameters in configs.
- No silent exception swallowing.
- Fail fast on incompatible configs or missing required fields.
- Error messages should say what failed, where, and how to fix it.
- Use deterministic seeds through a central utility.
- Use `pathlib` for paths.
- Use the standard `logging` module; no scattered debug `print()` in library code.
- Docstrings for mathematical/core public functions, including units and tensor shapes.

## 4. Library policy
### Preferred
- PyTorch
- NumPy
- SciPy when mathematically justified
- OpenCV for vision/video utilities
- pandas for analysis/aggregation
- matplotlib for plots
- pytest
- PyYAML or OmegaConf (choose one)

### Avoid unless justified
- Large orchestration frameworks that obscure experiment logic.
- Multiple overlapping config frameworks.
- NetworkX in the per-frame hot path unless profiling proves it acceptable; canonical edge dictionaries/tensors are simpler for this use case.
- pandas inside per-frame temporal inference loops.
- UI frameworks before research milestones are complete.

### External research repositories
Do not rewrite external baselines casually. Prefer adapters, isolated environments, submodules, or documented setup. Respect licenses. Do not commit large pretrained weights or datasets.

## 5. Git/change discipline
- Make focused changes by phase/task.
- Do not mass-refactor unrelated baseline code while implementing a feature.
- Before changing a public schema or metric definition, identify downstream effects.
- Never delete raw result artifacts to make results look cleaner.
- Generated large outputs, datasets, checkpoints, and caches belong in `.gitignore`.

## 6. Experiment rules
Every run must have:
- resolved config
- run ID
- git commit
- seed
- dataset/split identity
- base-model identity
- temporal method identity
- corruption manifest when relevant
- metrics and timing

All compared methods must receive the same underlying predictions/corruptions for a paired comparison where possible.

Do not rerun only failed/unfavorable seeds selectively. Predeclare seed sets in configs for final experiments.

## 7. Data rules
- Never leak validation/test annotations into training or threshold fitting.
- Dataset conversion must be deterministic and versioned.
- Keep original data immutable; derived data goes to separate cache/output directories.
- Verify track IDs and predicate vocabularies when adapting datasets/models.
- Do not assume occlusion annotations exist until verified from the dataset itself.
- Store synthetic noise separately as manifests rather than destructively editing source data.

## 8. Metric rules
Before large experiments, write `research/metric_definitions.md` containing exact formulas and edge cases.

For every custom metric, tests must cover:
- perfect prediction
- complete failure
- no-positive case
- no-prediction case
- short occlusion
- true termination
- repeated churn

Do not change a metric definition after seeing final results without documenting the change and rerunning all affected experiments.

## 9. Performance/ADSA rules
- First implement correct eager reference behavior.
- Then implement lazy/heap optimization.
- Compare outputs automatically.
- Heap entries must be versioned or otherwise support safe decrease/increase-key semantics.
- Report stale-pop behavior and heap growth.
- Complexity claims must count all relevant operations, including visibility regime changes and heap invalidation.
- Separate candidate-pair generation cost from temporal-state maintenance cost.

## 10. Testing gates
No phase is complete until its listed tests pass.

At minimum before final experiments:
- unit tests pass
- randomized eager-vs-lazy equivalence tests pass
- deterministic corruption tests pass
- metric toy tests pass
- tiny end-to-end integration test passes

## 11. Claude Code behavior rules
Claude must:
1. Read the relevant docs before coding.
2. State which phase/task it is implementing.
3. Inspect existing code before creating duplicate abstractions.
4. Make the smallest coherent change.
5. Run relevant tests after changes.
6. Report exact files changed and tests run.
7. Update `Memory.md` after a meaningful completed task once coding has begun.
8. Add unresolved research/mathematical questions to `Memory.md` rather than guessing.
9. Update `research/claims.md` when an experiment materially supports/refutes a paper claim.
10. Never mark a phase complete solely because code was written; acceptance criteria/tests must pass.

Claude must not:
- invent unavailable dataset files/checkpoints
- fake a successful command
- silently weaken tests
- replace an established metric with an easier one
- add a frontend/dashboard unless explicitly requested after research milestones
- prematurely optimize before reference correctness exists
- write the paper as if proposed results already occurred

## 12. Paper-first documentation rules
For each major method component, keep enough documentation to later write:
- motivation
- equation/algorithm
- assumptions
- implementation details
- complexity
- hyperparameters
- ablation mapping
- limitations

For each final experiment, save enough metadata to reproduce the corresponding table/figure.

## 13. Definition of done for a task
A task is done only when:
- implementation exists
- relevant tests pass
- config/example exists when needed
- docs are updated if behavior changed
- `Memory.md` records the status/decision
- no known correctness issue is being hidden
