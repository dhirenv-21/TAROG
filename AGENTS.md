# AGENTS.md — Persistent Rules for TAORG Development

## Project Identity
- **TAORG** — Temporal Adaptive Object Relationship Graph
- **V-HPED** — Visibility-aware Hawkes-Process Edge Dynamics
- **Type:** Paper-oriented research repository + ADSA academic project

## Before Any Work
1. Read `Memory.md` — it records verified reality, not aspirations.
2. Check `Phases.md` — identify the current phase and stay within it.
3. Read `Rules.md` — all rules are mandatory.

## Mandatory Rules

### Research Integrity
- Never fabricate results, citations, dataset properties, or benchmark numbers.
- Never mark a claim as supported without linking to experiment artifacts.
- If a baseline beats V-HPED, report it honestly.
- Preserve negative results. Do not delete failed experiments.
- Distinguish hypotheses from verified facts in all documentation.

### Mathematical Correctness
- Verify mathematical assumptions before encoding them in code.
- Do not assume the proposed equations are correct because they appear in spec files.
- Flag questionable derivations rather than silently patching them.
- Keep a simple eager reference implementation when optimizing algorithms.

### Reproducibility
- Record random seeds, git commits, configs, dataset versions.
- Every experiment result must be traceable to run artifacts.
- Do not cherry-pick seeds or selectively rerun unfavorable experiments.
- Use `taorg.utils.seed.set_seed()` for all randomness.

### Code Quality
- Run tests before marking any work complete.
- Update `Memory.md` only with verified facts.
- Prefer small, focused changes over large rewrites.
- Do not introduce dependencies without justification.
- Type hints on public functions. Docstrings on mathematical/core functions.

### Safety
- Do not perform destructive operations or major architectural rewrites without justification.
- Do not delete specification files (PRD.md, Architecture.md, Rules.md, Phases.md, Design.md).
- Do not modify `Memory.md` to fabricate progress.
- Do not alter `.gitignore` to hide important files.

## Canonical Documents (Root)
| File | Role |
|------|------|
| `PRD.md` | What to build and test |
| `Architecture.md` | Software architecture |
| `Rules.md` | Mandatory rules |
| `Phases.md` | Implementation order |
| `Design.md` | Figure/visualization standards |
| `Memory.md` | Verified implementation record |

These six files are the canonical specification. This AGENTS.md does not override them.
