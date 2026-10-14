# TAORG — Temporal Adaptive Object Relationship Graph

## Overview

TAORG is a research project investigating whether **visibility-aware temporal
modeling (V-HPED)** can distinguish observation failure (occlusion, tracker noise)
from genuine relationship termination in video scene graphs.

This is a **conference-paper-oriented research repository**, not a demo or product.

## Core Research Question

> When a video-scene-graph relationship edge stops being observed, can
> visibility-aware temporal modeling distinguish:
> - observation failure caused by occlusion, missed detection, or tracker loss,
> - from genuine termination of the relationship while objects remain visible?

## Project Structure

```
├── PRD.md, Architecture.md, Rules.md, Phases.md, Design.md, Memory.md
│                               # Canonical specification documents
├── src/taorg/                  # Main source code
│   ├── schemas/                # Canonical domain objects
│   ├── temporal/               # Temporal methods (baselines + V-HPED)
│   ├── metrics/                # Evaluation metrics
│   ├── noise/                  # Noise injection for robustness testing
│   ├── data/                   # Dataset loaders
│   ├── adapters/               # Base model adapters
│   ├── tracking/               # Tracker integration
│   ├── experiments/            # Run management and artifacts
│   └── utils/                  # Seeds, logging, git utilities
├── configs/                    # YAML experiment configurations
├── tests/                      # pytest test suite
├── research/                   # Claims ledger, metric definitions, derivations
├── scripts/                    # Experiment and analysis scripts
└── outputs/                    # Run artifacts (gitignored)
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```


## What you can run

**Local website:** double-click `start_web.cmd`, or run
`python -m taorg.web --open`. Upload a clip, follow processing, and explore or
download its graph. See [the website guide](docs/WEB.md) for setup and limits.

| Goal | Command | Docs |
|---|---|---|
| Video → tracks → EGTR relationships → temporal maintenance → graphs + viewer | `python -m taorg.pipeline run --video clip.mp4 --work W` | [docs/PIPELINE.md](docs/PIPELINE.md), [docs/SETUP.md](docs/SETUP.md) |
| Phase 5 lazy-vs-eager benchmark (synthetic) | `python scripts/phase5/benchmark.py` then `python scripts/phase5/figures.py` | [research/derivations/lazy_maintenance.md](research/derivations/lazy_maintenance.md) |
| EGTR source-reproduction check (VG subset) | `scripts/egtr/fetch_vg_subset.py`, then `reproduce_vg_subset.py` (EGTR env) | [research/taorg_egtr_integration.md](research/taorg_egtr_integration.md) |
| Synthetic presentation demo | `python demo/build_demo.py --open` | [demo/README.md](demo/README.md) (synthetic, not evidence) |
| Tests | `python -m pytest -q` | — |

**Where the project stands** (verified; details in [Memory.md](Memory.md) and the evidence ledger):
- Certified lazy maintenance is exact and benchmarked.
- On real tracker outputs, appearance, not filter complexity, separates identity takeovers from
  gaps.
- No novel temporal estimator is supported.
- The predictor-level final evaluation stopped at its development gate.

The working paper draft is [paper/taorg_paper_draft.md](paper/taorg_paper_draft.md).

## Current Status

See [Memory.md](Memory.md) for verified implementation status.

## Specification Documents

| Document | Purpose |
|----------|---------|
| [PRD.md](PRD.md) | What to build and test |
| [Architecture.md](Architecture.md) | Software architecture and data flow |
| [Rules.md](Rules.md) | Mandatory coding and research rules |
| [Phases.md](Phases.md) | Implementation order and exit criteria |
| [Design.md](Design.md) | Figure and visualization standards |
| [Memory.md](Memory.md) | Verified implementation record |
