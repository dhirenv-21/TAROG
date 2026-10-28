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

## Phase 3 pilot (Action Genome)

```bash
# 1. place the official AG annotations under data/action_genome/annotations/
# 2. verify the loader's format assumptions on the real files
python scripts/inspect_action_genome.py --root data/action_genome
# 3. run the pilot (all predeclared seeds, all gap lengths, all methods incl. V-HPED ablations)
python scripts/run_pilot_ag.py --config configs/experiments/pilot_action_genome.yaml --workers 4
```

Outputs go to `outputs/runs/<run_id>/` (curves, selected-threshold table, hypervolume
comparisons, manifests, fitted parameters, figures).

## Other experiments and the paper

```bash
python scripts/run_ablation_matrix.py --workers 5     # Phase 8 synthetic ablation matrix
python scripts/run_complexity_benchmark.py            # Phase 5 eager vs lazy scaling
python scripts/make_paper_artifacts.py                # paper/generated/ from run artifacts
```

The paper draft is `paper/main.tex` (all numbers come from `paper/generated/`).
Reproducibility status: `research/reproducibility_audit.md`.

## Dashboard

A web dashboard over the saved runs in `outputs/runs/`: headline results, the ablation
tables with threshold curves and paired comparisons, the lazy-maintenance benchmark,
parameter recovery, run history with downloads, and a filter explorer. The explorer replays
one saved synthetic test video through the real filters, using that run's fitted parameters
and thresholds, and checks each result against the per-video value the run stored. It never
starts an experiment or writes to `outputs/`.

```bash
pip install -e ".[dashboard]"
```
```bash
python scripts/run_dashboard.py
```

Then open http://127.0.0.1:8050. Options: `--runs <dir>` and `--port <n>`. The API docs
are at `/api/docs`. The frontend is plain HTML/CSS/JS in `src/taorg/dashboard/static/`, with
no build step. All results shown are synthetic.

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
