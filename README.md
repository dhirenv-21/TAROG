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
