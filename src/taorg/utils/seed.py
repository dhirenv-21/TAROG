"""Deterministic seed utility.

Central seed management for reproducibility.  All random number generators
used in experiments must be seeded through this module.

References: Rules.md §3 (deterministic seeds), PRD FR-11 (record seeds).
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and optionally PyTorch.

    Args:
        seed: Non-negative integer seed value.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    logger.info("Set Python random seed = %d, NumPy seed = %d", seed, seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: torch.backends.cudnn.deterministic = True may be needed
        # for full reproducibility but has performance cost.
        # Set explicitly per-experiment in config rather than globally here.
        logger.info("Set PyTorch manual seed = %d", seed)
    except ImportError:
        logger.debug("PyTorch not available; skipping torch seed.")
