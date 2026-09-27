"""TAORG temporal method modules.

All temporal methods implement the TemporalEdgeModel protocol
defined in base.py.

Registry mapping config method names to classes:
    METHODS["no_memory"]     -> NoMemory
    METHODS["fixed_window"]  -> FixedWindow
    METHODS["ema"]           -> EMA
    METHODS["gated_ema"]     -> GatedEMA
    METHODS["hmm"]           -> HMM
"""

from taorg.temporal.base import TemporalEdgeModel
from taorg.temporal.no_memory import NoMemory
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.ema import EMA
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.hmm import HMM

# Config-driven method registry.
# Evaluation code should use this instead of method-specific branches.
METHODS: dict[str, type] = {
    "no_memory": NoMemory,
    "fixed_window": FixedWindow,
    "ema": EMA,
    "gated_ema": GatedEMA,
    "hmm": HMM,
}

__all__ = [
    "TemporalEdgeModel",
    "NoMemory",
    "FixedWindow",
    "EMA",
    "GatedEMA",
    "HMM",
    "METHODS",
]
