"""TAORG temporal method modules.

All temporal methods implement the TemporalEdgeModel protocol
defined in base.py.

Registry mapping config method names to classes:
    METHODS["no_memory"]     -> NoMemory
    METHODS["fixed_window"]  -> FixedWindow
    METHODS["ema"]           -> EMA
    METHODS["gated_ema"]     -> GatedEMA
    METHODS["hmm"]           -> HMM
    METHODS["va_hmm"]        -> VisibilityAwareHMM (Phase 4 baseline M1)
    METHODS["vhped_m2".."vhped_m6"] -> V-HPED ladder rungs (Phase 4)
"""

from taorg.temporal.base import TemporalEdgeModel
from taorg.temporal.no_memory import NoMemory
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.ema import EMA
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.hmm import HMM
from taorg.temporal.va_hmm import VisibilityAwareHMM
from taorg.temporal.vhped import VHPED, vhped_m2, vhped_m3, vhped_m4, vhped_m5, vhped_m6

# Config-driven method registry.
# Evaluation code should use this instead of method-specific branches.
METHODS: dict[str, type] = {
    "no_memory": NoMemory,
    "fixed_window": FixedWindow,
    "ema": EMA,
    "gated_ema": GatedEMA,
    "hmm": HMM,
    "va_hmm": VisibilityAwareHMM,
    "vhped_m2": vhped_m2,
    "vhped_m3": vhped_m3,
    "vhped_m4": vhped_m4,
    "vhped_m5": vhped_m5,
    "vhped_m6": vhped_m6,
}

__all__ = [
    "TemporalEdgeModel",
    "NoMemory",
    "FixedWindow",
    "EMA",
    "GatedEMA",
    "HMM",
    "VisibilityAwareHMM",
    "VHPED",
    "METHODS",
]
