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
    Episode study (research/taorg_episode_formulation.md):
    METHODS["gated_ema_hyst"] -> GatedEMAHysteresis (B2)
    METHODS["hsmm"]           -> DurationHSMM (B4)
    METHODS["gated_ema_id"]   -> GatedEMAIdentity (B5)
    METHODS["episode".."episode_hist"] -> episode ladder P1-P4
"""

from taorg.temporal.base import TemporalEdgeModel
from taorg.temporal.no_memory import NoMemory
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.ema import EMA
from taorg.temporal.gated_ema import GatedEMA
from taorg.temporal.hmm import HMM
from taorg.temporal.va_hmm import VisibilityAwareHMM
from taorg.temporal.vhped import VHPED, vhped_m2, vhped_m3, vhped_m4, vhped_m5, vhped_m6
from taorg.temporal.gated_variants import GatedEMAHysteresis, GatedEMAIdentity
from taorg.temporal.hsmm import DurationHSMM
from taorg.temporal.episode import EpisodeEstimator, episode_p1, episode_p2, episode_p3, episode_p4

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
    "gated_ema_hyst": GatedEMAHysteresis,
    "hsmm": DurationHSMM,
    "gated_ema_id": GatedEMAIdentity,
    "episode": episode_p1,
    "episode_id": episode_p2,
    "episode_ctx": episode_p3,
    "episode_hist": episode_p4,
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
    "GatedEMAHysteresis",
    "GatedEMAIdentity",
    "DurationHSMM",
    "EpisodeEstimator",
    "METHODS",
]
