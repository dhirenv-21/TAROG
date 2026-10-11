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
    Next-method study (research/taorg_next_method_formulation.md):
    METHODS["b2_hold_km" | "_static" | "_pooled" | "_geometric" | "_buffer"] -> B2SurvivalHold modes
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
from taorg.temporal.cure import B2SurvivalHold


def _hold(mode: str):
    def make(**p):
        return B2SurvivalHold(**p, mode=mode)
    make.__name__ = f"b2_hold_{mode}"
    return make

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
    "b2_hold_km": _hold("km"),
    "b2_hold_static": _hold("static"),
    "b2_hold_pooled": _hold("pooled"),
    "b2_hold_geometric": _hold("geometric"),
    "b2_hold_buffer": _hold("buffer"),
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
    "B2SurvivalHold",
    "METHODS",
]
