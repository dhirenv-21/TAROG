"""V-HPED — Visibility-aware Hawkes-Process Edge Dynamics.

Split per Architecture.md §6:
    params.py      parameter container + ablations
    dynamics.py    excitation traces, intensity, hazard
    visibility.py  regime -> thinning / trace decay
    alive.py       alive/dead Bayesian filter step
    model.py       eager TemporalEdgeModel (reference oracle)
    expiry_heap.py closed-form untouched dynamics + versioned expiry heap
    lazy.py        LazyVHPED: touched-pair updates + expiry scheduling (Phase 5)
    likelihood.py  differentiable log-likelihood (PyTorch)
    trainer.py     maximum-likelihood fitting

Derivation: research/derivations/vhped_model.md
"""

from taorg.temporal.vhped.lazy import LazyVHPED
from taorg.temporal.vhped.model import HAND_SET_CONFIG, VHPED
from taorg.temporal.vhped.params import VHPEDParams
from taorg.temporal.vhped.trainer import FitResult, TrainerConfig, fit_vhped

__all__ = ["VHPED", "LazyVHPED", "VHPEDParams", "HAND_SET_CONFIG", "TrainerConfig", "FitResult", "fit_vhped"]
