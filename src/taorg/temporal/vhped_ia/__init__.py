"""IA-VHPED — identity-aware V-HPED (Stage 6 extension; baseline V-HPED unchanged).

    params.py      IAParams (V-HPED params + foreign rate, foreign birth, recovery)
    identity.py    per-track identity belief zeta from tracker identity scores
    model.py       IAVHPED (eager) and LazyIAVHPED (wake-up heap)
    likelihood.py  differentiable log-likelihood (PyTorch)
    trainer.py     maximum-likelihood fitting

Derivation: research/derivations/identity_aware.md
"""

from taorg.temporal.vhped_ia.model import IAVHPED, LazyIAVHPED
from taorg.temporal.vhped_ia.params import IAParams
from taorg.temporal.vhped_ia.trainer import IAFitResult, fit_ia_vhped

__all__ = ["IAVHPED", "LazyIAVHPED", "IAParams", "IAFitResult", "fit_ia_vhped"]
