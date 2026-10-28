"""IA-VHPED parameters: V-HPED parameters + identity-mixture parameters.

See research/derivations/identity_aware.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from taorg.temporal.vhped.model import HAND_SET_CONFIG
from taorg.temporal.vhped.params import VHPEDParams


@dataclass
class IAParams:
    """Identity-aware extension parameters.

    Attributes:
        base: Unchanged V-HPED parameters.
        foreign_rate: phi (1/s), detection rate of records on an identity-
            inconsistent pair (state-independent; thinned by visibility like V-HPED).
        foreign_birth: b, alive belief for a predicate first seen on an
            inconsistent pair.
        recovery_rate: omega (1/s), spontaneous return of a label to its home object.
        posterior_weight: True = exact within-frame posterior w; False = w := zeta
            (ablation).
        identity_births: True = inconsistent-branch births use `foreign_birth`;
            False = they use the V-HPED birth prediction (ablation).
        jump_calibration: (a, b) of P(jump | kappa) = sigmoid(a + b * logit(1 - kappa))
            (IA-v2, learned).  None = raw v1 mapping P(jump) = 1 - kappa.
    """

    base: VHPEDParams
    foreign_rate: float = 1.0
    foreign_birth: float = 0.1
    recovery_rate: float = 0.0
    posterior_weight: bool = True
    identity_births: bool = True
    jump_calibration: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.jump_calibration is not None:
            a, b = (float(x) for x in self.jump_calibration)
            if b < 0:
                raise ValueError(f"jump calibration slope must be >= 0, got {b}")
            self.jump_calibration = (a, b)
        if self.foreign_rate < 0:
            raise ValueError(f"foreign_rate must be >= 0, got {self.foreign_rate}")
        if not 0.0 < self.foreign_birth < 1.0:
            raise ValueError(f"foreign_birth must be in (0, 1), got {self.foreign_birth}")
        if self.recovery_rate < 0:
            raise ValueError(f"recovery_rate must be >= 0, got {self.recovery_rate}")

    @classmethod
    def default(cls, num_predicates: int, **kw) -> IAParams:
        base = VHPEDParams.from_config(dict(HAND_SET_CONFIG, num_predicates=num_predicates))
        return cls(base=base, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base.to_dict(), "foreign_rate": self.foreign_rate,
                "foreign_birth": self.foreign_birth, "recovery_rate": self.recovery_rate,
                "posterior_weight": self.posterior_weight,
                "identity_births": self.identity_births,
                "jump_calibration": None if self.jump_calibration is None
                else list(self.jump_calibration)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IAParams:
        return cls(base=VHPEDParams.from_dict(d["base"]), foreign_rate=d["foreign_rate"],
                   foreign_birth=d["foreign_birth"], recovery_rate=d["recovery_rate"],
                   posterior_weight=d["posterior_weight"], identity_births=d["identity_births"],
                   jump_calibration=None if d.get("jump_calibration") is None
                   else tuple(d["jump_calibration"]))

