"""V-HPED parameter container.

All quantities are in SI-like units: rates in 1/second, decays (beta) in
1/second, probabilities in [0, 1].  See research/derivations/vhped_model.md §2–§6.

Array shapes (P = num_predicates):
    mu, eta:      (P,)
    alpha, gamma: (P, P), indexed [source q, target p]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from taorg.schemas.tracks import VisibilityState

# Regime index order shared by NumPy inference and the PyTorch likelihood.
REGIME_INDEX: dict[VisibilityState, int] = {
    VisibilityState.VISIBLE: 0,
    VisibilityState.OCCLUDED_OR_UNRELIABLE: 1,
    VisibilityState.UNKNOWN: 2,
}


@dataclass
class VHPEDParams:
    """Parameters of the visibility-thinned Hawkes edge model.

    Attributes:
        mu: Baseline pre-activation of the observation intensity, shape (P,).
        alpha: Excitation/inhibition of predicate p's intensity by events of
            predicate q, shape (P, P).  Any sign is valid (softplus link).
        eta: Baseline pre-activation of the termination hazard, shape (P,).
        gamma: Effect of predicate-q events on predicate p's termination
            hazard, shape (P, P).
        beta: Trace decay rates [beta_vis, beta_occ, beta_unk] in 1/s, all > 0.
        rho: Detection thinning [rho_vis, rho_occ, rho_unk]; rho_vis is fixed
            to 1 by definition.
        eps_rate: Spurious-detection rate when the edge is dead (1/s, > 0).
        rebirth_rate: Dead -> alive transition rate (1/s, >= 0).
        prior_alive: Belief assigned when a predicate is first observed.
        use_marks: If True, events add predicate_score to the trace; else 1.
    """

    mu: np.ndarray
    alpha: np.ndarray
    eta: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.3, 1.0]))
    rho: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.05, 0.5]))
    eps_rate: float = 1.0
    rebirth_rate: float = 0.0
    prior_alive: float = 0.5
    use_marks: bool = True

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.eta = np.asarray(self.eta, dtype=np.float64)
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.gamma = np.asarray(self.gamma, dtype=np.float64)
        self.beta = np.asarray(self.beta, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        self.validate()

    @property
    def num_predicates(self) -> int:
        return int(self.mu.shape[0])

    def validate(self) -> None:
        """Fail fast on shape or constraint violations."""
        p = self.mu.shape[0] if self.mu.ndim == 1 else -1
        if p < 1:
            raise ValueError(f"mu must be a non-empty 1-D array, got shape {self.mu.shape}")
        if self.eta.shape != (p,):
            raise ValueError(f"eta must have shape ({p},), got {self.eta.shape}")
        for name, arr in (("alpha", self.alpha), ("gamma", self.gamma)):
            if arr.shape != (p, p):
                raise ValueError(f"{name} must have shape ({p}, {p}), got {arr.shape}")
        if self.beta.shape != (3,) or np.any(self.beta <= 0):
            raise ValueError(
                f"beta must be 3 positive rates [vis, occ, unk], got {self.beta.tolist()}"
            )
        if self.rho.shape != (3,) or np.any(self.rho < 0) or np.any(self.rho > 1):
            raise ValueError(f"rho must be 3 values in [0, 1], got {self.rho.tolist()}")
        if self.rho[0] != 1.0:
            raise ValueError("rho[VISIBLE] must be exactly 1.0 (detection is unthinned when visible)")
        if self.eps_rate <= 0:
            raise ValueError(f"eps_rate must be > 0, got {self.eps_rate}")
        if self.rebirth_rate < 0:
            raise ValueError(f"rebirth_rate must be >= 0, got {self.rebirth_rate}")
        if not 0.0 < self.prior_alive < 1.0:
            raise ValueError(f"prior_alive must be in (0, 1), got {self.prior_alive}")
        for name in ("mu", "eta", "alpha", "gamma"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains non-finite values")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VHPEDParams:
        """Build parameters from a config mapping (configs/temporal/vhped.yaml).

        Scalar entries are broadcast: `mu`/`eta` to (P,), `alpha_self`/
        `gamma_self` to the diagonal and `alpha_cross`/`gamma_cross` to the
        off-diagonal.  Full arrays may instead be given as `alpha`/`gamma`.
        """
        required = ("num_predicates", "mu", "eta")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(
                f"V-HPED config missing {missing}; see configs/temporal/vhped.yaml"
            )
        p = int(cfg["num_predicates"])
        eye = np.eye(p)

        def _vec(key: str) -> np.ndarray:
            val = np.asarray(cfg[key], dtype=np.float64)
            return np.full(p, float(val)) if val.ndim == 0 else val

        def _mat(full: str, self_key: str, cross_key: str) -> np.ndarray:
            if full in cfg:
                return np.asarray(cfg[full], dtype=np.float64)
            return float(cfg.get(self_key, 0.0)) * eye + float(cfg.get(cross_key, 0.0)) * (1 - eye)

        return cls(
            mu=_vec("mu"),
            eta=_vec("eta"),
            alpha=_mat("alpha", "alpha_self", "alpha_cross"),
            gamma=_mat("gamma", "gamma_self", "gamma_cross"),
            beta=np.array([cfg["beta_vis"], cfg["beta_occ"], cfg["beta_unk"]], dtype=np.float64),
            rho=np.array([1.0, cfg["rho_occ"], cfg["rho_unk"]], dtype=np.float64),
            eps_rate=float(cfg["eps_rate"]),
            rebirth_rate=float(cfg["rebirth_rate"]),
            prior_alive=float(cfg["prior_alive"]),
            use_marks=bool(cfg.get("use_marks", True)),
        )

    def apply_ablation(
        self,
        use_visibility: bool = True,
        use_excitation: bool = True,
        use_cross_predicate: bool = True,
    ) -> VHPEDParams:
        """Return a copy with ablated components switched off.

        - use_visibility=False: rho = 1 in every regime, all betas = beta_vis.
        - use_excitation=False: alpha = gamma = 0 (no Hawkes terms).
        - use_cross_predicate=False: off-diagonal alpha, gamma = 0.
        """
        alpha, gamma = self.alpha.copy(), self.gamma.copy()
        beta, rho = self.beta.copy(), self.rho.copy()
        if not use_visibility:
            rho[:] = 1.0
            beta[:] = beta[0]
        if not use_excitation:
            alpha[:] = 0.0
            gamma[:] = 0.0
        elif not use_cross_predicate:
            alpha = np.diag(np.diag(alpha))
            gamma = np.diag(np.diag(gamma))
        return VHPEDParams(
            mu=self.mu.copy(), alpha=alpha, eta=self.eta.copy(), gamma=gamma,
            beta=beta, rho=rho, eps_rate=self.eps_rate,
            rebirth_rate=self.rebirth_rate, prior_alive=self.prior_alive,
            use_marks=self.use_marks,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (for run artifacts)."""
        return {
            "mu": self.mu.tolist(),
            "alpha": self.alpha.tolist(),
            "eta": self.eta.tolist(),
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
            "rho": self.rho.tolist(),
            "eps_rate": self.eps_rate,
            "rebirth_rate": self.rebirth_rate,
            "prior_alive": self.prior_alive,
            "use_marks": self.use_marks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VHPEDParams:
        """Inverse of to_dict."""
        return cls(**{k: d[k] for k in (
            "mu", "alpha", "eta", "gamma", "beta", "rho",
            "eps_rate", "rebirth_rate", "prior_alive", "use_marks",
        )})
