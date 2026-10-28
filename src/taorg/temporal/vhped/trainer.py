"""Maximum-likelihood fitting of V-HPED parameters (PRD FR-6, hypothesis H3).

Unsupervised: maximises the observation-stream likelihood of TRAINING videos
only; no ground-truth alive labels are used.  The keep/drop threshold is not
fit here — select it on validation data.

Constraints (research/derivations/vhped_model.md §6) are enforced by
reparameterisation; ablations are enforced by fixing / masking parameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from taorg.schemas.edges import FrameGraph
from taorg.temporal.vhped.likelihood import (
    DTYPE,
    encode_video,
    sequence_log_likelihood,
    stack_videos,
)
from taorg.temporal.vhped.params import VHPEDParams

logger = logging.getLogger(__name__)

_BETA_FLOOR = 1e-6


def _inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


@dataclass
class TrainerConfig:
    """Fitting hyperparameters (configs/temporal/vhped.yaml `trainer`).

    Attributes:
        num_iters: Adam iterations.
        lr: Adam learning rate.
        l2_cross: L2 penalty on off-diagonal alpha and all of gamma (per term).
        fit_rebirth: Whether rebirth_rate is trainable (else fixed at init).
        use_visibility / use_excitation / use_cross_predicate: ablation switches.
        fit_gamma: If False, the termination hazard ignores traces (gamma = 0).
        rho_unk_fixed: If set, pin rho_unk to this value (0 = exact censoring).
        fit_prior: If False, prior_alive stays at its initial value (diagnostic).
            fit_gamma=False with rho_unk_fixed=0 gives a LazyVHPED-exact model.
        seed: Torch seed for determinism.
    """

    num_iters: int = 200
    lr: float = 0.05
    l2_cross: float = 1e-3
    fit_rebirth: bool = False
    use_visibility: bool = True
    use_excitation: bool = True
    use_cross_predicate: bool = True
    fit_gamma: bool = True
    rho_unk_fixed: float | None = None
    fit_prior: bool = True
    seed: int = 0


@dataclass
class FitResult:
    """Output of fit_vhped."""

    params: VHPEDParams
    nll_history: list[float] = field(default_factory=list)
    num_terms: int = 0


class _Raw(torch.nn.Module):
    """Unconstrained parameters and their constrained view."""

    def __init__(self, init: VHPEDParams, cfg: TrainerConfig) -> None:
        super().__init__()
        p = init.num_predicates
        self.cfg = cfg
        t = lambda x: torch.nn.Parameter(torch.as_tensor(x, dtype=DTYPE).clone())  # noqa: E731
        self.mu = t(init.mu)
        self.eta = t(init.eta)
        self.alpha = t(init.alpha)
        self.gamma = t(init.gamma)
        self.beta = t([_inv_softplus(float(b) - _BETA_FLOOR) for b in init.beta])
        rho = np.clip(init.rho[1:], 1e-4, 1 - 1e-4)
        self.rho = t([_logit(float(r)) for r in rho])
        self.eps = t(_inv_softplus(init.eps_rate))
        self.prior = t(_logit(init.prior_alive))
        self.prior.requires_grad_(cfg.fit_prior)
        self.rebirth_fixed = init.rebirth_rate
        self.rebirth = t(_inv_softplus(max(init.rebirth_rate, 1e-8)))
        self.rebirth.requires_grad_(cfg.fit_rebirth)

        eye = torch.eye(p, dtype=DTYPE)
        if not cfg.use_excitation:
            mask = torch.zeros_like(eye)
        elif not cfg.use_cross_predicate:
            mask = eye
        else:
            mask = torch.ones_like(eye)
        self.register_buffer("cross_mask", mask)
        self.register_buffer("gamma_mask", mask if cfg.fit_gamma else torch.zeros_like(eye))
        self.register_buffer("offdiag", 1 - eye)
        if not cfg.use_visibility:
            self.rho.requires_grad_(False)

    def constrained(self) -> dict[str, torch.Tensor]:
        beta = F.softplus(self.beta) + _BETA_FLOOR
        if self.cfg.use_visibility:
            rho = torch.cat([torch.ones(1, dtype=DTYPE), torch.sigmoid(self.rho)])
            if self.cfg.rho_unk_fixed is not None:
                rho = torch.cat([rho[:2], torch.full((1,), float(self.cfg.rho_unk_fixed),
                                                     dtype=DTYPE)])
        else:
            rho = torch.ones(3, dtype=DTYPE)
            beta = beta[0].expand(3)
        rebirth = F.softplus(self.rebirth) if self.cfg.fit_rebirth else torch.as_tensor(
            self.rebirth_fixed, dtype=DTYPE)
        return {
            "mu": self.mu,
            "eta": self.eta,
            "alpha": self.alpha * self.cross_mask,
            "gamma": self.gamma * self.gamma_mask,
            "beta": beta,
            "rho": rho,
            "eps_rate": F.softplus(self.eps),
            "rebirth_rate": rebirth,
            "prior_alive": torch.sigmoid(self.prior),
        }

    def penalty(self) -> torch.Tensor:
        th = self.constrained()
        return self.cfg.l2_cross * (
            ((th["alpha"] * self.offdiag) ** 2).sum() + (th["gamma"] ** 2).sum()
        )

    def to_params(self, use_marks: bool) -> VHPEDParams:
        with torch.no_grad():
            th = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.constrained().items()}
        return VHPEDParams(
            mu=th["mu"], alpha=th["alpha"], eta=th["eta"], gamma=th["gamma"],
            beta=th["beta"], rho=th["rho"], eps_rate=float(th["eps_rate"]),
            rebirth_rate=float(th["rebirth_rate"]),
            prior_alive=float(th["prior_alive"]), use_marks=use_marks,
        )


def fit_vhped(
    train_videos: list[list[FrameGraph]],
    init: VHPEDParams,
    cfg: TrainerConfig | None = None,
    fps: float = 30.0,
) -> FitResult:
    """Fit V-HPED parameters by maximum likelihood on training videos.

    Args:
        train_videos: Observation streams (corrupted, as the model will see them).
        init: Initial parameters (e.g. the hand-set configuration).
        cfg: Trainer hyperparameters and ablation switches.
        fps: Frame rate for each video's first interval.

    Returns:
        FitResult with fitted parameters and per-iteration mean NLL.
    """
    cfg = cfg or TrainerConfig()
    if not train_videos:
        raise ValueError("fit_vhped needs at least one training video")
    torch.manual_seed(cfg.seed)

    encoded = stack_videos(
        [encode_video(v, init.num_predicates, fps, init.use_marks) for v in train_videos]
    )
    num_terms = sum(e.num_terms for e in encoded)
    if num_terms == 0:
        raise ValueError("Training videos contain no observed edges; nothing to fit")

    raw = _Raw(init, cfg)
    opt = torch.optim.Adam([p for p in raw.parameters() if p.requires_grad], lr=cfg.lr)
    history: list[float] = []
    for it in range(cfg.num_iters):
        opt.zero_grad()
        theta = raw.constrained()
        ll = sum(sequence_log_likelihood(e, theta) for e in encoded)
        nll = -ll / num_terms
        loss = nll + raw.penalty()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite V-HPED loss at iteration {it}")
        loss.backward()
        opt.step()
        history.append(float(nll.detach()))
        if it % 50 == 0:
            logger.info("V-HPED fit iter %d: mean NLL %.5f", it, history[-1])

    return FitResult(params=raw.to_params(init.use_marks), nll_history=history, num_terms=num_terms)
