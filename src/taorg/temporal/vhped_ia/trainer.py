"""Maximum-likelihood fitting of IA-VHPED (training videos only).

Extends the V-HPED trainer's parameterisation with phi = softplus(raw) and
b = sigmoid(raw); all V-HPED ablation switches (TrainerConfig) still apply.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from taorg.schemas.edges import FrameGraph
from taorg.temporal.vhped.likelihood import DTYPE
from taorg.temporal.vhped.trainer import TrainerConfig, _inv_softplus, _logit, _Raw
from taorg.temporal.vhped_ia.likelihood import encode_ia_video, ia_log_likelihood, stack_ia_videos
from taorg.temporal.vhped_ia.params import IAParams

logger = logging.getLogger(__name__)


@dataclass
class IAFitResult:
    params: IAParams
    nll_history: list[float] = field(default_factory=list)
    num_terms: int = 0


class _IARaw(_Raw):
    def __init__(self, init: IAParams, cfg: TrainerConfig) -> None:
        super().__init__(init.base, cfg)
        self.phi = torch.nn.Parameter(torch.tensor(_inv_softplus(max(init.foreign_rate, 1e-6)),
                                                   dtype=DTYPE))
        self.fb = torch.nn.Parameter(torch.tensor(_logit(init.foreign_birth), dtype=DTYPE))
        self.calibrate = init.jump_calibration is not None
        if self.calibrate:
            a, b = init.jump_calibration
            self.ja = torch.nn.Parameter(torch.tensor(a, dtype=DTYPE))
            self.jb = torch.nn.Parameter(torch.tensor(_inv_softplus(max(b, 1e-6)), dtype=DTYPE))

    def constrained(self) -> dict[str, torch.Tensor]:
        th = super().constrained()
        th["foreign_rate"] = F.softplus(self.phi)
        th["foreign_birth"] = torch.sigmoid(self.fb)
        if self.calibrate:
            th["jump_a"] = self.ja
            th["jump_b"] = F.softplus(self.jb)
        return th


def fit_ia_vhped(train_videos: list[list[FrameGraph]], init: IAParams,
                 cfg: TrainerConfig | None = None, fps: float = 30.0) -> IAFitResult:
    """Fit IA-VHPED parameters by maximum likelihood."""
    cfg = cfg or TrainerConfig()
    if not train_videos:
        raise ValueError("fit_ia_vhped needs at least one training video")
    torch.manual_seed(cfg.seed)
    p = init.base
    encoded = stack_ia_videos([encode_ia_video(v, p.num_predicates, fps, p.use_marks,
                                               init.recovery_rate) for v in train_videos])
    num_terms = sum(e.base.num_terms for e in encoded)
    if num_terms == 0:
        raise ValueError("Training videos contain no observed edges; nothing to fit")

    raw = _IARaw(init, cfg)
    opt = torch.optim.Adam([q for q in raw.parameters() if q.requires_grad], lr=cfg.lr)
    history: list[float] = []
    for it in range(cfg.num_iters):
        opt.zero_grad()
        theta = raw.constrained()
        ll = sum(ia_log_likelihood(e, theta, init.posterior_weight, init.identity_births)
                 for e in encoded)
        nll = -ll / num_terms
        loss = nll + raw.penalty()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite IA-VHPED loss at iteration {it}")
        loss.backward()
        opt.step()
        history.append(float(nll.detach()))
        if it % 50 == 0:
            logger.info("IA-VHPED fit iter %d: mean NLL %.5f", it, history[-1])

    with torch.no_grad():
        th = raw.constrained()
        fitted = IAParams(base=raw.to_params(p.use_marks),
                          foreign_rate=float(th["foreign_rate"]),
                          foreign_birth=float(th["foreign_birth"]),
                          recovery_rate=init.recovery_rate,
                          posterior_weight=init.posterior_weight,
                          identity_births=init.identity_births,
                          jump_calibration=(float(th["jump_a"]), float(th["jump_b"]))
                          if raw.calibrate else None)
    if math.isnan(fitted.foreign_rate):
        raise FloatingPointError("Fitted foreign_rate is NaN")
    return IAFitResult(params=fitted, nll_history=history, num_terms=num_terms)
