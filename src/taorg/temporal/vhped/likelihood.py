"""Differentiable V-HPED log-likelihood (PyTorch).

Mirrors model.VHPED.step exactly (with prune_below = 0) but vectorised over
all pairs of a video, so parameters can be fit by gradient ascent.  A unit
test asserts both implementations agree to 1e-9.

log p(o_{1:T}) = sum_n sum_{tracked (pair, p)} log c_n   (forward algorithm)

Shapes: T frames, N pairs, P predicates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from taorg.schemas.edges import FrameGraph, PairKey
from taorg.temporal.vhped.params import REGIME_INDEX, VHPEDParams
from taorg.temporal.vhped.visibility import pair_regime

DTYPE = torch.float64


@dataclass
class EncodedVideo:
    """Dense tensors for one video.

    Attributes:
        dt: Interval lengths in seconds, (T,) or per pair (T, N) after stacking.
        regime: Pair regime index per frame (0 vis, 1 occ, 2 unk), (T, N) long.
        observed: Observation indicators, (T, N, P) bool.
        marks: Event marks added to traces after each frame, (T, N, P).
        pairs: Pair keys in column order, length N.
    """

    dt: torch.Tensor
    regime: torch.Tensor
    observed: torch.Tensor
    marks: torch.Tensor
    pairs: list[PairKey]

    @property
    def num_terms(self) -> int:
        """Number of (frame, pair, predicate) evidence terms in the likelihood."""
        tracked = torch.cummax(self.observed.to(torch.int8), dim=0).values.bool()
        return int(tracked.sum())


def encode_video(
    frames: list[FrameGraph], num_predicates: int, fps: float, use_marks: bool
) -> EncodedVideo:
    """Convert a FrameGraph sequence into dense likelihood tensors.

    Pairs enter at their first observed edge, exactly as in VHPED.step.
    """
    pair_index: dict[PairKey, int] = {}
    for frame in frames:
        for obs in frame.edge_observations:
            if obs.observed and obs.pair_key not in pair_index:
                pair_index[obs.pair_key] = len(pair_index)

    t_len, n_pairs = len(frames), len(pair_index)
    dt = torch.empty(t_len, dtype=DTYPE)
    regime = torch.full((t_len, n_pairs), REGIME_INDEX[pair_regime([])], dtype=torch.long)
    observed = torch.zeros((t_len, n_pairs, num_predicates), dtype=torch.bool)
    marks = torch.zeros((t_len, n_pairs, num_predicates), dtype=DTYPE)

    prev_ts: float | None = None
    for t, frame in enumerate(frames):
        dt[t] = 1.0 / fps if prev_ts is None else frame.timestamp - prev_ts
        if dt[t] <= 0:
            raise ValueError(f"Non-increasing timestamp at frame {frame.frame_idx}")
        prev_ts = frame.timestamp

        by_pair: dict[PairKey, list] = {}
        for obs in frame.edge_observations:
            if obs.pair_key in pair_index:
                by_pair.setdefault(obs.pair_key, []).append(obs)
        for pk, obs_list in by_pair.items():
            n = pair_index[pk]
            regime[t, n] = REGIME_INDEX[pair_regime(obs_list)]
            for obs in obs_list:
                if obs.observed:
                    p = obs.edge_key.predicate_id
                    if not 0 <= p < num_predicates:
                        raise ValueError(
                            f"predicate_id {p} outside vocabulary [0, {num_predicates})"
                        )
                    observed[t, n, p] = True
                    m = obs.predicate_score if use_marks else 1.0
                    marks[t, n, p] = max(float(marks[t, n, p]), m)

    return EncodedVideo(dt=dt, regime=regime, observed=observed, marks=marks,
                        pairs=list(pair_index))


def stack_videos(videos: list[EncodedVideo]) -> list[EncodedVideo]:
    """Merge videos into one batch along the pair axis, padding to equal length.

    Pairs are conditionally independent, so the summed log-likelihood is
    unchanged.  Each pair column keeps its own intervals (dt becomes (T, N)).
    Padding steps use dt = 0 with no observations, which is an exact no-op:
    no decay, no transition, and both emission likelihoods equal 1.
    """
    videos = [v for v in videos if v.observed.shape[1] > 0]
    if not videos:
        return []
    t_max = max(v.observed.shape[0] for v in videos)

    def pad(x: torch.Tensor, value) -> torch.Tensor:
        extra = t_max - x.shape[0]
        if extra == 0:
            return x
        filler = torch.full((extra, *x.shape[1:]), value, dtype=x.dtype)
        return torch.cat([x, filler], dim=0)

    dts = []
    for v in videos:
        n = v.observed.shape[1]
        dt = v.dt.unsqueeze(1).expand(-1, n) if v.dt.dim() == 1 else v.dt
        dts.append(pad(dt, 0.0))
    return [EncodedVideo(
        dt=torch.cat(dts, dim=1),
        regime=torch.cat([pad(v.regime, 0) for v in videos], dim=1),
        observed=torch.cat([pad(v.observed, False) for v in videos], dim=1),
        marks=torch.cat([pad(v.marks, 0.0) for v in videos], dim=1),
        pairs=[pk for v in videos for pk in v.pairs],
    )]


def params_to_tensors(params: VHPEDParams) -> dict[str, torch.Tensor]:
    """Constrained-space parameters as float64 tensors."""
    t = lambda x: torch.as_tensor(x, dtype=DTYPE)  # noqa: E731
    return {
        "mu": t(params.mu), "alpha": t(params.alpha),
        "eta": t(params.eta), "gamma": t(params.gamma),
        "beta": t(params.beta), "rho": t(params.rho),
        "eps_rate": t(params.eps_rate), "rebirth_rate": t(params.rebirth_rate),
        "prior_alive": t(params.prior_alive),
    }


def sequence_log_likelihood(video: EncodedVideo, theta: dict[str, torch.Tensor]) -> torch.Tensor:
    """Exact forward-algorithm log-likelihood of one encoded video.

    Args:
        video: Output of encode_video.
        theta: Constrained parameters (see params_to_tensors).

    Returns:
        Scalar tensor log p(observations), differentiable w.r.t. theta.
    """
    t_len, n_pairs, n_pred = video.observed.shape
    traces = torch.zeros((n_pairs, n_pred), dtype=DTYPE)
    belief = torch.zeros((n_pairs, n_pred), dtype=DTYPE)
    tracked = torch.zeros((n_pairs, n_pred), dtype=torch.bool)
    total = torch.zeros((), dtype=DTYPE)

    for t in range(t_len):
        dt = video.dt[t]  # scalar, or (N,) for stacked videos
        dtc = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        reg = video.regime[t]
        obs = video.observed[t]

        traces = traces * torch.exp(-theta["beta"][reg] * dt).unsqueeze(1)
        lam = F.softplus(theta["mu"] + traces @ theta["alpha"])
        haz = F.softplus(theta["eta"] + traces @ theta["gamma"])

        born = obs & ~tracked
        belief = torch.where(born, theta["prior_alive"].expand_as(belief), belief)
        tracked = tracked | obs

        rho = theta["rho"][reg].unsqueeze(1)
        rate = haz + theta["rebirth_rate"]
        pi = theta["rebirth_rate"] / rate
        pred = pi + (belief - pi) * torch.exp(-rate * dtc)
        log_miss_alive = -rho * lam * dtc
        log_miss_dead = (-rho * theta["eps_rate"] * dtc).expand_as(lam)
        lik_alive = torch.where(obs, -torch.expm1(log_miss_alive), torch.exp(log_miss_alive))
        lik_dead = torch.where(obs, -torch.expm1(log_miss_dead), torch.exp(log_miss_dead))

        num_alive = pred * lik_alive
        evidence = num_alive + (1 - pred) * lik_dead
        valid = tracked & (evidence > 0)
        safe_ev = torch.where(valid, evidence, torch.ones_like(evidence))
        total = total + torch.where(valid, torch.log(safe_ev), torch.zeros_like(safe_ev)).sum()
        belief = torch.where(valid, num_alive / safe_ev, torch.where(tracked, pred, belief))

        traces = traces + video.marks[t]

    return total
