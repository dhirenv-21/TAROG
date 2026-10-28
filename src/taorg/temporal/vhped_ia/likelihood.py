"""Differentiable IA-VHPED log-likelihood (PyTorch); mirrors IAVHPED with prune_below = 0.

Reuses V-HPED's encoding and adds per-track identity scores so that the track
identity beliefs zeta are computed inside the graph — differentiable in the
jump calibration (a, b) of IA-v2.  Recovery (omega > 0) is not supported in
training.  A unit test checks equality with the NumPy model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from taorg.schemas.edges import FrameGraph
from taorg.temporal.vhped.likelihood import DTYPE, EncodedVideo, encode_video, stack_videos
from taorg.temporal.vhped_ia.identity import KAPPA_CLIP


@dataclass
class EncodedIAVideo:
    """V-HPED encoding + identity inputs.

    Attributes:
        kappa: identity score per frame and track, (T, M); 1.0 where not supplied.
        update: track present in the frame and not at its first appearance, (T, M) bool.
        pair_tracks: (N, 2) indices of each pair's subject / object track.
    """

    base: EncodedVideo
    kappa: torch.Tensor
    update: torch.Tensor
    pair_tracks: torch.Tensor


def encode_ia_video(frames: list[FrameGraph], num_predicates: int, fps: float, use_marks: bool,
                    recovery_rate: float = 0.0) -> EncodedIAVideo:
    if recovery_rate != 0.0:
        raise NotImplementedError("IA-VHPED training supports recovery_rate = 0 only")
    base = encode_video(frames, num_predicates, fps, use_marks)
    track_index: dict[int, int] = {}
    for f in frames:
        for t in f.track_observations:
            track_index.setdefault(t.track_id, len(track_index))
    for pk in base.pairs:
        for tid in (pk.subject_track_id, pk.object_track_id):
            track_index.setdefault(tid, len(track_index))
    t_len, m = len(frames), len(track_index)
    kappa = torch.ones((t_len, m), dtype=DTYPE)
    update = torch.zeros((t_len, m), dtype=torch.bool)
    seen: set[int] = set()
    for i, f in enumerate(frames):
        for t in f.track_observations:
            j = track_index[t.track_id]
            if t.track_id in seen:
                update[i, j] = True
                if t.identity_confidence is not None:
                    kappa[i, j] = float(t.identity_confidence)
            seen.add(t.track_id)
    pair_tracks = torch.tensor([[track_index[pk.subject_track_id], track_index[pk.object_track_id]]
                                for pk in base.pairs], dtype=torch.long).reshape(-1, 2)
    return EncodedIAVideo(base=base, kappa=kappa, update=update, pair_tracks=pair_tracks)


def stack_ia_videos(videos: list[EncodedIAVideo]) -> list[EncodedIAVideo]:
    """Batch along pairs and tracks; padding frames do not update identity."""
    videos = [v for v in videos if v.base.observed.shape[1] > 0]
    if not videos:
        return []
    (base,) = stack_videos([v.base for v in videos])
    t_max = base.observed.shape[0]
    kappas, updates, pts, offset = [], [], [], 0
    for v in videos:
        pad = t_max - v.kappa.shape[0]
        kappas.append(torch.cat([v.kappa, torch.ones((pad, v.kappa.shape[1]), dtype=DTYPE)]))
        updates.append(torch.cat([v.update, torch.zeros((pad, v.update.shape[1]),
                                                        dtype=torch.bool)]))
        pts.append(v.pair_tracks + offset)
        offset += v.kappa.shape[1]
    return [EncodedIAVideo(base=base, kappa=torch.cat(kappas, 1), update=torch.cat(updates, 1),
                           pair_tracks=torch.cat(pts, 0))]


def jump_prob(kappa: torch.Tensor, theta: dict[str, torch.Tensor]) -> torch.Tensor:
    """P(jump | kappa); raw 1 - kappa unless theta has the IA-v2 calibration (jump_a, jump_b)."""
    if "jump_a" not in theta:
        return 1.0 - kappa
    k = kappa.clamp(KAPPA_CLIP, 1.0 - KAPPA_CLIP)
    j = torch.sigmoid(theta["jump_a"] + theta["jump_b"] * (torch.log1p(-k) - torch.log(k)))
    return torch.where(kappa >= 1.0, torch.zeros_like(j), j)


def ia_log_likelihood(video: EncodedIAVideo, theta: dict[str, torch.Tensor],
                      posterior_weight: bool = True, identity_births: bool = True) -> torch.Tensor:
    """Exact forward log-likelihood of IA-VHPED for one encoded video.

    theta: V-HPED constrained parameters plus "foreign_rate", "foreign_birth" and,
    for IA-v2, "jump_a", "jump_b".
    """
    v = video.base
    t_len, n_pairs, n_pred = v.observed.shape
    traces = torch.zeros((n_pairs, n_pred), dtype=DTYPE)
    belief = torch.zeros((n_pairs, n_pred), dtype=DTYPE)
    tracked = torch.zeros((n_pairs, n_pred), dtype=torch.bool)
    z = torch.ones(video.kappa.shape[1], dtype=DTYPE)
    total = torch.zeros((), dtype=DTYPE)
    r = theta["rebirth_rate"]
    s_idx, o_idx = video.pair_tracks[:, 0], video.pair_tracks[:, 1]

    for t in range(t_len):
        jp = jump_prob(video.kappa[t], theta)
        z = torch.where(video.update[t], (1 - jp) * z + jp * (1 - z), z)
        zeta = z[s_idx] * z[o_idx]

        dt = v.dt[t]
        dtc = dt.unsqueeze(-1) if dt.dim() == 1 else dt
        reg = v.regime[t]
        obs = v.observed[t]

        traces = traces * torch.exp(-theta["beta"][reg] * dt).unsqueeze(1)
        lam = F.softplus(theta["mu"] + traces @ theta["alpha"])
        haz = F.softplus(theta["eta"] + traces @ theta["gamma"])

        born = obs & ~tracked
        a_prev = torch.where(born, theta["prior_alive"].expand_as(belief), belief)
        tracked = tracked | obs

        rho = theta["rho"][reg].unsqueeze(1)
        rate = haz + r
        pi = r / rate
        pred = pi + (a_prev - pi) * torch.exp(-rate * dtc)
        log_miss_alive = -rho * lam * dtc
        log_miss_dead = (-rho * theta["eps_rate"] * dtc).expand_as(lam)
        lik_alive = torch.where(obs, -torch.expm1(log_miss_alive), torch.exp(log_miss_alive))
        lik_dead = torch.where(obs, -torch.expm1(log_miss_dead), torch.exp(log_miss_dead))
        num_alive = pred * lik_alive
        evidence = num_alive + (1 - pred) * lik_dead
        valid = tracked & (evidence > 0)
        safe_ev = torch.where(valid, evidence, torch.ones_like(evidence))
        log_c = torch.where(valid, torch.log(safe_ev), torch.zeros_like(safe_ev))
        post = torch.where(valid, num_alive / safe_ev, pred)

        q1 = (-torch.expm1(-rho * theta["foreign_rate"] * dtc)).expand_as(lam)
        log_q = torch.where(obs, torch.log(q1.clamp_min(1e-300)), torch.log1p(-q1))
        log_q = torch.where(tracked, log_q, torch.zeros_like(log_q))

        log_a = torch.log(zeta.clamp_min(1e-300)) + log_c.sum(1)
        log_b = torch.log1p(-zeta.clamp(max=1.0)) + log_q.sum(1)
        log_p = torch.logaddexp(log_a, log_b)
        w = torch.exp(log_a - log_p) if posterior_weight else zeta
        w_ = w.unsqueeze(1)
        a_bar = torch.where(born, theta["foreign_birth"].expand_as(pred), pred) \
            if identity_births else pred
        belief = torch.where(tracked, w_ * post + (1 - w_) * a_bar, belief)
        total = total + log_p.sum()
        traces = traces + w_ * v.marks[t]
    return total
