"""Simulated tracker identity-continuity scores for controlled ID-switch studies.

A real tracker reports an association confidence for every track update; it
tends to be low when a label is re-assigned to a different object.  This module
simulates such a score from an ID-switch CorruptionManifest so that the
informativeness of the signal is an explicit experimental factor:

    normal frames      kappa ~ U(normal_low, 1.0)
    jump frames        kappa ~ U(normal_low - s * drop, 1.0 - s * drop)
    false alarms       with prob false_alarm_prob a normal frame draws a jump-frame score

`s = informativeness` in [0, 1]: s = 0 makes jump frames indistinguishable from
normal ones (uninformative tracker), s = 1 makes them clearly lower.  A jump
frame is the first frame of a swap and the first frame after it (the label
changes object at both).  The score uses only what a tracker would know at that
frame; ground-truth relationships are never used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from taorg.noise.manifest import CorruptionManifest
from taorg.schemas.edges import FrameGraph


@dataclass(frozen=True)
class IdentityScoreConfig:
    """Parameters of the simulated identity-continuity score."""

    informativeness: float = 1.0
    normal_low: float = 0.9
    drop: float = 0.85
    false_alarm_prob: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= self.informativeness <= 1.0:
            raise ValueError(f"informativeness must be in [0, 1], got {self.informativeness}")
        if not 0.0 <= self.normal_low <= 1.0 or self.normal_low - self.drop < 0.0:
            raise ValueError("need 0 <= normal_low - drop and normal_low <= 1")


def jump_frames(manifest: CorruptionManifest) -> dict[int, set[int]]:
    """frame index -> track labels whose object assignment changes at that frame."""
    out: dict[int, set[int]] = {}
    for e in manifest.entries:
        if e.corruption_type != "id_switch" or not e.track_ids:
            continue
        for f in (e.start_frame, e.end_frame + 1):
            out.setdefault(f, set()).update(e.track_ids)
    return out


def annotate_identity_confidence(
    frames: list[FrameGraph], manifest: CorruptionManifest, seed: int,
    config: IdentityScoreConfig | None = None,
) -> list[FrameGraph]:
    """Return copies of `frames` whose TrackObservations carry identity_confidence."""
    cfg = config or IdentityScoreConfig()
    rng = random.Random(seed)
    jumps = jump_frames(manifest)
    shift = cfg.informativeness * cfg.drop

    def draw(low_regime: bool) -> float:
        if low_regime:
            return rng.uniform(cfg.normal_low - shift, 1.0 - shift)
        return rng.uniform(cfg.normal_low, 1.0)

    out = []
    for f in frames:
        jumped = jumps.get(f.frame_idx, set())
        tracks = []
        for t in f.track_observations:
            low = t.track_id in jumped or rng.random() < cfg.false_alarm_prob
            tracks.append(replace(t, identity_confidence=draw(low)))
        out.append(FrameGraph(f.video_id, f.frame_idx, f.timestamp, track_observations=tracks,
                              edge_observations=list(f.edge_observations)))
    return out
