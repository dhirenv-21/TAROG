"""Frozen adapter contract between base VidSGG models and the temporal layer.

Phase 6: every base model (STTran, DSG-DETR, TEMPURA, ...) plugs in by
producing canonical FrameGraph videos through this protocol.  Temporal
methods never see model internals (Architecture.md §5).

Contract (frozen 2026-10-28; changes must be recorded in Memory.md):
    - One list[FrameGraph] per video, frames in time order,
      frame_idx == position in the list, strictly increasing timestamps.
    - Observed edges: EdgeObservation(observed=True, predicate_score in [0, 1]).
    - For a pair present in a frame, predicates previously observed on it but
      not predicted now appear as observed=False records carrying the frame's
      visibility (so visible absence can be told from a missing pair).
    - Pairs absent from a frame have no records (temporal methods treat them
      as UNKNOWN; UNKNOWN never silently becomes VISIBLE).
    - source_model / source_checkpoint set on every observed record.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from taorg.schemas.edges import FrameGraph


@runtime_checkable
class PredictionAdapter(Protocol):
    """Source of canonical prediction streams for one dataset split."""

    source_model: str
    source_checkpoint: str

    def videos(self) -> Iterator[list[FrameGraph]]:
        """Yield one video at a time as a list of FrameGraphs."""
        ...


def validate_video(frames: list[FrameGraph]) -> None:
    """Raise ValueError if a video violates the adapter contract."""
    prev = None
    for i, f in enumerate(frames):
        if f.frame_idx != i:
            raise ValueError(f"{f.video_id}: frame_idx {f.frame_idx} at position {i}")
        if prev is not None and f.timestamp <= prev:
            raise ValueError(f"{f.video_id}: non-increasing timestamp at frame {i}")
        prev = f.timestamp
        for o in f.edge_observations:
            if o.observed and not 0.0 <= o.predicate_score <= 1.0:
                raise ValueError(f"{f.video_id}: score {o.predicate_score} outside [0, 1]")
