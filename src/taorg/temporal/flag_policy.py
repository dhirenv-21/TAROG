"""Endpoint-invalidation flag policy (research/taorg_downstream_preregistration.md §5).

Before the wrapped method processes frame t, every key held in its memory whose subject OR object
is a flagged track at t is DELETED (as if never seen). Unrelated keys are untouched. A deleted key
can only return through a new observation, after which the method's normal onset dynamics apply.

Supported wrapped methods (declared internal-state use; the methods themselves are unchanged):
    FixedWindow          -> ``_edge_history``
    GatedEMAHysteresis   -> ``_inner._edge_scores`` and ``_on``
"""

from __future__ import annotations

from typing import Mapping, Union

from taorg.schemas.edges import EdgeKey, FrameGraph
from taorg.temporal.fixed_window import FixedWindow
from taorg.temporal.gated_variants import GatedEMAHysteresis

Wrapped = Union[FixedWindow, GatedEMAHysteresis]


class EndpointInvalidation:
    """Wrap a temporal method; ``flags[frame_idx]`` is the set of flagged track ids at that frame."""

    def __init__(self, inner: Wrapped, flags: Mapping[int, frozenset[int] | set[int]]):
        if not isinstance(inner, (FixedWindow, GatedEMAHysteresis)):
            raise TypeError(f"EndpointInvalidation does not support {type(inner).__name__}")
        self.inner = inner
        self.flags = flags
        self.n_deleted = 0

    def _memory_keys(self) -> list[EdgeKey]:
        if isinstance(self.inner, FixedWindow):
            return list(self.inner._edge_history)
        return list(self.inner._inner._edge_scores)

    def _delete(self, k: EdgeKey) -> None:
        if isinstance(self.inner, FixedWindow):
            self.inner._edge_history.pop(k, None)
        else:
            self.inner._inner._edge_scores.pop(k, None)
            self.inner._on.discard(k)
        self.n_deleted += 1

    def reset_video(self, video_id: str) -> None:
        """Reset the wrapped method."""
        self.inner.reset_video(video_id)
        self.n_deleted = 0

    def step(self, frame: FrameGraph) -> FrameGraph:
        """Invalidate keys touching flagged endpoints, then run the wrapped method unchanged."""
        bad = self.flags.get(frame.frame_idx)
        if bad:
            for k in self._memory_keys():
                if k.subject_track_id in bad or k.object_track_id in bad:
                    self._delete(k)
        return self.inner.step(frame)

    def finalize_video(self) -> None:
        """Delegate."""
        self.inner.finalize_video()

    def state_dict(self) -> dict:
        """Wrapped configuration plus the policy name."""
        return {**self.inner.state_dict(), "flag_policy": "endpoint_invalidation", "n_deleted": self.n_deleted}
