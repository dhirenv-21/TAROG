"""Versioned expiry scheduling; callers establish the validity of deadlines.

This queue does not derive a model's expiry time or delete its posterior state.
Rescheduling invalidates old heap entries without changing scientific decisions.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey


@dataclass
class ExpiryQueue:
    """Min-heap with cancellation, replacement, and bounded stale-entry growth."""

    compact_ratio: int = 3
    compact_slack: int = 64
    _heap: list[tuple[float, int, EdgeKey]] = field(default_factory=list, init=False)
    _current: dict[EdgeKey, tuple[float, int]] = field(default_factory=dict, init=False)
    _serial: int = field(default=0, init=False)
    stale_pops: int = field(default=0, init=False)
    valid_pops: int = field(default=0, init=False)
    rebuilds: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.compact_ratio < 2 or self.compact_slack < 0:
            raise ValueError("compact_ratio must be >= 2 and compact_slack >= 0")

    def schedule(self, key: EdgeKey, deadline: float) -> None:
        """Replace a key's deadline (seconds); require a finite timestamp."""
        if not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        self._serial += 1
        self._current[key] = (deadline, self._serial)
        heapq.heappush(self._heap, (deadline, self._serial, key))
        self._compact()

    def __contains__(self, key: EdgeKey) -> bool:
        """Whether the key has a valid (current-generation) deadline."""
        return key in self._current

    def cancel(self, key: EdgeKey) -> None:
        """Invalidate a deadline, including any previously cancelled generations."""
        self._current.pop(key, None)
        self._compact()

    def pop_due(self, timestamp: float) -> list[EdgeKey]:
        """Consume valid deadlines <= timestamp, in deadline/insertion order."""
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        due = []
        while self._heap and self._heap[0][0] <= timestamp:
            deadline, serial, key = heapq.heappop(self._heap)
            if self._current.get(key) != (deadline, serial):
                self.stale_pops += 1
                continue
            del self._current[key]
            self.valid_pops += 1
            due.append(key)
        self._compact()
        return due

    def _compact(self) -> None:
        if len(self._heap) > self.compact_ratio * len(self._current) + self.compact_slack:
            self._heap = [(d, s, k) for k, (d, s) in self._current.items()]
            heapq.heapify(self._heap)
            self.rebuilds += 1

    def cancel_all(self) -> None:
        """Invalidate every deadline at once (instrumentation counters are kept)."""
        self._heap.clear()
        self._current.clear()

    def reset(self) -> None:
        """Clear deadlines and instrumentation for a new video."""
        self._heap.clear()
        self._current.clear()
        self._serial = self.stale_pops = self.valid_pops = self.rebuilds = 0

    def counters(self) -> dict[str, int]:
        """Current sizes and consumed-entry counts (rebuild removals excluded)."""
        return {"heap_size": len(self._heap), "scheduled_keys": len(self._current),
                "stale_pops": self.stale_pops, "valid_pops": self.valid_pops,
                "rebuilds": self.rebuilds}
