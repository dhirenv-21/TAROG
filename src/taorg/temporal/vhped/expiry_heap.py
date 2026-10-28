"""Closed-form untouched-pair dynamics and a versioned expiry min-heap.

Concern 4 of Architecture.md §6.  See research/derivations/lazy_expiry.md.

Untouched pair = a tracked pair with no record in a frame (regime UNKNOWN).
Under the LAZY-EXACTNESS CONDITIONS
    (L1) rho_unk = 0        absence of any record carries no evidence
    (L2) gamma  = 0         hazard does not depend on the excitation traces
the eager per-frame update of an untouched pair reduces to the two-state
continuous-time Markov chain transition with constant rates h_p = softplus(eta_p)
(alive -> dead) and r (dead -> alive), whose transition matrices form a
semigroup: P(d1) P(d2) = P(d1 + d2).  Hence the belief after any sequence of
untouched frames depends only on elapsed time:

    a(t0 + d) = pi + (a(t0) - pi) * exp(-(h + r) d),   pi = r / (h + r)

which is monotone in d, so the time at which it crosses a level is unique and
closed-form.  That is what makes exact expiry scheduling valid.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from taorg.schemas.edges import EdgeKey, ExpiryEntry


def ctmc_belief(a0: float, h: float, r: float, d: float) -> float:
    """Belief after `d` seconds untouched (closed form).  h, r >= 0 (1/s)."""
    k = h + r
    if k <= 0.0 or d <= 0.0:
        return a0
    pi = r / k
    return pi + (a0 - pi) * math.exp(-k * d)


def crossing_delay(a0: float, h: float, r: float, level: float) -> float:
    """Delay d > 0 after which the untouched belief crosses `level`, or inf.

    "Crosses" means a(d) moves strictly to the other side of `level` for all
    larger delays; the belief is monotone so there is at most one crossing.
    Returns inf if the belief never reaches the other side.
    """
    k = h + r
    if k <= 0.0:
        return math.inf
    pi = r / k
    above0 = a0 >= level
    if above0 == (pi >= level):
        # Asymptote on the same side: no strict crossing (incl. a0 == pi).
        return math.inf
    # Solve pi + (a0 - pi) e^{-k d} = level.
    ratio = (a0 - pi) / (level - pi)
    if ratio <= 0.0:
        return math.inf
    return max(math.log(ratio) / k, 0.0)


@dataclass
class HeapCounters:
    """Instrumentation (Rules §9: report stale pops and heap growth)."""

    pushes: int = 0
    pops: int = 0
    stale_pops: int = 0
    rebuilds: int = 0
    max_size: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class VersionedHeap:
    """Min-heap of ExpiryEntry with lazy invalidation by version number.

    An entry is valid iff its version equals the current version of its
    owner (supplied by the caller at pop time).  Stale entries are skipped
    when popped and the heap is rebuilt when stale entries dominate.
    """

    rebuild_factor: float = 4.0
    _heap: list[tuple[float, int, ExpiryEntry]] = field(default_factory=list)
    _seq: int = 0
    _next_check: int = 64
    counters: HeapCounters = field(default_factory=HeapCounters)

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, time: float, edge_key: EdgeKey, version: int) -> None:
        # Sequence number breaks ties deterministically (insertion order).
        heapq.heappush(self._heap, (time, self._seq, ExpiryEntry(time, edge_key, version)))
        self._seq += 1
        self.counters.pushes += 1
        self.counters.max_size = max(self.counters.max_size, len(self._heap))

    def pop_due(self, before: float, current_version) -> list[ExpiryEntry]:
        """Pop all valid entries with expiry_time < before, in time order.

        Args:
            before: Strict upper bound on expiry time.
            current_version: Callable EdgeKey -> int | None (None = owner gone).
        """
        due = []
        while self._heap and self._heap[0][0] < before:
            _, _, entry = heapq.heappop(self._heap)
            self.counters.pops += 1
            if current_version(entry.edge_key) != entry.version:
                self.counters.stale_pops += 1
                continue
            due.append(entry)
        return due

    def maybe_rebuild(self, current_version) -> None:
        """Drop stale entries once the heap has doubled since the last check.

        The O(|heap|) scan runs only when the size passes a doubling
        threshold, so its amortized cost per push is O(1).
        """
        if len(self._heap) <= self._next_check:
            return
        live = [item for item in self._heap
                if current_version(item[2].edge_key) == item[2].version]
        if len(self._heap) > self.rebuild_factor * max(len(live), 1):
            self._heap = live
            heapq.heapify(self._heap)
            self.counters.rebuilds += 1
        self._next_check = max(64, 2 * len(self._heap))

    def clear(self) -> None:
        self._heap.clear()
        self._next_check = 64
