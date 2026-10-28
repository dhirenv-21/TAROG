"""Lazy V-HPED: touched-pair updates + per-pair wake-up heap (Phase 5, ADSA core).

Produces the same keep/drop decisions and alive scores as the eager
reference `VHPED` (randomized equivalence harness in
tests/property/test_lazy_equivalence.py) while doing state-maintenance work
only for

    k_touch  pairs with a record in the frame (observed or absent), and
    k_wake   pair wake-ups that fall due,

instead of every active pair every frame.  Exactness requires the
lazy-exactness conditions of expiry_heap.py (rho_unk = 0, gamma = 0).  If
they do not hold, the model falls back to eager updates (still exact, no
savings) and records `lazy_exact = False`.

Heap discipline (research/derivations/lazy_expiry.md §4):
    - one authoritative wake-up per pair: the earliest future time at which
      any of its tracked predicates crosses the output threshold or the prune
      level, assuming the pair stays untouched;
    - a wake-up may be EARLY (pair touched since it was scheduled): on pop the
      pair is re-validated from the closed form and rescheduled, so early
      wake-ups are safe; superseded entries are skipped via a wake id
      (version) check;
    - push only when the new wake-up is earlier than the pending one, so a
      touch costs at most one push.

Per-frame order (matches eager semantics):
    1. touched pairs: materialize the closed-form untouched evolution up to the
       previous frame time, run the eager frame step, refresh output-set
       membership, schedule a wake-up if earlier than the pending one;
    2. pop wake-ups with time < frame time; re-validate those pairs at the
       frame time (apply crossings / prunes) and reschedule;
    3. emit the output set (enumeration cost O(|S|) counted separately).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph, PairKey
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.vhped import dynamics
from taorg.temporal.vhped.expiry_heap import VersionedHeap, crossing_delay, ctmc_belief
from taorg.temporal.vhped.model import VHPED, _PairState
from taorg.temporal.vhped.params import REGIME_INDEX

logger = logging.getLogger(__name__)

_UNK = REGIME_INDEX[VisibilityState.UNKNOWN]


@dataclass
class LazyCounters:
    """Work counters for the complexity evaluation."""

    frames: int = 0
    pair_steps: int = 0                   # eager frame steps actually executed
    eager_equivalent_pair_steps: int = 0  # steps an eager implementation would run
    materializations: int = 0
    wakeups: int = 0                      # valid wake-ups processed
    early_wakeups: int = 0                # wake-ups that found no crossing due
    output_edges: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class LazyVHPED(VHPED):
    """V-HPED with lazy materialization and wake-up scheduling."""

    _t_last: dict[PairKey, float] = field(default_factory=dict, init=False, repr=False)
    _wake: dict[PairKey, tuple[float, int]] = field(default_factory=dict, init=False, repr=False)
    _alive_set: set[EdgeKey] = field(default_factory=set, init=False, repr=False)
    _heap: VersionedHeap = field(default_factory=VersionedHeap, init=False, repr=False)
    counters: LazyCounters = field(default_factory=LazyCounters, init=False)
    lazy_exact: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        p = self.params
        self.lazy_exact = bool(p.rho[_UNK] == 0.0 and not np.any(p.gamma))
        if not self.lazy_exact:
            logger.warning(
                "LazyVHPED: lazy-exactness conditions violated (rho_unk=%.3g, any gamma=%s); "
                "falling back to eager updates of untouched pairs",
                p.rho[_UNK], bool(np.any(p.gamma)),
            )
        self._hazard = [float(x) for x in dynamics.softplus(p.eta)]  # valid since gamma == 0
        self._wake_seq = 0

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def reset_video(self, video_id: str) -> None:
        super().reset_video(video_id)
        self._t_last.clear()
        self._wake.clear()
        self._alive_set.clear()
        self._heap.clear()

    def step(self, frame: FrameGraph) -> FrameGraph:
        self.counters.frames += 1
        self.counters.eager_equivalent_pair_steps += len(self._pairs)
        prev_ts = self._last_timestamp
        dt = self._interval(frame.timestamp)
        ts = frame.timestamp
        by_pair, observed_map, absent_map = self._parse_frame(frame)

        touched = [pk for pk in by_pair if pk in self._pairs] if self.lazy_exact \
            else list(self._pairs)
        for pk, obs_list in by_pair.items():
            if pk not in self._pairs and any(o.observed for o in obs_list):
                self._pairs[pk] = self._new_pair()
                self._t_last[pk] = ts if prev_ts is None else prev_ts
                touched.append(pk)

        # 1. touched pairs
        for pk in touched:
            state = self._pairs[pk]
            if prev_ts is not None:
                self._materialize(pk, state, prev_ts)
            self.counters.pair_steps += 1
            if not self._update_pair(state, by_pair.get(pk, []), dt, pk=pk):
                self._delete_pair(pk)
                continue
            self._t_last[pk] = ts
            self._refresh(pk, state)
            if self.lazy_exact:
                self._schedule(pk, ts + self._next_delay(state), keep_earlier=True)

        # 2. due wake-ups of untouched (or recently touched) pairs
        if self.lazy_exact:
            for entry in self._heap.pop_due(ts, self._wake_id):
                self._revalidate(PairKey(entry.edge_key.subject_track_id,
                                         entry.edge_key.object_track_id), ts)
            self._heap.maybe_rebuild(self._wake_id)

        # 3. emit
        edges = self._emit_lazy(frame, ts, observed_map, absent_map)
        self.counters.output_edges += len(edges)
        return FrameGraph(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            track_observations=frame.track_observations,
            edge_observations=edges,
        )

    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update(method="vhped_lazy", lazy_exact=self.lazy_exact,
                 counters=self.counters.to_dict(), heap=self._heap.counters.to_dict())
        return d

    def alive_probability(self, edge_key: EdgeKey) -> float:
        pk = PairKey(edge_key.subject_track_id, edge_key.object_track_id)
        state = self._pairs.get(pk)
        if state is None or not state.tracked[edge_key.predicate_id]:
            return 0.0
        return self._belief_at(pk, state, edge_key.predicate_id, self._last_timestamp)

    # ------------------------------------------------------------------
    # Closed-form helpers
    # ------------------------------------------------------------------

    def _belief_at(self, pk: PairKey, state: _PairState, p: int, t: float | None) -> float:
        a0 = float(state.belief[p])
        if t is None or not self.lazy_exact:
            return a0
        return ctmc_belief(a0, self._hazard[p], self.params.rebirth_rate, t - self._t_last[pk])

    def _materialize(self, pk: PairKey, state: _PairState, t: float) -> None:
        """Apply the closed-form untouched evolution from t_last to t."""
        d = t - self._t_last[pk]
        if d <= 0.0 or not self.lazy_exact:
            return
        self.counters.materializations += 1
        state.traces = dynamics.decay_traces(state.traces, float(self.params.beta[_UNK]), d)
        r = self.params.rebirth_rate
        for p in np.flatnonzero(state.tracked).tolist():
            state.belief[p] = ctmc_belief(float(state.belief[p]), self._hazard[p], r, d)
        self._t_last[pk] = t

    def _next_delay(self, state: _PairState) -> float:
        """Earliest crossing delay (threshold or prune level) from the current state."""
        r = self.params.rebirth_rate
        best = math.inf
        for p in np.flatnonzero(state.tracked).tolist():
            a0, h = float(state.belief[p]), self._hazard[p]
            best = min(best, crossing_delay(a0, h, r, self.score_threshold))
            if self.prune_below > 0.0 and a0 >= self.prune_below:
                best = min(best, crossing_delay(a0, h, r, self.prune_below))
        return best

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _wake_id(self, key: EdgeKey) -> int | None:
        w = self._wake.get(PairKey(key.subject_track_id, key.object_track_id))
        return None if w is None else w[1]

    def _schedule(self, pk: PairKey, time: float, keep_earlier: bool) -> None:
        """Make `time` the pair's wake-up unless an earlier pending one may stay."""
        pending = self._wake.get(pk)
        if keep_earlier and pending is not None and pending[0] <= time:
            return  # early wake-up already pending: it will re-validate
        if not math.isfinite(time):
            self._wake.pop(pk, None)
            return
        self._wake_seq += 1
        self._wake[pk] = (time, self._wake_seq)
        self._heap.push(time, EdgeKey(pk.subject_track_id, pk.object_track_id, -1), self._wake_seq)

    def _revalidate(self, pk: PairKey, ts: float) -> None:
        """Apply all crossings due by `ts` for a pair and reschedule its wake-up."""
        state = self._pairs.get(pk)
        if state is None:
            return
        self.counters.wakeups += 1
        changed = False
        r = self.params.rebirth_rate
        d = ts - self._t_last[pk]
        for p in np.flatnonzero(state.tracked).tolist():
            b = ctmc_belief(float(state.belief[p]), self._hazard[p], r, d)
            key = EdgeKey(pk.subject_track_id, pk.object_track_id, p)
            if self.prune_below > 0.0 and b < self.prune_below:
                state.tracked[p] = False
                state.belief[p] = 0.0
                self._alive_set.discard(key)
                changed = True
            elif b >= self.score_threshold:
                changed |= key not in self._alive_set
                self._alive_set.add(key)
            else:
                changed |= key in self._alive_set
                self._alive_set.discard(key)
        if not changed:
            self.counters.early_wakeups += 1
        if not state.tracked.any():
            self._delete_pair(pk)
            return
        self._materialize(pk, state, ts)
        self._wake.pop(pk, None)
        self._schedule(pk, ts + self._next_delay(state), keep_earlier=False)

    def _refresh(self, pk: PairKey, state: _PairState) -> None:
        """Output-set membership of a just-updated pair."""
        s, o = pk.subject_track_id, pk.object_track_id
        tracked, belief, thr = state.tracked, state.belief, self.score_threshold
        for p in range(self.params.num_predicates):
            key = EdgeKey(s, o, p)
            if tracked[p] and belief[p] >= thr:
                self._alive_set.add(key)
            elif key in self._alive_set:
                self._alive_set.discard(key)

    def _delete_pair(self, pk: PairKey) -> None:
        del self._pairs[pk]
        self._t_last.pop(pk, None)
        self._wake.pop(pk, None)
        for p in range(self.params.num_predicates):
            self._alive_set.discard(EdgeKey(pk.subject_track_id, pk.object_track_id, p))

    def _emit_lazy(
        self,
        frame: FrameGraph,
        ts: float,
        observed_map: dict[EdgeKey, EdgeObservation],
        absent_map: dict[EdgeKey, EdgeObservation],
    ) -> list[EdgeObservation]:
        out: list[EdgeObservation] = []
        for key in self._alive_set:
            pk = PairKey(key.subject_track_id, key.object_track_id)
            belief = self._belief_at(pk, self._pairs[pk], key.predicate_id, ts)
            src = observed_map.get(key) or absent_map.get(key)
            if src is not None:
                out.append(EdgeObservation(
                    edge_key=key, video_id=frame.video_id, frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp, predicate_score=belief, observed=src.observed,
                    subject_visibility=src.subject_visibility,
                    object_visibility=src.object_visibility,
                    source_model=src.source_model if src.observed else "vhped",
                    source_checkpoint=src.source_checkpoint if src.observed else "",
                ))
            else:
                out.append(EdgeObservation(
                    edge_key=key, video_id=frame.video_id, frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp, predicate_score=belief, observed=False,
                    source_model="vhped",
                ))
        return out
