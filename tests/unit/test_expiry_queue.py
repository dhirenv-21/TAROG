"""Scheduler correctness against a dictionary oracle, including reused identities."""

import math
import random

import pytest

from taorg.schemas.edges import EdgeKey
from taorg.temporal.expiry_queue import ExpiryQueue
from taorg.utils.seed import set_seed


def test_reschedule_cancel_and_reuse():
    q = ExpiryQueue()
    k, other = EdgeKey(1, 2, 0), EdgeKey(2, 3, 0)
    q.schedule(k, 2)
    q.schedule(k, 4)
    q.schedule(other, 3)
    assert q.pop_due(2) == []
    q.cancel(k)
    q.schedule(k, 5)
    assert q.pop_due(4) == [other]
    assert q.pop_due(5) == [k]
    assert q.counters()["valid_pops"] == 2
    assert q.counters()["stale_pops"] == 2


def test_randomized_dictionary_equivalence():
    set_seed(51001)
    q = ExpiryQueue(compact_slack=8)
    oracle = {}
    serial = 0
    for tick in range(2000):
        k = EdgeKey(random.randrange(20), 99, 0)
        if random.random() < 0.2:
            q.cancel(k)
            oracle.pop(k, None)
        else:
            serial += 1
            deadline = tick + random.randrange(30)
            q.schedule(k, deadline)
            oracle[k] = (deadline, serial)
        expected = sorted((d, s, key) for key, (d, s) in oracle.items() if d <= tick)
        assert q.pop_due(tick) == [key for _, _, key in expected]
        for _, _, key in expected:
            del oracle[key]
        c = q.counters()
        assert c["heap_size"] <= 3 * c["scheduled_keys"] + 8
    assert q.pop_due(3000) == [k for k, _ in sorted(oracle.items(), key=lambda item: item[1])]


def test_growth_reset_and_validation():
    q = ExpiryQueue(compact_slack=0)
    k = EdgeKey(1, 2, 0)
    for t in range(100):
        q.schedule(k, t)
    assert q.counters()["rebuilds"] > 0
    q.reset()
    assert all(v == 0 for v in q.counters().values())
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            q.schedule(k, bad)
        with pytest.raises(ValueError):
            q.pop_due(bad)
