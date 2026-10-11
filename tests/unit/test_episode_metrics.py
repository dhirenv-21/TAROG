"""Episode metrics (prereg §6): matching, censoring, undefined cases, trivial-output guards."""

from __future__ import annotations

import math

import pytest

from taorg.metrics.episodes import episode_metrics, iou, match_episodes, runs

FPS = 30.0
N = 300
A, B = ("a",), ("b",)


def out_from(spec: dict, n: int = N) -> list[set]:
    """spec: key -> list of inclusive output intervals."""
    o = [set() for _ in range(n)]
    for k, ivs in spec.items():
        for s, e in ivs:
            for f in range(s, e + 1):
                o[f].add(k)
    return o


GT = {A: [(10, 99), (150, 249)]}


def m(spec, gt=GT, **kw):
    return episode_metrics(gt, out_from(spec), FPS, **kw)


def test_runs_iou_and_one_to_one_matching():
    assert runs([False, True, True, False, True]) == [(1, 2), (4, 4)]
    assert iou((0, 9), (5, 14)) == pytest.approx(5 / 15)
    assert iou((0, 9), (20, 29)) == 0.0
    # two halves of one GT episode cannot both match (IoU 0.5 each -> only one)
    assert len(match_episodes([(0, 9)], [(0, 4), (5, 9)], 0.5)) == 1


def test_perfect_output():
    r = m({A: GT[A]})
    assert r["J"] == 1.0 and r["frame_f1"] == 1.0 and r["episode_f1"] == 1.0
    assert r["frr"] == 0.0 and r["fpt_s"] == 0.0 and r["ghost_time_s"] == 0.0
    assert r["std_mean_s"] == pytest.approx(1 / FPS)            # drop right after the end
    assert r["rdd_mean_s"] == 0.0 and r["bridge_rate"] == 0.0
    assert r["missed_rate"] == 0.0 and r["fragmentation"] == 0.0


def test_always_off_and_always_on_cannot_win():
    off = m({})
    assert off["J"] == 0.0 and off["missed_rate"] == 1.0 and math.isnan(off["fragmentation"])
    on = m({A: [(0, N - 1)]})
    assert on["episode_f1"] == 0.0 and on["J"] < 0.5
    assert on["fpt_s"] == pytest.approx((N - 190) / FPS)
    assert on["bridge_rate"] == 1.0 and math.isnan(on["rdd_mean_s"])
    assert on["std_censored"] == 2.0


def test_false_reactivation_and_relapse_aware_termination_delay():
    # dip-and-return after the first episode: drop at 101, back on 105-120
    r = m({A: [(10, 100), (105, 120), (150, 249)]})
    assert r["frr"] == pytest.approx(0.5)                       # 1 of 2 evaluable terminations
    assert r["reactivation_frames"] == 16
    # first-drop delay would be 2 frames; sustained (>= 15 off) delay = 121 - 99 = 22 frames
    first_std = (121 - 99) / FPS
    second_std = 1 / FPS
    assert r["std_mean_s"] == pytest.approx((first_std + second_std) / 2)
    # no drop at all after the last episode -> censored, |D| + 1
    r2 = m({A: [(10, 99), (150, N - 1)]})
    assert r2["std_censored"] == 1.0
    assert r2["frr"] == 0.0                                     # 'no drop' is not a reactivation


def test_recurrence_delay_bridged_and_censored():
    r = m({A: [(10, 99), (160, 249)]})
    assert r["rdd_mean_s"] == pytest.approx(10 / FPS) and r["bridge_rate"] == 0.0
    r = m({A: [(10, 249)]})
    assert r["bridge_rate"] == 1.0 and math.isnan(r["rdd_mean_s"])
    r = m({A: [(10, 99)]})
    assert r["rdd_censored"] == 1.0 and r["rdd_mean_s"] == pytest.approx(100 / FPS)
    assert r["missed_rate"] == 0.5


def test_fragmentation_ghosts_and_undefined_cases():
    r = m({A: [(10, 50), (53, 99), (150, 249)], B: [(0, 29)]})
    assert r["fragmentation"] == pytest.approx(0.5)              # first episode in 2 pieces
    assert r["ghost_time_s"] == pytest.approx(30 / FPS)
    assert r["fpt_s"] == 0.0
    # short dead interval (< 10 frames) -> termination not evaluable
    r = episode_metrics({A: [(10, 99), (105, 200)]}, out_from({A: [(10, 99), (105, 200)]}), FPS)
    assert r["n_terminations"] == 1.0
    # open episode only -> FRR and STD undefined
    r = episode_metrics({A: [(10, N - 1)]}, out_from({A: [(10, N - 1)]}), FPS)
    assert math.isnan(r["frr"]) and math.isnan(r["std_mean_s"])
