"""Episode estimator (formulation §2-§5): gaps, hidden termination, misleading evidence,
recurrence, recovery from false termination, flicker, irregular timestamps, numerics,
determinism, causality."""

from __future__ import annotations

import math
import random

import pytest

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import TrackObservation, VisibilityState
from taorg.temporal import METHODS
from taorg.temporal.episode import EpisodeEstimator

K = EdgeKey(1, 3, 0)
VIS, OCC, UNK = VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE, VisibilityState.UNKNOWN


def rec(f: int, t: float, kind: str, key: EdgeKey = K) -> EdgeObservation:
    if kind == "D":
        return EdgeObservation(key, "v", f, t, 0.8, True)
    vis = {"V": VIS, "O": OCC, "U": UNK}[kind]
    return EdgeObservation(key, "v", f, t, 0.0, False, object_visibility=vis)


def stream(pattern: str, fps: float = 30.0, times=None, key=K, tracks=None) -> list[FrameGraph]:
    """pattern chars: D detection, V/O/U absence with flag, '-' no record."""
    out = []
    for f, ch in enumerate(pattern):
        t = times[f] if times is not None else f / fps
        edges = [] if ch == "-" else [rec(f, t, ch, key)]
        out.append(FrameGraph("v", f, t, tracks(f, t) if tracks else [], edges))
    return out


def run(model, frames) -> list[bool]:
    model.reset_video("v")
    res = [K in {o.edge_key for o in model.step(fr).edge_observations} for fr in frames]
    model.finalize_video()
    return res


BASE = dict(miss=0.05, false_pos=0.01, a_occ=0.98, a_unk=0.6, mu=0.3, end_threshold=0.2, birth_threshold=5.0,
            refractory_boost=0.0, refractory_tau=0.5)


def test_birth_needs_repeated_detections_and_temporary_gap_is_bridged():
    m = EpisodeEstimator(**BASE)
    # log((1-m)/f) = log(95) = 4.55 < 5 -> two detections needed for birth
    out = run(m, stream("DD" + "D" * 20 + "O" * 30 + "D" * 10))
    assert out[0] is False and out[1] is True
    assert all(out[22:52]), "a 1-s OCC gap must be bridged"
    assert m.episode_index(K) == 1


def test_hidden_termination_ends_eventually_and_monotonically():
    m = EpisodeEstimator(**BASE)
    m.reset_video("v")
    ps = []
    for fr in stream("D" * 30 + "O" * 600):
        m.step(fr)
        ps.append(m.belief(K))
    tail = [p for p in ps[30:] if not math.isnan(p)]
    assert all(b <= a + 1e-15 for a, b in zip(tail, tail[1:])), "Lemma 1: non-increasing without detections"
    assert math.isnan(ps[-1]), "termination must happen under a long OCC-only stretch"


def test_misleading_flags_after_termination_never_reactivate():
    m = EpisodeEstimator(**BASE)
    out = run(m, stream("D" * 40 + "V" * 10 + "O" * 200 + "U" * 100))
    first_off = out.index(False, 40)
    assert not any(out[first_off:]), "absences (any flag) can never re-open a terminated episode"
    assert m.episode_index(K) == 1


def test_genuine_recurrence_is_a_new_episode():
    m = EpisodeEstimator(**BASE)
    out = run(m, stream("D" * 40 + "V" * 30 + "D" * 40))
    assert out[39] and not out[60] and out[109]
    assert m.episode_index(K) == 2
    # recurrence detected after ceil(h / log(95)) = 2 detections
    assert out.index(True, 70) == 71


def test_recovery_from_false_termination():
    m = EpisodeEstimator(**BASE)
    # odds cap ~ Λ_det/(μΔ) = 95/0.01; four visible misses (Λ = 0.05/0.99 each) give odds ~0.06 < 0.2/0.8
    out = run(m, stream("D" * 40 + "V" * 4 + "D" * 40))
    assert not out[43], "a 4-frame visible miss run falsely terminates with these parameters"
    assert out[45] and out[-1] and m.episode_index(K) == 2, "recovered by a new episode after 2 detections"


def test_isolated_false_detections_do_not_create_episodes():
    m = EpisodeEstimator(**BASE)
    pat = "D" * 40 + "V" * 20 + ("DVVVVV" * 10) + "V" * 20
    out = run(m, stream(pat))
    assert m.episode_index(K) == 1 and not any(out[45:])


def test_refractory_raises_rebirth_threshold_only_after_termination():
    pat = "D" * 40 + "V" * 15 + "DD" + "V" * 10
    plain = EpisodeEstimator(**BASE)
    refr = EpisodeEstimator(**{**BASE, "refractory_boost": 10.0, "refractory_tau": 1.0})
    a, b = run(plain, stream(pat)), run(refr, stream(pat))
    assert a[56] and not b[56], "two detections re-birth without, but not with, the refractory term"
    assert b[1], "first birth (never terminated) is unaffected by the refractory term"


def test_irregular_timestamps_death_filter_is_partition_invariant():
    p = {**BASE, "a_occ": 1.0, "end_threshold": 0.02}        # OCC absence carries no evidence
    rng = random.Random(1)
    t_reg = [i / 30 for i in range(80)]
    t_irr = sorted(t_reg[:20] + [0.67 + rng.uniform(0, 1.9) for _ in range(30)])
    final = []
    for times in (t_reg, t_irr):
        m = EpisodeEstimator(**p)
        m.reset_video("v")
        n_det = 20
        for f, t in enumerate(times):
            m.step(FrameGraph("v", f, t, [], [rec(f, t, "D" if f < n_det else "O")]))
            if f == n_det - 1:
                p0, t0 = m.belief(K), t
        final.append((m.belief(K), p0, t0, times[-1]))
    for p_end, p0, t0, t1 in final:
        assert p_end == pytest.approx(p0 * math.exp(-p["mu"] * (t1 - t0)), rel=1e-9)


def test_numerical_stability_extremes():
    for mu in (0.01, 5.0):
        for m_ in (0.005, 0.5):
            for f_ in (0.0005, 0.2):
                m = EpisodeEstimator(**{**BASE, "mu": mu, "miss": m_, "false_pos": f_, "birth_threshold": 0.5})
                m.reset_video("v")
                for fr in stream("D" * 3000 + "O" * 3000 + "D" * 50 + "V" * 3000):
                    m.step(fr)
                    b = m.belief(K)
                    assert math.isnan(b) or 0.0 < b < 1.0
    with pytest.raises(ValueError):
        EpisodeEstimator(**{**BASE, "miss": 0.9, "false_pos": 0.2})
    with pytest.raises(ValueError):
        EpisodeEstimator(**{**BASE, "end_threshold": 0.7})
    with pytest.raises(ValueError):
        METHODS["episode_ctx"](**BASE)                      # context rung without fitted statistic


def test_determinism_and_causality():
    rng = random.Random(3)
    pat = "".join(rng.choice("DDDDVOU-") for _ in range(400))
    a = run(EpisodeEstimator(**BASE), stream(pat))
    b = run(EpisodeEstimator(**BASE), stream(pat))
    assert a == b
    alt = pat[:200] + "".join(rng.choice("DVO") for _ in range(200))
    c = run(EpisodeEstimator(**BASE), stream(alt))
    assert a[:200] == c[:200], "output at frame n must not depend on frames after n"


def _boxes(jump_at: int | None):
    def tracks(f, t):
        x = 100 + 60 * t                                    # smooth motion, 60 px/s
        if jump_at is not None and jump_at <= f < jump_at + 20:
            x += 300                                        # identity swapped to a far track
        return [TrackObservation("v", f, t, 1, 0, (x, 100, x + 60, 190)),
                TrackObservation("v", f, t, 3, 1, (300, 100, 360, 190))]
    return tracks


def test_identity_tempering_suppresses_ghost_births_but_not_ordinary_ones():
    p2 = METHODS["episode_id"](**BASE, sigma_id=0.05, r_id=1.0)
    ghost_pat = "-" * 100 + "D" * 10 + "-" * 50          # key only detected during the swap window
    out = run(p2, stream(ghost_pat, tracks=_boxes(100)))
    assert not any(out), "detections under an unreliable identity must not create an episode"
    p2b = METHODS["episode_id"](**BASE, sigma_id=0.05, r_id=1.0)
    out = run(p2b, stream(ghost_pat, tracks=_boxes(None)))
    assert any(out), "the same detections with a reliable identity do create an episode"


def test_transition_context_speeds_hidden_termination_only_with_partner():
    E = [[0.0] * 6 for _ in range(6)]
    E[0][1] = 0.8
    p = {**BASE, "sigma_id": 0.1, "r_id": 3.0, "kappa_c": 30.0, "transition": E}
    partner = EdgeKey(1, 3, 1)

    def frames(with_partner: bool):
        fs = stream("D" * 40 + "O" * 200)
        for f, fr in enumerate(fs):
            if with_partner and f >= 40:
                fr.edge_observations.append(rec(f, fr.timestamp, "D", partner))
        return fs
    a = run(METHODS["episode_ctx"](**p), frames(True))
    b = run(METHODS["episode_ctx"](**p), frames(False))
    assert a.index(False, 40) < b.index(False, 40)
