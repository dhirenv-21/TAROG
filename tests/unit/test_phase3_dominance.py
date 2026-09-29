"""Tests for the exploratory frontier-dominance classification."""

import numpy as np

from taorg.experiments.phase3.dominance import bootstrap_dominance, classify, set_dominates

REF = 40.0


def test_strict_dominance():
    g = [(0.9, 2.0), (0.95, 5.0)]
    m = [(0.8, 2.0), (0.9, 6.0)]
    assert set_dominates(g, m, REF)
    assert not set_dominates(m, g, REF)


def test_identical_frontiers_do_not_dominate():
    f = [(0.8, 2.0), (0.9, 5.0)]
    assert not set_dominates(f, list(f), REF)


def test_uncovered_point_breaks_dominance():
    g = [(0.9, 2.0)]
    m = [(0.8, 2.0), (0.99, 10.0)]  # high-F1 point of m is not covered
    assert not set_dominates(g, m, REF)


def test_points_outside_box_ignored():
    g = [(0.9, 2.0)]
    m = [(0.8, 2.0), (1.0, 45.0)]  # m's second point lies beyond the reference delay
    assert set_dominates(g, m, REF)
    assert not set_dominates([(1.0, 50.0)], m, REF)  # g has nothing in the box


def test_empty_m_in_box():
    assert set_dominates([(0.5, 3.0)], [(1.0, 41.0)], REF)


def test_bootstrap_fraction():
    rng = np.random.default_rng(0)
    n = 30
    Fg = np.full((1, n), 0.9)
    Dg = np.full((1, n), 2.0)
    Fm = np.full((1, n), 0.5)
    Dm = np.full((1, n), 2.0)
    idx = rng.integers(0, n, size=(50, n))
    assert bootstrap_dominance(Fg, Dg, Fm, Dm, idx, REF) == 1.0
    assert bootstrap_dominance(Fm, Dm, Fg, Dg, idx, REF) == 0.0


def test_classify():
    yes = {"dominates": True, "p_dom": 1.0}
    weak = {"dominates": True, "p_dom": 0.6}
    no = {"dominates": False, "p_dom": 0.0}
    comps = ["ema", "hmm"]
    assert classify({("C0", 1): {"ema": yes, "hmm": yes}}, comps) == "A"
    assert classify({("C0", 1): {"ema": weak, "hmm": yes}}, comps) == "B"
    assert classify({("C0", 1): {"ema": yes, "hmm": yes}, ("C0", 2): {"ema": no, "hmm": yes}}, comps) == "B"
    assert classify({("C0", 1): {"ema": yes, "hmm": no}}, comps) == "C"
