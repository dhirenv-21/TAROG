"""Tests for the Phase 5H-1 decision rule and predicate families."""

from taorg.hawkes.decision1 import decide_phase5h1, family_passes
from taorg.hawkes.families import FAMILIES, family_of


def st(m, lo, hi):
    return {"mean": m, "ci_lo": lo, "ci_hi": hi}


NULL = [0.001 * i for i in range(-5, 6)]  # max 0.005, median 0
FAM_OK = {"d_a": st(0.05, 0.01, 0.1), "d_b": st(0.04, 0.005, 0.08), "d_all": st(0.045, 0.02, 0.07),
          "null": NULL, "dur_gain": 0.2}
FAM_BAD = dict(FAM_OK, d_b=st(0.04, -0.01, 0.08))


def base(**kw):
    a = dict(d_cross=st(0.05, 0.03, 0.07), null_cross=NULL, d_time=st(0.04, 0.02, 0.06), null_time=NULL,
             dur_gain=0.2, n_discoveries=3, rho_real=0.5, rho_null=[0.1, 0.2], families={}, ratio=0.10)
    a.update(kw)
    return decide_phase5h1(**a)


def test_go():
    assert base()["label"] == "GO"


def test_narrow_requires_k1_and_a_family():
    r = base(d_time=st(0.01, 0.002, 0.02), families={"CONTACT_MANIPULATION": FAM_OK})
    assert r["label"] == "NARROW" and not r["K3"]
    assert base(d_time=st(0.01, 0.002, 0.02), families={"X": FAM_BAD})["label"] == "STOP"


def test_stop_without_k1():
    assert base(d_cross=st(0.004, 0.001, 0.007), families={"X": FAM_OK})["label"] == "STOP"


def test_family_rule_requires_both_halves_and_null():
    assert family_passes(FAM_OK, 0.1)
    assert not family_passes(FAM_BAD, 0.1)
    assert not family_passes(dict(FAM_OK, d_all=st(0.004, 0.001, 0.01)), 0.1)


def test_families_cover_vocab_once():
    names = [p for ps in FAMILIES.values() for p in ps]
    assert len(names) == len(set(names)) == 50
    assert family_of("hold") == "CONTACT_MANIPULATION" and family_of("next_to") == "SPATIAL"
