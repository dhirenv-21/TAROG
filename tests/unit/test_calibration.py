"""Edge cases for the frozen calibration metric (metric_definitions.md §6)."""

from __future__ import annotations

import numpy as np
import pytest

from taorg.metrics.calibration import CalibrationAccumulator, expected_calibration_error
from taorg.metrics.threshold_sweep import VideoScores


def _vs(scores, labels):
    s = np.array([scores], dtype=float)
    return VideoScores(gt=np.array([labels], dtype=bool), score=s,
                       corrupted=np.zeros_like(s, dtype=bool))


def test_no_units():
    assert CalibrationAccumulator().result() == {"ece": 0.0, "brier": 0.0, "num_units": 0}


def test_perfect():
    acc = CalibrationAccumulator()
    acc.add(_vs([1.0, 0.0, 1.0], [1, 0, 1]))
    assert acc.result()["ece"] == 0.0 and acc.result()["brier"] == 0.0


def test_constant_half():
    acc = CalibrationAccumulator()
    acc.add(_vs([0.5] * 4, [1, 0, 1, 0]))
    r = acc.result()
    assert r["ece"] == pytest.approx(0.0) and r["brier"] == pytest.approx(0.25)


def test_untracked_units_excluded():
    acc = CalibrationAccumulator()
    acc.add(_vs([-np.inf, 0.9, -np.inf], [1, 1, 0]))
    assert acc.result()["num_units"] == 1


def test_overconfident():
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([0.0, 0.0])) == pytest.approx(0.9)
