"""End-to-end Phase 3 pilot on a FAKE Action Genome tree (tiny settings)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from taorg.data.action_genome import ActionGenomeConfig, load_action_genome
from taorg.experiments.pilot import PilotData, run_pilot, split_train
from tests.fixtures.fake_action_genome import write_fake_action_genome

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tiny():
    cfg = yaml.safe_load((REPO / "configs/experiments/pilot_action_genome.yaml").read_text(
        encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = [42, 123]
    cfg["occlusion"]["gap_lengths"] = [1, 3]
    cfg["thresholds"] = [0.1, 0.5, 0.9]
    cfg["training"]["trainer"]["num_iters"] = 3
    cfg["val_fraction"] = 0.25
    return cfg


def _data(tmp_path) -> PilotData:
    write_fake_action_genome(tmp_path, n_train=8, n_test=4, seed=2)
    ag = ActionGenomeConfig(root=tmp_path)
    train, vocab, _ = load_action_genome(ag, "train")
    test, _, _ = load_action_genome(ag, "test")
    fit, val = split_train(train, 0.25, 0)
    return PilotData(fit, val, test, vocab.num_predicates, "fake_ag")


def test_pilot_runs_and_summarizes(tmp_path, tiny):
    res = run_pilot(tiny, _data(tmp_path), workers=1)
    assert set(res["fitted"][42]) == {"vt_hmm_fit", "vhped_novis_fit", "vhped_self_fit",
                                      "vhped_full_fit"}
    assert len(res["cells"]) == 2 * 2 * 2
    assert len(res["test_table"]) == 2 * 2 * len(tiny["methods"])
    names = {(c["a"], c["b"]) for c in res["comparisons"]}
    assert ("gated_ema", "ema") in names and ("vhped_full_fit", "gated_ema") in names
    for row in res["aggregate"]:
        assert 0.0 <= row["persistence_f1"][0] <= 1.0


def test_pilot_is_deterministic_and_parallel_matches(tmp_path, tiny):
    data = _data(tmp_path)
    a = run_pilot(tiny, data, workers=1)
    b = run_pilot(tiny, data, workers=2)
    assert a["aggregate"] == b["aggregate"]
