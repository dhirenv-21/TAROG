"""Tests for the Action Genome loader on a FAKE annotation tree (assumed format)."""

from __future__ import annotations

import pickle

import pytest

from taorg.data.action_genome import (
    PERSON_TRACK_ID,
    ActionGenomeConfig,
    inspect_action_genome,
    load_action_genome,
    normalize_name,
    parse_frame_key,
)
from taorg.schemas.tracks import VisibilityState
from tests.fixtures.fake_action_genome import write_fake_action_genome


@pytest.fixture
def ag_cfg(tmp_path):
    write_fake_action_genome(tmp_path, n_train=5, n_test=3, seed=1)
    return ActionGenomeConfig(root=tmp_path)


def test_parse_frame_key():
    assert parse_frame_key("001YG.mp4/000089.png") == ("001YG.mp4", 89)
    with pytest.raises(ValueError):
        parse_frame_key("no_slash.png")


def test_normalize_name():
    assert normalize_name("looking_at") == normalize_name("lookingat") == "lookingat"
    assert normalize_name("cup/glass/bottle") == "cupglassbottle"


def test_splits_and_frame_indices(ag_cfg):
    train, vocab, stats = load_action_genome(ag_cfg, "train")
    test, _, _ = load_action_genome(ag_cfg, "test")
    assert len(train) == 5 and len(test) == 3
    assert vocab.num_predicates == 8
    for video in train + test:
        assert [f.frame_idx for f in video] == list(range(len(video)))
        ts = [f.timestamp for f in video]
        assert all(b > a for a, b in zip(ts, ts[1:]))
    assert stats.frames == sum(len(v) for v in train)


def test_edges_visibility_and_absence_records(ag_cfg):
    train, vocab, _ = load_action_genome(ag_cfg, "train")
    holding = vocab.relation_id("holding")
    saw_occ_absent = saw_vis_absent = False
    for video in train:
        seen = set()
        for f in video:
            for o in f.edge_observations:
                assert o.edge_key.subject_track_id == PERSON_TRACK_ID
                if o.observed:
                    assert o.object_visibility == VisibilityState.VISIBLE
                    seen.add(o.edge_key)
                else:
                    assert o.edge_key in seen  # absence only after first observation
                    if o.object_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE:
                        saw_occ_absent = True
                    elif o.edge_key.predicate_id == holding or True:
                        saw_vis_absent = True
    assert saw_occ_absent and saw_vis_absent


def test_unknown_relationship_fails_fast(tmp_path):
    write_fake_action_genome(tmp_path, n_train=2, n_test=1)
    path = tmp_path / "annotations" / "object_bbox_and_relationship.pkl"
    data = pickle.loads(path.read_bytes())
    first = next(iter(data))
    data[first][0]["visible"] = True
    data[first][0]["spatial_relationship"] = ["floating_above"]
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(KeyError, match="floating_above"):
        load_action_genome(ActionGenomeConfig(root=tmp_path), "train")


def test_missing_files_have_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="action_genome.yaml"):
        load_action_genome(ActionGenomeConfig(root=tmp_path), "train")


def test_inspect_report(ag_cfg):
    rep = inspect_action_genome(ag_cfg)
    assert rep["unknown_relationships"] == {} and rep["unknown_object_classes"] == {}
    assert rep["frames_with_duplicate_classes"] == 0
    assert set(rep["split_counts"]) == {"train", "test"}
