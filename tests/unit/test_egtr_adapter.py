"""EGTR adapter (runs only in the EGTR environment with the checkpoint present; skipped elsewhere)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SRC = Path("D:/TAORG_runs/egtr/src")
CKPT = Path("D:/TAORG_runs/egtr/ckpt")
IMG = sorted(Path("D:/TAORG_runs/egtr/vg_subset/images").glob("*.jpg"))[:1] if Path("D:/TAORG_runs/egtr/vg_subset/images").exists() else []


def _env_ok() -> bool:
    try:
        import transformers
        return transformers.__version__ == "4.18.0" and SRC.exists() and CKPT.exists() and bool(IMG)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _env_ok(), reason="EGTR environment/checkpoint not available")


@pytest.fixture(scope="module")
def predictor():
    from taorg.adapters.egtr import EGTRPredictor
    return EGTRPredictor()


def test_predict_shapes_and_vocab(predictor):
    from PIL import Image
    rgb = np.asarray(Image.open(IMG[0]).convert("RGB"))
    g = predictor.predict([rgb])[0]
    assert g.boxes.shape == (200, 4) and g.pairs.shape == (100, 2) and g.rel_scores.shape == (100, 50)
    assert (g.pairs[:, 0] != g.pairs[:, 1]).all()                         # no self-pairs
    assert (np.diff(g.pair_scores) <= 1e-12).all()                        # ranked
    assert len(predictor.class_names) == 150 and len(predictor.predicate_names) == 50


def test_person_relations_consistent_with_full_scores(predictor):
    from PIL import Image

    from taorg.adapters.egtr import MAPPED, PREDICATE_MAP
    rgb = np.asarray(Image.open(IMG[0]).convert("RGB"))
    out = predictor.raw([rgb])
    pr = predictor.person_relations(out, 0, rgb.shape[:2])
    n = len(pr.obj_scores)
    assert pr.scores.shape == (n, n, len(MAPPED)) and np.allclose(np.diagonal(pr.scores, axis1=0, axis2=1), 0)
    assert all(predictor.class_names[c] in ("boy", "child", "girl", "guy", "kid", "lady", "man", "men", "people",
                                            "person", "player", "skier", "woman") for c in pr.classes)
    assert set(PREDICATE_MAP) == set(MAPPED)
