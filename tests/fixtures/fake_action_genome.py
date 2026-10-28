"""Writes a tiny FAKE Action Genome annotation tree in the ASSUMED format.

Mirrors the layout documented in taorg.data.action_genome so the loader and
pilot can be tested without the real dataset.  It is not real data and says
nothing about the real files' format — scripts/inspect_action_genome.py
checks that.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np

OBJECT_CLASSES = ["person", "bag", "cup/glass/bottle", "phone/camera", "table"]
# relationship_classes.txt spellings deliberately differ from the pickle
# spellings for some names to exercise name normalisation.
RELATIONSHIP_FILE = ["lookingat", "notlookingat", "unsure", "infrontof", "holding",
                     "touching", "drinkingfrom", "notcontacting"]
REL_PICKLE = {"lookingat": "looking_at", "notlookingat": "not_looking_at",
              "infrontof": "in_front_of", "drinkingfrom": "drinking_from",
              "notcontacting": "not_contacting"}
ATTENTION = ["looking_at", "not_looking_at", "unsure"]


def write_fake_action_genome(root: Path, n_train: int = 6, n_test: int = 4, seed: int = 0) -> Path:
    """Create <root>/annotations/* and return root."""
    rng = random.Random(seed)
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    (ann / "object_classes.txt").write_text("\n".join(OBJECT_CLASSES) + "\n", encoding="utf-8")
    (ann / "relationship_classes.txt").write_text("\n".join(RELATIONSHIP_FILE) + "\n",
                                                  encoding="utf-8")
    persons: dict = {}
    objects: dict = {}
    for v in range(n_train + n_test):
        split = "train" if v < n_train else "test"
        vid = f"V{v:04d}.mp4"
        n_frames = rng.randint(18, 30)
        frame_no = rng.randint(0, 20)
        classes = rng.sample(OBJECT_CLASSES[1:], k=rng.randint(2, 3))
        plan = {}
        for c in classes:
            end = rng.randint(8, n_frames + 5)          # may outlive the video
            hidden = set(range(s := rng.randint(3, n_frames - 4), s + rng.randint(0, 3)))
            plan[c] = (end, hidden, rng.choice(["holding", "touching", "drinking_from"]))
        for t in range(n_frames):
            frame_no += rng.randint(5, 30)
            key = f"{vid}/{frame_no:06d}.png"
            empty_person = rng.random() < 0.05
            persons[key] = {"bbox": np.zeros((0, 4)) if empty_person
                            else np.array([[10.0, 10.0, 50.0, 90.0]]),
                            "bbox_mode": "xyxy"}
            objs = []
            for c in classes:
                end, hidden, contact = plan[c]
                visible = t not in hidden
                alive = t <= end
                objs.append({
                    "class": c,
                    "bbox": (1.0, 2.0, 3.0, 4.0) if visible else None,
                    "visible": visible,
                    "attention_relationship": [rng.choice(ATTENTION)] if visible else None,
                    "spatial_relationship": ["in_front_of"] if visible else None,
                    "contacting_relationship": ([contact] if alive else ["not_contacting"])
                    if visible else None,
                    "metadata": {"tag": key, "set": split},
                })
            objects[key] = objs
    with open(ann / "person_bbox.pkl", "wb") as f:
        pickle.dump(persons, f)
    with open(ann / "object_bbox_and_relationship.pkl", "wb") as f:
        pickle.dump(objects, f)
    return root
