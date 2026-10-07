"""Synthetic presentation scenario: a person holds a cup, it is briefly hidden, then put down.

SYNTHETIC SCENARIO WITH CONTROLLED OBSERVATIONS. Nothing here is detected from pixels.
Two files are produced and kept separate:

* TRUTH        — what really happens in the scene (used ONLY to draw the animation and the
                 "scene truth" timeline row; never given to a model).
* OBSERVATIONS — what a hypothetical relationship detector reports per frame (observed or not,
                 a score, and a visibility flag). This is built from the truth plus an explicit,
                 hand-written list of observation failures, and it is the ONLY model input.
"""

from __future__ import annotations

FPS = 30
N_FRAMES = 360                      # 12 s

PERSON, CUP, TABLE = 1, 2, 3
HOLD, ON = 0, 1                     # predicate ids
PREDICATES = {HOLD: "holding", ON: "on"}
NODES = {PERSON: "person", CUP: "cup", TABLE: "table"}

# ---------------------------------------------------------------- truth (scene events)
TRUTH = {
    "pick_up_frame": 30,            # person picks the cup up
    "release_frame": 240,           # person puts the cup down on the table (genuine termination)
    "occluder": [105, 135],         # a board passes in front of the cup (frames, end exclusive)
    "relations": [
        {"subject": PERSON, "object": CUP, "predicate": HOLD, "start": 30, "end": 240},
        {"subject": CUP, "object": TABLE, "predicate": ON, "start": 240, "end": N_FRAMES},
    ],
}

# ---------------------------------------------------------------- observation failures (hand-written)
# Each entry: frames [start, end) during which the detector does NOT report person-holding-cup
# although it is true, and the visibility flag reported for that absence.
FAILURES = [
    {"start": 105, "end": 135, "visibility": "OCCLUDED_OR_UNRELIABLE",
     "why": "cup hidden behind the passing board (1.0 s)"},
    {"start": 175, "end": 178, "visibility": "VISIBLE",
     "why": "detector flicker: cup visible but relation missed for 3 frames"},
    {"start": 195, "end": 203, "visibility": "UNKNOWN",
     "why": "tracker unsure (motion blur): visibility unknown for 8 frames"},
]
SCORE = 0.85                         # constant detector score when a relation is reported (demo setting)


def truth_alive(rel: dict, f: int) -> bool:
    """True if relation ``rel`` holds in the scene at frame f."""
    return rel["start"] <= f < rel["end"]


def build_observations() -> list[dict]:
    """Per-frame detector reports (the model input). Deterministic.

    Every frame carries a record for both edges once the objects are in view: either
    ``observed=True`` with a score, or an absence record with a visibility flag.
    """
    frames = []
    for f in range(N_FRAMES):
        edges = []
        for rel in TRUTH["relations"]:
            alive = truth_alive(rel, f)
            observed, vis = alive, "VISIBLE"
            if alive and rel["predicate"] == HOLD:
                for fail in FAILURES:
                    if fail["start"] <= f < fail["end"]:
                        observed, vis = False, fail["visibility"]
            edges.append({"subject": rel["subject"], "object": rel["object"], "predicate": rel["predicate"],
                          "observed": observed, "score": SCORE if observed else 0.0, "visibility": vis})
        frames.append({"frame": f, "t": round(f / FPS, 6), "edges": edges})
    return frames
