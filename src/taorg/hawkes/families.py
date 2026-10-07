"""Phase 5H-1 predicate families (semantic rule; fixed before any family-level effect is computed).

Rule: a predicate is assigned by what physically links subject and object.

- SPATIAL: a static geometric relation.
- MOTION: relative movement.
- ATTENTION_COMMUNICATION: perception or communication, with no physical contact.
- VEHICLE_SUPPORT: the subject is supported by / mounts / dismounts the object.
- CONTACT_MANIPULATION: hand/body contact or manipulation.

This grouping is the project's own (not an official VidOR taxonomy) and every one of the 50 VidOR
predicates is assigned exactly once.
"""

from __future__ import annotations

FAMILIES: dict[str, list[str]] = {
    "SPATIAL": ["above", "beneath", "behind", "in_front_of", "inside", "next_to"],
    "MOTION": ["away", "towards", "chase"],
    "ATTENTION_COMMUNICATION": ["watch", "speak_to", "point_to", "wave", "wave_hand_to", "shout_at", "smell"],
    "VEHICLE_SUPPORT": ["ride", "drive", "get_on", "get_off", "lean_on"],
    "CONTACT_MANIPULATION": ["hold", "grab", "release", "touch", "push", "pull", "lift", "carry", "press", "pat",
                             "caress", "hit", "kick", "bite", "throw", "squeeze", "knock", "cut", "open", "close",
                             "clean", "use", "feed", "play(instrument)", "lick", "kiss", "hug", "hold_hand_of",
                             "shake_hand_with"],
}
_INDEX = {p: f for f, ps in FAMILIES.items() for p in ps}


def family_of(predicate: str) -> str:
    """Family of a VidOR predicate; raises KeyError for unknown predicates (fail fast)."""
    return _INDEX[predicate]
