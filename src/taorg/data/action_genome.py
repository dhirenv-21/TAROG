"""Action Genome annotation loader -> canonical FrameGraph streams.

ASSUMED FILE FORMAT — must be verified on the real files with
`python scripts/inspect_action_genome.py --root <AG root>` before results are
interpreted (Rules §7: verify dataset properties from the dataset itself).
The assumptions below follow the public Action Genome toolkit layout:

    <root>/<annotation_dir>/person_bbox.pkl
        dict  frame_key -> {"bbox": array (k, 4), ...}      k = 0 if no person
    <root>/<annotation_dir>/object_bbox_and_relationship.pkl
        dict  frame_key -> list of object dicts with keys
              "class"                    object class name
              "visible"                  bool
              "attention_relationship"   list[str] | None
              "spatial_relationship"     list[str] | None
              "contacting_relationship"  list[str] | None
              "metadata"                 {"set": "train" | "test", ...}
    <root>/<annotation_dir>/object_classes.txt          one name per line
    <root>/<annotation_dir>/relationship_classes.txt    one name per line
    frame_key = "<video_id>.mp4/<frame_number>.png"

Canonical mapping (decisions recorded in Memory.md):
    - subject = the annotated person, track_id 0
    - object  = one track per object class, track_id = 1 + class index
      (duplicate instances of a class in one frame are counted and the first kept)
    - predicate_id = index in relationship_classes.txt
    - frame_idx = position among the video's ANNOTATED frames (AG annotates a
      sparse subset of frames); timestamp = frame_number / fps
    - object "visible" == False  -> OCCLUDED_OR_UNRELIABLE
      person with an empty bbox   -> OCCLUDED_OR_UNRELIABLE
      track not listed in a frame -> UNKNOWN (never silently VISIBLE)
    - a predicate seen earlier on a pair but not annotated now gets an absence
      record (observed=False) carrying the frame's visibility
    - boxes are NOT loaded (no temporal method uses them; box format unverified)

Class/relationship names are matched after lower-casing and removing
non-alphanumerics, so e.g. "lookingat" and "looking_at" match.
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph
from taorg.schemas.tracks import VisibilityState

logger = logging.getLogger(__name__)

PERSON_TRACK_ID = 0
RELATION_FIELDS = ("attention_relationship", "spatial_relationship", "contacting_relationship")
_FRAME_KEY = re.compile(r"^(?P<video>[^/]+)/(?P<frame>\d+)\.\w+$")


@dataclass
class ActionGenomeConfig:
    """Loader configuration (configs/datasets/action_genome.yaml).

    Attributes:
        root: Dataset root directory.
        annotation_dir: Sub-directory holding the pickles and class lists.
        fps: Frame rate used to convert frame numbers to seconds.  UNVERIFIED
            per video; only relative intervals matter for the temporal methods.
        min_frames: Skip videos with fewer annotated frames.
    """

    root: Path
    annotation_dir: str = "annotations"
    fps: float = 30.0
    min_frames: int = 2

    @property
    def ann(self) -> Path:
        return Path(self.root) / self.annotation_dir


@dataclass
class LoadStats:
    """Counters reported with every load so data quirks are never silent."""

    videos: int = 0
    frames: int = 0
    skipped_short_videos: int = 0
    duplicate_class_instances: int = 0
    invisible_with_relations: int = 0
    frames_missing_person_entry: int = 0
    observed_edges: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def normalize_name(name: str) -> str:
    """Lower-case and strip non-alphanumerics: 'looking_at' -> 'lookingat'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Action Genome file missing: {path}. Download the AG annotations and set "
            f"`root`/`annotation_dir` in configs/datasets/action_genome.yaml"
        )
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_pickle(path: Path) -> Any:
    """Load an AG pickle.  Pickles execute code on load: only use official AG files."""
    if not path.exists():
        raise FileNotFoundError(
            f"Action Genome file missing: {path}. See configs/datasets/action_genome.yaml"
        )
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def parse_frame_key(key: str) -> tuple[str, int]:
    """'001YG.mp4/000089.png' -> ('001YG.mp4', 89)."""
    m = _FRAME_KEY.match(key)
    if m is None:
        raise ValueError(
            f"Unexpected Action Genome frame key {key!r}; expected '<video>.mp4/<frame>.png'"
        )
    return m.group("video"), int(m.group("frame"))


@dataclass
class Vocabulary:
    object_classes: list[str]
    relationship_classes: list[str]
    _obj: dict[str, int] = field(init=False, repr=False)
    _rel: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._obj = {normalize_name(n): i for i, n in enumerate(self.object_classes)}
        self._rel = {normalize_name(n): i for i, n in enumerate(self.relationship_classes)}

    def object_id(self, name: str) -> int:
        try:
            return self._obj[normalize_name(name)]
        except KeyError:
            raise KeyError(f"Object class {name!r} not in object_classes.txt") from None

    def relation_id(self, name: str) -> int:
        try:
            return self._rel[normalize_name(name)]
        except KeyError:
            raise KeyError(f"Relationship {name!r} not in relationship_classes.txt") from None

    @property
    def num_predicates(self) -> int:
        return len(self.relationship_classes)


def load_vocabulary(cfg: ActionGenomeConfig) -> Vocabulary:
    return Vocabulary(
        object_classes=_read_lines(cfg.ann / "object_classes.txt"),
        relationship_classes=_read_lines(cfg.ann / "relationship_classes.txt"),
    )


def _video_split(frames: list[list[dict]], video: str) -> str | None:
    sets = {o["metadata"]["set"] for objs in frames for o in objs
            if isinstance(o.get("metadata"), dict) and "set" in o["metadata"]}
    if len(sets) > 1:
        raise ValueError(f"Video {video} has objects from several splits: {sorted(sets)}")
    return next(iter(sets), None)


def load_action_genome(
    cfg: ActionGenomeConfig, split: str
) -> tuple[list[list[FrameGraph]], Vocabulary, LoadStats]:
    """Load one AG split ('train' or 'test') as canonical FrameGraph videos.

    Returns:
        (videos sorted by video id, vocabulary, load statistics)
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    vocab = load_vocabulary(cfg)
    persons: dict = load_pickle(cfg.ann / "person_bbox.pkl")
    objects: dict = load_pickle(cfg.ann / "object_bbox_and_relationship.pkl")

    by_video: dict[str, list[tuple[int, str]]] = {}
    for key in objects:
        video, frame_no = parse_frame_key(key)
        by_video.setdefault(video, []).append((frame_no, key))

    stats = LoadStats()
    videos: list[list[FrameGraph]] = []
    for video in sorted(by_video):
        keys = sorted(by_video[video])
        if _video_split([objects[k] or [] for _, k in keys], video) != split:
            continue
        if len(keys) < cfg.min_frames:
            stats.skipped_short_videos += 1
            continue
        videos.append(_build_video(video, keys, persons, objects, vocab, cfg.fps, stats))
        stats.videos += 1
        stats.frames += len(keys)

    logger.info("Loaded Action Genome %s: %s", split, stats.to_dict())
    return videos, vocab, stats


def _build_video(video: str, keys: list[tuple[int, str]], persons: dict, objects: dict,
                 vocab: Vocabulary, fps: float, stats: LoadStats) -> list[FrameGraph]:
    VIS, OCC, UNK = (VisibilityState.VISIBLE, VisibilityState.OCCLUDED_OR_UNRELIABLE,
                     VisibilityState.UNKNOWN)
    seen: dict[int, set[int]] = {}  # object track -> predicates seen so far
    frames: list[FrameGraph] = []
    for pos, (frame_no, key) in enumerate(keys):
        ts = frame_no / fps
        person = persons.get(key)
        if person is None:
            stats.frames_missing_person_entry += 1
            subj_vis = UNK
        else:
            bbox = person.get("bbox")
            subj_vis = VIS if bbox is not None and len(bbox) > 0 else OCC

        obj_vis: dict[int, VisibilityState] = {}
        obj_rels: dict[int, set[int]] = {}
        for obj in objects[key] or []:
            track = 1 + vocab.object_id(obj["class"])
            if track in obj_vis:
                stats.duplicate_class_instances += 1
                continue
            visible = bool(obj.get("visible", False))
            rels = {vocab.relation_id(r) for f in RELATION_FIELDS for r in (obj.get(f) or [])}
            if not visible and rels:
                stats.invisible_with_relations += 1
                rels = set()
            obj_vis[track] = VIS if visible else OCC
            obj_rels[track] = rels

        edges: list[EdgeObservation] = []
        for track in sorted(set(obj_rels) | set(seen)):
            vis = obj_vis.get(track, UNK)
            present = obj_rels.get(track, set())
            for p in sorted(present | seen.get(track, set())):
                observed = p in present
                edges.append(EdgeObservation(
                    edge_key=EdgeKey(PERSON_TRACK_ID, track, p), video_id=video,
                    frame_idx=pos, timestamp=ts, predicate_score=1.0 if observed else 0.0,
                    observed=observed, subject_visibility=subj_vis, object_visibility=vis,
                    source_model="action_genome_gt",
                ))
            stats.observed_edges += len(present)
            seen.setdefault(track, set()).update(present)
        frames.append(FrameGraph(video_id=video, frame_idx=pos, timestamp=ts,
                                 edge_observations=edges))
    return frames


def inspect_action_genome(cfg: ActionGenomeConfig, max_examples: int = 3) -> dict[str, Any]:
    """Report the real files' structure so the assumptions above can be checked."""
    report: dict[str, Any] = {"annotation_dir": str(cfg.ann)}
    vocab = load_vocabulary(cfg)
    report["num_object_classes"] = len(vocab.object_classes)
    report["num_relationship_classes"] = len(vocab.relationship_classes)
    report["relationship_classes"] = vocab.relationship_classes

    persons = load_pickle(cfg.ann / "person_bbox.pkl")
    objects = load_pickle(cfg.ann / "object_bbox_and_relationship.pkl")
    report["person_pickle_type"] = type(persons).__name__
    report["object_pickle_type"] = type(objects).__name__
    report["num_object_frames"] = len(objects)
    report["num_person_frames"] = len(persons)
    sample = list(objects)[:max_examples]
    report["sample_frame_keys"] = sample
    report["sample_person_entry_keys"] = sorted(persons[sample[0]].keys()) if sample and \
        isinstance(persons.get(sample[0]), dict) else None

    obj_keys, unknown_rel, unknown_obj, sets = Counter(), Counter(), Counter(), Counter()
    visible = Counter()
    dup_frames = bad_keys = 0
    steps: list[int] = []
    last: dict[str, int] = {}
    for key in sorted(objects, key=lambda k: (k.split("/")[0], k)):
        try:
            video, frame_no = parse_frame_key(key)
        except ValueError:
            bad_keys += 1
            continue
        if video in last:
            steps.append(frame_no - last[video])
        last[video] = frame_no
        classes = []
        for obj in objects[key] or []:
            obj_keys.update(obj.keys())
            visible[bool(obj.get("visible"))] += 1
            md = obj.get("metadata")
            sets[md.get("set") if isinstance(md, dict) else None] += 1
            classes.append(obj.get("class"))
            try:
                vocab.object_id(str(obj.get("class")))
            except KeyError:
                unknown_obj[obj.get("class")] += 1
            for f in RELATION_FIELDS:
                for r in obj.get(f) or []:
                    try:
                        vocab.relation_id(r)
                    except KeyError:
                        unknown_rel[r] += 1
        if len(classes) != len(set(classes)):
            dup_frames += 1
    steps.sort()
    report.update({
        "object_dict_keys": dict(obj_keys),
        "visible_counts": {str(k): v for k, v in visible.items()},
        "split_counts": {str(k): v for k, v in sets.items()},
        "unknown_object_classes": dict(unknown_obj),
        "unknown_relationships": dict(unknown_rel),
        "frames_with_duplicate_classes": dup_frames,
        "unparseable_frame_keys": bad_keys,
        "annotated_frame_step_median": steps[len(steps) // 2] if steps else None,
        "annotated_frame_step_p90": steps[int(len(steps) * 0.9)] if steps else None,
    })
    return report
