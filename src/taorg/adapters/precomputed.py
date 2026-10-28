"""Adapter for precomputed per-frame predictions in JSON Lines.

Any base model can be attached by dumping its predictions in this format
(no model code is imported here).  One JSON object per line:

    {"type": "frame", "video_id": "V1", "frame_idx": 0, "timestamp": 0.0,
     "visibility": {"7": "visible", "9": "occluded_or_unreliable"}}        (optional)
    {"type": "edge", "video_id": "V1", "frame_idx": 0, "timestamp": 0.0,
     "subject_track_id": 7, "object_track_id": 9, "predicate_id": 4, "score": 0.83}

`frame` lines declare frames that may have no edges and the per-track
visibility; tracks without a declared visibility are VISIBLE if they occur in
an edge of that frame.  Records may appear in any order.  Options:
    min_score   drop predictions below this score before temporal processing
    top_k       keep the k highest-scoring edges per frame (None = all)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from taorg.adapters.base import validate_video
from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph, PairKey
from taorg.schemas.tracks import VisibilityState


@dataclass
class PrecomputedJSONLAdapter:
    """Reads a JSONL prediction dump and yields canonical videos."""

    path: Path
    source_model: str
    source_checkpoint: str = ""
    min_score: float = 0.0
    top_k: int | None = None

    def videos(self) -> Iterator[list[FrameGraph]]:
        frames: dict[str, dict[int, dict]] = {}
        with open(self.path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                kind = rec.get("type")
                if kind not in ("frame", "edge"):
                    raise ValueError(f"{self.path}:{ln}: unknown record type {kind!r}")
                fr = frames.setdefault(rec["video_id"], {}).setdefault(
                    int(rec["frame_idx"]),
                    {"timestamp": float(rec["timestamp"]), "edges": [], "vis": {}})
                if kind == "frame":
                    fr["vis"].update({int(k): VisibilityState(v)
                                      for k, v in rec.get("visibility", {}).items()})
                elif float(rec["score"]) >= self.min_score:
                    fr["edges"].append(rec)
        for vid in sorted(frames):
            video = self._build(vid, frames[vid])
            validate_video(video)
            yield video

    def _build(self, vid: str, by_idx: dict[int, dict]) -> list[FrameGraph]:
        seen: dict[PairKey, set[int]] = {}
        out: list[FrameGraph] = []
        for pos, idx in enumerate(sorted(by_idx)):
            fr = by_idx[idx]
            edges = sorted(fr["edges"], key=lambda r: -float(r["score"]))
            if self.top_k is not None:
                edges = edges[: self.top_k]
            present: dict[PairKey, dict[int, float]] = {}
            for r in edges:
                pk = PairKey(int(r["subject_track_id"]), int(r["object_track_id"]))
                p = int(r["predicate_id"])
                present.setdefault(pk, {})[p] = max(present.get(pk, {}).get(p, 0.0),
                                                    float(r["score"]))
            tracks_here = {t for pk in present for t in pk} | set(fr["vis"])

            def vis(track: int) -> VisibilityState:
                if track in fr["vis"]:
                    return fr["vis"][track]
                return VisibilityState.VISIBLE if track in tracks_here else VisibilityState.UNKNOWN

            records: list[EdgeObservation] = []
            for pk in sorted(set(present) | {pk for pk in seen
                                              if pk.subject_track_id in tracks_here
                                              and pk.object_track_id in tracks_here}):
                preds = present.get(pk, {})
                for p in sorted(set(preds) | seen.get(pk, set())):
                    records.append(EdgeObservation(
                        edge_key=EdgeKey(pk.subject_track_id, pk.object_track_id, p),
                        video_id=vid, frame_idx=pos, timestamp=fr["timestamp"],
                        predicate_score=preds.get(p, 0.0), observed=p in preds,
                        subject_visibility=vis(pk.subject_track_id),
                        object_visibility=vis(pk.object_track_id),
                        source_model=self.source_model, source_checkpoint=self.source_checkpoint,
                    ))
                seen.setdefault(pk, set()).update(preds)
            out.append(FrameGraph(video_id=vid, frame_idx=pos, timestamp=fr["timestamp"],
                                  edge_observations=records))
        return out
