"""Corruption manifest schema for deterministic noise injection.

Every corruption run produces a CorruptionManifest that records exactly
what happened.  This is essential for reproducibility: any method
evaluated on corrupted data must have access to the same manifest.

The manifest is JSON-serializable for storage in run artifact directories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from taorg.schemas.edges import EdgeKey

logger = logging.getLogger(__name__)


@dataclass
class CorruptionEntry:
    """One corruption event applied to a sequence.

    Attributes:
        corruption_type: "occlusion" or "id_switch".
        video_id: Video/sequence identifier.
        edge_key_tuple: (subject_track_id, object_track_id, predicate_id)
            for occlusion corruptions, or None for id_switch.
        track_ids: Track IDs involved in an id_switch, or None.
        start_frame: First corrupted frame (inclusive).
        end_frame: Last corrupted frame (inclusive).
        duration: Number of corrupted frames.
        original_visibility: Visibility state before corruption.
        modified_visibility: Visibility state after corruption.
        id_mapping: {old_id: new_id} for id_switch corruptions, or None.
    """

    corruption_type: str
    video_id: str
    edge_key_tuple: Optional[tuple[int, int, int]] = None
    track_ids: Optional[tuple[int, ...]] = None
    start_frame: int = 0
    end_frame: int = 0
    duration: int = 0
    original_visibility: str = "VISIBLE"
    modified_visibility: str = "OCCLUDED_OR_UNRELIABLE"
    id_mapping: Optional[dict[int, int]] = None

    @property
    def edge_key(self) -> Optional[EdgeKey]:
        """Convert edge_key_tuple to EdgeKey if present."""
        if self.edge_key_tuple is not None:
            return EdgeKey(*self.edge_key_tuple)
        return None


@dataclass
class CorruptionManifest:
    """Complete record of all corruptions applied to a sequence.

    Attributes:
        seed: Random seed used for corruption selection.
        config: Configuration dict used for the corruption run.
        entries: List of individual corruption events.
        version: Schema version for forward compatibility.
    """

    seed: int
    config: dict = field(default_factory=dict)
    entries: list[CorruptionEntry] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "seed": self.seed,
            "config": self.config,
            "version": self.version,
            "entries": [asdict(e) for e in self.entries],
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            path: Output file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Corruption manifest saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> CorruptionManifest:
        """Load manifest from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded CorruptionManifest.
        """
        with open(path) as f:
            data = json.load(f)
        entries = [
            CorruptionEntry(
                corruption_type=e["corruption_type"],
                video_id=e["video_id"],
                edge_key_tuple=tuple(e["edge_key_tuple"]) if e.get("edge_key_tuple") else None,
                track_ids=tuple(e["track_ids"]) if e.get("track_ids") else None,
                start_frame=e["start_frame"],
                end_frame=e["end_frame"],
                duration=e["duration"],
                original_visibility=e.get("original_visibility", "VISIBLE"),
                modified_visibility=e.get("modified_visibility", "OCCLUDED_OR_UNRELIABLE"),
                id_mapping=e.get("id_mapping"),
            )
            for e in data.get("entries", [])
        ]
        return cls(
            seed=data["seed"],
            config=data.get("config", {}),
            entries=entries,
            version=data.get("version", "1.0"),
        )

    def __len__(self) -> int:
        return len(self.entries)
