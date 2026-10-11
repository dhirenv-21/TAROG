"""MOT17 public-detection + ByteTrack adapter producing contract ``TrackRecord`` streams.

INFERENCE-TIME ONLY: this module reads detections (``det/det.txt``) and ``seqinfo.ini``; it
never reads ground truth (``gt/gt.txt`` is read only by ``taorg.experiments.observation.oracle``).

Information classes (research/taorg_observation_audit.md §4):
    direct     detections, tracker outputs (tracker_id), the tracker's internal lost-buffer
               tracklets (time_since_update, Kalman-predicted box)  [internal state; declared]
    heuristic  EXIT / IDENTITY_SUSPECT / OCCLUSION_EVIDENCE from predicted boxes and other
               tracks' outputs (thresholds fixed a priori in the audit document)
    none       association confidence (ByteTrack provides none -> None)
"""

from __future__ import annotations

import configparser
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from taorg.schemas.observation import EvidenceSource, MissingReason, TrackRecord
from taorg.schemas.tracks import TrackerState

# a-priori heuristic thresholds (audit §4)
EXIT_OUTSIDE_FRACTION = 0.5
IDENTITY_IOU = 0.5
OCCLUSION_COVER = 0.3
HORIZON_S = 3.0


@dataclass(frozen=True)
class SeqInfo:
    """Sequence metadata from seqinfo.ini."""

    name: str
    fps: float
    width: int
    height: int
    length: int


def _read(z: zipfile.ZipFile, path: str) -> bytes:
    return z.read(path)


def load_detections(zip_path: Path, seq: str, detector: str) -> tuple[SeqInfo, dict[int, np.ndarray]]:
    """Public detections per frame (1-based) as (N, 5) arrays [x1, y1, x2, y2, conf]."""
    with zipfile.ZipFile(zip_path) as z:
        base = f"train/MOT17-{seq}-{detector}"
        ini = configparser.ConfigParser()
        ini.read_string(_read(z, f"{base}/seqinfo.ini").decode())
        q = ini["Sequence"]
        info = SeqInfo(q["name"], float(q["frameRate"]), int(q["imWidth"]), int(q["imHeight"]), int(q["seqLength"]))
        a = np.loadtxt(io.BytesIO(_read(z, f"{base}/det/det.txt")), delimiter=",", ndmin=2)
    dets: dict[int, np.ndarray] = {f: np.empty((0, 5)) for f in range(1, info.length + 1)}
    for f in np.unique(a[:, 0]).astype(int):
        r = a[a[:, 0] == f]
        dets[f] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]], axis=1)
    return info, dets


@dataclass
class FrameTracks:
    """Tracker state after one frame: confirmed outputs and the internal confirmed tracklet list."""

    frame: int
    outputs: dict[int, tuple[tuple[float, float, float, float], float]]      # tid -> (box, conf)
    internal: dict[int, tuple[tuple[float, float, float, float], int]]       # tid -> (predicted box, time_since_update)


def run_bytetrack(info: SeqInfo, dets: dict[int, np.ndarray]) -> list[FrameTracks]:
    """ByteTrack (trackers 2.6.1, library defaults) over public detections; fixed-rate timing."""
    import supervision as sv
    from trackers import ByteTrackTracker
    trk = ByteTrackTracker(frame_rate=info.fps)
    out: list[FrameTracks] = []
    for f in range(1, info.length + 1):
        d = dets[f]
        det = sv.Detections(xyxy=d[:, :4].astype(float), confidence=d[:, 4].astype(float)) if len(d) \
            else sv.Detections.empty()
        res = trk.update(det)
        outputs = {}
        for box, conf, tid in zip(res.xyxy, res.confidence if res.confidence is not None else [1.0] * len(res),
                                  res.tracker_id):
            if int(tid) >= 0:
                outputs[int(tid)] = (tuple(float(v) for v in box), float(conf))
        internal = {int(t.tracker_id): (tuple(float(v) for v in t.get_state_bbox()), int(t.time_since_update))
                    for t in trk.tracks if int(t.tracker_id) >= 0}
        out.append(FrameTracks(f, outputs, internal))
    return out


def iou(a, b) -> float:
    """IoU of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cover(a, b) -> float:
    """Fraction of box a covered by box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def outside_fraction(box, w: int, h: int) -> float:
    """Fraction of the box area outside the image."""
    return 1.0 - cover(box, (0.0, 0.0, float(w), float(h)))


def heuristic_reason(x: int, pred, last_det_frame: int, frame: FrameTracks, first_out: dict[int, int],
                     w: int, h: int) -> MissingReason:
    """Inference-time missing reason for undetected track x (audit §4; precedence fixed)."""
    cx, cy = 0.5 * (pred[0] + pred[2]), 0.5 * (pred[1] + pred[3])
    if outside_fraction(pred, w, h) >= EXIT_OUTSIDE_FRACTION or not (0 <= cx <= w and 0 <= cy <= h):
        return MissingReason.OUT_OF_VIEW_EVIDENCE
    for y, (box, _) in frame.outputs.items():
        if y != x and first_out.get(y, frame.frame) > last_det_frame and iou(pred, box) >= IDENTITY_IOU:
            return MissingReason.IDENTITY_SUSPECT
    for y, (box, _) in frame.outputs.items():
        if y != x and first_out.get(y, frame.frame) <= last_det_frame and cover(pred, box) >= OCCLUSION_COVER:
            return MissingReason.OCCLUSION_EVIDENCE
    return MissingReason.UNSPECIFIED


def build_records(info: SeqInfo, frames: list[FrameTracks], horizon_s: float = HORIZON_S
                  ) -> dict[int, list[TrackRecord]]:
    """Contract records per confirmed track id, from its first output to last output + horizon."""
    first_out: dict[int, int] = {}
    last_out: dict[int, int] = {}
    for fr in frames:
        for t in fr.outputs:
            first_out.setdefault(t, fr.frame)
            last_out[t] = fr.frame
    H = int(round(horizon_s * info.fps))
    recs: dict[int, list[TrackRecord]] = {}
    for x in sorted(first_out):
        rows: list[TrackRecord] = []
        last_det = None
        last_pred = None
        for f in range(first_out[x], min(info.length, last_out[x] + H) + 1):
            fr = frames[f - 1]
            ts = (f - 1) / info.fps
            if x in fr.outputs:
                box, conf = fr.outputs[x]
                last_det, last_pred = f, box
                rows.append(TrackRecord(x, f, ts, True, TrackerState.TRACKED, detector_confidence=conf, bbox=box,
                                        time_since_detection_s=0.0))
                continue
            if x in fr.internal:
                pred, _ = fr.internal[x]
                state = TrackerState.LOST
                last_pred = pred
            else:
                pred, state = last_pred, TrackerState.REMOVED
            reason = heuristic_reason(x, pred, last_det, fr, first_out, info.width, info.height)
            src = EvidenceSource.HEURISTIC
            if reason is MissingReason.UNSPECIFIED:
                reason = MissingReason.TRACK_LOST if state is TrackerState.LOST else MissingReason.TRACK_TERMINATED
                src = EvidenceSource.TRACKER
            rows.append(TrackRecord(x, f, ts, False, state, predicted_bbox=pred,
                                    time_since_detection_s=(f - last_det) / info.fps,
                                    missing_reason=reason, missing_reason_source=src))
        recs[x] = rows
    return recs
