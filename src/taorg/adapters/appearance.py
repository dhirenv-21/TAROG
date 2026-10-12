"""Appearance-enabled tracking adapter for the appearance audit (MOT17 frames + public detections).

INFERENCE-TIME ONLY: reads frames, public detections and seqinfo; never ground truth
(research/taorg_appearance_audit_preregistration.md §3.1; enforced by an AST test).

Provides
    check_alignment      frames on disk vs seqinfo (count, size)
    compute_embeddings   one ReID embedding per public detection (float16 cache)
    run_bytetrack_v1     the observation-audit ByteTrack run (V1) plus output detection indices
    run_botsort          BoxMOT BotSort with appearance off (V2) or on (V3), with recorded
                         first-association costs and lost-track internals (declared internal API)
    compute_cues         tracker-independent cues dt, a_glob, a_loc for every missing record

Costs are raw distances, not probabilities: geom = 1 - IoU in [0, 1]; emb = cosine distance in [0, 2].
BoxMOT is imported lazily; it lives in the isolated venv documented in the pre-registration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from taorg.adapters.mot17 import FrameTracks, SeqInfo
from taorg.schemas.observation import TrackRecord

# BoT-SORT settings shared by V2 and V3 (BoxMOT BotSort constructor defaults; prereg §2.3)
BOTSORT_SETTINGS: dict[str, Any] = {
    "track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.6, "track_buffer": 30,
    "match_thresh": 0.8, "proximity_thresh": 0.5, "appearance_thresh": 0.25, "use_cmc": True,
    "cmc_method": "ecc", "fuse_first_associate": False, "second_match_thresh": 0.5,
    "unconfirmed_match_thresh": 0.7, "unconfirmed_emb_scale": 2.0, "removed_stracks_buffer": 100,
}
HIGH_CONF = 0.5          # a detection carries appearance iff conf > HIGH_CONF (BoT-SORT split; amendment A1)
EMA_ALPHA = 0.9          # BoT-SORT smooth_feat rule
LOC_MOTION_PER_S = 2.0   # a_loc gate: R = h * (1 + LOC_MOTION_PER_S * dt)
EMB_DIM = 512


def frame_file(img_dir: Path, f: int) -> Path:
    """MOT image path of 1-based frame f."""
    return img_dir / f"{f:06d}.jpg"


def check_alignment(info: SeqInfo, img_dir: Path) -> dict[str, Any]:
    """Frames on disk must match seqinfo: count == length, first/last frame sizes == (h, w)."""
    import cv2
    files = sorted(p.name for p in img_dir.glob("*.jpg"))
    expected = [f"{f:06d}.jpg" for f in range(1, info.length + 1)]
    sizes = []
    for f in (1, info.length):
        im = cv2.imread(str(frame_file(img_dir, f)))
        sizes.append(None if im is None else tuple(im.shape[:2]))
    ok = files == expected and all(s == (info.height, info.width) for s in sizes)
    return {"seq": info.name, "n_files": len(files), "length": info.length, "sizes": sizes,
            "expected_size": (info.height, info.width), "ok": bool(ok)}


def compute_embeddings(info: SeqInfo, dets: dict[int, np.ndarray], img_dir: Path, reid: Any
                       ) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """L2-normalised ReID embedding (float16) for every public detection of every frame.

    Args:
        reid: a callable ``reid(image_bgr, boxes=(N, 4) xyxy) -> (N, D)`` (BoxMOT ``ReID``).
    Returns:
        frame -> (N_f, D) float16 array aligned with ``dets[frame]`` rows; timing summary.
    """
    import cv2
    out: dict[int, np.ndarray] = {}
    t_total = 0.0
    for f in range(1, info.length + 1):
        d = dets[f]
        if len(d) == 0:
            out[f] = np.zeros((0, EMB_DIM), np.float16)
            continue
        img = cv2.imread(str(frame_file(img_dir, f)))
        if img is None:
            raise FileNotFoundError(f"missing frame {frame_file(img_dir, f)}")
        t0 = time.perf_counter()
        e = np.asarray(reid(img, boxes=d[:, :4]), dtype=np.float32)
        t_total += time.perf_counter() - t0
        if e.shape != (len(d), EMB_DIM) or not np.isfinite(e).all():
            raise ValueError(f"bad embeddings at {info.name} frame {f}: shape {e.shape}")
        out[f] = e.astype(np.float16)
    return out, {"reid_s_total": t_total, "reid_ms_per_frame": 1000 * t_total / info.length}


def unit(e: np.ndarray) -> np.ndarray:
    """Float32 L2-normalised rows."""
    e = np.asarray(e, dtype=np.float32)
    n = np.linalg.norm(e, axis=-1, keepdims=True)
    return e / np.where(n > 1e-12, n, 1.0)


@dataclass
class TrackerRun:
    """Per-frame outputs of one tracker run, in the observation-audit FrameTracks form.

    ``frames[i].outputs`` maps track id -> (detection box, confidence) of the output's detection;
    ``out_det[i]`` maps track id -> detection row index in ``dets[frame]``.
    """

    frames: list[FrameTracks]
    out_det: list[dict[int, int]]
    timing: dict[str, float] = field(default_factory=dict)
    assoc_log: np.ndarray | None = None       # float32 rows: frame, track id, det idx, state, geom, emb_raw, final
    smooth_feats: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)  # (frame, tid) -> smooth_feat
    created: dict[int, tuple[int, int]] = field(default_factory=dict)  # tid -> (frame, det idx) before confirmation


def run_bytetrack_v1(info: SeqInfo, dets: dict[int, np.ndarray]) -> TrackerRun:
    """V1: the observation audit's ByteTrack run (adapter.run_bytetrack) plus detection indices.

    ByteTrack (trackers 2.6.1) returns the detection rows themselves, so each output box equals
    exactly one row of ``dets[frame]``; the index is recovered by exact equality.
    """
    from taorg.adapters.mot17 import run_bytetrack
    t0 = time.perf_counter()
    frames = run_bytetrack(info, dets)
    dt = time.perf_counter() - t0
    out_det = []
    for fr in frames:
        d = dets[fr.frame]
        m = {}
        for tid, (box, _) in fr.outputs.items():
            hit = np.where(np.all(np.isclose(d[:, :4], np.asarray(box)[None], rtol=0, atol=1e-6), axis=1))[0]
            if len(hit) < 1:
                raise RuntimeError(f"V1 output of track {tid} at frame {fr.frame} matches no detection")
            m[tid] = int(hit[0])
        out_det.append(m)
    return TrackerRun(frames, out_det, {"track_ms_per_frame": 1000 * dt / info.length})


def _make_botsort(use_embeddings: bool, frame_rate: int, log: list, feats: dict, frame_ref: list):
    """BoxMOT BotSort subclass that records first-association costs (return value unchanged)."""
    from boxmot.trackers.box.botsort.tracker import BotSort
    from boxmot.trackers.common.association.matching import embedding_distance

    class RecordingBotSort(BotSort):
        def _first_association_cost(self, tracks, detections):  # noqa: D401 - wrapped internal
            final = super()._first_association_cost(tracks, detections)
            if len(tracks) and len(detections):
                geom = self.association_distance(tracks, detections)
                raw = embedding_distance(tracks, detections) if self.use_embeddings \
                    else np.full_like(geom, np.nan, dtype=np.float64)
                nt, nd = geom.shape
                block = np.empty((nt * nd, 7), np.float32)
                block[:, 0] = frame_ref[0]
                block[:, 1] = np.repeat([int(t.id) for t in tracks], nd)
                block[:, 2] = np.tile([int(d.det_ind) for d in detections], nt)
                block[:, 3] = np.repeat([int(t.state) for t in tracks], nd)
                block[:, 4] = geom.ravel()
                block[:, 5] = raw.ravel()
                block[:, 6] = np.asarray(final).ravel()
                log.append(block)
            return final

    return RecordingBotSort(use_embeddings=use_embeddings, frame_rate=frame_rate, **BOTSORT_SETTINGS)


def run_botsort(info: SeqInfo, dets: dict[int, np.ndarray], img_dir: Path,
                embs: dict[int, np.ndarray] | None, use_embeddings: bool,
                feat_snapshot_every: int = 10) -> TrackerRun:
    """V2 (``use_embeddings=False``, embs ignored) or V3 (``True``; cached embeddings supplied).

    Frames are always supplied (ECC camera-motion compensation is on in both variants).
    ``internal`` holds confirmed lost tracks: id -> (Kalman-predicted xyxy box, frames since update).
    """
    import cv2
    import torch
    from boxmot.structures.detections import Detections
    from boxmot.structures.geometry import Boxes
    from boxmot.trackers.box.botsort.track import TrackState

    if use_embeddings and embs is None:
        raise ValueError("V3 requires cached embeddings")
    log: list = []
    feats: dict = {}
    created: dict[int, tuple[int, int]] = {}
    frame_ref = [0]
    trk = _make_botsort(use_embeddings, int(round(info.fps)), log, feats, frame_ref)
    frames: list[FrameTracks] = []
    out_det: list[dict[int, int]] = []
    t_track = 0.0
    for f in range(1, info.length + 1):
        frame_ref[0] = f
        img = cv2.imread(str(frame_file(img_dir, f)))
        if img is None:
            raise FileNotFoundError(f"missing frame {frame_file(img_dir, f)}")
        d = dets[f]
        e = None
        if use_embeddings:
            e = torch.from_numpy(embs[f].astype(np.float32))
        D = Detections(geometry=Boxes(torch.tensor(d[:, :4].reshape(-1, 4), dtype=torch.float32)),
                       scores=torch.tensor(d[:, 4], dtype=torch.float32),
                       class_ids=torch.zeros(len(d), dtype=torch.int64), sample_id=f"{info.name}:{f}",
                       embeddings=e)
        t0 = time.perf_counter()
        res = trk.update(D, img)
        t_track += time.perf_counter() - t0
        tids = res.track_ids.numpy().astype(int)
        dix = res.detection_indices.numpy().astype(int)
        outputs, m = {}, {}
        for tid, j in zip(tids, dix):
            if j < 0 or j >= len(d):
                raise RuntimeError(f"output of track {tid} at frame {f} has detection index {j}")
            outputs[int(tid)] = (tuple(float(v) for v in d[j, :4]), float(d[j, 4]))
            m[int(tid)] = int(j)
        for t in trk.active_tracks:
            if not t.is_activated and int(t.id) not in created:
                created[int(t.id)] = (f, int(t.det_ind))
        internal = {}
        for t in trk.lost_stracks:
            if t.state == TrackState.Lost and t.is_activated:
                internal[int(t.id)] = (tuple(float(v) for v in t.xyxy), int(f - t.end_frame))
                if use_embeddings and f % feat_snapshot_every == 0 and t.smooth_feat is not None:
                    feats[(f, int(t.id))] = np.asarray(t.smooth_feat, np.float32).copy()
        frames.append(FrameTracks(f, outputs, internal))
        out_det.append(m)
    arr = np.concatenate(log) if log else np.zeros((0, 7), np.float32)
    return TrackerRun(frames, out_det, {"track_ms_per_frame": 1000 * t_track / info.length}, arr, feats, created)


def stored_appearance_series(run: TrackerRun, dets: dict[int, np.ndarray], embs: dict[int, np.ndarray],
                             include_creation: bool = False
                             ) -> dict[int, list[tuple[int, np.ndarray | None]]]:
    """For each track: chronological (frame, s_X after that frame's output) for output frames.

    s_X = EMA(alpha=0.9, renormalised) of the embeddings of X's output detections with conf > 0.5
    (the registered, tracker-independent cue). ``include_creation=True`` also seeds the EMA with the
    pre-confirmation creation detection recorded in ``run.created`` (BoT-SORT's smooth_feat does
    this); it is used only for the implementation check (prereg amendment A2).
    """
    series: dict[int, list] = {}
    state: dict[int, np.ndarray | None] = {}
    if include_creation:
        for tid, (f, j) in run.created.items():
            if dets[f][j, 4] > HIGH_CONF:
                state[tid] = unit(embs[f][j])
    for fr, m in zip(run.frames, run.out_det):
        for tid, j in m.items():
            s = state.get(tid)
            if dets[fr.frame][j, 4] > HIGH_CONF:
                e = unit(embs[fr.frame][j])
                s = e if s is None else unit(EMA_ALPHA * s + (1 - EMA_ALPHA) * e)
                state[tid] = s
            series.setdefault(tid, []).append((fr.frame, s))
    return series


def compute_cues(info: SeqInfo, run: TrackerRun, recs: dict[int, list[TrackRecord]],
                 dets: dict[int, np.ndarray], embs: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    """Registered cues for every missing record (prereg §4). Causal: uses frames <= t only.

    Returns one dict per missing record: track, frame, dt, a_glob, a_loc (NaN when undefined),
    n_cand, n_loc, last_frame.
    """
    series = stored_appearance_series(run, dets, embs)
    cand_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def cands(f: int) -> tuple[np.ndarray, np.ndarray]:
        if f not in cand_cache:
            d = dets[f]
            keep = d[:, 4] > HIGH_CONF
            cand_cache[f] = (d[keep, :4], unit(embs[f][keep]) if keep.any() else np.zeros((0, EMB_DIM), np.float32))
        return cand_cache[f]

    rows = []
    for x, rr in recs.items():
        ser = dict((f, s) for f, s in series.get(x, []))
        out_frames = sorted(ser)
        k = 0
        last = None
        for r in rr:
            f = r.frame_idx
            while k < len(out_frames) and out_frames[k] <= f:
                last = out_frames[k]
                k += 1
            if r.detected:
                continue
            if last is None:
                raise RuntimeError(f"missing record of track {x} at {f} precedes its first output")
            dt = (f - last) / info.fps
            s = ser[last]
            j = run.out_det[last - 1][x]
            lb = dets[last][j, :4]
            boxes, E = cands(f)
            a_glob = a_loc = math.nan
            n_loc = 0
            if s is not None and len(E):
                dist = 1.0 - E @ s
                a_glob = float(dist.min())
                h = float(lb[3] - lb[1])
                c0 = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
                cc = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
                near = np.linalg.norm(cc - c0, axis=1) <= h * (1.0 + LOC_MOTION_PER_S * dt)
                n_loc = int(near.sum())
                if n_loc:
                    a_loc = float(dist[near].min())
            rows.append({"track": x, "frame": f, "last_frame": last, "dt": dt, "a_glob": a_glob,
                         "a_loc": a_loc, "n_cand": int(len(E)), "n_loc": n_loc,
                         "has_s": s is not None})
    return rows
