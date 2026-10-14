"""Build the website's sample clip from MOT17 frames already on D: (no download). Not committed.

    python scripts/web/make_sample_clip.py
Writes D:/TAORG_runs/web-samples/MOT17-09_sample.mp4 and its .json description (source, licence,
attribution). MOT17 is distributed under CC BY-NC-SA 3.0 (non-commercial use, attribution,
share-alike); the clip is for local, non-commercial demonstration. MOT17 has no relationship
annotations, so the website's predictions on this clip cannot be checked against ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SRC = Path("D:/TAORG_runs/appearance/MOT17Det/train/MOT17-09/img1")
OUT = Path("D:/TAORG_runs/web-samples")


def main() -> None:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300, help="frames from the start (30 fps)")
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()
    files = sorted(SRC.glob("*.jpg"))[:a.frames]
    if len(files) < a.frames:
        raise SystemExit(f"MOT17-09 frames not found under {SRC}; see docs/SETUP.md "
                         "(scripts/appearance/fetch_mot17_frames.py)")
    first = cv2.imread(str(files[0]))
    h = int(round(first.shape[0] * a.width / first.shape[1] / 2) * 2)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "MOT17-09_sample.mp4"
    # MPEG-4 Part 2: OpenCV writes it without extra DLLs; browsers may not preview it.
    for codec in ("mp4v",):
        w = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*codec), 30.0, (a.width, h))
        if w.isOpened():
            break
    else:
        raise SystemExit("OpenCV cannot write MP4 on this machine")
    for f in files:
        w.write(cv2.resize(cv2.imread(str(f)), (a.width, h), interpolation=cv2.INTER_AREA))
    w.release()
    meta = {"title": f"MOT17-09 pedestrian street scene (first {a.frames / 30:.0f} s, "
                     f"{a.width}×{h})",
            "source": f"MOTChallenge MOT17 (MOT17Det train, sequence MOT17-09, "
                      f"frames 1–{a.frames}), motchallenge.net",
            "licence": "CC BY-NC-SA 3.0 (non-commercial, attribution, share-alike)",
            "attribution": "Milan, Leal-Taixé, Reid, Roth, Schindler: "
                           "MOT16: A Benchmark for Multi-Object Tracking (2016).",
            "note": "No relationship annotations exist for MOT17; "
                    "predictions on this clip are unverified.",
            "codec": codec, "duration_s": round(a.frames / 30.0, 1),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    text = json.dumps(meta, indent=1, ensure_ascii=False)
    dst.with_suffix(".json").write_text(text, encoding="utf-8")
    print(dst, dst.stat().st_size, "bytes, codec", codec)


if __name__ == "__main__":
    main()
