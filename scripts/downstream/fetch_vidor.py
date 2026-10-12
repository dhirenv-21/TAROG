"""Selective retrieval of sampled VidOR videos from the HuggingFace mirror via HTTP range requests.

    python scripts/downstream/fetch_vidor.py --split training     (development, 30 videos)
    python scripts/downstream/fetch_vidor.py --split validation   (evaluation, 60; ONLY after the freeze)

Writes D:/TAORG_runs/downstream/videos/<video_path> and a manifest with member CRC-32 (verified by
zipfile) and SHA-256 of each video.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "appearance"))
from fetch_mot17_frames import HttpRangeFile  # noqa: E402

from taorg.adapters.vidor import sample  # noqa: E402

BASE = "https://huggingface.co/datasets/shangxd/vidor/resolve/main/"
ZIPS = {"validation": ["validation-video.zip"], "training": [f"training-video-part{i}.zip" for i in range(1, 9)]}
OUT = Path("D:/TAORG_runs/downstream/videos")


def head(u: str) -> tuple[int, str]:
    """Content length and final (redirected) URL, with retries for transient network errors."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=60) as r:
                return int(r.headers["Content-Length"]), r.url
        except OSError:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(ZIPS), required=True)
    a = ap.parse_args()
    want = {"video/" + p for p in sample(Path("data/vidor"), a.split)}
    manifest, transferred = [], 0
    for name in ZIPS[a.split]:
        size, url = head(BASE + name)
        f = HttpRangeFile(url, size, block=1 << 20)
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
        for i in z.infolist():
            if i.filename in want:
                data = z.read(i)
                dst = OUT / i.filename[len("video/"):]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                manifest.append({"zip": name, "member": i.filename, "bytes": i.file_size, "crc32": f"{i.CRC:08x}",
                                 "sha256": hashlib.sha256(data).hexdigest()})
                want.discard(i.filename)
        transferred += f.bytes_fetched
    if want:
        raise SystemExit(f"members not found: {sorted(want)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / f"manifest_videos_{a.split}.json").write_text(
        json.dumps({"source": BASE, "split": a.split, "bytes_transferred": transferred,
                    "files": sorted(manifest, key=lambda m: m["member"])}, indent=0), encoding="utf-8")
    print(a.split, len(manifest), "videos; bytes transferred", transferred)


if __name__ == "__main__":
    main()
