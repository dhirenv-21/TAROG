"""Selective retrieval of MOT17 frames from the official MOT17Det.zip via HTTP range requests.

Only the members of the requested sequences are transferred (the zip central directory plus
those members' local headers and data). Nothing else from the 1.95 GB archive is downloaded.

Usage:
    python scripts/appearance/fetch_mot17_frames.py --list
    python scripts/appearance/fetch_mot17_frames.py --seq 02 09 11 --out D:/TAORG_runs/appearance/MOT17Det

Records a manifest (member name, size, CRC-32 from the archive, SHA-256 of the extracted bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://motchallenge.net/data/MOT17Det.zip"
EXPECTED_SIZE = 1952547113


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP byte ranges (with a small read-ahead cache)."""

    def __init__(self, url: str, size: int, block: int = 1 << 20) -> None:
        self.url, self.size, self.block = url, size, block
        self.pos = 0
        self.cache_start, self.cache = -1, b""
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"server ignored Range request (HTTP {r.status})")
                    data = r.read()
                self.bytes_fetched += len(data)
                return data
            except (OSError, RuntimeError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, ce = self.cache_start, self.cache_start + len(self.cache)
        if not (cs <= self.pos and self.pos + n <= ce):
            start = self.pos
            end = min(self.size, start + max(n, self.block))
            self.cache_start, self.cache = start, self._get(start, end)
            cs = start
        out = self.cache[self.pos - cs:self.pos - cs + n]
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("D:/TAORG_runs/appearance/MOT17Det"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    if size != EXPECTED_SIZE:
        raise SystemExit(f"archive size changed: {size} != {EXPECTED_SIZE}; re-verify the source")
    f = HttpRangeFile(URL, size, block=4 << 20)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    names = z.namelist()
    if a.list:
        seqs = sorted({n.split("/")[1] for n in names if n.count("/") >= 2})
        for s in seqs:
            m = [i for i in z.infolist() if i.filename.split("/")[1:2] == [s]]
            print(s, len(m), sum(i.file_size for i in m))
        print("central directory bytes fetched:", f.bytes_fetched)
        return
    manifest = []
    for s in a.seq:
        prefix = f"train/MOT17-{s}/"
        members = [i for i in z.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
        if not members:
            raise SystemExit(f"no members under {prefix}")
        for i in members:
            dst = a.out / i.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(i)  # CRC-32 checked by zipfile
            dst.write_bytes(data)
            manifest.append({"name": i.filename, "size": i.file_size, "crc32": f"{i.CRC:08x}",
                             "sha256": hashlib.sha256(data).hexdigest()})
        print(s, len(members), "files")
    out = a.out / f"manifest_{'_'.join(a.seq)}.json"
    out.write_text(json.dumps({"source": URL, "archive_size": size, "bytes_transferred": f.bytes_fetched,
                               "files": manifest}, indent=0), encoding="utf-8")
    print("bytes transferred:", f.bytes_fetched, "->", out)


if __name__ == "__main__":
    main()
