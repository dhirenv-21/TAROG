"""Deterministic artifact I/O: gzip with mtime=0 and CONTENT hashes (SHA-256 of uncompressed bytes)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from pathlib import Path


def write_csv_gz(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> str:
    """Write rows deterministically; returns the content SHA-256 (of the uncompressed CSV text)."""
    buf = io.StringIO()
    names = fieldnames or (list(rows[0]) if rows else [])
    w = csv.DictWriter(buf, fieldnames=names, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
        gz.write(data)
    return hashlib.sha256(data).hexdigest()


def read_csv_gz(path: Path) -> list[dict]:
    """Rows as dicts of strings."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def content_sha256(path: Path) -> str:
    """SHA-256 of a .gz file's decompressed bytes, or of the raw bytes otherwise."""
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return hashlib.sha256(raw).hexdigest()
