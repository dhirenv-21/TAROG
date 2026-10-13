"""Seeded subset of the Visual Genome TEST split (RelTR/EGTR COCO-format annotations) for a small
EGTR reproduction check. Images are range-fetched from the official VG zips; only sampled members
are transferred.

    python scripts/egtr/fetch_vg_subset.py --n 200
Writes D:/TAORG_runs/egtr/vg_subset/{images/, test.json, rel.json, val.json, train.json(empty), manifest.json}
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "appearance"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "downstream"))
from fetch_mot17_frames import HttpRangeFile  # noqa: E402
from fetch_vidor import head  # noqa: E402

ZIPS = ["https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"]
ROOT = Path("D:/TAORG_runs/egtr")
SEED = 20261013


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    with zipfile.ZipFile(ROOT / "vg" / "vg.zip") as z:
        test = json.loads(z.read("vg/test.json"))
        rel = json.loads(z.read("vg/rel.json"))
    ids = sorted(im["id"] for im in test["images"])
    pick = set(random.Random(SEED).sample(ids, a.n))
    images = [im for im in test["images"] if im["id"] in pick]
    want = {im["file_name"] for im in images}
    out = ROOT / "vg_subset"
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest, transferred = [], 0
    for u in ZIPS:
        size, url = head(u)
        f = HttpRangeFile(url, size, block=1 << 20)
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
        for i in z.infolist():
            name = i.filename.rsplit("/", 1)[-1]
            if name in want:
                data = z.read(i)
                (out / "images" / name).write_bytes(data)
                manifest.append({"zip": u, "member": i.filename, "sha256": hashlib.sha256(data).hexdigest()})
                want.discard(name)
        transferred += f.bytes_fetched
    if want:
        raise SystemExit(f"not found: {sorted(want)[:5]} ... ({len(want)})")
    sub = {"images": images, "annotations": [x for x in test["annotations"] if x["image_id"] in pick],
           "categories": test["categories"]}
    for k in test:
        if k not in sub:
            sub[k] = test[k]
    (out / "test.json").write_text(json.dumps(sub), encoding="utf-8")
    (out / "val.json").write_text(json.dumps({**sub, "images": [], "annotations": []}), encoding="utf-8")
    (out / "train.json").write_text(json.dumps({**sub, "images": [], "annotations": []}), encoding="utf-8")
    (out / "rel.json").write_text(json.dumps({**{k: v for k, v in rel.items() if k not in ("train", "val", "test")},
                                              "train": {}, "val": {},
                                              "test": {k: v for k, v in rel["test"].items() if int(k) in pick}}),
                                  encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"seed": SEED, "n": a.n, "bytes_transferred": transferred,
                                                   "files": manifest}, indent=0), encoding="utf-8")
    print(len(manifest), "images; bytes transferred", transferred)


if __name__ == "__main__":
    main()
