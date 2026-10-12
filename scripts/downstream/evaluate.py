"""Downstream study: run the four frozen methods (FB, FB+F, B2, B2+F) on every video of a split.

    python scripts/downstream/evaluate.py --split eval   (registered; after the freeze)
    python scripts/downstream/evaluate.py --split dev    (descriptive; after the freeze)

Parameters come ONLY from results/downstream/freeze/frozen.json. Outputs per video-method outcome rows
and per-frame output arrays (deterministic gzip) for metric re-computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from taorg.adapters.vidor import sample
from taorg.experiments.appearance.labels import frame_label, purity
from taorg.experiments.downstream.io import write_csv_gz
from taorg.experiments.downstream.observations import gt_dict
from taorg.experiments.downstream.runner import METHODS, load_video_data, make_method, run_method
from taorg.experiments.observation.oracle import identity_of
from taorg.metrics.downstream import video_outcomes
from taorg.utils.seed import set_seed

SPLIT = {"dev": "training", "eval": "validation"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT), required=True)
    a = ap.parse_args()
    set_seed(20261015)
    fz = json.loads(Path("results/downstream/freeze/frozen.json").read_text(encoding="utf-8"))
    out = Path(f"results/downstream/{a.split}")
    heavy = Path(f"D:/TAORG_runs/downstream/{a.split}")
    rec = json.loads((out / "pipeline_records.json").read_text(encoding="utf-8"))
    rows, flag_rows, outs_rows = [], [], []
    for vp in sample(Path("data/vidor"), SPLIT[a.split]):
        vid = Path(vp).stem
        if rec["videos"].get(vid, {}).get("status") != "done":
            rows.append({"video": vid, "method": "ALL", "status": "failed:perception"})
            continue
        d = load_video_data(heavy / vid, Path(f"data/vidor/{SPLIT[a.split]}") / vp.replace(".mp4", ".json"))
        for m in METHODS:
            fam, flagged = m.split("+")[0], m.endswith("+F")
            o = run_method(make_method(fam, fz["params"][fam], d.flags if flagged else None), d)
            r = video_outcomes(o, d.labels, d.video.fps)
            rows.append({"video": vid, "method": m, "status": "done", **r})
            for k, arr in o.items():
                outs_rows.append({"video": vid, "method": m, "key": f"{k.subject_track_id}-{k.object_track_id}-{k.predicate_id}",
                                  "on_frames": " ".join(map(str, np.where(arr)[0]))})
        # flag behaviour on missing records (label of the missing track at that frame; descriptive)
        g, pur, gt = identity_of(d.stream.matches), purity(d.stream.matches), gt_dict(d.video)
        for r in d.missing:
            flag_rows.append({"video": vid, "track": r["track"], "frame": r["frame"], "dt": r["dt"],
                              "a_loc": r["a_loc"], "p_takeover": r["p_takeover"],
                              "appearance_available": r["appearance_available"], "flag": r["flag"],
                              "label": frame_label(int(r["track"]), int(r["frame"]), g, pur, d.stream.matches, gt)})
        print(vid, "done", flush=True)
    keys = sorted({k for r in rows for k in r})
    h1 = write_csv_gz(out / "outcomes.csv.gz", rows, keys)
    h2 = write_csv_gz(out / "flag_records.csv.gz", flag_rows,
                      ["video", "track", "frame", "dt", "a_loc", "p_takeover", "appearance_available", "flag", "label"])
    h3 = write_csv_gz(out / "method_outputs.csv.gz", outs_rows, ["video", "method", "key", "on_frames"])
    (out / "evaluate_record.json").write_text(json.dumps({"frozen": fz["freeze_id"], "outcomes_content_sha256": h1,
                                                          "flag_records_content_sha256": h2,
                                                          "method_outputs_content_sha256": h3}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
