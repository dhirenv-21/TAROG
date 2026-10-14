"""TAORG end-to-end pipeline entry point.

    python -m taorg.pipeline run --video clip.mp4 --work D:/TAORG_runs/pipeline/clip
    python -m taorg.pipeline run --video clip.mp4 --work W --method VH --no-flag
    python -m taorg.pipeline stage {perception,sgg,maintain,view} --work W [--video ...]

``run`` executes the four stages, each in its pinned environment (docs/SETUP.md): the perception and
sgg stages are launched as subprocesses with the interpreters in configs/pipeline/default.yaml
(override with TAORG_PY_PERCEPTION / TAORG_PY_EGTR). Stages are idempotent (cached outputs are reused).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["python_perception"] = os.environ.get("TAORG_PY_PERCEPTION", cfg["python_perception"])
    cfg["python_egtr"] = os.environ.get("TAORG_PY_EGTR", cfg["python_egtr"])
    return cfg


def run_stage_here(stage: str, a: argparse.Namespace, cfg: dict) -> dict:
    from taorg.pipeline import stages
    work = Path(a.work)
    if stage == "perception":
        return stages.perception(Path(a.video), work, ROOT / cfg["flag_artifact"], Path(cfg["reid_weights"]))
    if stage == "sgg":
        return stages.sgg(work)
    if stage == "maintain":
        m = a.method or cfg["method"]
        return stages.maintain(work, m, cfg["params"][m], float(cfg["tau_obs"]),
                               use_flag=(cfg["use_flag"] if a.flag is None else a.flag))
    if stage == "view":
        from taorg.pipeline.viewer import build_viewer
        return {"viewer": str(build_viewer(work, with_frames=not a.no_frames))}
    raise SystemExit(f"unknown stage {stage}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m taorg.pipeline")
    ap.add_argument("command", choices=["run", "stage"])
    ap.add_argument("stage", nargs="?", choices=["perception", "sgg", "maintain", "view"])
    ap.add_argument("--video")
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--method", choices=["NM", "FB", "B2", "VH"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--flag", dest="flag", action="store_true", default=None)
    g.add_argument("--no-flag", dest="flag", action="store_false")
    ap.add_argument("--no-frames", action="store_true", help="viewer without embedded frames")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))
    Path(a.work).mkdir(parents=True, exist_ok=True)
    if a.command == "stage":
        if a.stage is None:
            raise SystemExit("stage name required")
        print(json.dumps(run_stage_here(a.stage, a, cfg), indent=1, default=str))
        return
    if not a.video:
        raise SystemExit("--video is required for 'run'")
    common = ["--work", a.work, "--config", a.config]
    for stage, py in (("perception", cfg["python_perception"]), ("sgg", cfg["python_egtr"])):
        if not Path(py).exists():
            raise SystemExit(f"{stage} interpreter not found: {py} (set TAORG_PY_{'PERCEPTION' if stage == 'perception' else 'EGTR'}; docs/SETUP.md)")
        cmd = [py, "-m", "taorg.pipeline", "stage", stage, *common] + (["--video", a.video] if stage == "perception" else [])
        print("[taorg] ", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            raise SystemExit(f"{stage} stage failed (exit {r.returncode})")
    for stage in ("maintain", "view"):
        print(f"[taorg] {stage}", flush=True)
        print(json.dumps(run_stage_here(stage, a, cfg), indent=1, default=str))
    print(f"[taorg] done: open {Path(a.work) / 'viewer.html'}")


if __name__ == "__main__":
    main(sys.argv[1:])
