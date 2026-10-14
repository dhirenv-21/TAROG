"""Disk-backed, single-worker pipeline jobs. No annotations or model changes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import BinaryIO

import yaml

from taorg.pipeline.cli import ROOT, load_config

STAGES = ("perception", "sgg", "maintain", "view")
ACTIVE = {"uploading", "queued", "running", "cancelling"}
EXPORTS = {"viewer.html", "graphs.json", "stream.csv.gz", "config.yaml", "job.json", "run.log"}


class BusyError(RuntimeError):
    """Only one upload or GPU job is allowed at a time."""


def save_json(path: Path, value: dict) -> None:
    """Atomically replace a small metadata file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


class Jobs:
    """Persist run metadata and execute the pinned pipeline stages sequentially."""

    def __init__(self, root: Path, config: Path, torch_home: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config.resolve()
        self.torch_home = torch_home.resolve()
        self.lock = threading.RLock()
        self.process = None
        self.thread = None
        self.active = None
        self.cancelled = threading.Event()
        self.items = {}
        # Caller holds the workspace lock before constructing this manager.
        for p in self.root.glob("*/job.json"):
            item = json.loads(p.read_text(encoding="utf-8"))
            if item["status"] in ACTIVE:
                item.update(status="interrupted", error="Server stopped. Start a new run.")
                save_json(p, item)
            self.items[item["id"]] = item

    def readiness(self) -> dict:
        """Check local resources without loading models or downloading weights."""
        from taorg.adapters.egtr import DEFAULT_ARTIFACT, DEFAULT_SRC, find_artifact

        cfg = load_config(self.config)
        paths = {
            "Tracking environment": Path(cfg["python_perception"]),
            "Relationship environment": Path(cfg["python_egtr"]),
            "Appearance weights": Path(cfg["reid_weights"]),
            "Takeover model": ROOT / cfg["flag_artifact"],
            "EGTR source": DEFAULT_SRC / "model" / "egtr.py",
            "Person detector": self.torch_home / "hub" / "checkpoints" /
            "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        }
        checks = [{"name": name, "ok": path.is_file(), "path": str(path)}
                  for name, path in paths.items()]
        try:
            _, checkpoint = find_artifact(DEFAULT_ARTIFACT)
            ok = checkpoint.is_file()
        except FileNotFoundError:
            checkpoint, ok = DEFAULT_ARTIFACT, False
        checks.append({"name": "EGTR checkpoint", "ok": ok, "path": str(checkpoint)})
        return {"ready": all(c["ok"] for c in checks), "checks": checks,
                "work_root": str(self.root), "default_tau": cfg["tau_obs"]}

    def snapshot(self, job_id: str | None = None) -> dict | list:
        """Return a copy safe for HTTP serialization."""
        with self.lock:
            if job_id:
                return copy.deepcopy(self.items[job_id])
            return sorted(copy.deepcopy(list(self.items.values())),
                          key=lambda j: j["created"], reverse=True)

    def update(self, job_id: str, **values: object) -> None:
        with self.lock:
            self.items[job_id].update(values, updated=time.time())
            save_json(self.root / job_id / "job.json", self.items[job_id])

    def upload(self, source: BinaryIO, size: int, name: str, options: dict) -> str:
        """Stream a bounded request body to a unique run; preserve incomplete uploads."""
        with self.lock:
            if self.active:
                raise BusyError("A run is already active. Wait for it to finish or cancel it.")
            job_id = uuid.uuid4().hex
            work = self.root / job_id
            work.mkdir()
            self.items[job_id] = {"id": job_id, "name": name, "created": time.time(),
                                  "status": "uploading", "stage": None, "completed_stages": [],
                                  "options": options, "bytes": size, "error": None}
            try:
                self.update(job_id)
            except OSError:
                del self.items[job_id]
                raise
            self.active = job_id
            self.cancelled.clear()
        try:
            digest = hashlib.sha256()
            video = work / ("input" + Path(name).suffix.lower())
            with video.open("wb") as dst:
                remaining = size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload ended early. Choose the video and try again.")
                    dst.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            cfg = load_config(self.config)
            cfg["tau_obs"] = options["tau"]
            cfg["params"]["FB"]["window_size"] = options["window"]
            (work / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=True)
            self.update(job_id, status="queued", input_sha256=digest.hexdigest(),
                        git_commit=git.stdout.strip(), config_sha256=hashlib.sha256(
                            (work / "config.yaml").read_bytes()).hexdigest())
            self.thread = threading.Thread(target=self._run, args=(job_id, video, cfg), daemon=True)
            self.thread.start()
            return job_id
        except Exception as exc:
            try:
                self.update(job_id, status="failed", error=str(exc))
            finally:
                with self.lock:
                    self.active = None
            raise

    def _run(self, job_id: str, video: Path, cfg: dict) -> None:
        work = self.root / job_id
        options = self.items[job_id]["options"]
        env = dict(os.environ, PYTHONUNBUFFERED="1", TORCH_HOME=str(self.torch_home),
                   TAORG_TORCH_HOME=str(self.torch_home), HF_HUB_OFFLINE="1")
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        try:
            with (work / "run.log").open("ab", buffering=0) as log:
                for stage in STAGES:
                    with self.lock:
                        if self.cancelled.is_set():
                            break
                        self.update(job_id, status="running", stage=stage)
                        py = cfg.get("python_" + ("egtr" if stage == "sgg" else stage),
                                     sys.executable)
                        cmd = [py, "-u", "-m", "taorg.pipeline", "stage", stage,
                               "--work", str(work), "--video", str(video),
                               "--config", str(work / "config.yaml"),
                               "--method", options["method"],
                               "--flag" if options["flag"] else "--no-flag"]
                        log.write((f"\n[TAORG] {stage}\n").encode())
                        self.process = subprocess.Popen(
                            cmd, cwd=ROOT, env=env, stdout=log, stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                            start_new_session=os.name != "nt")
                    code = self.process.wait()
                    with self.lock:
                        self.process = None
                    if self.cancelled.is_set():
                        break
                    if code:
                        raise RuntimeError(f"{stage} failed (exit {code}). See the run log.")
                    self.update(job_id, completed_stages=list(STAGES[:STAGES.index(stage) + 1]))
            if self.cancelled.is_set():
                self.update(job_id, status="cancelled", stage=None)
            else:
                data = json.loads((work / "graphs.json").read_text(encoding="utf-8"))
                self.update(job_id, status="completed", stage=None,
                            summary={"steps": len(data["frames"]),
                                     "edges": sum(len(f["edges"]) for f in data["frames"])})
        except Exception as exc:
            self.update(job_id, status="cancelled" if self.cancelled.is_set() else "failed",
                        error=str(exc), stage=None)
        finally:
            with self.lock:
                self.process = None
                self.active = None

    def cancel(self, job_id: str) -> None:
        """Stop only this manager's current process tree, keeping its artifacts."""
        with self.lock:
            if self.active != job_id:
                raise ValueError("This run is no longer active.")
            if self.items[job_id]["status"] == "uploading":
                raise ValueError("Wait for the upload to finish before cancelling processing.")
            self.cancelled.set()
            self.update(job_id, status="cancelling")
            if self.process and self.process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)

    def close(self) -> None:
        if self.active and self.items[self.active]["status"] != "uploading":
            self.cancel(self.active)
        if self.thread:
            self.thread.join(timeout=15)
