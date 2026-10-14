"""Disk-backed, single-worker pipeline jobs. No annotations or model changes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import yaml

from taorg.pipeline.cli import ROOT, load_config

STAGES = ("perception", "sgg", "maintain", "view")
ACTIVE = {"uploading", "queued", "running", "cancelling"}
EXPORTS = {"viewer.html", "graphs.json", "stream.csv.gz", "config.yaml", "job.json", "run.log"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MIN_FRAMES = 3


class BusyError(RuntimeError):
    """Only one upload or GPU job is allowed at a time."""


class VideoError(ValueError):
    """The input cannot be analysed; the message says why and what to do."""


def save_json(path: Path, value: dict) -> None:
    """Atomically replace a small metadata file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def probe_video(path: Path, max_seconds: float) -> dict:
    """Decode the header and first frame with OpenCV (the decoder the pipeline uses).

    Raises VideoError for files the pipeline cannot process, so no GPU stage starts on them.
    Frame counts reported by containers can be approximate; they are used only for limits.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(
                "This file could not be opened as a video. "
                "Re-export it as MP4 (H.264) and try again."
            )
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(
                "No frames could be decoded from this file. "
                "Re-export it as MP4 (H.264) and try again."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
    finally:
        cap.release()
    if fps <= 0 or fps > 240:
        raise VideoError(
            "The video's frame rate could not be read. Re-export it as MP4 (H.264) and try again."
        )
    if 0 < frames < MIN_FRAMES:
        raise VideoError(f"The video has only {frames} frame(s); at least {MIN_FRAMES} are needed.")
    duration = frames / fps if frames > 0 else None
    if duration is not None and duration > max_seconds:
        raise VideoError(
            f"The video is {duration:.0f} s long; "
            f"this workspace accepts up to {max_seconds:.0f} s. "
            "Trim it to a short clip (10–60 s works best) and try again."
        )
    return {
        "width": int(w),
        "height": int(h),
        "fps": round(fps, 3),
        "frames": frames,
        "duration_s": round(duration, 2) if duration is not None else None,
    }


class Jobs:
    """Persist run metadata and execute the pinned pipeline stages sequentially."""

    def __init__(
        self,
        root: Path,
        config: Path,
        torch_home: Path,
        samples: Path | None = None,
        max_seconds: float = 300.0,
        probe: Callable[[Path, float], dict] = probe_video,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config.resolve()
        self.torch_home = torch_home.resolve()
        self.samples_dir = samples.resolve() if samples else None
        self.max_seconds = max_seconds
        self.probe = probe
        self.started = time.time()
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
                stopped = stop_leftover_worker(item.get("pid"), p.parent)
                item.update(
                    status="interrupted",
                    stage=None,
                    pid=None,
                    error="The server stopped while this run was active. Partial outputs are kept; "
                    "start a new run to analyse the video again."
                    + (" Its leftover processing worker was stopped." if stopped else ""),
                )
                save_json(p, item)
            self.items[item["id"]] = item

    # -- environment ---------------------------------------------------------------
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
            "Person detector": self.torch_home
            / "hub"
            / "checkpoints"
            / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        }
        checks = [
            {"name": name, "ok": path.is_file(), "path": str(path)} for name, path in paths.items()
        ]
        try:
            _, checkpoint = find_artifact(DEFAULT_ARTIFACT)
            ok = checkpoint.is_file()
        except FileNotFoundError:
            checkpoint, ok = DEFAULT_ARTIFACT, False
        checks.append({"name": "EGTR checkpoint", "ok": ok, "path": str(checkpoint)})
        free = shutil.disk_usage(self.root).free
        checks.append(
            {
                "name": "Free disk space (≥ 5 GB)",
                "ok": free >= 5 * 1024**3,
                "path": f"{free / 1024**3:.0f} GB free in {self.root}",
            }
        )
        return {
            "ready": all(c["ok"] for c in checks),
            "checks": checks,
            "work_root": str(self.root),
            "default_tau": cfg["tau_obs"],
            "default_window": cfg["params"]["FB"]["window_size"],
            "max_seconds": self.max_seconds,
            "server_started": self.started,
        }

    def selfcheck(self, timeout: float = 180.0) -> list[dict]:
        """Import each inference environment's key packages and report the GPU (pre-demo check)."""
        with self.lock:
            if self.active:
                raise BusyError(
                    "Wait for the active run to finish before running the environment check."
                )
        cfg = load_config(self.config)
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=str(ROOT / "src"))
        probes = {
            "Tracking environment": (
                cfg["python_perception"],
                "import torch, boxmot, cv2; print('boxmot', boxmot.__version__, '| torch',"
                " torch.__version__, '| CUDA', torch.cuda.is_available())",
            ),
            "Relationship environment": (
                cfg["python_egtr"],
                "import torch, transformers, timm; print('transformers',"
                " transformers.__version__, '| torch', torch.__version__, '| CUDA',"
                " torch.cuda.is_available())",
            ),
        }
        out = []
        for name, (py, code) in probes.items():
            t0 = time.time()
            try:
                r = subprocess.run(
                    [py, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=ROOT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                last = (r.stdout.strip().splitlines() or [""])[-1]
                err = (r.stderr.strip().splitlines() or [""])[-1]
                out.append(
                    {
                        "name": name,
                        "ok": r.returncode == 0 and "CUDA True" in last,
                        "detail": last if r.returncode == 0 else err,
                        "seconds": round(time.time() - t0, 1),
                    }
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                out.append(
                    {
                        "name": name,
                        "ok": False,
                        "detail": str(exc),
                        "seconds": round(time.time() - t0, 1),
                    }
                )
        return out

    # -- samples -----------------------------------------------------------------
    def samples(self) -> list[dict]:
        """Allowlisted sample clips: videos in the samples folder with a .json description."""
        if not self.samples_dir or not self.samples_dir.is_dir():
            return []
        out = []
        for p in sorted(self.samples_dir.iterdir()):
            meta = p.with_suffix(".json")
            if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file() and meta.is_file():
                info = json.loads(meta.read_text(encoding="utf-8"))
                out.append(
                    {
                        "name": p.name,
                        "title": info.get("title", p.name),
                        "source": info.get("source", ""),
                        "licence": info.get("licence", ""),
                        "attribution": info.get("attribution", ""),
                        "duration_s": info.get("duration_s"),
                        "bytes": p.stat().st_size,
                    }
                )
        return out

    # -- jobs --------------------------------------------------------------------
    def snapshot(self, job_id: str | None = None) -> dict | list:
        """Return a copy safe for HTTP serialization."""
        with self.lock:
            if job_id:
                return copy.deepcopy(self.items[job_id])
            return sorted(
                copy.deepcopy(list(self.items.values())), key=lambda j: j["created"], reverse=True
            )

    def update(self, job_id: str, **values: object) -> None:
        with self.lock:
            self.items[job_id].update(values, updated=time.time())
            save_json(self.root / job_id / "job.json", self.items[job_id])

    def _new_job(self, name: str, options: dict, size: int, origin: str) -> tuple[str, Path]:
        with self.lock:
            if self.active:
                raise BusyError("A run is already active. Wait for it to finish or cancel it.")
            job_id = uuid.uuid4().hex
            work = self.root / job_id
            work.mkdir()
            self.items[job_id] = {
                "id": job_id,
                "name": name,
                "created": time.time(),
                "origin": origin,
                "status": "uploading",
                "stage": None,
                "completed_stages": [],
                "stage_times": {},
                "options": options,
                "bytes": size,
                "error": None,
            }
            try:
                self.update(job_id)
            except OSError:
                del self.items[job_id]
                raise
            self.active = job_id
            self.cancelled.clear()
        return job_id, work

    def upload(self, source: BinaryIO, size: int, name: str, options: dict) -> str:
        """Stream a bounded request body to a unique run; preserve incomplete uploads."""
        job_id, work = self._new_job(name, options, size, "upload")
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
            return self._queue(job_id, work, video, digest.hexdigest())
        except Exception as exc:
            self._fail_start(job_id, exc)
            raise

    def start_sample(self, sample: str, options: dict) -> str:
        """Start a run on an allowlisted sample clip (copied into the run; never a client path)."""
        match = next((s for s in self.samples() if s["name"] == sample), None)
        if match is None:
            raise KeyError("Unknown sample clip.")
        src = self.samples_dir / match["name"]
        job_id, work = self._new_job(match["name"], options, src.stat().st_size, "sample")
        try:
            video = work / ("input" + src.suffix.lower())
            shutil.copyfile(src, video)
            self.update(
                job_id, sample={k: match[k] for k in ("title", "source", "licence", "attribution")}
            )
            return self._queue(job_id, work, video, hashlib.sha256(video.read_bytes()).hexdigest())
        except Exception as exc:
            self._fail_start(job_id, exc)
            raise

    def _fail_start(self, job_id: str, exc: Exception) -> None:
        try:
            self.update(job_id, status="failed", error=str(exc))
        finally:
            with self.lock:
                self.active = None

    def _queue(self, job_id: str, work: Path, video: Path, sha: str) -> str:
        options = self.items[job_id]["options"]
        cfg = load_config(self.config)
        cfg["tau_obs"] = options["tau"]
        cfg["params"]["FB"]["window_size"] = options["window"]
        (work / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "configs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.update(
            job_id,
            status="queued",
            input_sha256=sha,
            git_commit=git.stdout.strip(),
            git_dirty_src=bool(dirty),
            config_sha256=hashlib.sha256((work / "config.yaml").read_bytes()).hexdigest(),
            settings={
                "method": options["method"],
                "takeover_flag": options["flag"],
                "score_cutoff": options["tau"],
                "buffer_steps": options["window"],
                "relationship_stride": 3,
            },
        )
        self.thread = threading.Thread(target=self._run, args=(job_id, video, cfg), daemon=True)
        self.thread.start()
        return job_id

    def _run(self, job_id: str, video: Path, cfg: dict) -> None:
        work = self.root / job_id
        options = self.items[job_id]["options"]
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            TORCH_HOME=str(self.torch_home),
            TAORG_TORCH_HOME=str(self.torch_home),
            HF_HUB_OFFLINE="1",
        )
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        stage = None
        try:
            self.update(job_id, status="running", stage="check")
            self.update(job_id, video=self.probe(video, self.max_seconds))
            with (work / "run.log").open("ab", buffering=0) as log:
                for stage in STAGES:
                    with self.lock:
                        if self.cancelled.is_set():
                            break
                        times = dict(
                            self.items[job_id]["stage_times"], **{stage: [time.time(), None]}
                        )
                        self.update(job_id, status="running", stage=stage, stage_times=times)
                        py = cfg.get(
                            "python_" + ("egtr" if stage == "sgg" else stage), sys.executable
                        )
                        cmd = [
                            py,
                            "-u",
                            "-m",
                            "taorg.pipeline",
                            "stage",
                            stage,
                            "--work",
                            str(work),
                            "--video",
                            str(video),
                            "--config",
                            str(work / "config.yaml"),
                            "--method",
                            options["method"],
                            "--flag" if options["flag"] else "--no-flag",
                        ]
                        log.write((f"\n[TAORG] {stage}\n").encode())
                        self.process = subprocess.Popen(
                            cmd,
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                            start_new_session=os.name != "nt",
                        )
                        self.update(job_id, pid=self.process.pid)
                    code = self.process.wait()
                    with self.lock:
                        self.process = None
                        self.update(job_id, pid=None)
                        times = dict(self.items[job_id]["stage_times"])
                        times[stage] = [times[stage][0], time.time()]
                        self.update(job_id, stage_times=times)
                    if self.cancelled.is_set():
                        break
                    if code:
                        raise RuntimeError(
                            f"The '{STAGE_LABELS[stage]}' stage failed (exit code {code}). "
                            "Open the processing log for the error message."
                        )
                    self.update(job_id, completed_stages=list(STAGES[: STAGES.index(stage) + 1]))
            if self.cancelled.is_set():
                self.update(job_id, status="cancelled", stage=None)
            else:
                data = json.loads((work / "graphs.json").read_text(encoding="utf-8"))
                edges = [e for f in data["frames"] for e in f["edges"]]
                self.update(
                    job_id,
                    status="completed",
                    stage=None,
                    summary={
                        "steps": len(data["frames"]),
                        "edges": len(edges),
                        "current": sum(e["observed"] for e in edges),
                        "retained": sum(not e["observed"] for e in edges),
                        "steps_with_edges": sum(bool(f["edges"]) for f in data["frames"]),
                        "tracks": len({t for f in data["frames"] for t in f["tracks"]}),
                    },
                )
        except VideoError as exc:
            self.update(job_id, status="failed", error=str(exc), stage=None, failed_stage="check")
        except Exception as exc:
            self.update(
                job_id,
                status="cancelled" if self.cancelled.is_set() else "failed",
                error=str(exc),
                stage=None,
                failed_stage=stage,
            )
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
                    subprocess.run(
                        ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)

    def close(self) -> None:
        if self.active and self.items[self.active]["status"] != "uploading":
            self.cancel(self.active)
        if self.thread:
            self.thread.join(timeout=15)


def _command_line(pid: int) -> str:
    """Command line of a running process, or an empty string if it is gone (stdlib only)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return out.stdout.strip()
        return (
            Path(f"/proc/{int(pid)}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def stop_leftover_worker(pid: int | None, work: Path) -> bool:
    """After a server crash, stop the run's pipeline worker if it is still alive.

    The process is stopped only if its command line runs ``taorg.pipeline`` on this run's work
    folder, so a reused PID belonging to an unrelated program is never touched. Returns True if
    one was stopped.
    """
    if not pid:
        return False
    cmdline = _command_line(pid)
    if "taorg.pipeline" not in cmdline or work.name not in cmdline:
        return False
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except OSError:
            return False
    return True


STAGE_LABELS = {
    "check": "Check video",
    "perception": "Track people",
    "sgg": "Predict relationships",
    "maintain": "Maintain graph",
    "view": "Build viewer",
}
