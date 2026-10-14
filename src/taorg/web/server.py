"""Loopback-only HTTP API and static interface, using Python's standard library."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from taorg.pipeline.cli import DEFAULT_CONFIG, ROOT
from taorg.web.jobs import EXPORTS, BusyError, Jobs

STATIC = Path(__file__).parent / "static"
SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def options_from_query(query: dict) -> dict:
    """Validate the small allowlisted set of UI overrides."""
    method = query.get("method", ["FB"])[0]
    flag = query.get("flag", ["true"])[0]
    tau = float(query.get("tau", ["6.648e-7"])[0])
    window = int(query.get("window", ["10"])[0])
    if method not in {"FB", "NM", "B2", "VH"} or flag not in {"true", "false"}:
        raise ValueError("Invalid method or flag option.")
    if flag == "true" and method not in {"FB", "B2"}:
        raise ValueError("The takeover flag supports Fixed Buffer and GatedEMA only.")
    if not math.isfinite(tau) or not 0 < tau <= 1 or not 1 <= window <= 300:
        raise ValueError("Score cutoff must be > 0 and <= 1; buffer must be 1–300 steps.")
    return {"method": method, "flag": flag == "true", "tau": tau, "window": window}


class LocalServer(ThreadingHTTPServer):
    """Threaded requests with one pipeline worker and a per-launch request token."""

    daemon_threads = True

    def __init__(self, address: tuple, jobs: Jobs, max_upload: int) -> None:
        self.jobs = jobs
        self.token = secrets.token_urlsafe(32)
        self.max_upload = max_upload
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    """Serve only named assets and artifacts inside registered job directories."""

    def _trusted(self, mutation: bool = False) -> bool:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.send_error(403, "Use the localhost address printed by the server.")
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + h for h in hosts}:
            self.send_error(403, "Cross-origin requests are not allowed.")
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send_error(403)
            return False
        if mutation and not secrets.compare_digest(
                self.headers.get("X-TAORG-Token", ""), self.server.token):
            self.send_error(403, "Refresh the page before starting a run.")
            return False
        return True

    def _headers(self, code: int, mime: str, size: int) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")

    def _json(self, data: object, code: int = 200) -> None:
        content = json.dumps(data).encode()
        self._headers(code, "application/json; charset=utf-8", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _file(self, p: Path, mime: str, download: bool = False) -> None:
        self._headers(200, mime, p.stat().st_size)
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
        self.end_headers()
        with p.open("rb") as src:
            shutil.copyfileobj(src, self.wfile)

    def do_GET(self) -> None:
        if not self._trusted():
            return
        url = urlsplit(self.path)
        path = unquote(url.path)
        try:
            assets = {"/": ("index.html", "text/html"),
                      "/app.js": ("app.js", "text/javascript"),
                      "/style.css": ("style.css", "text/css")}
            if path in assets:
                name, mime = assets[path]
                return self._file(STATIC / name, mime + "; charset=utf-8")
            if path == "/api/status":
                return self._json({**self.server.jobs.readiness(), "token": self.server.token,
                                   "max_upload": self.server.max_upload})
            if path == "/api/jobs":
                return self._json(self.server.jobs.snapshot())
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "log":
                self.server.jobs.snapshot(parts[2])
                p = self.server.jobs.root / parts[2] / "run.log"
                if not p.exists():
                    return self._json({"text": "Waiting for processing…"})
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 24000))
                    return self._json({"text": f.read().decode("utf-8", errors="replace")})
            if len(parts) == 3 and parts[0] == "results" and parts[2] in EXPORTS:
                self.server.jobs.snapshot(parts[1])
                p = self.server.jobs.root / parts[1] / parts[2]
                mime = "text/html" if p.suffix == ".html" else "application/octet-stream"
                return self._file(p, mime, download="download" in parse_qs(url.query))
            self.send_error(404)
        except (KeyError, FileNotFoundError):
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:
        if not self._trusted(mutation=True):
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/api/jobs":
                readiness = self.server.jobs.readiness()
                if not readiness["ready"]:
                    message = "Setup is incomplete. Check the environment panel."
                    return self._json({"error": message}, 409)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= self.server.max_upload:
                    return self._json({"error": "Video exceeds the upload limit or is empty."}, 413)
                if size * 3 + 1024**3 > shutil.disk_usage(self.server.jobs.root).free:
                    return self._json({"error": "Not enough free disk space for this upload."}, 507)
                query = parse_qs(url.query)
                name = query.get("name", [""])[0].replace("\\", "/").split("/")[-1]
                if Path(name).suffix.lower() not in SUFFIXES or len(name) > 200:
                    raise ValueError("Choose an MP4, MOV, AVI, MKV, WebM or M4V video.")
                options = options_from_query(query)
                self.connection.settimeout(120)
                job_id = self.server.jobs.upload(self.rfile, size, name, options)
                return self._json({"id": job_id}, 202)
            parts = url.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                self.server.jobs.cancel(parts[2])
                return self._json({"ok": True})
            self.send_error(404)
        except BusyError as exc:
            self._json({"error": str(exc)}, 409)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)
        except OSError as exc:
            self._json({"error": str(exc)}, 500)


def main() -> None:
    """Launch only on loopback; reserve the work folder against a second server."""
    ap = argparse.ArgumentParser(description="TAORG local video workspace")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--work", type=Path, default=ROOT.parent / "TAORG_runs" / "web")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--torch-home", type=Path,
                    default=ROOT.parent / "TAORG_runs" / "downstream" / "torch_home")
    ap.add_argument("--max-upload-mb", type=int, default=2048)
    ap.add_argument("--open", action="store_true", help="Open the local website in your browser")
    args = ap.parse_args()
    if not 1 <= args.port <= 65535 or args.max_upload_mb < 1:
        ap.error("Choose a valid port and a positive upload limit.")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "server.lock").open("a+b") as lock:
        # An OS lock releases even after a crash; never trust a stale PID file.
        try:
            import os
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            ap.error("Another TAORG website is already using this work folder.")
        jobs = Jobs(args.work, args.config, args.torch_home)
        server = LocalServer(("127.0.0.1", args.port), jobs, args.max_upload_mb * 1024**2)
        url = f"http://127.0.0.1:{args.port}"
        print(f"TAORG local workspace: {url}\nRuns saved to {args.work}", flush=True)
        if args.open:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            jobs.close()
            server.server_close()
