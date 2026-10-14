"""Local web boundary and job lifecycle tests; no downloads or GPU required."""

import hashlib
import http.client
import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from taorg.web.jobs import BusyError, Jobs, VideoError, probe_video
from taorg.web.server import LocalServer, options_from_query

FAKE_VIDEO = {"width": 64, "height": 48, "fps": 30.0, "frames": 30, "duration_s": 1.0}


@pytest.fixture
def jobs(tmp_path):
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "python_perception": "tracking-python",
                "python_egtr": "egtr-python",
                "tau_obs": 1e-6,
                "params": {"FB": {"window_size": 10}},
            }
        )
    )
    return Jobs(
        tmp_path / "runs",
        config,
        tmp_path / "torch",
        samples=tmp_path / "samples",
        probe=lambda path, limit: dict(FAKE_VIDEO),
    )


def test_settings_and_flag_compatibility():
    assert options_from_query({})["method"] == "FB"
    for query in (
        {"tau": ["nan"]},
        {"tau": ["inf"]},
        {"window": ["0"]},
        {"method": ["shell"]},
        {"method": ["VH"], "flag": ["true"]},
    ):
        with pytest.raises(ValueError):
            options_from_query(query)
    assert options_from_query({"method": ["NM"], "flag": ["false"]})["flag"] is False


def test_upload_hash_and_saved_configuration(jobs, monkeypatch):
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    name = 'a video & quote".mp4'
    job_id = jobs.upload(io.BytesIO(b"video"), 5, name, options_from_query({}))
    jobs.thread.join()
    item = jobs.snapshot(job_id)
    assert item["name"] == name
    assert item["input_sha256"] == hashlib.sha256(b"video").hexdigest()
    assert (jobs.root / job_id / "input.mp4").read_bytes() == b"video"
    assert yaml.safe_load((jobs.root / job_id / "config.yaml").read_text())["tau_obs"] > 0
    with pytest.raises(BusyError):
        jobs.upload(io.BytesIO(b"x"), 1, "x.mp4", options_from_query({}))
    restarted = Jobs(jobs.root, jobs.config, jobs.torch_home, probe=jobs.probe)
    assert restarted.snapshot(job_id)["status"] == "interrupted"


def test_incomplete_upload_kept_and_slot_released(jobs):
    with pytest.raises(ValueError, match="ended early"):
        jobs.upload(io.BytesIO(b"x"), 5, "x.mp4", options_from_query({}))
    item = jobs.snapshot()[0]
    assert item["status"] == "failed"
    assert jobs.active is None
    assert (jobs.root / item["id"] / "input.mp4").read_bytes() == b"x"


@pytest.mark.parametrize("fail_at", [1, 2])
def test_metadata_write_failure_releases_slot(jobs, monkeypatch, fail_at):
    from taorg.web.jobs import save_json

    calls = 0

    def fail(path, value):
        nonlocal calls
        calls += 1
        if calls >= fail_at:
            raise OSError("Disk full")
        save_json(path, value)

    monkeypatch.setattr("taorg.web.jobs.save_json", fail)
    with pytest.raises(OSError, match="Disk full"):
        jobs.upload(io.BytesIO(b"v"), 1, "x.mp4", options_from_query({}))
    assert jobs.active is None
    assert all(j["status"] == "failed" for j in jobs.snapshot())


@pytest.mark.parametrize("failed_stage", [None, "sgg"])
def test_stage_execution_and_errors(jobs, monkeypatch, failed_stage):
    commands = []

    class Process:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            commands.append(cmd)
            self.stage = cmd[5]
            assert kwargs["env"]["TORCH_HOME"] == str(jobs.torch_home)
            assert "shell" not in kwargs
            if self.stage == "view":
                work = Path(cmd[cmd.index("--work") + 1])
                (work / "graphs.json").write_text(
                    json.dumps(
                        {
                            "frames": [
                                {
                                    "tracks": {"1": [0, 0, 1, 1], "2": [2, 2, 3, 3]},
                                    "edges": [
                                        {
                                            "s": 1,
                                            "o": 2,
                                            "p": "watch",
                                            "score": 1e-6,
                                            "observed": True,
                                        }
                                    ],
                                },
                                {
                                    "tracks": {"1": [0, 0, 1, 1]},
                                    "edges": [
                                        {
                                            "s": 1,
                                            "o": 2,
                                            "p": "watch",
                                            "score": 1e-6,
                                            "observed": False,
                                        }
                                    ],
                                },
                            ]
                        }
                    )
                )

        def wait(self):
            return 1 if self.stage == failed_stage else 0

    # Patch only Popen during the worker; git provenance still uses real subprocess.run.
    original = jobs._run

    def run(*args):
        with patch("taorg.web.jobs.subprocess.Popen", Process):
            original(*args)

    monkeypatch.setattr(jobs, "_run", run)
    job_id = jobs.upload(io.BytesIO(b"v"), 1, "x.mp4", options_from_query({}))
    jobs.thread.join(timeout=10)
    item = jobs.snapshot(job_id)
    assert item["status"] == ("failed" if failed_stage else "completed")
    assert commands[0][0] == "tracking-python"
    assert commands[1][0] == "egtr-python"
    assert jobs.active is None
    assert len(commands) == (2 if failed_stage else 4)
    assert item["video"] == FAKE_VIDEO and "perception" in item["stage_times"]
    assert item["pid"] is None
    if failed_stage:
        assert item["failed_stage"] == "sgg" and "Predict relationships" in item["error"]
    else:
        assert item["summary"] == {
            "steps": 2,
            "edges": 2,
            "current": 1,
            "retained": 1,
            "steps_with_edges": 2,
            "tracks": 2,
        }
        assert item["settings"]["method"] == "FB" and item["settings"]["relationship_stride"] == 3


def test_http_upload_origin_limits_and_artifact_allowlist(jobs, monkeypatch):
    monkeypatch.setattr(jobs, "readiness", lambda: {"ready": True})
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    server = LocalServer(("127.0.0.1", 0), jobs, 10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(*server.server_address)
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        status, data = response.status, response.read()
        conn.close()
        return status, data

    try:
        assert request("GET", "/")[0] == 200
        assert request("GET", "/api/status", headers={"Host": "evil.example"})[0] == 403
        assert request("POST", "/api/jobs?name=x.mp4", b"x")[0] == 403
        auth = {"X-TAORG-Token": server.token}
        assert (
            request(
                "POST", "/api/jobs?name=x.mp4", b"x", {**auth, "Origin": "https://evil.example"}
            )[0]
            == 403
        )
        assert request("POST", "/api/jobs?name=x.mp4", b"x" * 11, auth)[0] == 413
        assert request("POST", "/api/jobs?name=x.exe", b"x", auth)[0] == 400
        status, data = request("POST", "/api/jobs?name=x.mp4", b"x", auth)
        assert status == 202
        job_id = json.loads(data)["id"]
        assert request("GET", f"/results/{job_id}/input.mp4")[0] == 404
        assert request("GET", f"/results/{job_id}/config.yaml")[0] == 200
        assert request("GET", "/results/../Memory.md")[0] == 404
        assert request("POST", "/api/jobs?name=y.mp4", b"x", auth)[0] == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cancel_real_child_keeps_history(jobs, monkeypatch):
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    job_id = jobs.upload(io.BytesIO(b"x"), 1, "x.mp4", options_from_query({}))
    jobs.thread.join()
    jobs.update(job_id, status="running")
    import os
    import sys

    jobs.process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=os.name != "nt"
    )
    try:
        jobs.cancel(job_id)
        assert jobs.process.wait(timeout=10) != 0
        assert jobs.cancelled.is_set()
        assert (jobs.root / job_id / "job.json").exists()
    finally:
        if jobs.process.poll() is None:
            jobs.process.kill()


def _write_video(path, n, fps=30.0):
    import cv2
    import numpy as np

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), i * 5 % 255, np.uint8))
    w.release()


def test_probe_video_accepts_real_video_and_rejects_bad_ones(tmp_path):
    good = tmp_path / "good.mp4"
    _write_video(good, 30)
    info = probe_video(good, 300)
    assert (
        info["width"] == 64
        and info["height"] == 48
        and info["frames"] == 30
        and info["duration_s"] == 1.0
    )
    with pytest.raises(VideoError, match="up to 0 s"):
        probe_video(good, 0.5)
    short = tmp_path / "short.mp4"
    _write_video(short, 2)
    with pytest.raises(VideoError, match="at least"):
        probe_video(short, 300)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(VideoError, match="MP4"):
        probe_video(bad, 300)


def test_probe_failure_stops_before_any_stage(jobs, monkeypatch):
    def fail(path, limit):
        raise VideoError("No frames could be decoded from this file.")

    jobs.probe = fail
    job_id = jobs.upload(io.BytesIO(b"v"), 1, "x.mp4", options_from_query({}))
    jobs.thread.join(timeout=10)
    item = jobs.snapshot(job_id)
    assert item["status"] == "failed" and item["failed_stage"] == "check"
    assert (
        item["completed_stages"] == [] and not (jobs.root / job_id / "run.log").exists()
    )  # no stage started
    assert jobs.active is None


def test_samples_are_allowlisted_and_copied(jobs, monkeypatch):
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    d = jobs.samples_dir
    d.mkdir()
    (d / "clip.mp4").write_bytes(b"sample-bytes")
    (d / "clip.json").write_text(
        json.dumps({"title": "Clip", "source": "src", "licence": "CC", "attribution": "a"})
    )
    (d / "undescribed.mp4").write_bytes(b"x")  # no description -> not offered
    (d / "notes.json").write_text("{}")
    assert [s["name"] for s in jobs.samples()] == ["clip.mp4"]
    with pytest.raises(KeyError):
        jobs.start_sample("../clip.mp4", options_from_query({}))
    with pytest.raises(KeyError):
        jobs.start_sample("undescribed.mp4", options_from_query({}))
    job_id = jobs.start_sample("clip.mp4", options_from_query({}))
    jobs.thread.join()
    item = jobs.snapshot(job_id)
    assert item["origin"] == "sample" and item["sample"]["licence"] == "CC"
    assert (jobs.root / job_id / "input.mp4").read_bytes() == b"sample-bytes"
    assert item["input_sha256"] == hashlib.sha256(b"sample-bytes").hexdigest()


def test_selfcheck_refuses_while_busy_and_reports_failures(jobs, monkeypatch):
    import sys

    cfg = yaml.safe_load(jobs.config.read_text())
    cfg.update(python_perception=sys.executable, python_egtr=str(jobs.root / "missing-python.exe"))
    jobs.config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("taorg.web.jobs.load_config", lambda p: yaml.safe_load(Path(p).read_text()))
    res = jobs.selfcheck(timeout=60)
    assert [r["name"] for r in res] == ["Tracking environment", "Relationship environment"]
    assert not res[1]["ok"]  # missing interpreter is reported, not raised
    jobs.active = "x"
    with pytest.raises(BusyError):
        jobs.selfcheck()


def test_http_frames_and_samples_endpoints(jobs, monkeypatch):
    monkeypatch.setattr(jobs, "readiness", lambda: {"ready": True})
    monkeypatch.setattr(jobs, "_run", lambda *args: None)
    server = LocalServer(("127.0.0.1", 0), jobs, 10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, headers=None):
        conn = http.client.HTTPConnection(*server.server_address)
        conn.request(method, path, None, headers or {})
        r = conn.getresponse()
        out = r.status, r.read()
        conn.close()
        return out

    try:
        job_id = jobs.upload(io.BytesIO(b"v"), 1, "x.mp4", options_from_query({}))
        jobs.thread.join()
        (jobs.root / job_id / "frames").mkdir()
        (jobs.root / job_id / "frames" / "000001.jpg").write_bytes(b"jpg")
        assert request("GET", f"/results/{job_id}/frames/000001.jpg") == (200, b"jpg")
        assert request("GET", f"/results/{job_id}/frames/000002.jpg")[0] == 404
        assert request("GET", f"/results/{job_id}/frames/..%2Fjob.json")[0] == 404
        assert request("GET", f"/results/{job_id}/frames/1.jpg")[0] == 404
        assert request("GET", "/results/" + "0" * 32 + "/frames/000001.jpg")[0] == 404
        assert request("GET", "/api/samples")[0] == 200
        assert request("POST", "/api/samples/clip.mp4")[0] == 403  # token required
        assert request("POST", "/api/samples/nope.mp4", {"X-TAORG-Token": server.token})[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_restart_stops_only_the_runs_leftover_worker(tmp_path):
    """A crashed server's worker is stopped on restart; an unrelated process is not."""
    from taorg.web.jobs import stop_leftover_worker

    work = tmp_path / ("a" * 32)
    work.mkdir()
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    orphan = subprocess.Popen([*sleeper, "-m", "taorg.pipeline", "--work", str(work)])
    stranger = subprocess.Popen(sleeper)
    try:
        assert stop_leftover_worker(stranger.pid, work) is False
        assert stop_leftover_worker(None, work) is False
        (work / "job.json").write_text(
            json.dumps({"id": work.name, "status": "running", "pid": orphan.pid}), encoding="utf-8"
        )
        jobs = Jobs(tmp_path, tmp_path / "config.yaml", tmp_path / "torch")
        item = jobs.snapshot(work.name)
        assert item["status"] == "interrupted" and item["pid"] is None
        assert "leftover processing worker was stopped" in item["error"]
        assert orphan.wait(timeout=20) is not None
        assert stranger.poll() is None
    finally:
        for proc in (orphan, stranger):
            if proc.poll() is None:
                proc.kill()


def test_second_server_cannot_share_the_port(jobs):
    first = LocalServer(("127.0.0.1", 0), jobs, 1024)
    try:
        with pytest.raises(OSError):
            LocalServer(("127.0.0.1", first.server_address[1]), jobs, 1024)
    finally:
        first.server_close()
