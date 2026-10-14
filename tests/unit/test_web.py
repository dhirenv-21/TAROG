"""Local web boundary and job lifecycle tests; no downloads or GPU required."""

import hashlib
import http.client
import io
import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from taorg.web.jobs import BusyError, Jobs
from taorg.web.server import LocalServer, options_from_query


@pytest.fixture
def jobs(tmp_path):
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump({"python_perception": "tracking-python",
                                     "python_egtr": "egtr-python", "tau_obs": 1e-6,
                                     "params": {"FB": {"window_size": 10}}}))
    return Jobs(tmp_path / "runs", config, tmp_path / "torch")


def test_settings_and_flag_compatibility():
    assert options_from_query({})["method"] == "FB"
    for query in ({"tau": ["nan"]}, {"tau": ["inf"]}, {"window": ["0"]},
                  {"method": ["shell"]}, {"method": ["VH"], "flag": ["true"]}):
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
    restarted = Jobs(jobs.root, jobs.config, jobs.torch_home)
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
        def __init__(self, cmd, **kwargs):
            commands.append(cmd)
            self.stage = cmd[5]
            assert kwargs["env"]["TORCH_HOME"] == str(jobs.torch_home)
            assert "shell" not in kwargs
            if self.stage == "view":
                work = Path(cmd[cmd.index("--work") + 1])
                (work / "graphs.json").write_text('{"frames":[{"edges":[]}]}')

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
        assert request("POST", "/api/jobs?name=x.mp4", b"x", {
            **auth, "Origin": "https://evil.example"})[0] == 403
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
    jobs.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                    start_new_session=os.name != "nt")
    try:
        jobs.cancel(job_id)
        assert jobs.process.wait(timeout=10) != 0
        assert jobs.cancelled.is_set()
        assert (jobs.root / job_id / "job.json").exists()
    finally:
        if jobs.process.poll() is None:
            jobs.process.kill()
