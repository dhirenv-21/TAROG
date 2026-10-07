"""Presentation demo (demo/): displayed outputs equal the repository models' outputs, inputs are
identical across methods, truth is not a model input, and the rendered player stays in sync."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[2] / "demo"
sys.path.insert(0, str(DEMO))

import build_demo as B  # noqa: E402
import scenario as S  # noqa: E402

from taorg.schemas.edges import EdgeKey  # noqa: E402
from taorg.temporal import METHODS  # noqa: E402


@pytest.fixture(scope="module")
def data() -> dict:
    return B.build()


def test_embedded_outputs_equal_a_fresh_run_of_each_repository_method(data):
    frames = B.to_framegraphs(S.build_observations())
    for name, params in B.DEMO_METHODS.items():
        model = METHODS[name](**params)
        model.reset_video(B.VIDEO_ID)
        for f, fg in enumerate(frames):
            out = {o.edge_key: o for o in model.step(fg).edge_observations}
            for j, (s, o, p) in enumerate(B.EDGES):
                rec = data["methods"][name]["frames"][f][j]
                assert rec["present"] == (EdgeKey(s, o, p) in out), (name, f, j)
                if rec["present"]:
                    assert rec["out_score"] == pytest.approx(out[EdgeKey(s, o, p)].predicate_score, abs=1e-6)


def test_display_state_follows_model_output_only(data):
    for name, m in data["methods"].items():
        for f in range(S.N_FRAMES):
            for j in range(len(B.EDGES)):
                present, state = m["frames"][f][j]["present"], m["states"][f][j]
                assert (state in ("observed", "retained")) == present, (name, f, j)
                if state == "retained":
                    assert not data["observations"][f]["edges"][j]["observed"]


def test_every_method_gets_identical_inputs_and_no_truth(data):
    hashes = {m["input_sha256"] for m in data["methods"].values()}
    assert hashes == {data["meta"]["observations_sha256"]}
    allowed = {"subject", "object", "predicate", "observed", "score", "visibility"}
    for fr in S.build_observations():
        for e in fr["edges"]:
            assert set(e) == allowed                     # no truth fields leak into model input


def test_scenario_has_temporary_loss_and_genuine_termination(data):
    obs = data["observations"]
    hold = [fr["edges"][0]["observed"] for fr in obs]
    assert not any(hold[105:135]) and hold[104] and hold[135]          # temporary loss, then reappears
    assert not any(hold[S.TRUTH["release_frame"]:])                   # genuine release
    vis = {fr["edges"][0]["visibility"] for fr in obs[105:135]}
    assert vis == {"OCCLUDED_OR_UNRELIABLE"}
    # no-memory removes immediately; at least one method shows delayed removal after the release
    delays = {k: m["summary"]["removal_delay_frames"] for k, m in data["methods"].items()}
    assert delays["no_memory"] == 0 and max(delays.values()) > 0


def test_lengths_and_timestamps_are_synchronised(data):
    assert len(data["observations"]) == S.N_FRAMES
    for m in data["methods"].values():
        assert len(m["frames"]) == len(m["states"]) == S.N_FRAMES
    for f, fr in enumerate(data["observations"]):
        assert fr["frame"] == f and fr["t"] == pytest.approx(f / S.FPS, abs=1e-6)


def test_written_html_embeds_exactly_the_computed_data(data, tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path)
    html = B.write(data).read_text(encoding="utf-8")
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m and json.loads(m.group(1)) == json.loads(json.dumps(data))
    assert "Synthetic scenario with controlled observations" in html


CHROME = next((p for p in (shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("chromium"),
                           "C:/Program Files/Google/Chrome/Application/chrome.exe",
                           "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
               if p and Path(p).exists()), None)


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome/Edge available")
@pytest.mark.parametrize("frame,method", [(0, "no_memory"), (120, "gated_ema"), (120, "ema"), (177, "va_hmm"),
                                          (199, "fixed_window"), (245, "ema"), (359, "gated_ema")])
def test_rendered_player_shows_the_model_output_for_that_frame(data, tmp_path, monkeypatch, frame, method):
    monkeypatch.setattr(B, "OUT", tmp_path)
    url = B.write(data).as_uri() + f"#frame={frame}&method={method}"
    dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--dump-dom", url],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
    m = re.search(r'<pre id="render-state">(.*?)</pre>', dom, re.S)
    assert m, "player did not render"
    shown = json.loads(m.group(1).replace("&quot;", '"'))
    assert shown["frame"] == frame and shown["method"] == method
    assert shown["output"] == data["methods"][method]["states"][frame]
    assert shown["present"] == [r["present"] for r in data["methods"][method]["frames"][frame]]
    assert shown["input"] == [{"observed": e["observed"], "visibility": e["visibility"]}
                              for e in data["observations"][frame]["edges"]]


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome/Edge available")
def test_playback_clock_advances_at_video_rate(data, tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT", tmp_path)
    url = B.write(data).as_uri() + "#autoplay=1&method=gated_ema"
    dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=4000", "--dump-dom", url],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
    shown = json.loads(re.search(r'<pre id="render-state">(.*?)</pre>', dom, re.S).group(1).replace("&quot;", '"'))
    assert abs(shown["frame"] - 4 * S.FPS) <= 3                       # 4 s of (virtual) wall time -> ~frame 120
    assert shown["output"] == data["methods"]["gated_ema"]["states"][shown["frame"]]
