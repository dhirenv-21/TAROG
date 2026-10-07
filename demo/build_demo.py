"""Build the TAORG presentation demo (synthetic scenario, real temporal models).

Steps:
1. Write the scenario truth and the separate observation file (demo/output/).
2. Convert observations into taorg FrameGraphs and run each repository method
   (taorg.temporal.METHODS) through the standard reset_video / step / finalize_video loop.
   Every method receives the identical FrameGraph list.
3. Embed truth, observations and model outputs into a self-contained HTML player.

Usage:
    python demo/build_demo.py            # build demo/output/taorg_demo.html
    python demo/build_demo.py --open     # build and open it in the default browser
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import scenario as S  # noqa: E402

from taorg.schemas.edges import EdgeKey, EdgeObservation, FrameGraph  # noqa: E402
from taorg.schemas.tracks import VisibilityState  # noqa: E402
from taorg.temporal import METHODS  # noqa: E402

OUT = ROOT / "output"
VIDEO_ID = "synthetic_demo"

# Illustrative DEMO SETTINGS (hand-picked to make behaviour visible; not tuned, not research results).
DEMO_METHODS: dict[str, dict] = {
    "no_memory": {},
    "ema": {"alpha": 0.5, "decay": 0.9, "score_threshold": 0.3},
    "gated_ema": {"alpha": 0.5, "decay_visible": 0.5, "decay_occluded": 0.98, "decay_unknown": 0.9,
                  "score_threshold": 0.3},
    "fixed_window": {"window_size": 10},
    "va_hmm": {"p_stay": 0.995, "p_birth": 0.001, "prior": 0.5, "theta_vis": 0.05, "theta_occ": 0.99,
               "theta_unk": 0.6, "false_pos": 0.01, "threshold": 0.5},
}
LABELS = {"no_memory": "No Memory", "ema": "EMA", "gated_ema": "GatedEMA (visibility-aware)",
          "fixed_window": "Fixed Window", "va_hmm": "Visibility-aware HMM"}
EDGES = [(r["subject"], r["object"], r["predicate"]) for r in S.TRUTH["relations"]]


def to_framegraphs(obs: list[dict]) -> list[FrameGraph]:
    """Observation file -> FrameGraphs (the only thing a model sees)."""
    out = []
    for fr in obs:
        edges = []
        for e in fr["edges"]:
            vis = VisibilityState[e["visibility"]]
            edges.append(EdgeObservation(
                edge_key=EdgeKey(e["subject"], e["object"], e["predicate"]), video_id=VIDEO_ID,
                frame_idx=fr["frame"], timestamp=fr["t"], predicate_score=e["score"], observed=e["observed"],
                subject_visibility=VisibilityState.VISIBLE, object_visibility=vis, source_model="demo_observations"))
        out.append(FrameGraph(VIDEO_ID, fr["frame"], fr["t"], [], edges))
    return out


def internal_score(model, key: EdgeKey) -> float | None:
    """Model memory value for an edge if the model exposes one (EMA score / HMM belief)."""
    for attr in ("_edge_scores", "_belief"):
        d = getattr(model, attr, None)
        if isinstance(d, dict):
            v = d.get(key)
            return None if v is None else float(v)
    return None


def run_method(method: str, params: dict, frames: list[FrameGraph]) -> dict:
    """Run one repository method over the frames; record outputs per frame and edge."""
    model = METHODS[method](**params)
    model.reset_video(VIDEO_ID)
    per_frame = []
    for fg in frames:
        out = model.step(fg)
        present = {o.edge_key: o for o in out.edge_observations}
        row = []
        for (s, o, p) in EDGES:
            k = EdgeKey(s, o, p)
            e = present.get(k)
            row.append({"present": e is not None, "out_score": None if e is None else round(float(e.predicate_score), 6),
                        "memory": None if internal_score(model, k) is None else round(internal_score(model, k), 6)})
        per_frame.append(row)
    model.finalize_video()
    return {"label": LABELS[method], "params": params, "frames": per_frame}


def display_states(obs: list[dict], result: dict) -> list[list[str]]:
    """Display state per frame and edge, derived only from model output + model input.

    observed = output present and input observed now; retained = output present, input missing
    (temporal memory); removed = output absent after having been present; inactive = never output yet.
    """
    states, ever = [], [False] * len(EDGES)
    for fr, row in zip(obs, result["frames"]):
        srow = []
        for j, (e, r) in enumerate(zip(fr["edges"], row)):
            if r["present"]:
                ever[j] = True
                srow.append("observed" if e["observed"] else "retained")
            else:
                srow.append("removed" if ever[j] else "inactive")
        states.append(srow)
    return states


def summarise(obs: list[dict], states: list[list[str]]) -> dict:
    """Descriptive scenario facts for the holding edge (demo only, not a metric)."""
    rel = S.TRUTH["relations"][0]
    fails = {}
    for fl in S.FAILURES:
        kept = sum(states[f][0] in ("observed", "retained") for f in range(fl["start"], fl["end"]))
        fails[fl["why"]] = f"kept {kept}/{fl['end'] - fl['start']} frames"
    rel_f = rel["end"]
    drop = next((f for f in range(rel_f, S.N_FRAMES) if states[f][0] not in ("observed", "retained")), None)
    delay = None if drop is None else drop - rel_f      # frames still shown after the true release
    return {"gaps": fails, "removal_delay_frames": delay,
            "removal_delay_s": None if delay is None else round(delay / S.FPS, 3)}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT.parent, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance only
        return "unknown"


def build() -> dict:
    """Compute everything and return the data embedded in the player."""
    obs = S.build_observations()
    frames = to_framegraphs(obs)
    obs_hash = hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest()
    methods = {}
    for m, params in DEMO_METHODS.items():
        res = run_method(m, params, frames)
        res["states"] = display_states(obs, res)
        res["summary"] = summarise(obs, res["states"])
        res["input_sha256"] = obs_hash          # identical input for every method
        methods[m] = res
    return {
        "meta": {"label": "Synthetic scenario with controlled observations",
                 "disclaimer": "No video understanding: the animation is drawn from the scenario truth; "
                               "the observations are hand-specified; only the temporal outputs are computed "
                               "by the repository's methods (taorg.temporal) with illustrative demo settings. "
                               "This demo is not research evidence.",
                 "fps": S.FPS, "n_frames": S.N_FRAMES, "git_commit": git_commit(),
                 "observations_sha256": obs_hash},
        "nodes": {str(k): v for k, v in S.NODES.items()},
        "predicates": {str(k): v for k, v in S.PREDICATES.items()},
        "edges": [{"subject": s, "object": o, "predicate": p} for s, o, p in EDGES],
        "truth": S.TRUTH, "failures": S.FAILURES, "observations": obs, "methods": methods,
    }


def write(data: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenario_truth.json").write_text(json.dumps({"truth": data["truth"], "nodes": data["nodes"],
                                                        "predicates": data["predicates"]}, indent=1), encoding="utf-8")
    (OUT / "observations.json").write_text(json.dumps({"label": data["meta"]["label"],
                                                       "failures": data["failures"],
                                                       "frames": data["observations"]}, indent=1), encoding="utf-8")
    (OUT / "demo_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    html = (ROOT / "player.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    path = OUT / "taorg_demo.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true", help="open the player in the default browser")
    a = ap.parse_args()
    data = build()
    path = write(data)
    print(f"Synthetic scenario with controlled observations — demo written to {path}")
    for m, r in data["methods"].items():
        print(f"  {r['label']:30s} removal after release: {r['summary']['removal_delay_frames']} frames; "
              + "; ".join(f"{k.split(':')[0]}: {v}" for k, v in r["summary"]["gaps"].items()))
    if a.open:
        webbrowser.open(path.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
