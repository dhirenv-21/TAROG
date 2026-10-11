"""Episode world, corruption injectors, instance construction, sampler and fitting/evaluation separation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from taorg.data.episode_world import EpisodeWorldConfig, render, sample_world
from taorg.experiments.episodes.conditions import EpisodeCondition, build_instance, fit_transition
from taorg.experiments.episodes.search import half_indices, method_configs
from taorg.schemas.tracks import VisibilityState

CFG = yaml.safe_load(Path("configs/experiments/episodes.yaml").read_text(encoding="utf-8"))
WCFG = EpisodeWorldConfig.from_dict(CFG["world"])
COND = {n: EpisodeCondition.from_dict(n, d) for n, d in CFG["conditions"].items()}


def inst(name: str, seed: int = 9300):
    return build_instance(COND[name], seed, CFG["world"], CFG["base_corruption"])


def test_world_is_deterministic_with_valid_recurring_episodes():
    a, b = sample_world(11, WCFG), sample_world(11, WCFG)
    assert [(k.edge_key, k.episodes) for k in a.keys] == [(k.edge_key, k.episodes) for k in b.keys]
    n_rec = 0
    for s in range(20):
        w = sample_world(s, WCFG)
        for ke in w.keys:
            eps = ke.episodes
            assert all(0 <= x <= y < WCFG.num_frames for x, y in eps)
            assert all(eps[i][1] < eps[i + 1][0] for i in range(len(eps) - 1)), "disjoint, sorted"
            n_rec += len(eps) - 1
    assert n_rec > 20, "the world contains genuine recurrences"
    assert any(r for s in range(20) for r in sample_world(s, WCFG).succession if "skipped" not in r)


def test_render_marks_alive_frames_only():
    w = sample_world(5, WCFG)
    frames = render(w, "v")
    for ke in w.keys:
        for f in range(0, WCFG.num_frames, 7):
            o = next(o for o in frames[f].edge_observations if o.edge_key == ke.edge_key)
            assert o.observed == ke.alive(f)


def test_gap_and_termination_injectors_respect_ground_truth():
    i = inst("C2_hidden_termination")
    for e in i.manifests["occlusion"]["entries"]:
        k = tuple(e["edge_key_tuple"])
        eps = next(v for kk, v in i.gt_episodes.items() if tuple(kk) == k)
        assert any(a <= e["start_frame"] and e["end_frame"] <= b for a, b in eps), "gaps lie inside episodes"
    # background false detections are injected LATER in the registered order and may land in a block
    fd = {(tuple(e["edge_key_tuple"]), e["start_frame"]) for e in i.manifests["false_detection"]["entries"]}
    for e in i.manifests["termination_occlusion"]["entries"]:
        k = tuple(e["edge_key_tuple"])
        eps = next(v for kk, v in i.gt_episodes.items() if tuple(kk) == k)
        assert any(e["start_frame"] <= b < e["end_frame"] for _, b in eps), "spans a genuine end"
        for f in range(e["start_frame"], e["end_frame"] + 1):
            o = next(o for o in i.input_frames[f].edge_observations if tuple(o.edge_key) == k)
            if (k, f) in fd:
                assert o.observed
                continue
            assert not o.observed and o.edge_visibility == VisibilityState.OCCLUDED_OR_UNRELIABLE


def test_misleading_flags_only_on_dead_frames_and_false_detections_only_when_dead():
    i = inst("C3_misleading_post")
    alive = {tuple(k): v for k, v in i.gt_episodes.items()}
    for e in i.manifests["post_termination_flags"]["entries"]:
        k = tuple(e["edge_key_tuple"])
        assert not any(a <= f <= b for f in range(e["start_frame"], e["end_frame"] + 1) for a, b in alive[k])
    for e in i.manifests["false_detection"]["entries"]:
        k, f = tuple(e["edge_key_tuple"]), e["start_frame"]
        assert not any(a <= f <= b for a, b in alive[k])
        assert any(o.observed for o in i.input_frames[f].edge_observations if tuple(o.edge_key) == k)


def test_boundary_flicker_is_near_boundaries():
    i = inst("C5_boundary_flicker")
    alive = {tuple(k): v for k, v in i.gt_episodes.items()}
    w = CFG["conditions"]["C5_boundary_flicker"]["boundary_flicker"]["window"]
    for e in i.manifests["boundary_flicker"]["entries"]:
        k, f = tuple(e["edge_key_tuple"]), e["start_frame"]
        bounds = [a for a, _ in alive[k]] + [b + 1 for _, b in alive[k]]
        assert min(abs(f - x) for x in bounds) <= w


def test_instances_deterministic_and_paired_across_conditions():
    a, b = inst("C1_gaps", 6000), inst("C1_gaps", 6000)
    assert a.manifest_sha256 == b.manifest_sha256
    c = inst("C3_misleading_post", 6000)
    assert c.gt_episodes == a.gt_episodes, "same seed -> same world in conditions without world overrides"
    assert inst("C6_id_switch").manifests["id_switch"]["entries"], "ID swaps injected"
    m = inst("C7_motion", 6000)
    assert m.gt_episodes == a.gt_episodes


def test_model_input_carries_no_truth():
    i = inst("C6_id_switch")
    fields = {"edge_key", "video_id", "frame_idx", "timestamp", "predicate_score", "observed",
              "subject_visibility", "object_visibility", "source_model", "source_checkpoint"}
    for fr in i.input_frames[:5]:
        for o in fr.edge_observations:
            assert set(o.__slots__) == fields


def test_sampler_common_random_numbers_and_half_control():
    mc = {c.config_id: c.params for c in method_configs(CFG, CFG["methods_order"])}
    for i in (0, 7, 199):
        g, h, d = mc[f"gated_ema#{i:04d}"], mc[f"gated_ema_hyst#{i:04d}"], mc[f"gated_ema_id#{i:04d}"]
        assert all(h[k] == v and d[k] == v for k, v in g.items())
        p1, p2, p3, p4 = (mc[f"{m}#{i:04d}"] for m in ("episode", "episode_id", "episode_ctx", "episode_hist"))
        assert all(p2[k] == v for k, v in p1.items()) and all(p3[k] == v for k, v in p2.items())
        assert all(p4[k] == v for k, v in p3.items())
        assert (p2["sigma_id"], p2["r_id"]) == (d["sigma_id"], d["r_id"]), "identical identity draws as B5"
    assert len([k for k in mc if k.startswith("episode#")]) == 200
    h = half_indices(CFG)
    assert len(h) == 100 and h == half_indices(CFG)


def test_transition_fit_uses_only_given_ground_truth():
    gts = []
    for s in CFG["seeds"]["calibration"][:5]:
        w = sample_world(s, WCFG)
        gts.append({ke.edge_key: list(ke.episodes) for ke in w.keys})
    E = fit_transition(gts, WCFG.num_frames, WCFG.num_predicates, 10, 15)
    assert len(E) == 6 and all(0.0 <= v < 1.0 for row in E for v in row)
    partners = WCFG.predicate_partners
    assert sum(E[r][partners[r]] for r in range(6)) > sum(E[r][q] for r in range(6) for q in range(6)
                                                          if q not in (r, partners[r]))
    assert set(CFG["seeds"]["calibration"]).isdisjoint(CFG["seeds"]["evaluation"])
    assert set(CFG["seeds"]["smoke"]).isdisjoint(CFG["seeds"]["evaluation"])


def test_run_fit_records_only_calibration_instances(tmp_path):
    from taorg.experiments.episodes.pipeline import load_config, run_fit
    cfg = load_config(Path("configs/experiments/episodes.yaml"))
    run_fit(cfg, tmp_path)
    rec = json.loads((tmp_path / "configs" / "transition.json").read_text(encoding="utf-8"))
    used_seeds = {int(x.split("|s")[1]) for x in rec["fitted_on"]}
    assert used_seeds == set(cfg["seeds"]["calibration"])
    assert not any(x.startswith("S") for x in rec["fitted_on"]), "transfer conditions never used for fitting"
