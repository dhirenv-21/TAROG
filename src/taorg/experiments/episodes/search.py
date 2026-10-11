"""Episode-study configuration sampler (prereg §5; common random numbers).

Each method = (base family, [extensions]).  Configuration i of a method combines
draw i of its base family's stream with draw i of each extension's stream.
Streams are seeded by (sampler seed, "episodes_search", family/extension name),
so e.g. GatedEMA#i, GatedEMA+hyst#i and GatedEMA+ID#i share alpha/decays/threshold,
and Episode-P2#i .. P4#i share P1#i's draws plus the same identity draws as B5#i.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from taorg.experiments.phase3.conditions import derive_seed
from taorg.experiments.phase3.grids import MethodConfig, expand_grid


def _draw(rng: random.Random, spec: list[Any]) -> float:
    kind, lo, hi = spec[0], float(spec[1]), float(spec[2])
    if kind == "uniform":
        return rng.uniform(lo, hi)
    if kind == "loguniform":
        return math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if kind == "one_minus_loguniform":
        return 1.0 - math.exp(rng.uniform(math.log(lo), math.log(hi)))
    raise ValueError(f"Unknown distribution '{kind}'")


def draw_stream(name: str, space: dict[str, list[Any]], n: int, sampler_seed: int) -> list[dict[str, float]]:
    """n parameter dicts from one named stream (keys drawn in declared order)."""
    rng = random.Random(derive_seed(sampler_seed, "episodes_search", name))
    return [{k: _draw(rng, v) for k, v in space.items()} for _ in range(n)]


def method_configs(cfg: dict[str, Any], methods: list[str]) -> list[MethodConfig]:
    """All configurations of the given methods (enumerated or sampled), in order."""
    s = cfg["search"]
    n = int(s["n_samples"])
    seed = int(cfg["seeds"]["sampler"])
    cache: dict[str, list[dict[str, float]]] = {}

    def stream(name: str, space: dict) -> list[dict[str, float]]:
        if name not in cache:
            cache[name] = draw_stream(name, space, n, seed)
        return cache[name]

    out: list[MethodConfig] = []
    for m in methods:
        if m in s["enumerated"]:
            out.extend(expand_grid(m, s["enumerated"][m]))
            continue
        if m not in s["methods"]:
            raise KeyError(f"No search definition for method '{m}'")
        family, exts = s["methods"][m]
        base = stream(f"base:{family}", s["base_spaces"][family])
        ext = [stream(f"ext:{e}", s["extension_spaces"][e]) for e in exts]
        for i in range(n):
            params = dict(base[i])
            for e in ext:
                params.update(e[i])
            out.append(MethodConfig(config_id=f"{m}#{i:04d}", method=m,
                                    params_json=json.dumps(params, sort_keys=True), grid_index=i))
    return out


def half_indices(cfg: dict[str, Any]) -> set[int]:
    """The registered seeded random half of the sample indices (same for every method)."""
    n = int(cfg["search"]["n_samples"])
    rng = random.Random(derive_seed(int(cfg["seeds"]["half_control"]), "episodes_half"))
    return set(rng.sample(range(n), int(round(float(cfg["criteria"]["half_control_fraction"]) * n))))
