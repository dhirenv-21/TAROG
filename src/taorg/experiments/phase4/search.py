"""Phase 4 configuration search spaces (equal random-search budget).

Every tunable method receives exactly ``n_samples`` configurations drawn from
the distributions declared in ``configs/experiments/phase4.yaml`` by a seeded
stream derived from (sampler seed, method name).  Parameters are drawn in the
declared key order.  V-HPED samples violating ``threshold > nu/(mu+nu) +
margin`` are redrawn in full (formulation §3), from the same stream.
Enumerated spaces (No Memory, Fixed Window) use the Phase 3 grid expander.
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


def _violates(method: str, p: dict[str, float], margin: float) -> bool:
    if not method.startswith("vhped"):
        return False
    return p["threshold"] <= p["nu"] / (p["mu"] + p["nu"]) + margin


def sample_method(method: str, space: dict[str, list[Any]], n: int, sampler_seed: int,
                  margin: float) -> list[MethodConfig]:
    """Draw ``n`` configurations of one method.

    Args:
        method: Registered method name.
        space: {param: [distribution, lo, hi]} in declared order.
        n: Number of samples.
        sampler_seed: Base sampler seed.
        margin: V-HPED threshold margin.

    Returns:
        MethodConfigs ``<method>#0000`` ... in draw order.
    """
    rng = random.Random(derive_seed(sampler_seed, "phase4_search", method))
    out: list[MethodConfig] = []
    for i in range(n):
        for _ in range(10_000):
            params = {k: _draw(rng, v) for k, v in space.items()}
            if not _violates(method, params, margin):
                break
        else:  # pragma: no cover - would indicate an unsatisfiable space
            raise RuntimeError(f"Could not satisfy constraints for {method}")
        out.append(MethodConfig(config_id=f"{method}#{i:04d}", method=method,
                                params_json=json.dumps(params, sort_keys=True), grid_index=i))
    return out


def method_configs(cfg: dict[str, Any], methods: list[str]) -> list[MethodConfig]:
    """All configurations for the given methods, in the given order."""
    s = cfg["search"]
    out: list[MethodConfig] = []
    for m in methods:
        if m in s["enumerated"]:
            out.extend(expand_grid(m, s["enumerated"][m]))
        elif m in s["spaces"]:
            out.extend(sample_method(m, s["spaces"][m], int(s["n_samples"]), int(cfg["seeds"]["sampler"]),
                                     float(s["vhped_threshold_margin"])))
        else:
            raise KeyError(f"No search space for method '{m}'")
    return out
