"""Parameter grids for the Phase 3 tuning protocol.

Every method is searched over an explicit, config-declared Cartesian grid.
Expansion order is deterministic (YAML key order, then value order), so the
integer index inside ``config_id`` is stable and is the final tie-breaker in
selection.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodConfig:
    """One concrete method configuration.

    Attributes:
        config_id: Stable identifier ``<method>#<index>``.
        method: Registered method name (``taorg.temporal.METHODS``).
        params_json: Canonical JSON of constructor kwargs.
        grid_index: Position in the method's expanded grid.
    """

    config_id: str
    method: str
    params_json: str
    grid_index: int

    @property
    def params(self) -> dict[str, Any]:
        """Constructor kwargs."""
        return json.loads(self.params_json)


def expand_grid(method: str, grid: dict[str, list[Any]]) -> list[MethodConfig]:
    """Expand a {param: [values]} grid into MethodConfigs.

    Args:
        method: Method name.
        grid: Mapping from parameter name to candidate values.  An empty
            mapping yields a single default configuration.

    Returns:
        List of MethodConfig in deterministic order.
    """
    names = list(grid.keys())
    for n in names:
        if not isinstance(grid[n], list) or not grid[n]:
            raise ValueError(f"Grid for {method}.{n} must be a non-empty list")
    combos = itertools.product(*[grid[n] for n in names]) if names else [()]
    out = []
    for i, combo in enumerate(combos):
        params = dict(zip(names, combo))
        out.append(MethodConfig(
            config_id=f"{method}#{i:04d}", method=method,
            params_json=json.dumps(params, sort_keys=True), grid_index=i,
        ))
    return out


def expand_all(grids: dict[str, dict[str, list[Any]]]) -> list[MethodConfig]:
    """Expand grids for all methods (method order as declared)."""
    out: list[MethodConfig] = []
    for method, grid in grids.items():
        out.extend(expand_grid(method, grid or {}))
    return out
