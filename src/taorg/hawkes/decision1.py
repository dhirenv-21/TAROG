"""Phase 5H-1 pre-registered decision rule (research/phase5h1_preregistration.md §9)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _ok(x: float) -> bool:
    return x is not None and not math.isnan(x)


def _beats(d: Mapping[str, float], null: Sequence[float]) -> bool:
    return _ok(d["ci_lo"]) and d["ci_lo"] > 0 and len(null) > 0 and d["mean"] > max(null)


def _median(x: Sequence[float]) -> float:
    s = sorted(x)
    n = len(s)
    return (s[(n - 1) // 2] + s[n // 2]) / 2 if n else math.nan


def family_passes(fam: Mapping[str, Any], ratio: float) -> bool:
    """fam: {"d_a": stats, "d_b": stats, "d_all": stats, "null": [...], "dur_gain": float}."""
    if not (_ok(fam["d_a"]["ci_lo"]) and fam["d_a"]["ci_lo"] > 0 and _ok(fam["d_b"]["ci_lo"]) and fam["d_b"]["ci_lo"] > 0):
        return False
    if not fam["null"] or fam["d_all"]["mean"] <= max(fam["null"]):
        return False
    excess = fam["d_all"]["mean"] - _median(fam["null"])
    return excess >= ratio * fam["dur_gain"] if fam["dur_gain"] > 0 else excess > 0


def decide_phase5h1(
    d_cross: Mapping[str, float], null_cross: Sequence[float],
    d_time: Mapping[str, float], null_time: Sequence[float],
    dur_gain: float, n_discoveries: int, rho_real: float, rho_null: Sequence[float],
    families: Mapping[str, Mapping[str, Any]], ratio: float,
) -> dict[str, Any]:
    """Return {label, K1..K4, families_passing}.

    d_cross = Δ(G4x, G2); d_time = Δ(G4, G3); dur_gain = Δ(G2, G1) point estimate.
    Null lists combine both registered nulls.
    """
    k1 = _beats(d_cross, null_cross)
    k2 = _beats(d_time, null_time)
    excess = d_time["mean"] - _median(null_time) if null_time else math.nan
    k3 = _ok(excess) and (excess >= ratio * dur_gain if dur_gain > 0 else excess > 0)
    k4 = n_discoveries >= 1 and len(rho_null) > 0 and _ok(rho_real) and rho_real > max(rho_null)
    fams = sorted(f for f, v in families.items() if family_passes(v, ratio))
    if k1 and k2 and k3 and k4:
        label = "GO"
    elif k1 and fams:
        label = "NARROW"
    else:
        label = "STOP"
    return {"label": label, "K1": k1, "K2": k2, "K3": k3, "K4": k4, "time_excess": excess,
            "families_passing": fams}
