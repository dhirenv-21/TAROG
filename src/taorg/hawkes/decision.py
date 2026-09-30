"""Phase 5H Stage 0 pre-registered decision rule (research/phase5h_stage0_preregistration.md §8).

Pure function over already-computed quantities; committed with the
pre-registration, before any model was fitted to VidOR.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _ok(x: float) -> bool:
    return x is not None and not math.isnan(x)


def decide_stage0(
    g_cross: Mapping[str, float],
    g_cross_null: Sequence[float],
    g_dur: Mapping[str, float],
    g_base: Mapping[str, float],
    g_self: float,
    s_real: int,
    s_null: Sequence[int],
    rho_real: float,
    rho_null: Sequence[float],
    c3_ratio: float,
) -> dict[str, Any]:
    """Return the HAWKES SIGNAL label and every criterion.

    Args:
        g_cross: {"mean", "ci_lo", "ci_hi"} of ΔLL/event(HK2 - HK1) on TEST.
        g_cross_null: ΔLL/event(HK2 - HK1) on each C1 null copy.
        g_dur: stats of ΔLL/event(HK2 - DUR).
        g_base: stats of ΔLL/event(HK2 - HK0).
        g_self: ΔLL/event(HK1 - HK0) point estimate.
        s_real / s_null: stable cross-cell counts (real, per C4 null copy).
        rho_real / rho_null: mean inter-fold Spearman of cross alphas.
        c3_ratio: registered ratio (0.10).
    """
    null_sorted = sorted(g_cross_null)
    null_max = null_sorted[-1] if null_sorted else math.nan
    n = len(null_sorted)
    null_median = (null_sorted[(n - 1) // 2] + null_sorted[n // 2]) / 2 if n else math.nan
    c1 = (_ok(g_cross["ci_lo"]) and g_cross["ci_lo"] > 0 and _ok(null_max) and g_cross["mean"] > null_max)
    c2 = (_ok(g_dur["ci_lo"]) and g_dur["ci_lo"] > 0 and _ok(g_base["ci_lo"]) and g_base["ci_lo"] > 0)
    excess = g_cross["mean"] - null_median if _ok(null_median) else math.nan
    if _ok(g_self) and g_self > 0:
        c3 = _ok(excess) and excess >= c3_ratio * g_self
    else:
        c3 = _ok(excess) and excess > 0
    c4 = (len(s_null) > 0 and s_real > max(s_null) and len(rho_null) > 0 and _ok(rho_real)
          and rho_real > max(rho_null))
    if not c1:
        label = "ABSENT"
    elif c2 and c3 and c4:
        label = "STRONG"
    else:
        label = "WEAK-MIXED"
    return {"label": label, "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "C4": bool(c4),
            "null_max": null_max, "null_median": null_median, "excess": excess,
            "excess_over_self": (excess / g_self) if _ok(g_self) and g_self > 0 else math.nan,
            "perm_p": (1 + sum(v >= g_cross["mean"] for v in g_cross_null)) / (1 + n) if n else math.nan}
