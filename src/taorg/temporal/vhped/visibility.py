"""V-HPED visibility policy.

Concern 2 of Architecture.md §6: maps a pair's visibility regime to the
detection-thinning probability rho_v and trace decay beta_v.  Absence is
never a Hawkes event; its evidential weight comes from rho_v (see
research/derivations/vhped_model.md §4).
"""

from __future__ import annotations

from collections.abc import Iterable

from taorg.schemas.edges import EdgeObservation
from taorg.schemas.tracks import VisibilityState
from taorg.temporal.vhped.params import REGIME_INDEX, VHPEDParams

# Most conservative regime wins when a pair's observations disagree.
_PRECEDENCE = {
    VisibilityState.OCCLUDED_OR_UNRELIABLE: 2,
    VisibilityState.UNKNOWN: 1,
    VisibilityState.VISIBLE: 0,
}


def pair_regime(observations: Iterable[EdgeObservation]) -> VisibilityState:
    """Combine the edge visibilities of one pair's observations in a frame.

    A pair with no observation records in the frame is UNKNOWN (PRD FR-2:
    UNKNOWN must not silently become VISIBLE).
    """
    regime: VisibilityState | None = None
    for obs in observations:
        vis = obs.edge_visibility
        if regime is None or _PRECEDENCE[vis] > _PRECEDENCE[regime]:
            regime = vis
    return VisibilityState.UNKNOWN if regime is None else regime


def thinning(regime: VisibilityState, params: VHPEDParams) -> float:
    """Detection probability rho_v for the regime."""
    return float(params.rho[REGIME_INDEX[regime]])


def trace_decay(regime: VisibilityState, params: VHPEDParams) -> float:
    """Trace decay rate beta_v (1/s) for the regime."""
    return float(params.beta[REGIME_INDEX[regime]])
