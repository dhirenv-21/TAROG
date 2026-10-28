"""V-HPED state dynamics: excitation traces, intensity, and hazard.

Concern 1 of Architecture.md §6.  Pure functions over NumPy arrays so they
can be unit-tested in isolation and mirrored exactly by likelihood.py.

Shapes: P = number of predicates.
    traces: (P,)  — e_q, decayed sum of past event marks of predicate q
"""

from __future__ import annotations

import numpy as np

from taorg.temporal.vhped.params import VHPEDParams


def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus log(1 + exp(x)); strictly positive."""
    return np.logaddexp(0.0, x)


def decay_traces(traces: np.ndarray, beta: float, dt: float) -> np.ndarray:
    """Analytically decay traces over an interval of `dt` seconds.

    e_q(t + dt) = e_q(t) * exp(-beta * dt)   (no events inside the interval)
    """
    return traces * np.exp(-beta * dt)


def preactivation(traces: np.ndarray, params: VHPEDParams) -> np.ndarray:
    """z_p = mu_p + sum_q alpha[q, p] e_q.  Shape (P,)."""
    return params.mu + traces @ params.alpha


def intensity(traces: np.ndarray, params: VHPEDParams) -> np.ndarray:
    """Observation intensity lambda_p = softplus(z_p) in 1/s.  Shape (P,).

    Always > 0 for finite inputs, so negative alpha (inhibition) is safe.
    """
    return softplus(preactivation(traces, params))


def hazard(traces: np.ndarray, params: VHPEDParams) -> np.ndarray:
    """Termination hazard h_p = softplus(eta_p + sum_q gamma[q, p] e_q) in 1/s."""
    return softplus(params.eta + traces @ params.gamma)
