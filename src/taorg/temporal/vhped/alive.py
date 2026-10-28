"""V-HPED alive mapping: Bayesian alive/dead filter with thinned Hawkes emissions.

Concern 3 of Architecture.md §6.  Converts intensity/hazard into the
edge-alive probability a_p = P(alive | observations so far).  The keep/drop
threshold is applied to a_p by the model, not here.

Equations (research/derivations/vhped_model.md §4):
    a_pred = pi + (a - pi) * exp(-(h + r) dt),  pi = r / (h + r)
           (exact two-state CTMC transition; equals a * exp(-h dt) when r = 0)
    p1 = 1 - exp(-rho * lambda * dt)     P(observed | alive)
    p0 = 1 - exp(-rho * eps * dt)        P(observed | dead)
    c  = a_pred * L1 + (1 - a_pred) * L0  one-step predictive evidence
    a  = a_pred * L1 / c
"""

from __future__ import annotations

import numpy as np


def filter_step(
    belief: np.ndarray,
    observed: np.ndarray,
    lam: np.ndarray,
    haz: np.ndarray,
    rho: float,
    eps_rate: float,
    rebirth_rate: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One predict + update step for a vector of predicates on one pair.

    Args:
        belief: Prior beliefs a_{n-1}, shape (K,).
        observed: Boolean observation indicators o_n, shape (K,).
        lam: Observation intensities lambda (1/s), shape (K,).
        haz: Termination hazards h (1/s), shape (K,).
        rho: Visibility thinning in [0, 1].
        eps_rate: Spurious-detection rate (1/s).
        rebirth_rate: Dead -> alive rate (1/s).
        dt: Interval length in seconds.

    Returns:
        (posterior beliefs a_n, log predictive evidence log c_n), each (K,).
    """
    rate = haz + rebirth_rate  # > 0 in the model since haz = softplus(.) > 0
    pi = np.where(rate > 0.0, rebirth_rate / np.where(rate > 0.0, rate, 1.0), 0.0)
    pred = pi + (belief - pi) * np.exp(-rate * dt)

    # log(1 - p) computed exactly as the negative compensator.
    log_miss_alive = -rho * lam * dt
    log_miss_dead = np.full_like(lam, -rho * eps_rate * dt)
    p1 = -np.expm1(log_miss_alive)
    p0 = -np.expm1(log_miss_dead)

    lik_alive = np.where(observed, p1, np.exp(log_miss_alive))
    lik_dead = np.where(observed, p0, np.exp(log_miss_dead))

    num_alive = pred * lik_alive
    evidence = num_alive + (1.0 - pred) * lik_dead
    # evidence == 0 only if both likelihoods are 0 (observed with rho == 0);
    # then the observation is impossible under the model — keep the prior.
    safe = evidence > 0.0
    post = np.where(safe, num_alive / np.where(safe, evidence, 1.0), pred)
    log_ev = np.where(safe, np.log(np.where(safe, evidence, 1.0)), 0.0)
    return post, log_ev
