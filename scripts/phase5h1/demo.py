"""DEMO (SYNTHETIC DATA): can the Stage 0 Hawkes models detect a planted cross-relation effect?

We simulate two synthetic "worlds" of relationship events:
  * WITH a planted effect: when relation A starts on a pair, relation B becomes more likely
    to start in the next ~1 s (cross-excitation alpha = 0.6, kernel rate beta = 1/s);
  * WITHOUT it (alpha = 0): A and B are independent.
For each world we fit HK1 (each relation's own history only) and HK2 (multivariate Hawkes,
also uses the other relation's history) on 60 training videos, and measure held-out
log-likelihood per event on 30 new videos. Expected: HK2 recovers alpha and gains
held-out likelihood only in the world WITH the planted effect.

This is the same code used for the real VidOR Stage 0 run, but the data here are
SYNTHETIC and say nothing about real videos. Runs in about 10 seconds.

Usage:
    python scripts/phase5h1/demo.py
"""

from __future__ import annotations

import numpy as np

from taorg.hawkes.dataset import build
from taorg.hawkes.models import fit, ll_per_video
from taorg.hawkes.simulate import simulate


def world(alpha: float, seed: int) -> tuple[float, float]:
    train, test = simulate(alpha, seed), simulate(alpha, seed + 1, n_videos=30)
    d_tr, d_te = build(train, 2, 1.0), build(test, 2, 1.0)
    idx = np.arange(len(train))
    hk1, hk2 = fit(d_tr, idx, "HK1", 0.0, 1e-10), fit(d_tr, idx, "HK2", 0.0, 1e-10)
    gain = (ll_per_video(d_te, hk2).sum() - ll_per_video(d_te, hk1).sum()) / len(d_te.ev_type)
    return float(hk2.A[2, 0]), float(gain)       # A[ONSET(B), ONSET(A)]


def main() -> None:
    print("=" * 72)
    print("DEMO — SYNTHETIC DATA (not VidOR, not a result): planted-effect recovery")
    print("=" * 72)
    for alpha, seed in ((0.6, 1), (0.0, 3)):
        a_hat, gain = world(alpha, seed)
        label = "WITH planted effect" if alpha else "WITHOUT effect (control)"
        print(f"{label:28s} true alpha = {alpha:.1f}   recovered alpha = {a_hat:.3f}   "
              f"held-out gain HK2-HK1 = {gain:+.4f} nats/event")
    print("-" * 72)
    print("Reading: the multivariate model should recover the planted effect and gain held-out\n"
          "likelihood only when the effect exists. On real VidOR annotations (Stage 0) the gain was\n"
          "+0.074 nats/event, but ~54% of it also appeared after shuffling timing — see\n"
          "research/phase5h_stage0_report.md.")


if __name__ == "__main__":
    main()
