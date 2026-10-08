"""Phase 5H-1 N-jit null (research/phase5h1_preregistration.md §6; construction details: amendment A2).

Rigid per-set shift: every (pair, predicate) interval set is shifted by one offset
d ~ U[-J, +J] seconds (rounded to frames), clipped to the pair window, which is kept unchanged.

This is a *shift-jitter variant*, not the interval-jitter procedure of Amarasingham et al. (2012):
interval jitter resamples each event independently and uniformly inside fixed windows, which gives
an exact conditional test. Here a whole interval set moves together, so durations and the set's
internal structure are kept and only the relative timing *between* predicates is randomised.
No exactness property of interval jitter is claimed.

Censoring rule (A2): an endpoint that is censored in the data (begin frame 0 = left-censored,
end frame T = right-censored) is not an event, so it is pinned and not shifted. Shifting it would
manufacture an ONSET/OFFSET event that the annotation never contained. Observed endpoints are
shifted. Clipping to the pair window can (i) drop intervals that leave the window and (ii) move an
observed onset to frame 0 when w0 = 0 (it then becomes censored). Both are counted in `diag`.

Randomness is drawn per video from random.Random(f"{seed}|{part}|{vid}"), so the result does not
depend on video order, chunking or worker scheduling.
"""

from __future__ import annotations

import random
from typing import Iterable

from taorg.hawkes.vidor import Interval, PairRecord, VideoRecord


def jitter_video(v: VideoRecord, seed: int | str, part: str, J: float, diag: dict[str, int] | None = None
                 ) -> VideoRecord:
    """N-jit transform of one video (pairs and predicates visited in sorted order)."""
    rng = random.Random(f"{seed}|{part}|{v.vid}")
    T = v.T
    pairs = {}
    for key in sorted(v.pairs):
        pr = v.pairs[key]
        w0, w1 = pr.window
        new: dict[int, list[Interval]] = {}
        for r in sorted(pr.rels):
            s = int(round(rng.uniform(-J, J) * v.fps))
            ivs: list[Interval] = []
            for b, e in pr.rels[r]:
                nb = b if b == 0 else b + s
                ne = e if e >= T else e + s
                cb, ce = max(nb, w0), min(ne, w1)
                if diag is not None:
                    diag["intervals"] = diag.get("intervals", 0) + 1
                    if b > 0 and cb == 0:
                        diag["onset_lost_to_censoring"] = diag.get("onset_lost_to_censoring", 0) + 1
                    if e < T and ce >= T:
                        diag["offset_lost_to_censoring"] = diag.get("offset_lost_to_censoring", 0) + 1
                    if (cb, ce) != (nb, ne):
                        diag["clipped"] = diag.get("clipped", 0) + 1
                if ce <= cb:
                    if diag is not None:
                        diag["dropped"] = diag.get("dropped", 0) + 1
                    continue
                ivs.append((cb, ce))
            ivs.sort()
            merged: list[list[int]] = []
            for b, e in ivs:                        # pinned endpoints can create overlaps
                if merged and b <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                    if diag is not None:
                        diag["merged"] = diag.get("merged", 0) + 1
                else:
                    merged.append([b, e])
            if merged:
                new[r] = [(b, e) for b, e in merged]
        pairs[key] = PairRecord((w0, w1), new)
    return VideoRecord(v.vid, v.fps, T, pairs, dict(v.cats))


def jitter_shift(videos: Iterable[VideoRecord], seed: int | str, part: str, J: float,
                 diag: dict[str, int] | None = None) -> list[VideoRecord]:
    """N-jit transform of a list of videos."""
    return [jitter_video(v, seed, part, J, diag) for v in videos]
