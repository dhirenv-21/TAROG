"""POST-HOC, descriptive (not registered; cannot change the decision): how close in time are
cross-predicate events on the same pair? Quantifies how much of the short-lag history signal could be
annotation boundary alignment rather than scene dynamics.

For every modelled target event on EVAL (and, for reference, TRAIN), the lag to the most recent event of a
*different* predicate on the same ordered pair is binned:
    tie (lag 0, excluded from history by the strict-history rule), <= 1 frame, (1 frame, 0.5 s],
    (0.5, 2], (2, 10] s, none within 10 s.
Also reports the share of the most frequent (source -> target) short-lag transitions.

Usage:
    python scripts/phase5h1/posthoc_boundary.py
"""

from __future__ import annotations

import bisect
import json
from collections import Counter
from pathlib import Path

RUN = Path("D:/TAORG_runs/phase5h1/main")
OUT = Path("results/phase5h1/full/posthoc_boundary.json")


def main() -> None:
    from taorg.hawkes.phase5h1_run import _meta, load_part
    m = _meta(RUN)
    vocab = m["layout"]["vocab"]
    targets = set(m["targets"])
    name = lambda c: f"{'ON' if c % 2 == 0 else 'OFF'}({vocab[c // 2]})"
    res = {"LABEL": "POST-HOC descriptive analysis (not registered; does not enter the decision)"}
    for part in ("EVAL", "TRAIN"):
        bins = Counter()
        top = Counter()
        n = 0
        for v in load_part(RUN, part):
            f = v.fps
            for pr in v.pairs.values():
                ev = sorted([(b, 2 * r) for r, ivs in pr.rels.items() for b, e in ivs if b > 0]
                            + [(e, 2 * r + 1) for r, ivs in pr.rels.items() for b, e in ivs if e < v.T])
                times = [t for t, _ in ev]
                for i, (t, c) in enumerate(ev):
                    if c not in targets:
                        continue
                    n += 1
                    j = bisect.bisect_right(times, t) - 1          # include ties, then classify
                    best = None
                    while j >= 0 and t - times[j] <= 10 * f:
                        if ev[j][1] // 2 != c // 2:
                            best = (t - times[j], ev[j][1])
                            break
                        j -= 1
                    if best is None:
                        bins["none_within_10s"] += 1
                        continue
                    lag_f, src = best
                    lag = lag_f / f
                    key = ("tie" if lag_f == 0 else "le_1frame" if lag_f <= 1 else "1frame_0.5s" if lag <= 0.5
                           else "0.5_2s" if lag <= 2 else "2_10s")
                    bins[key] += 1
                    if key in ("tie", "le_1frame", "1frame_0.5s"):
                        top[(name(src), name(c), key)] += 1
        res[part] = {"n_target_events": n, "share": {k: round(bins[k] / n, 4) for k in
                     ("tie", "le_1frame", "1frame_0.5s", "0.5_2s", "2_10s", "none_within_10s")},
                     "counts": dict(bins),
                     "top_short_lag_transitions": [{"source": s, "target": t, "bin": b, "n": k}
                                                   for (s, t, b), k in top.most_common(15)]}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({p: res[p]["share"] for p in ("EVAL", "TRAIN")}, indent=1))
    print(json.dumps(res["EVAL"]["top_short_lag_transitions"][:10], indent=1))


if __name__ == "__main__":
    main()
