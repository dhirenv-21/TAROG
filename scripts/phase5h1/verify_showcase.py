"""Check every number shown in the Phase 5H-1 showcase against saved artifacts.

Each row: (claim as displayed, value recomputed from an artifact, formatter). Exits non-zero on mismatch.

Usage:
    python scripts/phase5h1/verify_showcase.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

S0 = json.loads(Path("results/phase5h/stage0/summary.json").read_text(encoding="utf-8"))
AU = json.loads(Path("results/phase5h1/audit/audit.json").read_text(encoding="utf-8"))
P3 = json.loads(Path("results/phase3/aggregated/h1_decision.json").read_text(encoding="utf-8"))
P4 = json.loads(Path("results/phase4/aggregated/decisions.json").read_text(encoding="utf-8"))
SM = json.loads(Path("results/phase5h1/smoke/smoke.json").read_text(encoding="utf-8"))
T = S0["test"]
D = S0["decision"]

CHECKS = [
    ("Stage 0 observed gain +0.074", f"{T['G_cross']['mean']:+.3f}", "+0.074"),
    ("95% CI 0.066–0.083", f"{T['G_cross']['ci_lo']:.3f}–{T['G_cross']['ci_hi']:.3f}", "0.066–0.083"),
    ("null range 0.039–0.043", f"{min(S0['c1_null']):.3f}–{max(S0['c1_null']):.3f}", "0.039–0.043"),
    ("null median 0.040", f"{D['null_median']:.3f}", "0.040"),
    ("about 54% of observed", f"{100 * D['null_median'] / T['G_cross']['mean']:.0f}%", "54%"),
    ("timing-specific excess 0.034", f"{D['excess']:.3f}", "0.034"),
    ("excess / within-relation gain 7%", f"{100 * D['excess_over_self']:.0f}%", "7%"),
    ("within-relation gain 0.482", f"{T['G_self']['mean']:.3f}", "0.482"),
    ("practical line 0.089", f"{D['null_median'] + 0.10 * T['G_self']['mean']:.3f}", "0.089"),
    ("TEST events 50,509", f"{T['n_events']:,}", "50,509"),
    ("TEST videos 835", str(S0["n_test"]), "835"),
    ("stable cells 404 vs null max 422", f"{S0['c4']['S_real']} vs {max(S0['c4']['S_null'])}", "404 vs 422"),
    ("Stage 0 label WEAK-MIXED", D["label"], "WEAK-MIXED"),
    ("audit videos 5,600", f"{AU['n_videos']:,}", "5,600"),
    ("audit events 356,040", f"{AU['n_events']:,}", "356,040"),
    ("targets 34 onset / 35 offset", f"{AU['n_onset_targets']} / {AU['n_offset_targets']}", "34 / 35"),
    ("top 6 predicates 86%", f"{100 * AU['top6_event_share']:.0f}%", "86%"),
    ("spatial share 63.5%", f"{100 * AU['family_event_counts']['SPATIAL'] / AU['n_events']:.1f}%", "63.5%"),
    ("Phase 3 H1 = PASS", P3["decision"], "PASS"),
    ("Phase 4 H2/H3/H4", "/".join(P4["decisions"][h]["decision"] for h in ("H2", "H3", "H4")), "MIXED/MIXED/FAIL"),
    ("smoke = 300 TRAIN videos", str(SM["n_fit_videos"] + SM["n_check_videos"]), "300"),
]


def main() -> int:
    bad = 0
    for claim, got, shown in CHECKS:
        ok = got == shown
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'}  {claim:40s} artifact={got:22s} shown={shown}")
    peri = next(e for e in S0["descriptive"]["peri_event_top20"]
                if e["source"] == "ON(in_front_of)" and e["target"] == "ON(next_to)")
    ok = (peri["onsets_per_source_event_by_lag"][0], peri["baseline_onsets_per_source_event_by_lag"][0]) == (0.0001, 0.00871)
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'}  short-lag suppression 0.0001 vs 0.0087      artifact="
          f"{peri['onsets_per_source_event_by_lag'][0]} vs {peri['baseline_onsets_per_source_event_by_lag'][0]}")
    print("ALL SHOWCASE NUMBERS VERIFIED" if bad == 0 else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
