"""Verify headline Phase 5H-1 numbers against saved artifacts AND their presence in the documents.

Each check: (claim, value formatted from an artifact, documents that must contain that exact string).
Exits non-zero on any mismatch.

Usage:
    python scripts/phase5h1/verify_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

A = json.loads(Path("results/phase5h1/full/analysis.json").read_text(encoding="utf-8"))
CP = json.loads(Path("results/phase5h1/full/completeness.json").read_text(encoding="utf-8"))
PB = json.loads(Path("results/phase5h1/full/posthoc_boundary.json").read_text(encoding="utf-8"))
FZ = json.loads(Path("results/phase5h1/full/frozen/frozen.json").read_text(encoding="utf-8"))
C, D, S, ST = A["comparisons"], A["decision"], A["sample"], A["stability"]
EXIT = "research/phase5h1_exit_report.md"
SUP = "research/phase5h1_supervisor_update.md"
PAPER = "research/paper_notes/phase5h1_exploratory_section.md"
ALL = (EXIT, SUP, PAPER)


def ci3(s):
    return f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"


CHECKS = [
    ("decision label", "NARROW", (EXIT, SUP)),
    ("EVAL events", f"{int(S['eval_events']):,}", ALL),
    ("EVAL-A events", f"{int(S['eval_events_A']):,}", (EXIT,)),
    ("EVAL-B events", f"{int(S['eval_events_B']):,}", (EXIT,)),
    ("K1 Δ(G4x,G2)", f"+{C['G4x-G2']['EVAL']['mean']:.3f}", ALL),
    ("K1 CI", ci3(C["G4x-G2"]["EVAL"]), (EXIT, PAPER)),
    ("K2 Δ(G4,G3)", f"+{C['G4-G3']['EVAL']['mean']:.3f}", ALL),
    ("K2 CI", ci3(C["G4-G3"]["EVAL"]), ALL),
    ("duration gain", f"+{C['G2-G1']['EVAL']['mean']:.3f}", (EXIT,)),
    ("ratio G4-G3 / G2-G1", f"{C['G4-G3']['EVAL']['mean'] / C['G2-G1']['EVAL']['mean']:.3f}", (EXIT,)),
    ("G4+ - G4", f"−{-C['G4+-G4']['EVAL']['mean']:.3f}", ALL),
    ("null max Δ(G4x,G2)", f"+{D['null_cross_max']:.3f}", (EXIT, SUP)),
    ("null max Δ(G4,G3)", f"+{D['null_time_max']:.3f}", ALL),
    ("K3 excess", f"{D['time_excess']:.3f}", (EXIT,)),
    ("K3 threshold", f"{D['k3_threshold']:.3f}", (EXIT,)),
    ("sign-agree real", f"{ST['n_sign_agree_real']:,}", (EXIT, SUP)),
    ("sign-agree null range", f"{min(ST['n_sign_agree_null']):,}–{max(ST['n_sign_agree_null']):,}", (EXIT, SUP)),
    ("min FDR", f"{ST['fdr_scan_min']:.2f}", ALL),
    ("rho real", f"{ST['rho_real']:.3f}", (EXIT, SUP)),
    ("rho null max", f"{max(ST['rho_null']):.3f}", (EXIT, SUP)),
    ("families passing", " and ".join(sorted(D["families_passing"], reverse=True)), (EXIT,)),
    ("ON>OFF block", f"+{A['ablations']['G3+H:ON>OFF-G3']['EVAL']['mean']:.3f}", ALL),
    ("lag0 block", f"+{A['ablations']['G3+H:lag0-G3']['EVAL']['mean']:.3f}", (EXIT, PAPER)),
    ("lag1 block", f"+{A['ablations']['G3+H:lag1-G3']['EVAL']['mean']:.3f}", (EXIT, PAPER)),
    ("lag2 block", f"+{A['ablations']['G3+H:lag2-G3']['EVAL']['mean']:.3f}", (EXIT,)),
    ("tie share", f"{100 * PB['EVAL']['share']['tie']:.1f}%", ALL),
    ("H-a RR", f"{A['hypotheses']['results'][0]['rr']:.2f}", (EXIT, SUP)),
    ("H-b RR", f"{A['hypotheses']['results'][1]['rr']:.1f}", (EXIT, SUP)),
    ("N-circ event inflation", f"+{100 * (min(r['n_events'] for r in A['nulls']['replicates'] if r['kind'] == 'circ') / S['eval_events'] - 1):.0f}%", ALL),
    ("N-jit event loss", f"−{100 * (1 - max(r['n_events'] for r in A['nulls']['replicates'] if r['kind'] == 'jit') / S['eval_events']):.1f}% to "
                         f"−{100 * (1 - min(r['n_events'] for r in A['nulls']['replicates'] if r['kind'] == 'jit') / S['eval_events']):.1f}%", ALL),
    ("tune fits complete", f"{CP['tune']['complete']}/{CP['tune']['expected']}", (EXIT,)),
    ("null fits", f"{CP['nulls']['complete']}", (EXIT,)),
    ("freeze commit", "289319e", ALL),
    ("selected ridge G4", f"G4 {FZ['selected_ridge']['G4']:g}", (EXIT,)),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    docs = {p: Path(p).read_text(encoding="utf-8") for p in ALL if Path(p).exists()}
    bad = 0
    for claim, val, where in CHECKS:
        for doc in where:
            ok = doc in docs and val in docs[doc]
            bad += not ok
            if not ok:
                print(f"BAD  {claim:28s} value={val!r} not found in {doc}")
    if A["decision"]["label"] != "NARROW" or not CP["all_complete"]:
        bad += 1
        print("BAD  decision/completeness")
    n = sum(len(w) for _, _, w in CHECKS)
    print(f"{n - bad}/{n} checks OK" + ("" if bad else " — ALL PHASE 5H-1 NUMBERS VERIFIED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
