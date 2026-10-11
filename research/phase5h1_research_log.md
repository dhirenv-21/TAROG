# Phase 5H-1 research log

## 2026-09-30
- Phase 5H-1 approved. Literature searches for frailty, point-process GLMs, jitter nulls and
  relational event models were started; the session was interrupted before pre-registration.

## 2026-10-07
- Resumed from `3623f0a` (clean). Stage 0 artifacts untouched.
- Literature notes completed: `research/phase5h1_literature.md` (verification level per source).
  The interval-jitter paper's full text was paywalled (HTTP 403), so the exact mechanics of
  published interval jitter are recorded as "verify in full text"; our null is a rigid
  per-set shift and is described as a variant.
- Fitting-split audit (Stage 0 FIT only; no effects estimated): `results/phase5h1/audit/`.
  - 5,600 videos, 139,566 pairs, 356,040 events; top 6 predicates = 86.4% of events; spatial
    family = 63.5%; 34 onset / 35 offset target types ≥ 200 events.
  - TRAIN support of the Stage-0 hypotheses (events within 10 s): H-a 495, H-b 425, H-c 15,
    H-d 11.
- `07bf369` — **pre-registration** (all comparisons EXPLORATORY: every annotated VidOR video was
  used in Stage 0). Pushed to origin.
- `4d810ce` — GLM engine + tests (eager-reference likelihood, inhibition validity, positive-only
  bounds, reproducibility, leakage guard). Full suite 234 passed.
- Smoke run (300 TRAIN videos, G0–G3, ridge 1): finite, reproducible. About 20 s per rung on
  200 videos, so the full registered run is estimated at 30–60 single-thread CPU-hours (not
  started). **Smoke numbers are not evidence and are not reported as results.**
- Showcase: deck (6 slides, rendered through PowerPoint and inspected; two layout defects
  fixed), two figures, synthetic demo, 90-second script, supervisor update. All 22 showcased
  numbers verified against artifacts by `scripts/phase5h1/verify_showcase.py`.
- Not started: full registered run, nulls, stability, hypothesis replication. Next decision:
  compute approval and an independent confirmatory dataset.

## 2026-10-08
- Moved heavy run state to `D:/TAORG_runs/phase5h1/main` (C: had 6.8 GB free; user request).
- Benchmark on 1000 TRAIN videos replaced the 30–60 CPU-h estimate: G4 fit ≤ 0.75 s per target,
  design build 21 s, aggregation 64 s (`results/phase5h1/full/provenance/benchmark_train1000.json`).
- `c546591` execution infrastructure + amendments A1–A3 (aggregation, runner, N-jit construction,
  analysis details) before any registered fit. Tests: 258 pass; serial = parallel on synthetic data.
- First tuning attempt (max_iter 500): 35/157 early fits hit the iteration limit, plus a ~34-min
  system pause. `8721fcb` A4 → 5000 iterations (decided from optimiser diagnostics only); the
  attempt is preserved. A4 run: 2208/2208 converged; ridge selection identical to the attempt.
- `289319e` FREEZE committed; EVAL unlocked afterwards. evaluate 1 min, nulls 3.7 h (7 workers),
  stability 8 min, hypotheses 12 s. Matrix complete: 0 missing, 0 failed, 0 duplicate.
- Registered decision: **NARROW** (K1–K3 pass, K4 fails, SPATIAL + MOTION pass).
- Post-hoc (labelled): 17.0% exact cross-predicate boundary ties; fold-null strongest cells are
  sparse ridge-shrunk cells (K4 statistic flaw); N-circ +16% / N-jit −9.3% to −9.8% events.
- Exit report `research/phase5h1_exit_report.md`; supervisor materials updated. STOP here per
  instructions; confirmatory protocol proposed, not started.
