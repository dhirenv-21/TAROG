# TAORG downstream study — research log

Branch `codex/taorg-downstream`, created 2026-10-12 from `codex/taorg-appearance-audit`
@ `b85ffba` (verified; PR #6). The new PR is stacked on #6 → #5 → #4 → #3.

Workspace verified: `D:\ADSA project`; `taorg` imports from D:.

Unrelated items are never staged:
- `demo/output/*`;
- `outputs/`, `scripts/phase5h1/visual_demo.py`.

## Log

- 2026-10-12 — **Resources:** 0.9 GB free RAM (of 31 GB), C: 10 GB free, D: 264 GB free; RTX 4060
  Laptop 8 GB.
- 2026-10-12 — **Data suitability:**
  - VidOR is the only local relationship dataset; Phase 5H already used all of its annotations.
  - MOT17 has no relationship labels.
  - The report was written, and the user approved the downloads (about 216 MB of videos and
    175 MB of detector weights).
- 2026-10-12 — **Development downloads:**
  - 30 videos fetched by range requests from the HF mirror (134 MB transferred; one transient DNS
    failure was retried after adding HEAD retries);
  - detector weights downloaded (SHA-256 `dd69338a…`).
- 2026-10-12 — **Engineering check** (label-free): 30/30 development videos decode to exactly
  the annotated frame count and size.
- 2026-10-12 — **Pre-registration written** before any observation stream, method output or
  label was computed.
- 2026-10-12 — **Implementation** committed at `bf7749d` (16 new tests; 353 passed, 2 skipped).
- 2026-10-12 — **Development perception at `bf7749d` (clean):** 30/30 videos, 22,017 frames.
  - Missing records: 83,338; appearance available: 64,567; flagged: 11,712.
  - Timing: detection 115 ms/frame, ReID 31 ms/frame, tracker 9 ms/frame.
- 2026-10-12 — **Calibration** (unflagged only).
  - Selected: FB W = 5 (mean J 0.574); B2 α 0.5, decay_visible 0.3, decay_unknown 0.8, θ_on 0.5
    (mean J 0.679).
  - **Both are at the low-hold edge of their grids.** This is disclosed; the grids were not
    extended after seeing it.
- 2026-10-12 — **Gate PASS:**
  - F2: 20,517 takeover-incident runs on scored keys in 28 videos; IGT(B2) 56.8 s;
  - F3: coverage 0.775.
- 2026-10-12 — **Freeze** (`results/downstream/freeze/frozen.json`), committed before any
  evaluation video was fetched.
- 2026-10-12 — **Evaluation videos fetched** after the freeze push: 60 videos, 233 MB transferred.
- 2026-10-13 — **Evaluation perception.**
  - The session ended after 32/60 videos. The partial directory was removed and the run resumed
    from the checkpoint at `0942146` (amendment A1).
  - 60/60 videos aligned. `swscaler` warnings were disclosed (A2).
- 2026-10-13 — **Registered evaluation: 240/240 runs. Decision GO** (C1 and C2 pass).
- 2026-10-13 — **Development methods** (descriptive, after the freeze): same direction.
- 2026-10-13 — **POST-HOC** (`eval/posthoc.json`):
  - relative PGG loss 25–27%;
  - 2.4–2.7 ghost-seconds removed per added false-termination second;
  - 69% of terminations and 97% of recurrences are matching flicker; genuine-only effects ≈ 0.
- 2026-10-13 — Wrote the exit report, supervisor update, ledger L65–L68, results notes and
  Memory. Clarified the appearance audit's interval terminology (its exit report §11).
