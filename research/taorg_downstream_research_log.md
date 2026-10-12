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
