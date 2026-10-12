# TAORG appearance-based tracking audit — research log

Branch `codex/taorg-appearance-audit`, created 2026-10-12 from `codex/taorg-observation-audit`
@ `00a59c2` (PR #5). The new PR is stacked on #5, which is stacked on #4 and #3.

Workspace verified: `D:\ADSA project`; `taorg` imports from D:. The old C: copy was not touched.

Unrelated items were left untouched and are never staged:
- `demo/output/*` (modified);
- `outputs/`, `scripts/phase5h1/visual_demo.py` (untracked).

## Log

- 2026-10-12 — **State.**
  - RTX 4060 Laptop 8 GB (CUDA available to torch 2.14.0+cu132).
  - RAM: 31 GB total, about 1.1 GB free at the start.
  - Disk: D: 266 GB free, C: 2.3 GB free. All new state goes on D:.
- 2026-10-12 — **Correction.** `trackers` 2.6.1 `BoTSORTTracker` is IoU + CMC only, with no
  re-ID. The observation exit report §9 was wrong to imply otherwise; its decision is unaffected.
- 2026-10-12 — **Tracker selected:** BoxMOT 25.0.0 `BotSort`.
  - It has a `use_embeddings` switch, accepts precomputed embeddings, and returns detection
    indices.
  - Installed with `--no-deps` into an isolated D: venv. `filterpy` failed to build and is not
    needed.
- 2026-10-12 — **ReID model selected:** OSNet x1.0 MSMT17 from the torchreid zoo, downloaded
  (11.0 MB, SHA-256 in the manifest).
  - It loads under `weights_only=True`.
  - Its classifier has 1041 classes, matching the MSMT17 training identities. It was not
    trained on MOT17.
- 2026-10-12 — **Frames.** `MOT17Det.zip` (1.95 GB, range requests supported). Fetched dev
  sequences 02/09/11 only (about 345 MB).
- 2026-10-12 — **Engineering smoke test** (no labels): about 16 ms/frame tracker time, about
  20–27 ms/frame ReID time, and 233 MB peak GPU memory.
- 2026-10-12 — **Pre-registration written** before any comparative analysis.
- 2026-10-12 — **Implementation.** Amendments A1 (strict conf > 0.5) and A2 (the creation-frame
  seed of `smooth_feat`; the registered cue is unchanged) were made before any label was computed.
  - Two out-of-memory aborts (C: full, commit limit) and one JSON bug produced no results.
  - Implementation commit `e7ae883`.
- 2026-10-12 — **Development run stage at `e7ae883` (clean):**
  - all checks pass (raw-cost and `smooth_feat` mirror 100%);
  - appearance changes 2–9% of first-association costs, and V2/V3 outputs on 0–99% of frames
    (09-FRCNN: 0%);
  - ReID 17–25 ms/frame, tracker 10–18 ms/frame, GPU peak ≤ 357 MB, peak working set 6.0 GB.
- 2026-10-12 — **Registered development analysis. Gate PASS:**
  - V3 ΔAUC(T+A − T): FRCNN +0.177 [97.5% 0.102, 0.251]; SDP +0.127 [0.065, 0.187];
  - coverage 0.97.
- 2026-10-12 — **Freeze** (`results/appearance/freeze/`) committed before any held-back access.
- 2026-10-12 — **Held-back stage.**
  - Frames for 04/05/10/13 were fetched only after the freeze was pushed (568 MB).
  - The first attempt aborted on out-of-memory in `load_embs` (per-frame decompression).
  - Fixed in A3 (`1ff4387`). The development re-run is content-identical: 54/54 files, 12/12
    association logs. The gzip file hashes differ only by the timestamp.
  - Held-back run: 24/24 runs complete at `1ff4387` (clean); all implementation checks 100%.
- 2026-10-12 — **Frozen held-back analysis. Decision NARROW.**
  - FRCNN meets all GO criteria. SDP fails only takeover recall (0.468 < 0.50).
  - V3 ΔAUC: +0.194 [0.151, 0.240] (FRCNN); +0.112 [0.045, 0.183] (SDP).
- 2026-10-12 — **EXPLORATORY** proximity-only control (`exploratory_proximity.json`): T+G
  0.70/0.70 vs T+A 0.86/0.77 on held-back data. Labelled; no effect on the decision.
- 2026-10-12 — Wrote the exit report, supervisor update, ledger L61–L64, results notes and
  Memory.
