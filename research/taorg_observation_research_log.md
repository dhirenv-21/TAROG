# TAORG observation-interface audit — research log

Branch `codex/taorg-observation-audit`, created 2026-10-12 from `codex/taorg-next-method`
@ `ee65631` (PR #4). The new PR is stacked on #4.

Workspace verified: `D:\ADSA project`; `taorg` imports from D:.

Unrelated items were left untouched and are never staged:
- `demo/output/demo_data.json`, `demo/output/taorg_demo.html` (modified);
- `outputs/`, `scripts/phase5h1/visual_demo.py` (untracked).

## Commits (full hashes)

| # | Checkpoint | Commit |
|---|---|---|
| 1 | Audit part 1 (code trace, sampling rule, pre-stated criteria) and contract | `bbf545e1e59b240d5809e0a8252ab84103f29452` |
| 2 | Tested adapter/oracle and development audit results | `550f9f3b2584c3db2f8955ca9f64db0f7d6058d4` |
| 3 | Exit report, audit part 2, supervisor update, records | the commit adding this row (hash in the PR description) |

## Log

- 2026-10-12 — Inventory:
  - The only local videos are personal screen captures (games, exam platform, video chats showing
    third parties). They were not used, for privacy reasons and because they are unsuitable.
  - No detector or tracker weights were cached.
- 2026-10-12 — With user approval in chat, downloaded:
  - `MOT17Labels.zip` (10.1 MB, CC BY-NC-SA 3.0, SHA-256 in `results/observation/manifests/sources.json`);
  - `trackers==2.6.1` (Apache-2.0), with `supervision` 0.30.9, `av` 17.1.0 and `pydeprecate` 0.13.0.
- 2026-10-12 — Fixed the sequence split by a seeded rule before producing any tracker output:
  - dev {02, 09, 11}; eval {04, 05, 10, 13};
  - DPM excluded by its score range.
- 2026-10-12 — Code trace, contract and schema (`src/taorg/schemas/observation.py`, 4 tests).
  No tracker output yet.
- 2026-10-12 — Adapter (MOT17 public detections → ByteTrack → contract records with a-priori
  heuristics) and a separate oracle module.
- 2026-10-12 — Development audit on 02/09/11 × FRCNN/SDP (39,534 absence frames).
  - **Pre-stated verdict NOT USABLE.**
  - Two test/figure issues were fixed in the test and figure code (no change to results):
    - the AST-based ground-truth check had matched a docstring;
    - a shared-axis figure label bug.
- 2026-10-12 — Exploratory separability (labelled; after the verdict): elapsed time is the only
  signal better than chance.
- 2026-10-12 — **No pre-registration and no comparison**, because the pre-stated rule forbids
  them. The evaluation sequences were never processed.
  - Wrote the exit report, supervisor update, ledger L57–L60, results notes and Memory.
  - 326 tests pass.
