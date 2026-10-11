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
| 1 | Audit part 1 (code trace, sampling rule, pre-stated criteria) and contract | (recorded after commit) |

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
