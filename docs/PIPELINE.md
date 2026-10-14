# TAORG end-to-end pipeline

```bash
python -m taorg.pipeline run --video my_clip.mp4 --work D:/TAORG_runs/pipeline/my_clip
```

```
video ──► frames ──► person detections ──► OSNet + BoT-SORT tracks ──► missing records + takeover flag
                                                       │                                  │
frames ──► EGTR scene graph (every 3rd frame) ─────────┴──► canonical relationship stream ─┴──► temporal method
                                                                                                │
                                                                         graphs.json + viewer.html ◄┘
```

## Stages

| Stage | Environment | Reads | Writes |
|---|---|---|---|
| `perception` | perception venv (BoxMOT) | the video | `frames/`, `perception/{outputs,missing_records}.csv.gz`, `perception/meta.json` |
| `sgg` | EGTR venv | `frames/` | `sgg/relations.npz`: person queries and scores for 4 mapped predicates |
| `maintain` | main | perception and sgg outputs | `stream.csv.gz` (canonical stream), `graphs.json` (maintained graphs) |
| `view` | main | `graphs.json`, `frames/` | `viewer.html` (offline, self-contained) |

**Running stages separately.** Run a single stage with
`python -m taorg.pipeline stage <name> --work W [--video V]`.

**Caching.** Stages reuse cached outputs. Delete a stage's output folder to recompute it.

**Environments.** `run` launches `perception` and `sgg` with the interpreters configured in
`configs/pipeline/default.yaml` (see docs/SETUP.md).

## Options

| Option | Meaning |
|---|---|
| `--method {NM,FB,B2,VH}` | Temporal method: no memory, fixed window, GatedEMA + hysteresis, V-HPED M3 (certified lazy maintenance) |
| `--flag / --no-flag` | Endpoint-invalidation policy using the frozen appearance takeover flag (FB, B2 only) |
| `--no-frames` | Viewer without embedded frames (use when the video must not be redistributed) |
| `--config` | Alternative YAML (threshold τ_obs, method parameters, interpreter paths) |

## What the outputs mean

**Stream (`stream.csv.gz`).** EGTR predictions mapped to **tracker IDs**, with three kinds of row:
- **observed** (score ≥ τ_obs);
- **evaluated negative**: both people were detected by EGTR, but no relation was predicted;
- **no row**: the pair was not evaluated.

The stream contains person–person relations only, for four predicates: `behind`, `in_front_of`,
`next_to`, `watch`.

**Graphs (`graphs.json`).** For each relationship step:
- tracker boxes;
- takeover-flagged tracks;
- the maintained edges with scores. `observed: false` marks an edge retained by temporal memory.

**Scores are not calibrated probabilities.** EGTR's score is a ranking quantity (relation ×
connectivity × object scores).

## Limits (read before relying on the output)

- **Domain shift.** EGTR is trained on still Visual Genome images; it predicts frame by frame, with
  no temporal context, and covers 4 of VidOR's person–person predicates.
- **Takeover flag.** It was calibrated on MOT17. On VidOR it fires on about 32% of genuine
  observation gaps (`research/taorg_downstream_exit_report.md`).
- **Speed.** About 0.28 s/frame for EGTR and about 0.15 s/frame for perception on an RTX 4060. A
  30 s clip takes a few minutes.
- **Evidence.** For which methods help, see `research/taorg_final_exit_report.md`. The pipeline
  does not by itself establish any method's superiority.

## Smoke run (reproducible)

```bash
python -m taorg.pipeline run --video D:/TAORG_runs/downstream/videos/<a VidOR dev video>.mp4 --work D:/TAORG_runs/pipeline/smoke --no-frames
```

A recorded smoke run (no frames) is in `results/pipeline_smoke/`.
