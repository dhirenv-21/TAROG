# Downstream study artifacts (`results/downstream/`)

**This is a controlled downstream test on real annotations.**
- The relationship observations come from an oracle predictor on real tracker outputs.
- It is not end-to-end scene-graph validation.

Documents:
- Pre-registration: `research/taorg_downstream_preregistration.md`.
- Report: `research/taorg_downstream_exit_report.md`.

| Path | Content |
|---|---|
| `manifests/` | Sources (detector, ReID, flag artifact checksums) and per-video SHA-256 of the fetched VidOR videos (dev; eval after the freeze) |
| `dev/pipeline_records.json`, `eval/pipeline_records.json` | Perception runs: commit, timings, counts, **content** hashes of the outputs |
| `dev/calibration.csv.gz`, `dev/gate.json` | Unflagged baseline grids, selection, gate F2/F3 |
| `freeze/frozen.json` | Frozen parameters, flag artifact and θ, thresholds, evaluation sample (commit `0942146`) |
| `eval/outcomes.csv.gz` | Per video × method registered outcomes (240 rows) |
| `eval/method_outputs.csv.gz` | Per video × method × key output frames, for re-scoring |
| `eval/flag_records.csv.gz` | Every V3 missing record: cues, P, flag, **oracle label** |
| `eval/analysis.json` | Registered contrasts, completeness, C3, interaction, flag behaviour: **GO** |
| `eval/posthoc.json` | POST-HOC: relative harm (D1), genuine-vs-flicker event audit (D2) |
| `eval/examples.json`, `plots_eval/` | Predefined examples (timelines only) and figures |
| `dev/outcomes.csv.gz`, `dev/analysis.json` | Development results computed after the freeze (descriptive) |
| `checksums.json` | **Content** SHA-256 (decompressed for `.gz`) of every artifact here |

**Outside git** (`D:/TAORG_runs/downstream/`):
- the videos and extracted frames;
- detections and tracker outputs;
- the detector weights;
- the logs.

**Not redistributed:** no VidOR frames.

**Gzip:** artifacts are written with mtime 0, so they are byte-reproducible.
