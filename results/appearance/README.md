# Appearance-based tracking audit artifacts (`results/appearance/`)

**Tracker-information audit only.** No scene-graph predictor was run, and there are no
relationship results.
- Pre-registration: `research/taorg_appearance_audit_preregistration.md`.
- Report: `research/taorg_appearance_audit_exit_report.md`.

| Path | Content |
|---|---|
| `manifests/sources.json` | Tracker, ReID model (SHA-256, provenance), frame archive, labels, hardware |
| `manifests/frames_*.json` | Per-file SHA-256 of every frame fetched (development; held-back only after the freeze) |
| `dev/run_records.json` | Development run stage at `e7ae883`: alignment, embedding counts, timings, implementation checks, output/cue file hashes |
| `dev/run_records_check4_failed.json` | The run that failed check 4 (before amendment A2); kept |
| `dev/run_records_A3_verify.json` | The re-run after the A3 fix. Its gzip file hashes differ only by the timestamp; the decompressed contents are identical (exit report §7.7) |
| `dev/analysis.json` | Registered development analysis (LOSO): gate G1/G2 = PASS |
| `dev/frames_V*.csv.gz` | One row per missing record: cues, **oracle label**, out-of-fold scores and thresholds |
| `dev/incidents_V*.csv.gz` | Incidents and their oracle labels |
| `dev/examples.json` | Predefined examples (prereg §10) |
| `freeze/frozen_models.pkl`, `freeze/frozen_params.json` | Frozen logistic models, θ, settings, cue constants (commit `85a839c`) |
| `eval/run_records.json`, `eval/analysis.json`, `eval/frames_V*.csv.gz`, `eval/incidents_V*.csv.gz` | Held-back run (at `1ff4387`) and frozen analysis: **decision NARROW** |
| `exploratory_proximity.json` | EXPLORATORY proximity-only control (not registered) |
| `plots_dev/`, `plots_eval/` | Figures 1–3 (both splits); Fig. 4 examples (development) |
| `checksums.json` | SHA-256 of the committed artifacts |

**Heavy state, outside git** (`D:/TAORG_runs/appearance/`):
- frames;
- the BoxMOT venv;
- the ReID weights;
- embedding caches;
- per-run outputs, internals and association-cost logs;
- logs of the failed attempts.

**Redistribution.** Only cropped, annotated example frames are redistributed (Fig. 4), under
CC BY-NC-SA 3.0, with attribution to MOTChallenge.

**Information classes:**
- the `a_*` / `dt` cues and the scores are inference-time;
- `label` columns are the oracle (MOT17 manual annotations) and are used for evaluation only.
