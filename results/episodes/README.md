# Episode-study artifacts (`results/episodes/`)

Pre-registration: `research/taorg_episode_preregistration.md`. Config:
`configs/experiments/episodes.yaml` (SHA-256 in `configs/config_sha256.txt` and in every
`raw/stage_*.json`).

**Evidence class: controlled synthetic evidence. Not research evidence about real video.**

## Layout

| Path | Content | Written by |
|---|---|---|
| `configs/transition.json` | Fitted transition statistic E and the calibration instances it was fitted on | `fit` |
| `configs/frozen_parameters.json` | Frozen operating point per method (pooled calibration J), top-5 ranking, half-sample control | `select` |
| `configs/episodes.resolved.yaml`, `configs/config_sha256.txt` | Resolved config and hash | every stage |
| `raw/calibration_summary.csv.gz` | Pooled mean of every metric for every configuration (1801 rows), plus per-condition J | `calibrate` |
| `raw/calibration_instances.json` | Instance ids and manifest SHA-256 of the 140 calibration instances | `calibrate` |
| `raw/calibration_parts_sha256.json` | SHA-256 of every heavy per-instance calibration matrix (outside git; see below) | `calibrate` |
| `raw/evaluation.csv.gz` | One row per (condition, seed, variant): all metrics | `evaluate` |
| `raw/evaluation_parts_sha256.json` | SHA-256 of every per-instance evaluation checkpoint (outside git) | `evaluate` |
| `raw/latency.csv.gz` | Per-frame latency rows | `latency` |
| `raw/stage_*.json` | Provenance: run id, git commit, dirty flag, config hash, environment, timings | each stage |
| `aggregated/completeness.json` | Expected / actual / missing / duplicate / failed runs | `analyze` |
| `aggregated/analysis.json` | Registered decisions, ladder, answers Q1–Q6, Holm-adjusted p-values | `analyze` |
| `aggregated/descriptive.csv`, `aggregated/paired.csv` | Descriptive and paired statistics | `analyze` |
| `tables/*.md` | Tables generated from the artifacts | `analyze` |
| `plots/fig1…fig6.{pdf,png}` | Registered figures | `figures` |
| `failures/` | Representative failures (predefined rule) | `failures` |

## Heavy artifacts (outside git)

Per-instance checkpoints are written to `D:/TAORG_runs/episodes/`. The system drive was almost
full. They comprise:
- `calibration/<condition>__s<seed>.npz`: the metric matrix of all 1801 configurations;
- `evaluation/<condition>__s<seed>.json`.

Each file has a `.sha256` sidecar, and the checksums are committed in `raw/*_parts_sha256.json`.
These files are **not needed** to reproduce any table or decision: everything downstream reads
the committed compact artifacts. They can be regenerated bit-for-bit with the commands below,
since all randomness is seeded.

## Reproduction

```bash
python scripts/episodes/run_episodes.py fit calibrate select --workers 6
```

```bash
python scripts/episodes/run_episodes.py evaluate latency analyze figures failures --workers 6
```

Resume is automatic: checksum-valid per-instance parts are skipped. `--smoke` runs the pipeline
on the smoke seeds (9300–9301); those results are never analysed, and the completeness gate
refuses them.
