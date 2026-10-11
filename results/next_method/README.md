# Next-method study artifacts (`results/next_method/`)

- Pre-registration: `research/taorg_next_method_preregistration.md`
- Config: `configs/experiments/next_method.yaml` (SHA-256 in `configs/config_sha256.txt` and in
  every `raw/stage_*.json`)

**Evidence class: controlled synthetic evidence (the episode-study world). Not evidence about
real video.** The development audit outputs in `audit/` are development data (seeds 7000–7019),
kept separate from the registered study.

| Path | Content |
|---|---|
| `audit/` | Failure audit and development pilot (dev seeds only; `research/taorg_next_method_failure_audit.md`) |
| `configs/prior.json` | Prior hazards fitted on calibration **input** streams (no labels), with the list of streams used |
| `configs/frozen_parameters.json`, `configs/frozen_parameters.md` | Frozen operating points (pooled calibration J), the half-sample control, and the episode-study reference points |
| `raw/calibration_summary.csv.gz` | Pooled and per-condition calibration means for all 1200 configurations |
| `raw/calibration_instances.json`, `raw/calibration_parts_sha256.json` | Calibration bookkeeping and checksums of the heavy parts |
| `raw/evaluation.csv.gz`, `raw/evaluation_parts_sha256.json` | Evaluation rows (seeds 8000–8049) and part checksums |
| `raw/latency.csv.gz`, `raw/stage_*.json` | Latency and per-stage provenance |
| `aggregated/` | Completeness, paired statistics, gate decision, hold horizons |
| `tables/`, `plots/`, `failures/` | Generated tables, figures and predefined failure cases |

**Heavy per-instance checkpoints** are in `D:/TAORG_runs/next_method/`, with SHA-256 sidecars and
committed manifests. They are not needed to reproduce any table.

## Reproduction

```bash
python scripts/next_method/run_next_method.py fit calibrate select --workers 4
```

```bash
python scripts/next_method/run_next_method.py evaluate latency analyze figures failures --workers 4
```

Development audit and pilot (dev seeds only):

```bash
python scripts/next_method/audit_dev.py 4
```

```bash
python scripts/next_method/dev_pilot.py 4
```

Note that `dev_pilot.py` now imports the final module name. The exact pilot prototype is
snapshotted in `scripts/next_method/dev_snapshots/`.
