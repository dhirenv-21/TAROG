# Reproducibility

## Scope
Every number in the paper comes from one of the saved run directories listed in
`provenance.json`. `make_artifacts.py` rebuilds all tables and figures from those files.
All runs are **synthetic**. The code checkout is not under version control, so every run
records `git_commit = unknown`. Commit the code and re-run before relying on exact
provenance.

## Software
* Python 3.10 (runs used 3.10.11); NumPy, PyTorch (runs used 2.12 dev, CPU), PyYAML,
  matplotlib, pytest. See `pyproject.toml`.
* The package is not pip-installed in the original environment, so commands use
  `PYTHONPATH=src` from the project root.

## Data
* Synthetic data are generated deterministically from seeds; no external dataset is needed
  for any reported result.
* Action Genome (needed only for the not-yet-run pilot) is **not** redistributed. Obtain the
  annotations from the official release, accept its terms, and place them under
  `data/action_genome/annotations/`. Charades videos are under a non-commercial licence (AI2).
  Check the AG annotation licence before use.

## Seeds and splits
Predeclared seeds: 42, 123, 456, 789, 2024. Per seed: 20 train / 10 validation / 20 test
synthetic videos (video seeds `seed*10000 + {0..19}`, `+1000+{0..9}`, `+2000+{0..19}`).
Thresholds are selected on validation only. Bootstrap: 10,000 resamples, seed 0.

## Commands (from the project root)
| Result | Command | Saved run |
|---|---|---|
| Tables I, II, III, Fig. 2 | `PYTHONPATH=src python scripts/run_ablation_matrix.py --workers 5` | `outputs/runs/20261027_202753_a7314fe5` |
| Table IV (probes) | `... run_ablation_matrix.py --config configs/experiments/stage3_hid_probe.yaml --workers 5` and `... stage3_hid2_probe.yaml` | `20261028_044818_696b2840`, `20261028_045023_e10fb99f` |
| Table V, Fig. 3 (IA-v1) | `... run_ablation_matrix.py --config configs/experiments/ia_matrix_synthetic.yaml --workers 10` | `20261028_050001_0c54b136` |
| Table VI, Fig. 4 (efficiency) | `PYTHONPATH=src python scripts/run_complexity_benchmark.py` | `outputs/runs/complexity/20261027T202322Z` |
| Table VII (identifiability) | `PYTHONPATH=src python scripts/run_param_recovery.py` | `outputs/runs/param_recovery/20261028T051308Z` |
| Fig. 5 (qualitative) | `PYTHONPATH=src python scripts/make_paper_artifacts.py` | uses the baseline run's fitted parameters |
| All paper tables/figures | `cd ieee_paper && python make_artifacts.py` | — |
| Tests | `PYTHONPATH=src python -m pytest` | 282 passed at the last full run |

Re-running the baseline matrix cell (ID-switch condition, seeds 42 and 123) from its saved
config reproduced every saved metric exactly (maximum absolute difference 0). That check
was made during development and is not re-run here.

## Hardware
CPU only. Matrix runs used 5–10 worker processes on a 24-logical-core Windows 11 machine
(≈6 min for the baseline matrix, ≈23 min for the IA-v1 matrix). Latencies in Table VI are
specific to that machine.

## Known limitations
* No real-data results; the AG loader's annotation format is assumed until checked with
  `scripts/inspect_action_genome.py`.
* IA-v2 has no evaluation run (two attempts were interrupted or cancelled; see the marker
  files in those run directories).
* Memory was not measured in any saved run.
* The diagnostics on parameter identifiability beyond Table VII were exploratory and not
  archived; they are not reported as results.
