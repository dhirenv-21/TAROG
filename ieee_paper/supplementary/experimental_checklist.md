# Experimental Checklist — Missing or Incomplete Evidence

Status legend: **DONE** (saved artifact exists), **PARTIAL**, **NOT RUN**.
Nothing marked NOT RUN appears as a result in the paper.

| # | Experiment | Status | Evidence / blocker | How to run |
|---|---|---|---|---|
| A | Functional correctness (propositions, likelihood agreement, nesting) | DONE (tests) | `tests/unit/test_vhped.py`; 282 tests passed at the last full run | `PYTHONPATH=src python -m pytest` |
| B | Eager vs lazy equivalence | DONE (synthetic) | `tests/property/test_lazy_equivalence.py`; benchmark 0 mismatches at 16 grid points | same |
| C | Persistence through occlusion | DONE (synthetic) | run `20261027_202753_a7314fe5` | `python scripts/run_ablation_matrix.py --workers 5` |
| D | Synthetic occlusion / disappearance | DONE (synthetic) | same run | same |
| E | Tracker noise / identity switches | PARTIAL | synthetic swaps only (runs `…_202753…`, `…_044818…`, `…_045023…`, `…_050001…`); **no real tracker** | requires tracker outputs |
| F | Recovery after temporary disappearance | PARTIAL | covered by persistence F1 in occlusion gaps; no separate recovery-latency metric | add metric |
| G | False persistence / expiration | DONE (synthetic) | termination delay, FPR in the same runs | — |
| H | Runtime | DONE (one machine, Python) | `outputs/runs/complexity/20261027T202322Z` | `python scripts/run_complexity_benchmark.py` |
| I | Memory | **NOT RUN** | benchmark supports it (`--memory`) but no saved run includes it | `python scripts/run_complexity_benchmark.py --memory` |
| I2 | Adversarial alternating workload | **NOT RUN** | implemented (`--pattern alternating`) | `python scripts/run_complexity_benchmark.py --pattern alternating --fractions 0.5` |
| J | Baselines | DONE (synthetic) | no memory, fixed window, EMA, gated EMA, HMM | — |
| J2 | Comparison with recent VidSGG networks (WSGG, RTAformer, TCDSG) | **NOT RUN** | not reproduced; post-hoc layer would need their predictions | needs code/checkpoints |
| K | Hyper-parameter sensitivity | PARTIAL | threshold sweep curves only; no sweep over hand-set parameters or training hyper-parameters | extend config |
| L | Real videos (Action Genome) | **NOT RUN** | annotations not available locally; licence to be checked | `python scripts/inspect_action_genome.py --root data/action_genome` then `python scripts/run_pilot_ag.py --workers 4` |
| L2 | Real base-model predictions (STTran/DSG-DETR/TEMPURA) | **NOT RUN** | needs checkpoints + Charades frames + GPU | export predictions to the JSONL adapter format (`src/taorg/adapters/precomputed.py`) |
| M | Ablations | DONE (synthetic) | no-visibility, VT-HMM, self-only, hand-set, lazy-exact | — |
| N | Identity-aware extension IA-v2 (learned calibration) | **NOT RUN** | implemented and unit-tested; the evaluation run was interrupted by a session restart and a re-run was cancelled (`outputs/runs/20261028_053040_deae1e15/INTERRUPTED.txt`, `outputs/runs/20261028_162146_fc7b5ffd/STOPPED.txt`) | `python scripts/run_ablation_matrix.py --config configs/experiments/ia2_matrix_synthetic.yaml --workers 10` |
| O | Parameter identifiability | PARTIAL | run `param_recovery/20261028T051308Z`; cause of the μ/α trade-off not established | `python scripts/run_param_recovery.py` |
| P | Statistical tests | PARTIAL | paired bootstrap CIs; no multiple-comparison correction | — |

## Highest-value next experiments (in order)
1. **L**, the Action Genome pilot: it decides whether the visibility gating seen in the synthetic study holds on real annotations.
2. **N** with real tracker scores: run a tracker (e.g. ByteTrack) on AG, Charades or VidOR, define κ from association quality on training videos, and run the IA-v2 matrix on real streams.
3. **L2**: run V-HPED as post-processing on one base model's predictions and report Recall@K plus the temporal metrics.
4. **I / I2**: measure memory, and lazy behavior on real touch patterns.
5. **K**: sensitivity to the threshold-selection criterion and to hand-set parameters.
