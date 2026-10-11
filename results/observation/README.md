# Observation-interface audit artifacts (`results/observation/`)

**Detector/tracker-level only.** No scene-graph predictor was run, and no relationship-performance
result exists here.

| Path | Content | Information class |
|---|---|---|
| `manifests/sources.json` | Data source (MOT17Labels.zip: URL, bytes, SHA-256, licence), detector sets, tracker package and version, split rule | provenance |
| `dev_audit/absence_frames.csv.gz` | One row per absence frame of each confirmed ByteTrack track, on the dev sequences 02/09/11 × FRCNN/SDP: inference-time reason, tracker state, time since detection, **oracle category** | reason/state = automatic + heuristic; category = oracle (benchmark manual annotations) |
| `dev_audit/summary.json` | Confusion, precision/recall of each reason, coverage, the U1–U3 verdict, phenomenon counts | derived |
| `dev_audit/separability.json`, `dev_audit/hold_vs_time.csv` | **Exploratory** single-signal AUCs and composition vs elapsed time (defined after the verdict) | derived |
| `dev_audit/examples.json` | Representative successes and failures (predefined rule in `scripts/observation/figures.py`) | derived |
| `plots/fig1…fig4` | Confusion, signal AUCs, composition vs time, example timelines | figures |
| `checksums.json` | SHA-256 of every artifact here, and of the source archive | provenance |

**Not redistributed.** No MOT17 frames were downloaded or redistributed. The source archive stays
outside git at `D:/TAORG_runs/observation/MOT17Labels.zip`.

**The untouched evaluation sequences (04, 05, 10, 13) were never processed.** The pre-stated
criteria returned NOT USABLE, so no comparison was run.

## Reproduction

```bash
python scripts/observation/dev_audit.py
```

```bash
python scripts/observation/signal_separability.py
```

```bash
python scripts/observation/figures.py
```
