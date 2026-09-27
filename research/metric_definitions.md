# TAORG Research — Metric Definitions

> **Status:** FROZEN for Phase 2–3 pilot experiments.
>
> Do NOT interpret pilot or final results using metric definitions that have
> not been reviewed and frozen.  Changing a metric definition after seeing
> results requires documenting the change and rerunning all affected experiments.
>
> Frozen: 2026-09-28

## 1. Recall@K / mean Recall@K
- **Status:** DEFERRED — requires dataset-specific evaluation protocol.
- **Definition:** TBD — must match the exact evaluation protocol of the
  baseline benchmark (e.g., Action Genome).
- **Reference:** Ji et al., Action Genome.

---

## 2. Edge Persistence Precision / Recall / F1
**Status: FROZEN**

### Scope
Evaluated over **corrupted intervals only** — frames where an edge's ground-truth
state is ALIVE but observations were synthetically suppressed (occlusion injection).

### Definitions
For each edge `e` at each frame `t` within a corruption interval:

- **Ground truth `GT(e,t)`:** 1 if the edge is alive in the uncorrupted sequence at frame `t`, else 0.
- **Prediction `P(e,t)`:** 1 if the temporal method outputs edge `e` at frame `t`, else 0.

Per-frame classification:
- **TP:** `GT(e,t) = 1` AND `P(e,t) = 1` — correctly persisted through corruption
- **FP:** `GT(e,t) = 0` AND `P(e,t) = 1` — falsely persisted (should not occur within
  corruption intervals where GT is alive by construction, but possible if corruption
  intervals overlap with genuine termination)
- **FN:** `GT(e,t) = 1` AND `P(e,t) = 0` — failed to persist through corruption

### Aggregation
- **Micro-averaged** across all edges and all corrupted frames.
- Precision = TP / (TP + FP), or 0.0 if TP + FP = 0
- Recall = TP / (TP + FN), or 0.0 if TP + FN = 0
- F1 = 2 * P * R / (P + R), or 0.0 if P + R = 0

### Edge cases
| Case | Expected |
|------|----------|
| No corruption intervals | F1 = 0.0 (undefined, return 0) |
| Perfect persistence | F1 = 1.0 |
| No edges persisted | Recall = 0.0, F1 = 0.0 |
| Edge never observed pre-corruption | Excluded from evaluation |

---

## 3. Relationship Churn Rate
**Status: FROZEN**

### Definition
A **flip** occurs when the predicted state of an edge changes between consecutive frames:
- alive → dead (predicted at `t`, not predicted at `t+1`)
- dead → alive (not predicted at `t`, predicted at `t+1`)

### Counting rules
- **Initial appearance** (first frame where edge is predicted): does NOT count as a flip.
- **Final disappearance** (last frame of sequence): does NOT count as a flip.
- Only **consecutive frame pairs** within the evaluation window are considered.

### Formula
```
churn_rate(e) = num_flips(e) / (num_evaluation_frames(e) - 1)
```

Where `num_evaluation_frames(e)` is the number of frames between the first and last
frame where the edge appears in the ground truth (inclusive). The denominator is the
number of consecutive frame pairs.

### Aggregation
- Per-edge churn rate, then **mean** across edges.
- Lower is better (0 = perfectly stable).

### Edge cases
| Case | Expected |
|------|----------|
| Single-frame edge | churn = 0.0 (no consecutive pairs) |
| Constant prediction | churn = 0.0 |
| Alternating every frame | churn = 1.0 |
| Edge never predicted | churn = 0.0 (no flips) |

---

## 4. Termination Delay
**Status: FROZEN**

### Definition
For each edge with a known ground-truth termination frame `t_end`:

```
termination_delay(e) = t_drop(e) - t_end
```

Where:
- `t_end` = last frame where GT edge is alive (inclusive)
- `t_drop` = first frame after `t_end` where the method does NOT output edge `e`
- If the method never drops edge `e` before the sequence ends:
  `t_drop = sequence_length`, and the result is **flagged as censored**

### Properties
- **Zero delay is valid:** method correctly drops at `t_end + 1`, delay = 1 frame
  (one frame of false persistence). Delay = 0 means drop at exactly `t_end`, which
  means the method dropped while GT was still alive (early termination, should be
  flagged separately).
- **Negative delay** (early termination): recorded and flagged, not clipped.

### Aggregation
- **Mean** and **median** across edges with known termination.
- Censored edges (never dropped) are reported separately as a count/fraction.
- For mean computation, censored edges use `sequence_length - t_end` as the delay
  value but are flagged.

### Edge cases
| Case | Expected |
|------|----------|
| Immediate correct drop at t_end+1 | delay = 1 |
| Never drops | delay = seq_len - t_end (censored) |
| Drops before GT termination | delay < 0 (early, flagged) |
| No GT termination in sequence | Edge excluded from this metric |

---

## 5. False Persistence Rate
**Status: FROZEN**

### Definition
For each edge with a known ground-truth termination frame `t_end`:

```
false_persistence_rate(e) = (frames where P(e,t)=1 AND t > t_end) / (seq_len - t_end)
```

### Properties
- 0.0 = perfect (method never falsely persists after termination)
- 1.0 = method persists for every frame after true termination
- Only evaluated for edges with known GT termination within the sequence

### Aggregation
- Per-edge rate, then **mean** across edges.

### Edge cases
| Case | Expected |
|------|----------|
| Immediate correct drop | 0.0 |
| Never drops | 1.0 |
| Sequence ends at termination | Excluded (denominator = 0) |

---

## 6. Calibration / ECE
**Status:** DEFERRED — only meaningful if the method produces a calibrated alive
probability. Will be implemented when V-HPED or HMM produces probability outputs.

---

## 7. Per-frame Temporal-Module Latency
**Status: FROZEN**

### Definition
Wall-clock time for the temporal method's `step()` call, excluding base model inference.

### Reporting
- Mean, median, p95, p99 across frames within a video.
- Aggregate: mean of per-video means across videos.

### Implementation
Use `time.perf_counter()` for high-resolution timing.

---

## Freezing Protocol
Before the pilot (Phase 3), each metric must have:
1. Exact mathematical formula. ✅
2. Defined edge cases and expected values. ✅
3. Unit tests covering edge cases. (Phase 2 implementation)
4. Documented in this file with "FROZEN" status. ✅

After freezing, any change requires:
1. Documenting the change and rationale here.
2. Rerunning all affected experiments.
3. Recording the change in Memory.md.
