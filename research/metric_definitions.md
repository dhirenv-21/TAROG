# TAORG Research — Metric Definitions

> **Status:** STUB — definitions to be frozen before pilot experiments (Phase 2).
>
> Do NOT interpret pilot or final results using metric definitions that have
> not been reviewed and frozen.  Changing a metric definition after seeing
> results requires documenting the change and rerunning all affected experiments.

## Planned Metrics

### 1. Recall@K / mean Recall@K
- **Definition:** TBD — must match the exact evaluation protocol of the
  baseline benchmark (e.g., Action Genome).
- **Edge cases:** TBD.
- **Reference:** Ji et al., Action Genome.

### 2. Edge Persistence Precision / Recall / F1
- **Definition:** TBD.
- **Positive instance definition:** TBD — what exactly counts as a
  "correctly persisted" edge?
- **Edge cases:**
  - Edge never observed.
  - Edge observed once then never again.
  - Edge persisted past true termination.

### 3. Relationship Churn Rate
- **Definition:** TBD.
- **Denominator:** TBD.
- **Edge cases:**
  - Continuous observation (churn = 0).
  - Alternating every frame.
  - Short video with one edge.

### 4. Termination Delay
- **Definition:** TBD (in frames and/or seconds).
- **Edge cases:**
  - Method never terminates the edge.
  - Method terminates correctly at the exact frame.
  - Method terminates early (before ground truth termination).

### 5. False Persistence
- **Definition:** TBD — proportion of edges kept alive past true termination.
- **Relation to Edge Persistence F1:** TBD.

### 6. Calibration / ECE
- **Definition:** TBD — only meaningful if the method produces a
  calibrated alive probability.
- **Applicable methods:** V-HPED, possibly HMM.

### 7. Per-frame Temporal-Module Latency
- **Definition:** Wall-clock time for the temporal method's `step()`
  call, excluding base model inference.
- **Reporting:** Mean, median, p95, p99 across frames.

---

## Freezing Protocol
Before the pilot (Phase 3), each metric must have:
1. Exact mathematical formula.
2. Defined edge cases and expected values.
3. Unit tests covering edge cases.
4. Documented in this file with "FROZEN" status.

After freezing, any change requires:
1. Documenting the change and rationale here.
2. Rerunning all affected experiments.
3. Recording the change in Memory.md.
