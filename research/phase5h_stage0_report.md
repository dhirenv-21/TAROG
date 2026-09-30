# Phase 5H Stage 0 — Report: VidOR ground-truth Hawkes signal probe

# **HAWKES SIGNAL = WEAK-MIXED**

> **Evidence class: real human annotations (VidOR ground truth); no perception, visibility,
> identity or detector involved.** Artifacts: `results/phase5h/stage0/` (`H/`). Pre-registration
> `19e77d5`; code `5d3bfbc`; amendment A1 `06ad683`; run `…phase5h_stage0_20260930_043902_0732b0`
> (commit `06ad683`, clean; config SHA-256 `a21856f8…4d2239`; 2109 s).

## 1. Data audit

- **Downloaded:** VidOR annotations only, from the HuggingFace mirror `shangxd/vidor` (card licence
  CC-BY-SA-4.0).
  - training-annotation.zip: 150,728,902 B, MD5 `85f39fdd81a780bbb9b975cca8f219a2`;
  - validation-annotation.zip: 17,212,872 B, MD5 `96c6870910e8d4fb3836878215432c1f`.

  Both match the published MD5s. They are stored under `data/vidor/`, which is gitignored and not
  redistributed.
- **Video files are not technically required.** The JSONs contain fps, frame_count, per-frame
  trajectories and relation `begin_fid` (inclusive) / `end_fid` (exclusive). No video was
  downloaded.
- **Training:** 7000 videos, 267,210 relations, 50 predicates, 171,382 ordered pairs.
- **TEST (the official validation split):** 835 videos, 30,142 relations, 18,498 pairs.
- **Consistency:** no unknown predicates; 0 relations outside their pair's co-existence span;
  no overlapping or touching same-(pair, predicate) intervals.
- **Imbalance:** `next_to`, `in_front_of`, `watch`, `behind`, `away` and `towards` account for
  most events (e.g. `next_to` has 93,790 onsets+offsets).

## 2. Protocol and deviations

- Everything followed `research/phase5h_stage0_preregistration.md`.
- **Amendment A1 (engineering):**
  - The config key `null:` parsed as a YAML null, which crashed the first run after the TEST
    evaluation. The key was renamed to `null_control`; the seeds are unchanged.
  - The deterministic rerun reproduced every TEST number bit-for-bit (e.g. HK2 −3.78893728885604
    nats/event in both runs).
  - Disclosed in A1: TEST values from the crashed run were seen before the null. The decision rule
    was already frozen and was not changed.
- **Disclosed:** a timing check fitted HK2 on 700 training videos after registration. No
  likelihood was evaluated.
- **Implementation checks (tests):**
  - The fast likelihood matches an eager brute-force reference (direct intensity sums plus
    numerical compensator).
  - Simulation recovers a planted cross-excitation (α̂ > 0.25 for a true 0.6), and the null gain is
    smaller.
  - Extraction rules (censoring, PAIR_BIRTH, risk sets) are tested; the circular shift preserves
    on-time.
  - Full suite: 224 passed.

## 3. Results (TEST, 50,509 events; `H/stage0_tables.md`)

| Model | Held-out LL per event (nats) |
|---|---|
| HK0 (base rates) | −4.3451 |
| DUR (independent duration hazard) | −4.2975 |
| HK1 (within-relation exponential Hawkes + PAIR_BIRTH) | −3.8632 |
| **HK2 (multivariate exponential Hawkes)** | **−3.7889** |

| Quantity | Mean [95% video-bootstrap CI] |
|---|---|
| G_cross = HK2 − HK1 | **+0.0742 [+0.0660, +0.0827]** (per-event density ratio ×1.077) |
| G_self = HK1 − HK0 | +0.4820 [+0.4378, +0.5263] |
| G_dur = HK2 − DUR | +0.5086 [+0.4610, +0.5539] |
| G_base = HK2 − HK0 | +0.5562 [+0.5120, +0.6026] |
| HK1 − DUR (descriptive) | +0.4344 [+0.3875, +0.4819] |

**DEV selection:** HK1 β = 0.1 s⁻¹, λ = 1; HK2 β = 0.1 s⁻¹, λ = 10. **Both chose the smallest β in
the grid**, i.e. the slowest kernel (a memory of about 10 s, in videos with median length 29 s).

**C1 null (19 circular shifts):** G_cross^null ∈ [0.0392, 0.0431], median 0.0404. The observed
0.0742 exceeds all 19 (p = 0.05, the smallest attainable with 19 shifts).

**C4 stability:**

| Quantity | Real | Null (5 shifts) |
|---|---|---|
| Stable cross cells S (present in ≥ 4/5 folds) | 404 (170 in 5/5) | 421, 404, 406, 406, 422 |
| Mean inter-fold Spearman ρ̄ | 0.368 | 0.338–0.349 |

## 4. Criterion-by-criterion decision (`src/taorg/hawkes/decision.py`)

| Criterion | Required | Observed | Met? |
|---|---|---|---|
| C1 | CI_lo(G_cross) > 0 and G_cross > max null | 0.0660 > 0; 0.0742 > 0.0431 | **Yes** |
| C2 | CI_lo(G_dur) > 0 and CI_lo(G_base) > 0 | 0.461, 0.512 | **Yes** |
| C3 | excess = G_cross − median null ≥ 0.10 · G_self | 0.0338 vs 0.0482 (ratio 0.070) | **No** |
| C4 | S_real > max S_null **and** ρ̄_real > max ρ̄_null | 404 vs 422 (no); 0.368 vs 0.349 (yes) | **No** |
| **Label** | C1 holds, but C3 and C4 fail | | **WEAK-MIXED** |

Descriptive only (cannot change the label): C3 would pass at a 0.05 ratio and fail at 0.20.

## 5. Interpretation (observation vs hypothesis)

1. **OBSERVATION:** about 54% of the HK2 − HK1 gain (0.040 of 0.074 nats/event) is reproduced when
   cross-relation timing is destroyed but each (pair, predicate)'s own intervals are kept.
   **HYPOTHESIS:** the multivariate model partly absorbs **pair-level heterogeneity** (pairs that
   carry more relations produce more events of every type). This is a classic confound between
   Hawkes excitation and unobserved heterogeneity. It is consistent with DEV choosing the slowest
   kernel, and with the null producing as many "stable" cross cells as the real data (C4).
2. **OBSERVATION:** a **timing-specific** component remains: the excess of 0.034 nats/event
   exceeds every null. It is real but small, 7% of what within-relation history explains.
3. **OBSERVATION (exploratory, post-hoc reading of the top cells):** the strongest, fold-stable
   cells are semantically coherent **offset-triggering transitions**:
   - OFF(grab) ← ON(hold), α/β = 14.1, 5/5 folds;
   - OFF(hold) ← ON(release), 8.7, 5/5;
   - OFF(release) ← OFF(hold), 6.8, 5/5;
   - OFF(hold) ← ON(throw), 6.8;
   - OFF(ride) ← ON(get_off), 5.7;
   - OFF(behind) ← ON(inside), 3.6;
   - OFF(above) ← ON(beneath), 3.5.

   **HYPOTHESIS:** mutual exclusivity between predicates is expressed as *excitation of the
   other relation's offset*. That is a valid linear-Hawkes representation of what the design had
   framed as inhibition. It is concentrated in contact/action predicates, which are rare relative
   to the dominant spatial predicates, so it contributes little to the per-event average.
4. **OBSERVATION (descriptive):**
   - `ON(towards)` is followed by `ON(away)` within 2–5 s at 0.141 per source event, against a
     baseline of 0.011 (≈ 13×).
   - Many onset pairs are **suppressed at lags < 0.5 s**, e.g. ON(in_front_of) → ON(next_to):
     0.0001 vs a baseline of 0.0087.

   **HYPOTHESIS:** short-lag refractoriness or inhibition exists, which a non-negative (α ≥ 0)
   linear Hawkes model cannot express. That caps HK2's gain.
5. **Annotation caveat (threat 1):** VidOR boundaries are human-segmented. Correlated boundaries
   (e.g. `hold` starting exactly as `grab` ends) may partly reflect annotation convention rather
   than scene dynamics.

## 6. What Stage 0 does and does not establish

- **Established (ground truth only):** relationship onset/offset events in VidOR carry cross-relation
  timing information beyond within-relation history and beyond a pair-heterogeneity-preserving
  null. The effect is small on average and concentrated in a few coherent transitions.
- **Not established:**
  - a broad, stable interaction structure (C4 fails against the heterogeneity null);
  - a practically large cross-relation contribution (C3 fails);
  - any usefulness on perceived (detector) outputs;
  - any visibility/identity benefit.

## 7. Status changes (ledger)

| Claim | Change |
|---|---|
| L29 (multivariate Hawkes beyond independent/exponential controls) | UNTESTED → **PARTIAL** (VidOR GT: beats controls and null; small; not stable at the matrix level) |
| L34 (new): Hawkes cross-excitation in relationship GT is confounded with pair heterogeneity | **SUPPORTED** (≈ 54% of the gain reproduced under the null) |

## 8. Stop

Stage 0 is complete. Not started: visibility-aware Hawkes, detector integration, Phase 5B, paper
finalisation.
