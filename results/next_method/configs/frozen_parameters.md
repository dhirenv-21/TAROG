# Frozen operating points (selection rationale) — generated BEFORE any evaluation run

Rule (pre-registration §4): the maximum pooled mean J over 140 calibration instances (C1–C7 × 5000–5019); ties broken by churn, then index.

| Method | selected | cal. J | cal. churn | runner-up (J) | half-sample pick |
|---|---|---|---|---|---|
| B2 GatedEMA+hyst | gated_ema_hyst#0091 | 0.9029 | 0.0168 | gated_ema_hyst#0051 (0.8938) | gated_ema_hyst#0091 |
| BUF B2+per-flag buffer | b2_hold_buffer#0091 | 0.9235 | 0.0153 | b2_hold_buffer#0167 (0.9120) | b2_hold_buffer#0091 |
| GEO B2+geometric adaptive | b2_hold_geometric#0091 | 0.9224 | 0.0151 | b2_hold_geometric#0167 (0.9079) | b2_hold_geometric#0091 |
| K1-static (prior only) | b2_hold_static#0091 | 0.9233 | 0.0151 | b2_hold_static#0167 (0.9020) | b2_hold_static#0091 |
| K1-pooled (one stratum) | b2_hold_pooled#0091 | 0.9249 | 0.0152 | b2_hold_pooled#0167 (0.8964) | b2_hold_pooled#0091 |
| K1 B2+online KM hold | b2_hold_km#0091 | 0.9233 | 0.0151 | b2_hold_km#0167 (0.9020) | b2_hold_km#0091 |

## Parameters

- **B2 GatedEMA+hyst**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "off_ratio": 0.4703, "score_threshold": 0.4308}`
- **BUF B2+per-flag buffer**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "n_occ": 40.2073, "n_unk": 46.4374, "off_ratio": 0.4703, "score_threshold": 0.4308}`
- **GEO B2+geometric adaptive**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}`
- **K1-static (prior only)**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}`
- **K1-pooled (one stratum)**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}`
- **K1 B2+online KM hold**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}`

## Observations at calibration (not decisions)

- Every B2 extension selected B2's own base draw (#0091); only the non-visible-absence decision differs.
- K1, K1-static, K1-pooled and GEO all selected the same extension draw: τ = 0.376 and **κ = 838** pseudo-runs. A prior this strong outweighs the tens of runs a single video provides, so **pooled tuning effectively disabled online learning**. K1 and K1-static have identical calibration J (0.9233).
- The half-sample control selected the same configurations.

Episode-study reference points (descriptive only): ref_hsmm=hsmm#0116, ref_episode=episode#0176, ref_va_hmm=va_hmm#0086
