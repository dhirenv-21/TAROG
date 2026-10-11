# Frozen operating points (pooled calibration J; committed before evaluation)

| Method | config | cal. J | cal. churn | half-sample pick | params |
|---|---|---|---|---|---|
| B2 GatedEMA+hyst | gated_ema_hyst#0091 | 0.9029 | 0.0168 | gated_ema_hyst#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "off_ratio": 0.4703, "score_threshold": 0.4308}` |
| BUF B2+per-flag buffer | b2_hold_buffer#0091 | 0.9235 | 0.0153 | b2_hold_buffer#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "n_occ": 40.2073, "n_unk": 46.4374, "off_ratio": 0.4703, "score_threshold": 0.4308}` |
| GEO B2+geometric adaptive | b2_hold_geometric#0091 | 0.9224 | 0.0151 | b2_hold_geometric#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}` |
| K1-static (prior only) | b2_hold_static#0091 | 0.9233 | 0.0151 | b2_hold_static#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}` |
| K1-pooled (one stratum) | b2_hold_pooled#0091 | 0.9249 | 0.0152 | b2_hold_pooled#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}` |
| K1 B2+online KM hold | b2_hold_km#0091 | 0.9233 | 0.0151 | b2_hold_km#0091 | `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "kappa": 837.8401, "off_ratio": 0.4703, "score_threshold": 0.4308, "tau": 0.3765}` |