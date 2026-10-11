# Frozen operating points (selection rationale)

Rule (pre-registration §5): the maximum pooled mean J over the 140 calibration instances (C1–C7 × seeds 5000–5019); ties broken by lower mean churn, then lower index.
The same parameters are used in every condition. This table was generated from `configs/frozen_parameters.json` **before any evaluation run**.

| Method | selected | cal. J | cal. churn | runner-up J (gap) | half-sample pick (J) |
|---|---|---|---|---|---|
| No Memory | no_memory#0000 | 0.5313 | 0.0649 | — | no_memory#0000 (0.5313) |
| B1 GatedEMA | gated_ema#0144 | 0.8778 | 0.0268 | 0.8619 (0.0159) | gated_ema#0154 (0.8356) |
| B2 GatedEMA+hyst | gated_ema_hyst#0091 | 0.9029 | 0.0168 | 0.8938 (0.0091) | gated_ema_hyst#0092 (0.8869) |
| B3 VA-HMM | va_hmm#0086 | 0.8915 | 0.0110 | 0.8908 (0.0007) | va_hmm#0150 (0.8908) |
| B4 HSMM | hsmm#0116 | 0.8925 | 0.0126 | 0.8823 (0.0102) | hsmm#0116 (0.8925) |
| B5 GatedEMA+ID | gated_ema_id#0042 | 0.8575 | 0.0169 | 0.8355 (0.0220) | gated_ema_id#0154 (0.8355) |
| P1 Episode | episode#0176 | 0.9286 | 0.0163 | 0.9286 (0.0000) | episode#0176 (0.9286) |
| P2 +identity | episode_id#0101 | 0.9291 | 0.0153 | 0.9241 (0.0050) | episode_id#0056 (0.9218) |
| P3 +transition ctx | episode_ctx#0101 | 0.9303 | 0.0156 | 0.9279 (0.0024) | episode_ctx#0056 (0.9265) |
| P4 +event history | episode_hist#0101 | 0.9302 | 0.0157 | 0.9278 (0.0023) | episode_hist#0056 (0.9263) |

## Selected parameters

- **No Memory**: `{"score_threshold": 0.0}`
- **B1 GatedEMA**: `{"alpha": 0.2221, "decay_occluded": 0.9989, "decay_unknown": 0.0704, "decay_visible": 0.4537, "score_threshold": 0.2223}`
- **B2 GatedEMA+hyst**: `{"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "off_ratio": 0.4703, "score_threshold": 0.4308}`
- **B3 VA-HMM**: `{"false_pos": 0.0842, "p_birth": 0.0006, "p_stay": 0.9835, "prior": 0.2488, "theta_occ": 0.8849, "theta_unk": 0.8466, "theta_vis": 0.1972, "threshold": 0.1272}`
- **B4 HSMM**: `{"false_pos": 0.0695, "off_scale": 17.2474, "off_shape": 1.5211, "on_scale": 2.1508, "on_shape": 1.7355, "prior": 0.7175, "theta_occ": 0.9225, "theta_unk": 0.3252, "theta_vis": 0.1152, "threshold": 0.0822}`
- **B5 GatedEMA+ID**: `{"alpha": 0.1229, "decay_occluded": 0.9985, "decay_unknown": 0.5738, "decay_visible": 0.834, "r_id": 3.6697, "score_threshold": 0.2758, "sigma_id": 0.8129}`
- **P1 Episode**: `{"a_occ": 0.8549, "a_unk": 0.8182, "birth_threshold": 6.0958, "end_threshold": 0.0343, "false_pos": 0.0101, "miss": 0.0113, "mu": 0.0195, "refractory_boost": 4.4955, "refractory_tau": 0.3773}`
- **P2 +identity**: `{"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733}`
- **P3 +transition ctx**: `{"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "kappa_c": 24.3719, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733}`
- **P4 +event history**: `{"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "kappa_c": 24.3719, "kappa_h": 11.2815, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733, "tau_h": 0.6513}`

## Fitted transition statistic E (calibration ground truth only)

| r \ r' | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.350 | 0.044 | 0.008 | 0.044 | 0.040 |
| 1 | 0.270 | 0.000 | 0.036 | 0.071 | 0.034 | 0.036 |
| 2 | 0.021 | 0.021 | 0.000 | 0.331 | 0.050 | 0.023 |
| 3 | 0.041 | 0.047 | 0.320 | 0.000 | 0.031 | 0.033 |
| 4 | 0.014 | 0.036 | 0.005 | 0.058 | 0.000 | 0.304 |
| 5 | 0.025 | 0.035 | 0.033 | 0.050 | 0.279 | 0.000 |
