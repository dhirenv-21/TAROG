# Frozen operating points (pooled calibration J)

| Method | config | cal. J | cal. churn | params |
|---|---|---|---|---|
| No Memory | no_memory#0000 | 0.531 | 0.0649 | {"score_threshold": 0.0} |
| B1 GatedEMA | gated_ema#0144 | 0.878 | 0.0268 | {"alpha": 0.2221, "decay_occluded": 0.9989, "decay_unknown": 0.0704, "decay_visible": 0.4537, "score_threshold": 0.2223} |
| B2 GatedEMA+hyst | gated_ema_hyst#0091 | 0.903 | 0.0168 | {"alpha": 0.256, "decay_occluded": 0.9594, "decay_unknown": 0.4485, "decay_visible": 0.7225, "off_ratio": 0.4703, "score_threshold": 0.4308} |
| B3 VA-HMM | va_hmm#0086 | 0.892 | 0.0110 | {"false_pos": 0.0842, "p_birth": 0.0006, "p_stay": 0.9835, "prior": 0.2488, "theta_occ": 0.8849, "theta_unk": 0.8466, "theta_vis": 0.1972, "threshold": 0.1272} |
| B4 HSMM | hsmm#0116 | 0.893 | 0.0126 | {"false_pos": 0.0695, "off_scale": 17.2474, "off_shape": 1.5211, "on_scale": 2.1508, "on_shape": 1.7355, "prior": 0.7175, "theta_occ": 0.9225, "theta_unk": 0.3252, "theta_vis": 0.1152, "threshold": 0.0822} |
| B5 GatedEMA+ID | gated_ema_id#0042 | 0.858 | 0.0169 | {"alpha": 0.1229, "decay_occluded": 0.9985, "decay_unknown": 0.5738, "decay_visible": 0.834, "r_id": 3.6697, "score_threshold": 0.2758, "sigma_id": 0.8129} |
| P1 Episode | episode#0176 | 0.929 | 0.0163 | {"a_occ": 0.8549, "a_unk": 0.8182, "birth_threshold": 6.0958, "end_threshold": 0.0343, "false_pos": 0.0101, "miss": 0.0113, "mu": 0.0195, "refractory_boost": 4.4955, "refractory_tau": 0.3773} |
| P2 +identity | episode_id#0101 | 0.929 | 0.0153 | {"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733} |
| P3 +transition ctx | episode_ctx#0101 | 0.930 | 0.0156 | {"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "kappa_c": 24.3719, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733} |
| P4 +event history | episode_hist#0101 | 0.930 | 0.0157 | {"a_occ": 0.8826, "a_unk": 0.4103, "birth_threshold": 5.2805, "end_threshold": 0.1848, "false_pos": 0.0094, "kappa_c": 24.3719, "kappa_h": 11.2815, "miss": 0.0404, "mu": 0.0138, "r_id": 1.4284, "refractory_boost": 11.5081, "refractory_tau": 1.5028, "sigma_id": 0.733, "tau_h": 0.6513} |