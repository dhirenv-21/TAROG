# Decision matrix: T = P1 Episode vs registered comparators

Cell entries: W (win), L (loss), · (neutral); route J/harm; failed guards in brackets.

| Condition | role | B1 GatedEMA | B2 GatedEMA+hyst | B3 VA-HMM | B4 HSMM | B5 GatedEMA+ID | cell |
|---|---|---|---|---|---|---|---|
| C1_gaps | ni | · ΔJ 0.017 [0.011, 0.024] | W J ΔJ 0.143 [0.131, 0.154] | · [churn_noninferior] ΔJ 0.029 [0.024, 0.034] | · [churn_noninferior] ΔJ 0.023 [0.018, 0.028] | — | **neutral** |
| C2_hidden_termination | benefit | W J ΔJ 0.036 [0.029, 0.044] | L (std_mean_s) ΔJ 0.006 [-0.002, 0.015] | · [churn_noninferior] ΔJ 0.027 [0.022, 0.033] | · [churn_noninferior] ΔJ 0.024 [0.019, 0.029] | — | **loss** |
| C3_misleading_post | benefit | W J ΔJ 0.044 [0.035, 0.053] | · ΔJ 0.011 [-0.005, 0.027] | · [churn_noninferior] ΔJ 0.037 [0.028, 0.047] | · [churn_noninferior] ΔJ 0.017 [0.011, 0.023] | — | **neutral** |
| C4_recurrence | ni | W J ΔJ 0.026 [0.021, 0.031] | W J ΔJ 0.021 [0.017, 0.025] | · [churn_noninferior] ΔJ 0.026 [0.023, 0.030] | W J ΔJ 0.025 [0.021, 0.028] | — | **neutral** |
| C5_boundary_flicker | benefit | W J ΔJ 0.189 [0.181, 0.197] | L [J_noninferior,churn_noninferior] (J,frr) ΔJ -0.062 [-0.070, -0.054] | · [gap_recall_noninferior] ΔJ 0.069 [0.061, 0.077] | · [gap_recall_noninferior] ΔJ 0.081 [0.074, 0.089] | — | **loss** |
| C6_id_switch | benefit | L (ghost_time_s) ΔJ 0.063 [0.051, 0.075] | L (ghost_time_s) ΔJ 0.055 [0.043, 0.068] | · [churn_noninferior] ΔJ 0.022 [0.018, 0.026] | L (ghost_time_s) ΔJ 0.055 [0.046, 0.064] | L (ghost_time_s) ΔJ 0.058 [0.045, 0.070] | **loss** |
| C7_motion | ni | W J ΔJ 0.034 [0.028, 0.041] | W J ΔJ 0.022 [0.016, 0.028] | · [churn_noninferior] ΔJ 0.027 [0.023, 0.031] | · [churn_noninferior] ΔJ 0.026 [0.022, 0.029] | W J ΔJ 0.022 [0.019, 0.025] | **neutral** |
| S0_mixed | transfer | W J ΔJ 0.153 [0.143, 0.163] | L [J_noninferior] (J) ΔJ -0.024 [-0.034, -0.014] | · [gap_recall_noninferior] ΔJ 0.092 [0.082, 0.103] | · [gap_recall_noninferior] ΔJ 0.079 [0.071, 0.088] | — | **loss** |
| S1_mixed_flagnoise | transfer | W J ΔJ 0.104 [0.096, 0.112] | L [J_noninferior,gap_recall_noninferior] (J) ΔJ -0.038 [-0.045, -0.032] | · [gap_recall_noninferior,churn_noninferior] ΔJ 0.038 [0.031, 0.044] | · [gap_recall_noninferior] ΔJ 0.044 [0.038, 0.049] | — | **loss** |
| S2_mixed_unknown | transfer | W J ΔJ 0.291 [0.275, 0.305] | W J ΔJ 0.127 [0.110, 0.144] | W J ΔJ 0.079 [0.070, 0.088] | W J ΔJ 0.113 [0.098, 0.127] | — | **win** |
| S3_mixed_bursty | transfer | W J ΔJ 0.121 [0.113, 0.130] | L [J_noninferior,gap_recall_noninferior] (J) ΔJ -0.025 [-0.033, -0.018] | · [gap_recall_noninferior] ΔJ 0.057 [0.050, 0.064] | · [gap_recall_noninferior] ΔJ 0.065 [0.057, 0.073] | — | **loss** |

**Study decision: STOP**

GO checks: {"benefit_wins": false, "relapse_wins": false, "no_calibrated_losses": false, "transfer_losses": false, "latency": true, "half_control_not_stop": false}

STOP checks: {"benefit_wins_le_max": true, "losses_ge_wins": true, "simple_baseline_not_beaten": true}

Half-sample control decision: STOP (T = episode)