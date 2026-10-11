# Latency (mean ms/frame, C1, one machine)

| Method | ms/frame | × B2 |
|---|---|---|
| B2 GatedEMA+hyst | 0.0169 | 1.00 |
| BUF B2+per-flag buffer | 0.0293 | 1.73 |
| GEO B2+geometric adaptive | 0.0339 | 2.01 |
| K1-static (prior only) | 0.0553 | 3.27 |
| K1-pooled (one stratum) | 0.0617 | 3.65 |
| K1 B2+online KM hold | 0.0627 | 3.71 |