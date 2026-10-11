# Latency (mean ms/frame, C1 evaluation instances, one machine)

| Method | ms/frame | × GatedEMA |
|---|---|---|
| No Memory | 0.0009 | 0.12 |
| B1 GatedEMA | 0.0077 | 1.00 |
| B2 GatedEMA+hyst | 0.0131 | 1.69 |
| B3 VA-HMM | 0.0115 | 1.49 |
| B4 HSMM | 0.1120 | 14.53 |
| B5 GatedEMA+ID | 0.0204 | 2.65 |
| P1 Episode | 0.0159 | 2.07 |
| P2 +identity | 0.0274 | 3.56 |
| P3 +transition ctx | 0.0303 | 3.93 |
| P4 +event history | 0.0330 | 4.28 |