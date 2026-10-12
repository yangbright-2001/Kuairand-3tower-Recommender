**video / long view** — users 5,936, GT_H pairs 7,536 on 434 H videos

| group | model | Full R@20 [95% CI] | Full R@50 | Full N@10 | H-only R@20 [95% CI] | H-only R@50 | H-only N@10 |
|---|---|---|---:|---:|---|---:|---:|
| cold video model | Cold 3T-U0 (H removed) | 0.0041 [0.0026, 0.0058] | 0.0099 | 0.0003 | 0.1045 [0.0971, 0.1116] | 0.1964 | 0.0371 |
| cold condition baseline | Cold Pop [a] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0207 [0.0173, 0.0241] | 0.1009 | 0.0028 |
| cold condition baseline | Cold ALS (pure) | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| cold condition baseline | Random (analytical) | 0.0027 [0.0027, 0.0027] | 0.0066 | 0.0006 | 0.0269 [0.0269, 0.0269] | 0.0672 | 0.0065 |
| seen video reference | Seen 3T-U0 (reference) | 0.1112 [0.1037, 0.1186] | 0.1945 | 0.0361 | 0.4332 [0.4213, 0.4444] | 0.6687 | 0.1664 |
| seen video reference | Seen Pop (reference) | 0.1141 [0.1063, 0.1215] | 0.1632 | 0.0117 | 0.3060 [0.2942, 0.3166] | 0.4560 | 0.1248 |
| seen video reference | Seen ALS pure (reference) | 0.1170 [0.1096, 0.1246] | 0.2266 | 0.0375 | 0.4743 [0.4631, 0.4857] | 0.6567 | 0.1920 |

**video / like** — users 305, GT_H pairs 336 on 132 H videos

| group | model | Full R@20 [95% CI] | Full R@50 | Full N@10 | H-only R@20 [95% CI] | H-only R@50 | H-only N@10 |
|---|---|---|---:|---:|---|---:|---:|
| cold video model | Cold 3T-U0 (H removed) | 0.0066 [0.0000, 0.0164] | 0.0098 | 0.0011 | 0.0672 [0.0426, 0.0951] | 0.1732 | 0.0266 |
| cold condition baseline | Cold Pop [a] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0082 [0.0000, 0.0180] | 0.0443 | 0.0026 |
| cold condition baseline | Cold ALS (pure) | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| cold condition baseline | Cold ALS (aux) | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |
| cold condition baseline | Random (analytical) | 0.0027 [0.0027, 0.0027] | 0.0066 | 0.0006 | 0.0268 [0.0268, 0.0269] | 0.0671 | 0.0062 |
| seen video reference | Seen 3T-U0 (reference) | 0.0836 [0.0541, 0.1148] | 0.1607 | 0.0261 | 0.3902 [0.3393, 0.4427] | 0.5896 | 0.1458 |
| seen video reference | Seen Pop (reference) | 0.0328 [0.0131, 0.0557] | 0.0836 | 0.0154 | 0.2814 [0.2344, 0.3322] | 0.5093 | 0.0997 |
| seen video reference | Seen ALS pure (reference) | 0.0590 [0.0361, 0.0869] | 0.0951 | 0.0190 | 0.2279 [0.1852, 0.2721] | 0.3328 | 0.0892 |
| seen video reference | Seen ALS aux (reference) | 0.1148 [0.0820, 0.1525] | 0.1984 | 0.0305 | 0.3869 [0.3360, 0.4410] | 0.5929 | 0.1543 |

Paired differences, cold 3T-U0 minus each cold-condition baseline (95% CI; ** = CI excludes 0):

| task | candidates | vs | ΔR@20 | ΔR@50 | ΔN@10 |
|---|---|---|---|---|---|
| video / long view | full catalog | Cold Pop | **+0.0041 [+0.0026, +0.0058]** | **+0.0099 [+0.0077, +0.0123]** | **+0.0003 [+0.0001, +0.0006]** |
| video / long view | full catalog | Cold ALS (pure) | **+0.0041 [+0.0026, +0.0058]** | **+0.0099 [+0.0077, +0.0123]** | **+0.0003 [+0.0001, +0.0006]** |
| video / long view | full catalog | Random (analytical) | +0.0014 [-0.0000, +0.0032] | **+0.0032 [+0.0010, +0.0057]** | **-0.0003 [-0.0005, -0.0000]** |
| video / long view | H only | Cold Pop | **+0.0838 [+0.0756, +0.0923]** | **+0.0955 [+0.0831, +0.1081]** | **+0.0342 [+0.0303, +0.0383]** |
| video / long view | H only | Cold ALS (pure) | **+0.1045 [+0.0971, +0.1116]** | **+0.1964 [+0.1872, +0.2053]** | **+0.0371 [+0.0332, +0.0409]** |
| video / long view | H only | Random (analytical) | **+0.0776 [+0.0703, +0.0848]** | **+0.1293 [+0.1200, +0.1381]** | **+0.0306 [+0.0268, +0.0345]** |
| video / like | full catalog | Cold Pop | +0.0066 [+0.0000, +0.0164] | +0.0098 [+0.0000, +0.0230] | +0.0011 [+0.0000, +0.0033] |
| video / like | full catalog | Cold ALS (pure) | +0.0066 [+0.0000, +0.0164] | +0.0098 [+0.0000, +0.0230] | +0.0011 [+0.0000, +0.0033] |
| video / like | full catalog | Random (analytical) | +0.0039 [-0.0027, +0.0137] | +0.0032 [-0.0066, +0.0163] | +0.0005 [-0.0006, +0.0027] |
| video / like | H only | Cold Pop | **+0.0590 [+0.0328, +0.0869]** | **+0.1290 [+0.0809, +0.1820]** | **+0.0240 [+0.0095, +0.0421]** |
| video / like | H only | Cold ALS (pure) | **+0.0672 [+0.0426, +0.0951]** | **+0.1732 [+0.1306, +0.2191]** | **+0.0266 [+0.0121, +0.0443]** |
| video / like | H only | Random (analytical) | **+0.0404 [+0.0158, +0.0682]** | **+0.1061 [+0.0635, +0.1520]** | **+0.0203 [+0.0058, +0.0381]** |
