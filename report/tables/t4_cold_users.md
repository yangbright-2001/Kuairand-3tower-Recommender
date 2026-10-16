Formal **test** split, full catalog, items seen before T_test excluded. Values are means over users; 95% CIs come from a paired bootstrap over 1,000 user resamples. R@K = Recall@K, N@K = NDCG@K. U0 = three-tower model (user ID + history); U1 = U0 + 21 undated user-snapshot fields. Pure ALS = ALS on the task's action; Aux ALS = like ALS + 0.3 × long-view signal (like tasks only). Cold users = users with test positives but no event before T_test (a separate frame, excluded from the main population). Absolute test R@20 [95% CI].

| task | users | Pop | U0 | U1 | U0 no-history | U0 no-music |
|---|---:|---:|---:|---:|---:|---:|
| video / long view | 437 | 0.1417 [0.1174, 0.1683] | 0.0393 [0.0275, 0.0549] | 0.0638 [0.0470, 0.0821] | 0.1470 [0.1221, 0.1735] | 0.0457 [0.0314, 0.0617] |
| video / like | 68 | 0.2221 [0.1353, 0.3187] | 0.0478 [0.0101, 0.0920] | 0.0648 [0.0219, 0.1167] | 0.1514 [0.0809, 0.2337] | 0.0140 [0.0041, 0.0258] |
| music / long view | 434 | 0.1480 [0.1204, 0.1758] | 0.0408 [0.0274, 0.0540] | 0.0659 [0.0488, 0.0869] | 0.1584 [0.1306, 0.1873] | 0.0435 [0.0299, 0.0571] |
| music / like | 67 | 0.2254 [0.1383, 0.3267] | 0.0130 [0.0033, 0.0248] | 0.0678 [0.0262, 0.1240] | 0.1161 [0.0565, 0.1861] | 0.0093 [0.0017, 0.0182] |
