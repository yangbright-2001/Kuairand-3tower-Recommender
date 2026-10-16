Formal **test** split, full catalog, items seen before T_test excluded. Values are means over users; 95% CIs come from a paired bootstrap over 1,000 user resamples. R@K = Recall@K, N@K = NDCG@K. U0 = three-tower model (user ID + history); U1 = U0 + 21 undated user-snapshot fields. Pure ALS = ALS on the task's action; Aux ALS = like ALS + 0.3 × long-view signal (like tasks only). Absolute test R@20 by history-length bucket: the number of events fed to the model at T_test (capped at 100).

| task | bucket | users | Pop | Pure ALS | Aux ALS | U0 | U1 | U0 no-history | U0 no-music |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| video / long view | 1-5 | 1,162 | 0.1243 | 0.1285 | – | 0.1569 | 0.1919 | 0.1387 | 0.1765 |
| video / long view | 6-20 | 4,053 | 0.0951 | 0.1417 | – | 0.1358 | 0.1531 | 0.1181 | 0.1531 |
| video / long view | >20 | 13,537 | 0.0725 | 0.1188 | – | 0.1016 | 0.1142 | 0.0916 | 0.1082 |
| video / like | 1-5 | 113 | 0.1389 | 0.0199 | 0.0352 | 0.1382 | 0.1362 | 0.0929 | 0.1527 |
| video / like | 6-20 | 360 | 0.0727 | 0.0512 | 0.1331 | 0.1128 | 0.1081 | 0.0872 | 0.1197 |
| video / like | >20 | 1,451 | 0.0661 | 0.0593 | 0.0979 | 0.0824 | 0.0765 | 0.0889 | 0.0893 |
| music / long view | 1-5 | 1,141 | 0.1326 | 0.1273 | – | 0.1647 | 0.2055 | 0.1402 | 0.1795 |
| music / long view | 6-20 | 4,011 | 0.1007 | 0.1514 | – | 0.1392 | 0.1633 | 0.1236 | 0.1547 |
| music / long view | >20 | 13,443 | 0.0771 | 0.1213 | – | 0.1052 | 0.1166 | 0.0967 | 0.1115 |
| music / like | 1-5 | 109 | 0.1442 | 0.0217 | 0.0525 | 0.1434 | 0.1244 | 0.1011 | 0.1296 |
| music / like | 6-20 | 357 | 0.0737 | 0.0705 | 0.1533 | 0.1395 | 0.1099 | 0.0832 | 0.1168 |
| music / like | >20 | 1,429 | 0.0668 | 0.0584 | 0.0974 | 0.0897 | 0.0777 | 0.0845 | 0.0856 |
