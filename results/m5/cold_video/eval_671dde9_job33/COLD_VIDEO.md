# M5 controlled cold-video experiment (P1-9)

H: 750 videos, sha256 `2cac2d6f7a87a56e`, cold condition_id `1eee3839b6cc5c90`. Frames built once from the unfiltered log (GT_H = test positives on H minus seen; population = non-empty GT_H and >= 1 non-H history event).

## Frames

| frame | users | GT_H pairs | GT_H videos | excluded (only-H history) | mean candidates | normal-train OOV | cold-train OOV | label-conflict possible GT_H pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| video_long_view/cold_video_full | 5,936 | 7,536 | 434 | 5 | 7530 | 168 | 172 | 1 (of 1 conflict events) |
| video_long_view/cold_video_h | 5,936 | 7,536 | 434 | 5 | 745 | 168 | 172 | 1 (of 1 conflict events) |
| video_is_like/cold_video_full | 305 | 336 | 132 | 0 | 7532 | 7 | 7 | 0 (of 0 conflict events) |
| video_is_like/cold_video_h | 305 | 336 | 132 | 0 | 745 | 7 | 7 | 0 (of 0 conflict events) |

History buckets (non-H history = the cold model's input, cap 100): video_long_view/cold_video_full: {'0': 0, '1-5': 325, '6-20': 1216, '>20': 4395}; video_is_like/cold_video_full: {'0': 0, '1-5': 21, '6-20': 66, '>20': 218}

## Metrics (mean over evaluated users)

| task | frame | group | model | R@10 | R@20 | R@50 | N@20 | GT unscorable |
|---|---|---|---|---:|---:|---:|---:|---:|
| video_long_view | cold_video_full | cold_condition_baseline | cold_pop | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| video_long_view | cold_video_full | cold_condition_baseline | cold_als | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7,536 |
| video_long_view | cold_video_full | seen_video_reference | seen_pop | 0.0324 | 0.1141 | 0.1632 | 0.0328 | 0 |
| video_long_view | cold_video_full | seen_video_reference | seen_als | 0.0707 | 0.1170 | 0.2266 | 0.0496 | 28 |
| video_long_view | cold_video_full | cold_video_model | cold_threetower | 0.0010 | 0.0041 | 0.0099 | 0.0011 | 0 |
| video_long_view | cold_video_full | seen_video_reference | seen_threetower | 0.0715 | 0.1112 | 0.1945 | 0.0467 | 0 |
| video_long_view | cold_video_full | cold_condition_baseline | random_expected (analytical) [b] | 0.0013 | 0.0027 | 0.0066 | 0.0010 | 0 |
| video_long_view | cold_video_h | cold_condition_baseline | cold_pop (fixed tie-break ranking, all H scores 0) [a] | 0.0088 | 0.0207 | 0.1009 | 0.0059 | 0 |
| video_long_view | cold_video_h | cold_condition_baseline | cold_als | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7,536 |
| video_long_view | cold_video_h | seen_video_reference | seen_pop | 0.2156 | 0.3060 | 0.4560 | 0.1493 | 0 |
| video_long_view | cold_video_h | seen_video_reference | seen_als | 0.3425 | 0.4743 | 0.6567 | 0.2274 | 28 |
| video_long_view | cold_video_h | cold_video_model | cold_threetower | 0.0622 | 0.1045 | 0.1964 | 0.0482 | 0 |
| video_long_view | cold_video_h | seen_video_reference | seen_threetower | 0.2917 | 0.4332 | 0.6687 | 0.2039 | 0 |
| video_long_view | cold_video_h | cold_condition_baseline | random_expected (analytical) [b] | 0.0134 | 0.0269 | 0.0672 | 0.0100 | 0 |
| video_is_like | cold_video_full | cold_condition_baseline | cold_pop | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| video_is_like | cold_video_full | cold_condition_baseline | cold_als | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 336 |
| video_is_like | cold_video_full | cold_condition_baseline | cold_als_like_aux | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 336 |
| video_is_like | cold_video_full | seen_video_reference | seen_pop | 0.0295 | 0.0328 | 0.0836 | 0.0161 | 0 |
| video_is_like | cold_video_full | seen_video_reference | seen_als | 0.0344 | 0.0590 | 0.0951 | 0.0252 | 30 |
| video_is_like | cold_video_full | seen_video_reference | seen_als_like_aux | 0.0656 | 0.1148 | 0.1984 | 0.0435 | 0 |
| video_is_like | cold_video_full | cold_video_model | cold_threetower | 0.0033 | 0.0066 | 0.0098 | 0.0020 | 0 |
| video_is_like | cold_video_full | seen_video_reference | seen_threetower | 0.0459 | 0.0836 | 0.1607 | 0.0355 | 0 |
| video_is_like | cold_video_full | cold_condition_baseline | random_expected (analytical) [b] | 0.0013 | 0.0027 | 0.0066 | 0.0010 | 0 |
| video_is_like | cold_video_h | cold_condition_baseline | cold_pop (fixed tie-break ranking, all H scores 0) [a] | 0.0082 | 0.0082 | 0.0443 | 0.0026 | 0 |
| video_is_like | cold_video_h | cold_condition_baseline | cold_als | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 336 |
| video_is_like | cold_video_h | cold_condition_baseline | cold_als_like_aux | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 336 |
| video_is_like | cold_video_h | seen_video_reference | seen_pop | 0.1749 | 0.2814 | 0.5093 | 0.1270 | 0 |
| video_is_like | cold_video_h | seen_video_reference | seen_als | 0.1525 | 0.2279 | 0.3328 | 0.1093 | 30 |
| video_is_like | cold_video_h | seen_video_reference | seen_als_like_aux | 0.2689 | 0.3869 | 0.5929 | 0.1853 | 0 |
| video_is_like | cold_video_h | cold_video_model | cold_threetower | 0.0377 | 0.0672 | 0.1732 | 0.0345 | 0 |
| video_is_like | cold_video_h | seen_video_reference | seen_threetower | 0.2590 | 0.3902 | 0.5896 | 0.1798 | 0 |
| video_is_like | cold_video_h | cold_condition_baseline | random_expected (analytical) [b] | 0.0134 | 0.0268 | 0.0671 | 0.0097 | 0 |

[a] On the H-only candidate set every cold-condition Pop score is 0 (H has no train interactions), so its ranking is the deterministic tie-break (catalog row = video_id ascending): a fixed arbitrary order, NOT a random ranking.
[b] Closed-form expectation of a uniformly random ranking of each user's own candidates (P1-9); not a sampled model.


## Paired bootstrap differences (a - b), recall@20 / ndcg@20

| task | frame | a - b | metric | diff [95% CI] |
|---|---|---|---|---|
| video_long_view | cold_video_full | cold_threetower - cold_pop | recall@20 | 0.0041 [0.0026, 0.0058] |
| video_long_view | cold_video_full | cold_threetower - cold_pop | ndcg@20 | 0.0011 [0.0007, 0.0016] |
| video_long_view | cold_video_full | cold_threetower - cold_als | recall@20 | 0.0041 [0.0026, 0.0058] |
| video_long_view | cold_video_full | cold_threetower - cold_als | ndcg@20 | 0.0011 [0.0007, 0.0016] |
| video_long_view | cold_video_full | cold_threetower - random_expected | recall@20 | 0.0014 [-0.0000, 0.0032] |
| video_long_view | cold_video_full | cold_threetower - random_expected | ndcg@20 | 0.0001 [-0.0003, 0.0006] |
| video_long_view | cold_video_full | seen_threetower - cold_threetower | recall@20 | 0.1071 [0.0996, 0.1146] |
| video_long_view | cold_video_full | seen_threetower - cold_threetower | ndcg@20 | 0.0455 [0.0421, 0.0491] |
| video_long_view | cold_video_h | cold_threetower - cold_pop | recall@20 | 0.0838 [0.0756, 0.0923] |
| video_long_view | cold_video_h | cold_threetower - cold_pop | ndcg@20 | 0.0423 [0.0382, 0.0464] |
| video_long_view | cold_video_h | cold_threetower - cold_als | recall@20 | 0.1045 [0.0971, 0.1116] |
| video_long_view | cold_video_h | cold_threetower - cold_als | ndcg@20 | 0.0482 [0.0443, 0.0523] |
| video_long_view | cold_video_h | cold_threetower - random_expected | recall@20 | 0.0776 [0.0703, 0.0848] |
| video_long_view | cold_video_h | cold_threetower - random_expected | ndcg@20 | 0.0382 [0.0343, 0.0423] |
| video_long_view | cold_video_h | seen_threetower - cold_threetower | recall@20 | 0.3287 [0.3153, 0.3412] |
| video_long_view | cold_video_h | seen_threetower - cold_threetower | ndcg@20 | 0.1557 [0.1485, 0.1630] |
| video_is_like | cold_video_full | cold_threetower - cold_pop | recall@20 | 0.0066 [0.0000, 0.0164] |
| video_is_like | cold_video_full | cold_threetower - cold_pop | ndcg@20 | 0.0020 [0.0000, 0.0051] |
| video_is_like | cold_video_full | cold_threetower - cold_als | recall@20 | 0.0066 [0.0000, 0.0164] |
| video_is_like | cold_video_full | cold_threetower - cold_als | ndcg@20 | 0.0020 [0.0000, 0.0051] |
| video_is_like | cold_video_full | cold_threetower - cold_als_like_aux | recall@20 | 0.0066 [0.0000, 0.0164] |
| video_is_like | cold_video_full | cold_threetower - cold_als_like_aux | ndcg@20 | 0.0020 [0.0000, 0.0051] |
| video_is_like | cold_video_full | cold_threetower - random_expected | recall@20 | 0.0039 [-0.0027, 0.0137] |
| video_is_like | cold_video_full | cold_threetower - random_expected | ndcg@20 | 0.0010 [-0.0010, 0.0041] |
| video_is_like | cold_video_full | seen_threetower - cold_threetower | recall@20 | 0.0770 [0.0492, 0.1066] |
| video_is_like | cold_video_full | seen_threetower - cold_threetower | ndcg@20 | 0.0335 [0.0198, 0.0505] |
| video_is_like | cold_video_h | cold_threetower - cold_pop | recall@20 | 0.0590 [0.0328, 0.0869] |
| video_is_like | cold_video_h | cold_threetower - cold_pop | ndcg@20 | 0.0320 [0.0167, 0.0502] |
| video_is_like | cold_video_h | cold_threetower - cold_als | recall@20 | 0.0672 [0.0426, 0.0951] |
| video_is_like | cold_video_h | cold_threetower - cold_als | ndcg@20 | 0.0345 [0.0195, 0.0527] |
| video_is_like | cold_video_h | cold_threetower - cold_als_like_aux | recall@20 | 0.0672 [0.0426, 0.0951] |
| video_is_like | cold_video_h | cold_threetower - cold_als_like_aux | ndcg@20 | 0.0345 [0.0195, 0.0527] |
| video_is_like | cold_video_h | cold_threetower - random_expected | recall@20 | 0.0404 [0.0158, 0.0682] |
| video_is_like | cold_video_h | cold_threetower - random_expected | ndcg@20 | 0.0249 [0.0098, 0.0430] |
| video_is_like | cold_video_h | seen_threetower - cold_threetower | recall@20 | 0.3230 [0.2689, 0.3770] |
| video_is_like | cold_video_h | seen_threetower - cold_threetower | ndcg@20 | 0.1453 [0.1179, 0.1759] |

Seen-video reference rows are context only (not an upper bound, not a cold baseline).

