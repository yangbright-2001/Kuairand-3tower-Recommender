# Wave 3 gate review: E2 music fusion (post-hoc exploratory; read-only; no training)

**Status: review only. No Wave 3 GPU experiment has been executed. E-1 (no-music seeds 1/2) awaits
user approval; gated E2 and E-2 (stop-gradient) are not implemented.**

Evidence: `scripts/explore/diagnose_longview.py` -> this directory (val-only; `selection.json`,
val per-user files, train logs; an untrained model for init scales). Code read: `models/towers.py`
(MusicTower, VideoTower, `ThreeTowerModel.encode_items/_music_forward`), `models/train.py`,
`configs/threetower_abl_nomusic.yaml`. `src/` is identical between 671dde9 (no-music job 35) and
a158e5e.

## A. What the formal no-music ablation removes
`model.use_music: false` removes only the 128-d `music_base` input of the video tower's first Linear
(`d_in` 266 -> 138). Consequences:
- the music tower and music heads are still trained, but **only by the two music losses**: video
  losses (and, through `v_base` history tokens, the user tower's video-side signal) no longer
  back-propagate into the music tower;
- video candidates and history tokens are built from video id + author + metadata only;
- the first layer has 128 fewer inputs -> different parameter init and different RNG consumption,
  so job 35 is not a paired counterpart of U0 seed 0 (it shares only the data order).

## B. What the no-music observation supports
Observed (val, one no-music seed vs three U0 seeds, `nomusic_vs_u0_epochwise.json`):
- video_lv: no-music higher at 10/10, 10/10, 9/9 epochs vs U0 seeds 0/1/2 (mean +.007 to +.010) and
  above the per-epoch max of the 3 U0 seeds at 9/9 epochs; music_lv: 9/10, 9/10, 9/9 (envelope 8/9).
  At the selected epoch: video_lv .1334 vs U0 seed range .1241-.1267; music_lv .1352 vs .1203-.1344.
- likes: mixed (envelope 5/9 on both).
- train loss (`train_log.jsonl`, epochs 3 and 6): no-music is lower on all four tasks, including the
  two music tasks, than every U0 seed.
- init scale (`init_fusion_scale.json`): `music_base` is a LayerNorm output (mean norm 11.3) while
  ID embeddings start at norm 0.16; at init `music_base` carries 20% of the first-layer
  pre-activation variance vs 0.7% each for video id / author.
Supported hypothesis: "with the current fusion, adding the music-tower output to the video tower lowers
long-view val retrieval", robust to the U0 seed spread on video_lv and largely on music_lv.
Not yet evidenced (all compatible with the data):
- (i) representation interference: the near-ID music path (112 shared ids only) competes with
  video id;
- (ii) gradient interference: video losses pull the shared music tower away from what the music
  tasks need (the lower music-task train loss without the input is consistent with this);
- (iii) optimisation / scale: a 70x larger input dominates early training;
- (iv) no-music seed luck: a single seed. Its long-view lead is about 2-4x the U0 seed spread on
  video_lv, but no-music has no seed replicates yet.
Selection trade-off (`selection_vs_task_best.csv`) exists in every run: the long-view tasks peak 2-5
epochs after the selected epoch in most runs (val left on the table 0-.011; 0 where the selected epoch
is already the long-view best, e.g. tau 0.1), likes peak at epochs 3-6. But the
no-music lead holds at every epoch, so it is not a selection artifact. The trade-off is a separate,
protocol-level limitation.

## C/D. The planned 128-d gate before the Linear
`Linear(W)([.., g ⊙ m, ..]) = W[:, music] diag(g) m + ...`: any (W, g) equals (W diag(g), 1), so the
gate is a pure reparameterisation and adds no expressiveness. With `gate_init = 1` it is
forward-identical to U0. It can still change training through the optimiser alone:
- Adam gives g its own per-coordinate step;
- weight decay acts on W but not on g;
- g's gradient enters the global clip norm, and clipping is active on 100% of steps.

Any effect would therefore measure optimiser dynamics, not a fusion mechanism, and it would not
separate hypotheses (i)-(iii). Verdict: **not worth implementing**.

## E. Minimal controlled alternatives (no protocol change)
1. **No-music seed replicates** (config only: `threetower_abl_nomusic.yaml` + seed 1/2). Tests (iv)
   and makes B a 3-vs-3 seed comparison. Zero code risk.
2. **Stop-gradient fusion** (`music_base.detach()` inside the video tower; one config switch, no new
   parameter). The video tower still composes the music-tower output, as the assignment requires, but
   video losses no longer update the music tower. Decision rule:
   - detach ≈ no-music on long-view -> (ii) gradient interference;
   - detach ≈ U0 -> the input itself, i.e. (i)/(iii).

   Deviates from D1-1 ("video losses back-propagate into the music tower"), so it is exploratory
   only. Needs code, tests, CPU smoke and parity before any GPU run.

## Gate decision (time: ~21:00 PT, deadline end of day, >= 60 min reserved for submission)
- Gated E2 (C): **No-Go** (reparameterisation; uninterpretable).
- E-1 no-music s1/s2: **Go** (fits the window). Realistic timeline:

  | step | time |
  |---|---|
  | config + registry/test | 15 min |
  | commit / push / preflight | 10 min |
  | 2 parallel jobs | ~25 min |
  | backup (rsync ~0.5 MB/s, 2 x ~115 MB) | ~10 min |
  | val analysis | 10 min |
  | **total** | **~70 min** |

- E-2 stop-gradient: **No-Go in this window**. Realistic total is ~2 h:

  | step | time |
  |---|---|
  | code | 15 min |
  | tests | 25 min |
  | full pytest | 3 min |
  | CPU real-data smoke + parity | ~15 min |
  | push / preflight | 10 min |
  | 3-seed GPU run | 25 min |
  | backup | 15 min |
  | analysis | 15 min |

  That leaves no margin before the 60-min reserve. Record it as the next experiment.
