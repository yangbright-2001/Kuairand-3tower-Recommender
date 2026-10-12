# Local reproduction of the formal results (final compliance audit, 2026-10-12)

All reruns used the working tree at HEAD `671dde9` (+ uncommitted docs/scripts only; no source
module that produces data, baselines, frames or metrics was changed), wrote to `/tmp/kr_audit/`
only, and were compared against the committed / formal files. Nothing in `results/` or
`artifacts/<run>` was overwritten.

| step | command (output redirected to /tmp) | compared with | result |
|---|---|---|---|
| M1 dataset audit | `python -m kuairand_rec.audit.run --out /tmp/...` (120 s) | `results/audit/audit.json` | identical except `meta` (runtime); 30/30 consistency checks |
| M2 data pipeline | `python -m kuairand_rec.data.pipeline --config <data.yaml with output_dir/report_dir in /tmp>` | `data/processed/*` | all 11 parquet + `vocab_meta.json` byte-identical (md5); `meta.json` identical except `git_sha` and output paths; 10/10 checks |
| M3 baselines | `python -m kuairand_rec.baselines.run --config <baselines.yaml, output_dir in /tmp>` (604 s) | `results/baselines/` | `metrics.csv`, `als_tuning.csv` (val columns), `frames_summary.json`, `als_selected`, all 46 per-user parquet files identical |
| M5 main analysis | `python -m kuairand_rec.experiments.analysis --config <m5_analysis.yaml, output_dir in /tmp> --run abl_nohist=artifacts/abl_nohist/671dde9_seed0_job34 --run abl_nomusic=artifacts/abl_nomusic/671dde9_seed0_job35` | `results/m5/main_formal/` (server job 37) | 744 + 888 rows matched; means / diffs identical; CI bounds within 3.1e-16; `ci_excludes_0` identical everywhere; `bucket_counts.csv` and user sets byte-identical. One `p_two_sided` differs by 0.002 (= one replicate at exactly 0, float rounding; descriptive column) |
| M5 cold-video evaluation | `python -m kuairand_rec.experiments.cold_video evaluate --config configs/cold_video.yaml --cold_run artifacts/cold_u0/671dde9_seed0_job33 --seen_run artifacts/u0/9218a76_seed0_job27 --out_dir /tmp/...` | `results/m5/cold_video/eval_671dde9_job33/` | `cold_metrics.csv`, `cold_bootstrap_models.csv`, `cold_bootstrap_diffs.csv` byte-identical; 12/12 removal checks; seen Pop/ALS reproduce M3 bitwise |
| Re-score of the 5 formal exports | `python scripts/audit_final_compliance.py rescore` | formal per-user parquet (scored on Linux) | at most 2 users differ per run and frame; largest mean change 5.4e-5 (abl_nomusic, *val* music LV, 1 user); every test-frame mean change <= 1.1e-7. Evidence: `rescore.json` |
| CPU smoke (real data) | `RUN_ROOT=/tmp/... python -m kuairand_rec.models.train --config configs/threetower_smoke.yaml --max_steps 5` | parity checks 1-4 | 6 min 45 s; parity ok, max abs diff 0.0, 0 top-50 mismatches on 2,048 stratified users |
| Independent checks | `python scripts/audit_final_compliance.py checks` | raw CSV / pandas recomputation | 16/16 pass (`checks.json`) |
| Test suite | `pytest` | | 259 passed (`pytest.log`) |
| Report claims | `python scripts/check_report_claims.py` | formal CSV / JSON | 321 numbers parsed from REPORT.md (283 table cells in Tables 1-3, the bucket, cold-user, §1.5 population and B1 tables, plus 38 anchored prose numbers) match the formal CSV/JSON, incl. significance stars; remaining text is manual-review scope (listed in `report_claims.json`). Negative controls: 6/6 injected edits detected |
