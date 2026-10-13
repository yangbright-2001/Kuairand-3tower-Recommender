# Provenance of `results/explore/` (read-only copies)

**What these files are.** Byte-identical copies of the post-hoc exploratory (S9) **validation**
results on branch `explore/s9-opt` (worktree `kuairand-explore`), taken at that branch's final
commit `9a1a6b75858bbc8c083a659fc0f866bd58dd1393`. They are here so the extended report, the
README and their figures can be regenerated from `main`. The explore branch is not merged.

| folder | content | runs (Slurm jobs) | training code SHA | last changed on the branch |
|---|---|---|---|---|
| `wave1/` | seeds 1/2 at lr 1e-3; lr 5e-4, τ 0.03, τ 0.1 at seed 0 | 39–43 | `1051f86` | `9a1a6b7` (terminology note) |
| `wave2/` | `x_u0_repro` (bitwise = formal job 27); lr 5e-4 at seeds 1/2 | 44–46 | `a158e5e` | `9a1a6b7` (terminology note) |
| `wave3_review/` | read-only diagnostics on existing runs (epoch curves, long view vs ALS, no-music gate review) | none | n/a | `9a1a6b7` (terminology note) |
| `wave3/` | no-music ablation at seeds 0/1/2: `x_nomusic_repro` (bitwise = formal job 35), `x_nomusic_s1`, `x_nomusic_s2` | 47–49 | `be5ceb9` | `9a1a6b7` |

Scripts copied from the branch (`scripts/explore/`; identical except that the author's local repo path in the usage docstrings is replaced by `<formal_repo>`):
- `analyze_wave3.py`, plus its helper `analyze_wave1.py` (shared verifier and the test-file guard);
- `plot_wave3.py`.

The Wave 1/2 analysis scripts and diagnostics live on the branch. Training code (`src/`) is
identical to the formal `671dde9` throughout.

**Contents.** All numbers are **validation** metrics on the M3 val masked frames (frame ids
asserted). The analysis scripts refuse `metrics.csv` and `per_user/test_*`. No file here is, or was
selected with, a test result. All runs are post-hoc exploratory: they were designed after the formal
test results were analysed.

**Run artifacts.** Checkpoints, embeddings, per-user files and Slurm logs of all 11 explore runs are
in the submission bundle `05_Exploratory_Experiments.zip`. They are not in git.

**Known content.** The `run_verification.json` files and `wave1/validation_analysis.md` record the
absolute paths used at analysis time (the author's local artifact directories and the GPU server's
`/local-scratch/...` run directories). They are kept unchanged so the copies stay byte-identical.
They contain no credentials, and nothing on `main` reads those paths.
