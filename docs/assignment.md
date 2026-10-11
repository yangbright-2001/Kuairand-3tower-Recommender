# Research Engineer Take-Home Assessment

## Overview

This role places a high value on learning speed: the ability to get up to speed quickly in an unfamiliar domain, absorb the technical fundamentals, apply sound engineering principles to deliver results, and communicate what you learned and built clearly and concisely. The take-home is designed to test exactly that.

## The Assignment

Build a small-scale, multi-action, three-tower recommender on the KuaiRand-Pure dataset ([website](https://kuairand.com/), [download](https://zenodo.org/records/10439422)) that retrieves both videos and their background-music IDs for two engagement actions, long view and like.

That gives four retrieval tasks: video/long view, video/like, music/long view, and music/like.

Please use PyTorch.

The core pieces are:

- **Dataset audit:** Report entity counts, action distributions, music-ID coverage (especially how many videos share each music ID), and data-quality issues, along with any filtering you apply.

- **Temporal split and leakage control:** Split the standard logs chronologically into roughly 70/15/15. User histories should include only events strictly before the prediction time, capped at the most recent 100. Don't use the supplied monthly video-statistics aggregates as features, since they leak future engagement. Compute popularity from training data instead.

- **Baselines:** Popularity and implicit ALS for both user-video and user-music.

- **Three-tower model:**
  - A user tower using ID, usable metadata, and interaction history.
  - A music tower.
  - A video tower that composes video ID, the music tower's output, an author embedding, and video metadata.
  - 128-d base embeddings with separate 64-d projections per target and action, trained with a contrastive retrieval objective. Handle duplicate positives and false negatives in the batch, and explain how you balance the two actions.

- **Evaluation:**
  - Full-catalog Recall and NDCG at 10, 20, and 50 for all four tasks, with counts of evaluated users and of excluded or out-of-catalog positives.
  - Bootstrap confidence intervals for the main comparison.
  - A breakdown by history length (1–5, 6–20, and over 20 events).
  - A controlled cold-video experiment, where selected videos are held out of training and encoded at test time from metadata, music, and author.
  - At least one controlled ablation, such as removing history or removing the music input to the video tower.

- **Retrieval interface:** `retrieve(user_id, target, action, k)` backed by exported embeddings and ID mappings, with a check that exported scores match direct model scoring.

- **Engineering:** Reproducible commands, configs, a CPU smoke test, and focused tests for leakage, masking, and metric correctness.

- **Report:** About 5–8 pages covering design, results, error analysis, limitations, and what you'd do next.

**Correctness of data handling and evaluation matters more to us than beating ALS. A well-explained negative result is perfectly fine.**

## Resources

- We highly encourage using AI agents. An OpenRouter API key with $1,000 in credit has been provided separately for that purpose. The key must not be stored in this repository.
- You are also welcome to use your own personal AI agent subscriptions.
- An 8×A100 node is provided for training and testing. Access details are provided separately by Junfei.

## Timeline & Submission

**Deadline:** End of day Monday, October 12, 2026 (Pacific Time).

Please submit all artifacts produced during the take-home. That includes:

- Code repository
- Configs
- Checkpoints
- Exported embeddings
- Results files
- Report
- Coding-agent session logs

Please ask your coding agents how to export their session logs; they should be locally cached in your home folder.

You can upload the artifacts to a Google Drive folder or anywhere else that can be quickly retrieved. All submitted artifacts will be used in the evaluation.

## Follow-up Interview

Please provide availability from Tuesday, October 13, 2026 onward for a follow-up interview of approximately 60 minutes, where you will walk the team through your work.