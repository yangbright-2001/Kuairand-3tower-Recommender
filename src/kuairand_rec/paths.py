"""Canonical project paths. Raw data is read-only."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Override with KUAIRAND_RAW_DIR on the server (e.g. /local-scratch/liang/...).
RAW_DIR = Path(
    os.environ.get(
        "KUAIRAND_RAW_DIR", PROJECT_ROOT / "data" / "raw" / "KuaiRand-Pure" / "data"
    )
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Files allowed in the main pipeline.
STANDARD_LOGS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
USER_FEATURES = "user_features_pure.csv"
VIDEO_FEATURES_BASIC = "video_features_basic_pure.csv"

# Excluded from the main pipeline:
# - video_features_statistic_pure.csv: future-leaking aggregates, never used as features.
# - log_random_4_22_to_5_08_pure.csv: out of scope for main experiments.
FORBIDDEN_FEATURE_FILES = ("video_features_statistic_pure.csv",)
EXCLUDED_LOGS = ("log_random_4_22_to_5_08_pure.csv",)
